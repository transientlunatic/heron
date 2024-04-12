from pycbc.waveform import get_td_waveform
from elk.waveform import Timeseries
import torch
import scipy.interpolate as interp
import lalsimulation
import lal

from heron.models import Model

from lal import cached_detector_by_prefix, TimeDelayFromEarthCenter, LIGOTimeGPS


def generate_waveform(q, M, distance, apx, times):
    unaligned = {}
    m1 = M / (1 + q)
    m2 = M / (1 + 1 / q)
    assert ((m1 + m2) - M) < 1e-4
    delta_t = float(times[1] - times[0])
    unaligned["hp"], unaligned["hc"] = get_td_waveform(
        approximant=apx,
        mass1=m1,
        mass2=m2,
        spin1z=0,
        distance=distance,
        delta_t=delta_t,
        f_lower=20,
    )
    waveform_a = unaligned
    # idx = (waveform_a['hp'].sample_times >= float(times[0])) & (waveform_a['hp'].sample_times <= float(times[-1]))
    # print(times[:10], waveform_a['hp'].sample_times[idx][:10])
    # times = waveform_a['hp'].sample_times[idx]
    return waveform_a["hp"].sample_times, waveform_a["hp"], waveform_a["hp"]


class IMRPhenomPv2(Model):
    def __init__(self, device="cpu"):
        self.device = device

    def interpolate(self, x_old, y_old, x_new):
        """
        Convenience funtion to avoid repeated code
        """
        interpolator = interp.interp1d(x_old, y_old)
        return interpolator(x_new)

    def time_domain_waveform(self, p, times=None):
        params = {
            "total_mass": 20,
            "mass_ratio": 1.0,
            "inclination": 0,
            "distance": 1000,
        }

        epoch = p["gpstime"]
        times = torch.linspace(
            -p["before"],
            p["after"],
            int(p["sample rate"] * (p["before"] + p["after"])),
        )
        params.update(p)
        p = params
        m1 = p["total_mass"] / (1 + p["mass_ratio"]) * lal.MSUN_SI
        m2 = p["total_mass"] / (1 + 1 / p["mass_ratio"]) * lal.MSUN_SI
        params["distance"] = 1E6 * params["distance"] * lal.PC_SI
        dt = float(times[1] - times[0]) #float(1/(p["sample rate"]))
        approximant = lalsimulation.GetApproximantFromString("IMRPhenomPv2")

        args = {
            "m1": m1,
            "m2": m2,
            "S1x": 0.,
            "S1y": 0.,
            "S1z": 0.,
            "S2x": 0.,
            "S2y": 0.,
            "S2z": 0.,
            "distance": p["distance"],
            "inclination": float(p["inclination"]),
            "phi ref": 0.,
            "longAscNodes": 0.,
            "eccentricity": 0.,
            "meanPerAno": 0.,
            "delta T": dt,
            "f_min": 50.0,
            "f_ref": 50.0,
            "params": lal.CreateDict(),
            "approximant": approximant,
        }

        hp, hx = lalsimulation.SimInspiralChooseTDWaveform(
            *list(args.values())
        )
        sample_times = (
            hp.epoch.gpsNanoSeconds * 1e-9
            + hp.epoch.gpsSeconds
            + torch.linspace(
                0,
                (len(hp.data.data)) * hp.deltaT,
                len(hp.data.data),
                dtype=torch.float64,
            )
        )

        idx = (sample_times >= float(times[0])) & (
            sample_times <= float(times[-1]) + dt / 2
        )
        hp_data = torch.zeros(len(times))
        hp_data[: sum(idx)] = torch.tensor(hp.data.data[idx], dtype=torch.float64)

        hx_data = torch.zeros(len(times))
        hx_data[: sum(idx)] = torch.tensor(hx.data.data[idx], dtype=torch.float64)

        if "ra" in p.keys():
            ra, dec, psi, gpstime = (
                p["ra"],
                p["dec"],
                torch.tensor(p["psi"]),
                p["gpstime"],
            )
            detector = cached_detector_by_prefix[p["detector"]]
            response = self._get_antenna_response(
                detector, ra, dec, float(psi), gpstime
            )
            dt = TimeDelayFromEarthCenter(
                detector.location, ra, dec, LIGOTimeGPS(gpstime)
            )
            waveform_mean = hp_data * response.plus * torch.cos(
                psi
            ) + hx_data * response.cross * torch.sin(psi)

        times = torch.linspace(
            -p["before"],
            p["after"],
            int(p["sample rate"] * (p["before"] + p["after"])),
            device=self.device
        ) + p["gpstime"]
            
        detection = Timeseries(
            data=torch.tensor(waveform_mean, device=self.device),
            times=times,
            detector=p["detector"],
        )
        detection.variance = torch.zeros(len(detection.times), device=self.device)
        detection.covariance = torch.zeros(
            (len(detection.times), len(detection.times)), device=self.device
        )
        return detection
