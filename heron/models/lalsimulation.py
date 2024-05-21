import logging

import numpy as array_library
import torch
from scipy.interpolate import CubicSpline

# Astropy to handle units sanely
from astropy import units as u

# Import LAL tools
import lalsimulation
import lal
from lal import cached_detector_by_prefix, TimeDelayFromEarthCenter, LIGOTimeGPS

# Import heron types
from ..types import Waveform, WaveformDict, PSD
from . import WaveformApproximant, PSDApproximant


class LALSimulationApproximant(WaveformApproximant):
    """
    This is the base class for LALSimulation-based approximants.
    """

    def __init__(self):
        self._cache_key = {}
        self._args = {
            "m1": 10 * u.solMass,
            "m2": 10 * u.solMass,
            "S1x": 0.0,
            "S1y": 0.0,
            "S1z": 0.0,
            "S2x": 0.0,
            "S2y": 0.0,
            "S2z": 0.0,
            "distance": 10 * u.Mpc,
            "inclination": 0,
            "phi ref": 0.0 * u.Hertz,
            "longAscNodes": 0.0,
            "eccentricity": 0.0,
            "meanPerAno": 0.0,
            "delta T": 1 / (4096.0 * u.Hertz),
            "f_min": 50.0 * u.Hertz,
            "f_ref": 50.0 * u.Hertz,
            "params": lal.CreateDict(),
            "approximant": None,
        }
        self.allowed_parameters = list(self._args.keys())

        self.supported_converstions = {
            "mass_ratio",
            "total_mass",
            "luminosity_distance",
        }

        self.logger = logger = logging.getLogger(
            "heron.models.LALSimulationApproximant"
        )

    def _convert_units(self, args):

        units = {
            "mass": u.kilogram,
            "distance": u.meter,
            "frequency": u.Hertz,
            "time": u.second,
        }

        mappings = {
            "m1": "mass",
            "m2": "mass",
            "distance": "distance",
            "delta T": "time",
            "f_min": "frequency",
            "f_ref": "frequency",
            "phi ref": "frequency",
        }

        for name, argument in args.items():
            if isinstance(argument, u.quantity.Quantity):
                args[name] = argument.to_value(units[mappings[name]])

        return args

    @property
    def args(self):
        """
        Provide the arguments list converted to appropriate units.
        """
        args = {}
        args.update(self._args)

        # Convert total mass and mass ratio to the component masses
        args = self._convert(args)
        args = self._convert_units(args)
        # Remove sky and extrinsic parameters
        for par in ("ra", "dec", "phase", "psi", "theta_jn"):
            if par in args:
                args.pop(par)

        for key in list(args.keys()):
            if not key in self.allowed_parameters:
                args.pop(key)
        return args

    def time_domain(self, parameters, times=None):
        """
        Retrieve a time domain waveform for a given set of parameters.
        """
        self._args.update(parameters)
        if not (self.args == self._cache_key):
            self.logger.info(f"Generating new waveform at {self.args}")
            self._cache_key = self.args.copy()

            hp, hx = lalsimulation.SimInspiralChooseTDWaveform(
                *list(self.args.values())
            )
            if not isinstance(times, type(None)):
                # If we provide times then we'll need to interpolate the waveform generated by
                # lalsimulation in order to evaluate it at the same points.
                times_wf = (
                    array_library.arange(len(hp.data.data)) * hp.deltaT + hp.epoch
                )
                spl_hp = CubicSpline(times_wf, hp.data.data)
                spl_hx = CubicSpline(times_wf, hx.data.data)
                hp_data = spl_hp(times)
                hx_data = spl_hx(times)
                hp_ts = Waveform(data=hp_data, times=times)
                hx_ts = Waveform(data=hx_data, times=times)
            elif ("time" in parameters):
                t = parameters["time"]
                times_wf = (
                    array_library.arange(len(hp.data.data)) * hp.deltaT + hp.epoch
                )

                times = array_library.linspace(
                    t["lower"], t["upper"], t["number"]
                    )
                spl_hp = CubicSpline(times_wf, hp.data.data)
                spl_hx = CubicSpline(times_wf, hx.data.data)
                hp_data = spl_hp(times)
                hx_data = spl_hx(times)
                hp_ts = Waveform(data=hp_data, times=times)
                hx_ts = Waveform(data=hx_data, times=times)
            else:
                hp_data = hp.data.data
                hx_data = hx.data.data

                hp_ts = Waveform(data=hp_data, dt=hp.deltaT, t0=hp.epoch)
                hx_ts = Waveform(data=hx_data, dt=hx.deltaT, t0=hx.epoch)

            self._cache = WaveformDict(parameters=parameters, plus=hp_ts, cross=hx_ts)

        return self._cache


class IMRPhenomPv2(LALSimulationApproximant):
    def __init__(self):
        super().__init__()
        self._args["approximant"] = lalsimulation.GetApproximantFromString(
            "IMRPhenomPv2"
        )


class SEOBNRv3(LALSimulationApproximant):
    def __init__(self):
        super().__init__()
        self._args["approximant"] = lalsimulation.GetApproximantFromString("SEOBNRv3")


class IMRPhenomPv2_FakeUncertainty(IMRPhenomPv2):
    def time_domain(self, parameters, times=None):
        waveform_dict = super().time_domain(parameters, times)
        covariance = torch.eye(len(waveform_dict["plus"].times)) * 1e-24
        for wave in waveform_dict.waveforms.values():
            # Artificially add a covariance function to each of these
            wave.covariance = covariance
        return waveform_dict
