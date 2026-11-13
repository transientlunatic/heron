"""
LALSimulation-based reference waveform approximants.

These are used to generate training data for surrogate models.
Requires lalsuite (optional dependency).
"""

import logging

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from astropy import units as u

from ..types import Waveform, WaveformDict
from . import WaveformApproximant

try:
    import lalsimulation
    import lal

    HAS_LAL = True
except ImportError:
    HAS_LAL = False

logger = logging.getLogger("heron.models.lalsimulation")


def _require_lal():
    if not HAS_LAL:
        raise ImportError(
            "lalsuite is required for LALSimulation waveforms. "
            "Install with: pip install lalsuite"
        )


PEAK_ALIGNMENTS = ("sample", "interpolated")


def interpolated_peak_time(times, hp, hx, half_width=8):
    """Time of the maximum of ``hp**2 + hx**2``, resolved below one sample.

    Fits a cubic spline to ``log(hp**2 + hx**2)`` over ``half_width``
    samples either side of the largest sample and returns the root of its
    derivative that lies within one sample of it. Verified against the same
    IMRPhenomD/XAS waveforms generated at 8x the sample rate: agreement is
    <=0.01 us at 4096 Hz, versus 1-5 us bias for a three-point log-parabola.

    Falls back to the largest sample's time if it sits too close to either
    end of the array to fit a window (no sub-sample refinement possible).
    """
    times = np.asarray(times, dtype=np.float64)
    hp = np.asarray(hp, dtype=np.float64)
    hx = np.asarray(hx, dtype=np.float64)
    a2 = hp**2 + hx**2
    i = int(np.argmax(a2))
    lo, hi = i - half_width, i + half_width + 1
    if lo < 0 or hi > len(a2) or a2[i] <= 0:
        return float(times[i])
    window = a2[lo:hi]
    if np.any(window <= 0):
        return float(times[i])
    dt = times[1] - times[0]
    spline = CubicSpline(times[lo:hi], np.log(window))
    roots = spline.derivative().roots(extrapolate=False)
    roots = roots[np.abs(roots - times[i]) <= dt]
    if len(roots) == 0:
        return float(times[i])
    return float(roots[np.argmax(spline(roots))])


class LALSimulationApproximant(WaveformApproximant):
    """Base class for LALSimulation-based approximants.

    Parameters
    ----------
    peak_alignment : {"sample", "interpolated"}
        Time origin of the returned waveform. ``"sample"`` (default, LAL's
        own convention) keeps LAL's epoch, which for IMRPhenomD/XAS puts
        t=0 on the *sample* nearest the peak of ``hp**2 + hx**2``; the true
        peak then sits up to +-dt/2 from t=0, drifting with the parameters
        and jumping by one sample wherever it crosses dt/2 (see CLAUDE.md,
        LAL archaeology). ``"interpolated"`` shifts the time axis so the
        sub-sample-interpolated peak is exactly at t=0
        (:func:`interpolated_peak_time`), making the time origin a smooth
        function of the parameters.
    """

    def __init__(self, peak_alignment="sample"):
        _require_lal()
        if peak_alignment not in PEAK_ALIGNMENTS:
            raise ValueError(
                f"peak_alignment must be one of {PEAK_ALIGNMENTS}, "
                f"got {peak_alignment!r}"
            )
        self.peak_alignment = peak_alignment
        self._cache_key = {}
        self._args = {
            "m1": None,
            "m2": None,
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
            "f_min": 20.0 * u.Hertz,
            "f_ref": 20.0 * u.Hertz,
            "params": lal.CreateDict(),
            "approximant": None,
        }
        self.allowed_parameters = list(self._args.keys())

        self.supported_converstions = {
            "mass_ratio",
            "total_mass",
            "luminosity_distance",
        }

        self.logger = logging.getLogger(
            "heron.models.LALSimulationApproximant"
        )

    def __getstate__(self):
        """Drop the un-picklable SWIG state so the approximant (and anything
        holding it -- surrogates, mean functions, likelihoods) can be sent to
        a multiprocessing worker.

        ``_args["params"]`` is a ``lal.Dict`` (SWIG-wrapped, not picklable),
        and ``_cache``/``_cache_key`` hold SWIG-backed waveform data and a copy
        of that dict. All are cheap to rebuild, so we serialise everything else
        and reconstruct them in ``__setstate__``. This is what unblocks
        ``n_pool`` process parallelism in nessai/bilby, where the whole
        likelihood object is pickled to each worker.
        """
        state = self.__dict__.copy()
        args = dict(state.get("_args", {}))
        args["params"] = None  # lal.Dict -- rebuilt on load
        state["_args"] = args
        state["_cache_key"] = {}
        state.pop("_cache", None)  # SWIG-backed WaveformDict -- regenerated lazily
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if HAS_LAL and self._args.get("params") is None:
            self._args["params"] = lal.CreateDict()

    def _convert_units(self, args):
        default_units = {
            "mass": u.solMass,
            "distance": u.megaparsec,
            "frequency": u.Hertz,
            "time": u.second,
        }

        units = {
            "mass": u.kilogram,
            "distance": u.meter,
            "frequency": u.Hertz,
            "time": u.second,
        }

        mappings = {
            "m1": "mass",
            "m2": "mass",
            "luminosity_distance": "distance",
            "distance": "distance",
            "delta T": "time",
            "f_min": "frequency",
            "f_ref": "frequency",
            "phi ref": "frequency",
        }

        for name, argument in args.items():
            if isinstance(argument, u.quantity.Quantity) and name in mappings:
                args[name] = float(argument.to_value(units[mappings[name]]))
            elif name in mappings and argument:
                args[name] = (argument * default_units[mappings[name]]).to_value(
                    units[mappings[name]]
                )

        return args

    @property
    def args(self):
        args = {}
        args.update(self._args)
        args = self._convert_units(args)
        args = self._convert(args)
        for par in ("ra", "dec", "phase", "psi", "theta_jn"):
            if par in args:
                args.pop(par)

        for key in list(args.keys()):
            if key not in self.allowed_parameters:
                args.pop(key)
        return args

    def time_domain(self, parameters, times=None):
        """Generate a time-domain waveform for given parameters."""
        epoch = parameters.get("gpstime", parameters.get("epoch", 0))
        self._args.update(parameters)
        if not (self._args == self._cache_key):
            self._cache_key = self.args.copy()

            hp, hx = lalsimulation.SimInspiralChooseTDWaveform(
                *list(self.args.values())
            )

            # Native LAL time axis, relative to LAL's epoch, optionally
            # re-referenced so the interpolated peak is at t=0.
            native_times = (
                np.arange(len(hp.data.data)) * hp.deltaT + hp.epoch.ns() / 1e9
            )
            if getattr(self, "peak_alignment", "sample") == "interpolated":
                native_times = native_times - interpolated_peak_time(
                    native_times, hp.data.data, hx.data.data
                )

            if times is not None:
                times_wf = native_times + epoch

                spl_hp = CubicSpline(times_wf, hp.data.data, extrapolate=False)
                spl_hx = CubicSpline(times_wf, hx.data.data, extrapolate=False)

                hp_data = np.nan_to_num(spl_hp(times))
                hx_data = np.nan_to_num(spl_hx(times))
                hp_ts = Waveform(data=hp_data, times=np.asarray(times))
                hx_ts = Waveform(data=hx_data, times=np.asarray(times))

            elif "time" in parameters:
                t = parameters["time"]
                times_wf = native_times + epoch

                times_arr = np.linspace(t["lower"], t["upper"], t["number"])

                spl_hp = CubicSpline(times_wf, hp.data.data)
                spl_hx = CubicSpline(times_wf, hx.data.data)
                hp_data = spl_hp(times_arr)
                hx_data = spl_hx(times_arr)
                hp_ts = Waveform(data=hp_data, times=times_arr)
                hx_ts = Waveform(data=hx_data, times=times_arr)
                parameters.pop("time")

            else:
                hp_data = hp.data.data
                hx_data = hx.data.data
                times_arr = native_times
                hp_ts = Waveform(data=hp_data, times=times_arr, dt=hp.deltaT, t0=float(times_arr[0]) + epoch)
                hx_ts = Waveform(data=hx_data, times=times_arr, dt=hx.deltaT, t0=float(times_arr[0]) + epoch)

            self._cache = WaveformDict(parameters=parameters, plus=hp_ts, cross=hx_ts)
        return self._cache


class IMRPhenomPv2(LALSimulationApproximant):
    def __init__(self, peak_alignment="sample"):
        super().__init__(peak_alignment=peak_alignment)
        self._args["approximant"] = lalsimulation.GetApproximantFromString(
            "IMRPhenomPv2"
        )


class SEOBNRv3(LALSimulationApproximant):
    def __init__(self, peak_alignment="sample"):
        super().__init__(peak_alignment=peak_alignment)
        self._args["approximant"] = lalsimulation.GetApproximantFromString("SEOBNRv3")


class SEOBNRv4(LALSimulationApproximant):
    def __init__(self, peak_alignment="sample"):
        super().__init__(peak_alignment=peak_alignment)
        self._args["approximant"] = lalsimulation.GetApproximantFromString("SEOBNRv4")


class IMRPhenomD(LALSimulationApproximant):
    def __init__(self, peak_alignment="sample"):
        super().__init__(peak_alignment=peak_alignment)
        self._args["approximant"] = lalsimulation.GetApproximantFromString("IMRPhenomD")


class IMRPhenomXAS(LALSimulationApproximant):
    def __init__(self, peak_alignment="sample"):
        super().__init__(peak_alignment=peak_alignment)
        self._args["approximant"] = lalsimulation.GetApproximantFromString(
            "IMRPhenomXAS"
        )
