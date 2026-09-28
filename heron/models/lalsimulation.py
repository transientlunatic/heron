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


class LALSimulationApproximant(WaveformApproximant):
    """Base class for LALSimulation-based approximants."""

    def __init__(self):
        _require_lal()
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

            if times is not None:
                times_wf = (
                    np.arange(len(hp.data.data)) * hp.deltaT
                    + epoch + hp.epoch.ns() / 1e9
                )

                spl_hp = CubicSpline(times_wf, hp.data.data, extrapolate=False)
                spl_hx = CubicSpline(times_wf, hx.data.data, extrapolate=False)

                hp_data = np.nan_to_num(spl_hp(times))
                hx_data = np.nan_to_num(spl_hx(times))
                hp_ts = Waveform(data=hp_data, times=np.asarray(times))
                hx_ts = Waveform(data=hx_data, times=np.asarray(times))

            elif "time" in parameters:
                t = parameters["time"]
                times_wf = (
                    np.arange(len(hp.data.data)) * hp.deltaT
                    + hp.epoch
                    + epoch
                )

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
                n = len(hp_data)
                times_arr = np.arange(n) * hp.deltaT + float(hp.epoch)
                hp_ts = Waveform(data=hp_data, times=times_arr, dt=hp.deltaT, t0=float(hp.epoch) + epoch)
                hx_ts = Waveform(data=hx_data, times=times_arr, dt=hx.deltaT, t0=float(hx.epoch) + epoch)

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


class SEOBNRv4(LALSimulationApproximant):
    def __init__(self):
        super().__init__()
        self._args["approximant"] = lalsimulation.GetApproximantFromString("SEOBNRv4")


class IMRPhenomD(LALSimulationApproximant):
    def __init__(self):
        super().__init__()
        self._args["approximant"] = lalsimulation.GetApproximantFromString("IMRPhenomD")


class IMRPhenomXAS(LALSimulationApproximant):
    def __init__(self):
        super().__init__()
        self._args["approximant"] = lalsimulation.GetApproximantFromString(
            "IMRPhenomXAS"
        )
