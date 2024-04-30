import numpy as array_library
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
        self._args = {
            "m1": 10 * u.solMass,
            "m2": 10 * u.solMass,
            "S1x": 0.,
            "S1y": 0.,
            "S1z": 0.,
            "S2x": 0.,
            "S2y": 0.,
            "S2z": 0.,
            "distance": 10 * u.Mpc,
            "inclination": 0,
            "phi ref": 0. * u.Hertz,
            "longAscNodes": 0.,
            "eccentricity": 0.,
            "meanPerAno": 0.,
            "delta T": 1/(4096. * u.Hertz),
            "f_min": 50.0 * u.Hertz,
            "f_ref": 50.0 * u.Hertz,
            "params": lal.CreateDict(),
            "approximant": None,
        }
        self.allowed_parameters = list(self._args.keys())

    @property
    def args(self):
        """
        Provide the arguments list converted to appropriate units.
        """
        args = {}
        args.update(self._args)
        # Convert total mass and mass ratio to the component masses
        if "total mass" in args and "mass ratio" in args:
            args['m1'] = (args["total mass"] / (1 + args["mass ratio"])).to_value(u.kilogram)
            args['m2'] = (args["total mass"] / (1 + 1 / args["mass ratio"])).to_value(u.kilogram)
            args.pop("total mass")
            args.pop("mass ratio")
        # Remove sky and extrinsic parameters
        for par in ("ra", "dec", "phase", "psi", "theta_jn"):
            if par in args: args.pop(par)
        args['distance'] = args['distance'].to_value(u.meter)
        args['delta T'] = args['delta T'].to_value(u.second)
        args['f_min'] = args['f_min'].to_value(u.Hertz)
        args['f_ref'] = args['f_ref'].to_value(u.Hertz)
        args['phi ref'] = args['phi ref'].to_value(u.Hertz)

        for key in args.keys():
            if not key in self.allowed_parameters:
                args[key].pop()
        
        return args
        
    def time_domain(self, parameters, times=None):
        """
        Retrieve a time domain waveform for a given set of parameters.
        """
        self._args.update(parameters)
        hp, hx = lalsimulation.SimInspiralChooseTDWaveform(
            *list(self.args.values())
        )

        if not isinstance(times, type(None)):
            times_wf = array_library.arange(hp.epoch, (len(hp.data.data))*hp.deltaT + hp.epoch, hp.deltaT)
            spl_hp = CubicSpline(times_wf, hp.data.data)
            spl_hx = CubicSpline(times_wf, hx.data.data)
            hp_data = spl_hp(times)
            hx_data = spl_hx(times)
        else:
            hp_data = hp.data.data
            hx_data = hx.data.data
            
        hp_ts = Waveform(data=hp_data,
                         dt=hp.deltaT,
                         t0=hp.epoch)
        hx_ts = Waveform(data=hx_data,
                         dt=hx.deltaT,
                         t0=hx.epoch)
        
        return WaveformDict(parameters=parameters, plus=hp_ts, cross=hx_ts)


class IMRPhenomPv2(LALSimulationApproximant):
    def __init__(self):
        super().__init__()
        self._args["approximant"] = lalsimulation.GetApproximantFromString("IMRPhenomPv2")

class SEOBNRv3(LALSimulationApproximant):
    def __init__(self):
        super().__init__()
        self._args["approximant"] = lalsimulation.GetApproximantFromString("SEOBNRv3")        
