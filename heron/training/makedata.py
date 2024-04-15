"""
Utilities for generating waveform training data from conventional waveform sources.
"""

import scipy.interpolate as interp

# Astropy to handle units sanely
from astropy import units as u

# Import LAL tools
import lalsimulation
import lal
from lal import cached_detector_by_prefix, TimeDelayFromEarthCenter, LIGOTimeGPS

# Import heron types
from ..types import Waveform, WaveformDict

class WaveformApproximant:
    """
    This class handles a waveform approximant model.
    """
    pass

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
            "phi ref": 0.,
            "longAscNodes": 0.,
            "eccentricity": 0.,
            "meanPerAno": 0.,
            "delta T": 1/(4096.*u.Hertz),
            "f_min": 50.0,
            "f_ref": 50.0,
            "params": lal.CreateDict(),
            "approximant": None,
        }

    @property
    def args(self):
        """
        Provide the arguments list converted to appropriate units.
        """
        args = {}
        args.update(self._args)
        args['m1'] = args['m1'].to_value(u.kilogram)
        args['m2'] = args['m2'].to_value(u.kilogram)
        args['distance'] = args['distance'].to_value(u.Mpc)
        args['delta T'] = args['delta T'].to_value(u.second)
        return args
        
    def time_domain(self, parameters):
        """
        Retrieve a time domain waveform for a given set of parameters.
        """
        self._args.update(parameters)
        hp, hx = lalsimulation.SimInspiralChooseTDWaveform(
            *list(self.args.values())
        )
        hp_ts = Waveform(data=hp.data.data,
                         dt=hp.deltaT,
                         t0=hp.epoch)
        hx_ts = Waveform(data=hx.data.data,
                         dt=hx.deltaT,
                         t0=hx.epoch)
        
        return WaveformDict(plus=hp_ts, cross=hx_ts)


class IMRPhenomPv2(LALSimulationApproximant):
    def __init__(self):
        super().__init__()
        self._args["approximant"] = lalsimulation.GetApproximantFromString("IMRPhenomPv2")
