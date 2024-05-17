"""
Utilities for generating waveform training data from conventional waveform sources.
"""

import scipy.signal

import astropy.units as u
import numpy as array_library

from ..types import Waveform, WaveformDict, WaveformManifold
from ..models.lalsimulation import IMRPhenomPv2


def make_manifold(approximant=IMRPhenomPv2, fixed={}, varied={}):
    """
    Make a manifold of waveforms across a range of parameters.

    Notes
    -----

    This isn't finished; I really need to make this produce
    the cartesian product of all the various parameters.
    """
    approximant = approximant()
    approximant._args.update(fixed)
    for parameter, kwargs in varied.items():
        xaxis = array_library.arange(kwargs["lower"], kwargs["upper"], kwargs["step"])
        # Update with the fixed parameters
        manifold = WaveformManifold()
        for x in xaxis:
            manifold.add_waveform(
                approximant.time_domain({parameter: x * kwargs.get("unit", 1)})
            )
    return manifold


def make_optimal_manifold(approximant=IMRPhenomPv2, fixed={}, varied={}):
    """
    Make a manifold of waveforms across a range of parameters.

    Notes
    -----

    This isn't finished; I really need to make this produce
    the cartesian product of all the various parameters.
    """
    approximant = approximant()
    approximant._args.update(fixed)
    for parameter, kwargs in varied.items():
        xaxis = array_library.arange(kwargs["lower"], kwargs["upper"], kwargs["step"])
        # Update with the fixed parameters
        manifold = WaveformManifold()
        for x in xaxis:
            waveform = approximant.time_domain({parameter: x * kwargs.get("unit", 1)})
            peaks, _ = scipy.signal.find_peaks(waveform["plus"].value ** 2)
            new_waveform = Waveform(
                data=waveform["plus"][peaks], times=waveform["plus"].times[peaks]
            )
            manifold.add_waveform(
                WaveformDict(plus=new_waveform, parameters=waveform.parameters)
            )
    return manifold
