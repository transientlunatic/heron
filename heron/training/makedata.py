"""
Utilities for generating waveform training data from conventional waveform sources.
"""

import scipy.signal

import astropy.units as u
import numpy as np

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
    manifold = WaveformManifold()
    for parameter, kwargs in varied.items():
        xaxis = np.arange(kwargs["lower"], kwargs["upper"], kwargs["step"])
        # Update with the fixed parameters
        for x in xaxis:
            manifold.add_waveform(
                approximant.time_domain({parameter: x * kwargs.get("unit", 1)})
            )
    return manifold


def make_optimal_manifold(approximant=IMRPhenomPv2, fixed={}, varied={}, warp_factor=5):
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
        xaxis = np.arange(kwargs["lower"], kwargs["upper"], kwargs["step"])
        # Update with the fixed parameters
        manifold_plus = WaveformManifold()
        manifold_cross = WaveformManifold()
        for x in xaxis:
            waveform = approximant.time_domain({parameter: x * kwargs.get("unit", 1)})
            peaks, _ = scipy.signal.find_peaks(waveform["plus"].value ** 2)
            peaks_interp = np.interp(
                np.arange(warp_factor * len(peaks)), np.arange(len(peaks)), peaks
            )
            new_waveform = Waveform(
                data=waveform["plus"][peaks], times=waveform["plus"].times[peaks]
            )
            manifold_plus.add_waveform(
                WaveformDict(plus=new_waveform, parameters=waveform.parameters)
            )

            peaks, _ = scipy.signal.find_peaks(waveform["cross"].value ** 2)
            peaks_interp = np.interp(
                np.arange(warp_factor * len(peaks)), np.arange(len(peaks)), peaks
            )

            new_waveform = Waveform(
                data=waveform["cross"][peaks], times=waveform["cross"].times[peaks]
            )
            manifold_cross.add_waveform(
                WaveformDict(cross=new_waveform, parameters=waveform.parameters)
            )

    return manifold_plus, manifold_cross
