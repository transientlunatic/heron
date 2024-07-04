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
    for parameter, kwargs in varied.items():
        xaxis = np.arange(kwargs["lower"], kwargs["upper"], kwargs["step"])
        # Update with the fixed parameters
        manifold = WaveformManifold()
        for x in xaxis:
            manifold.add_waveform(
                approximant.time_domain({parameter: x * kwargs.get("unit", 1), "gpstime": 0})
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
            waveform = approximant.time_domain({parameter: x * kwargs.get("unit", 1),
                                                "gpstime": 0})
            peaks, _ = scipy.signal.find_peaks(waveform["plus"].value ** 2)
            peaks_interp = np.interp(
                np.arange(warp_factor*len(peaks))/warp_factor,
                np.arange(len(peaks)),
                peaks)
            peaks_interp = np.unique(peaks_interp).astype(int)
            # print("peaks", len(peaks), "new points", len(peaks_interp)) 
            new_waveform = Waveform(
                data=waveform["plus"][peaks_interp], times=waveform["plus"].times[peaks_interp]
            )
            manifold_plus.add_waveform(
                WaveformDict(plus=new_waveform, parameters=waveform.parameters)
            )

            peaks, _ = scipy.signal.find_peaks(waveform["cross"].value ** 2)
            peaks_interp = np.interp(
                np.arange(warp_factor*len(peaks))/warp_factor,
                np.arange(len(peaks)),
                peaks)
            peaks_interp = np.unique(peaks_interp).astype(int)
            new_waveform = Waveform(
                data=waveform["cross"][peaks_interp], times=waveform["cross"].times[peaks_interp]
            )
            manifold_cross.add_waveform(
                WaveformDict(cross=new_waveform, parameters=waveform.parameters)
            )
            
    return manifold_plus, manifold_cross
