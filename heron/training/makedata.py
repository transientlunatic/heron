"""
Utilities for generating waveform training data from conventional waveform sources.
"""
import astropy.units as u

from ..types import WaveformManifold
from ..models.lalsimulation import IMRPhenomPv2

def make_manifold(approximant=IMRPhenomPv2, **kwargs):
    xaxis = range(kwargs["lower"], kwargs["upper"], kwargs["step"])
    approximant = approximant()
    manifold = WaveformManifold()
    for x in xaxis:
        manifold.add_waveform(approximant.time_domain({kwargs["parameter"]: x * kwargs["unit"]}))
    return manifold

    
