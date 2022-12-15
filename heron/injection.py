"""
Code to create injections into data.
"""

from heron.types import PSD
from heron.utils import noise_psd

import lalsimulation
import torch


def psd_from_lalinference(name: str, frequencies, lower_frequency: float = 20):
    """Create a PSD object from a LALInference PSD class

    Parameters
    ----------
    name : str
       The name of the PSD from LALInference.
    frequencies : array-like
       The frequencies at which the PSD should be evaluated.
    lower_frequency : float
       The lowest frequency at which the PSD should be evaluated.

    Notes
    -----
    Frequently the PSD will either be invalid below a given frequency,
    or will be unusable. You should set a lower_frequency in order to
    account for this.
    """
    if name in dir(lalsimulation):
        psd_func = getattr(lalsimulation, name)
    else:
        raise NotImplementedError(
            "This noise source is not implemented in LALSimulation"
        )

    masked_psd = lambda f: psd_func(f) if f >= lower_frequency else psd_func(lower_frequency)
    psd = torch.tensor([masked_psd(float(f)) for f in frequencies], dtype=torch.float64)
    psd = PSD(data=psd, frequencies=frequencies)
    return psd


def frequencies_from_times(times):
    """Get the frequency axies which corresponds to a given time axis."""
    dt = (times[1] - times[0])
    frequencies = torch.arange((len(times) + 1) // 2) / (dt * len(times))
    return frequencies


def create_noise_series(psd: PSD, times, device="cuda"):
    """Create a timeseries filled with noise from a specific PSD.
    Parameters
    ----------
    psd : `heron:PSD`
       The power spectral density of the noise to be generated in the time
       series.
    times : array-like
       The time axis of the timeseries.
    device : str or torch.device
       The device which the noise series should be stored in.
    """
    frequencies = frequencies_from_times(times)
    noise = torch.tensor(noise_psd(len(times), frequencies=frequencies, psd=psd.data), device=device)
    return noise
