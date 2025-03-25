"""
Noise models from lalsimulation.
"""

import torch
import scipy
import numpy as np

import lal
import lalsimulation

from ..types import PSD, TimeSeries
from . import PSDApproximant


class LALSimulationPSD(PSDApproximant):
    """A class to generate power spectral densities using LALSimulation."""

    def __init__(self):

        super().__init__()

    def frequency_domain(
        self,
        df=1,
        frequencies=None,
        lower_frequency=20,
        upper_frequency=1024,
        mask_below=20,
    ):
        if frequencies is None:
            frequencies = torch.arange(lower_frequency, upper_frequency + df, df)

        N = int(len(frequencies))
        df = float(frequencies[1] - frequencies[0])
        psd_data = lal.CreateREAL8FrequencySeries(
            None, lal.LIGOTimeGPS(0), lower_frequency, df, lal.HertzUnit, N
        )
        self.psd_function(psd_data, flow=lower_frequency)
        psd_data = psd_data.data.data
        psd_data[frequencies < mask_below] = psd_data[frequencies > mask_below][0]
        psd = PSD(psd_data, frequencies=frequencies)
        return psd

    def twocolumn(self, *args, **kwargs):
        """
        Produce the PSD in two-column format.
        """
        psd = self.frequency_domain(*args, **kwargs)
        frequencies = psd.frequencies.value
        data = np.array(psd.data)

        return np.vstack([frequencies, data]).T
    
    def covariance_matrix(self, times):
        """
        Return a time-domain representation of this power spectral density.
        """
        dt = times[1] - times[0]
        N = len(times)
        T = times[-1] - times[0]
        df = 1 / T
        frequencies = torch.arange(len(times) // 2 + 1) * df.value
        psd = self.frequency_domain(df=df, frequencies=frequencies)
        ts = np.fft.irfft(psd, n=(N))  # * (N*N/dt/dt/2), n=(N))
        return scipy.linalg.toeplitz(ts)

    def time_domain(self, times):
        return self.covariance_matrix(times)

    def time_series(self, times):
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

        Notes
        -----
        In order to follow with the conventions in lalsuite this has been
        adapted from the code in lalnoise.
        """

        dt = times[1] - times[0]
        N = len(times)
        print(N)
        T = times[-1] - times[0]
        df = 1 / T
        frequencies = torch.arange(len(times) // 2 + 1) * df
        reals = np.random.randn(len(frequencies))
        imags = np.random.randn(len(frequencies))

        psd = self.frequency_domain(frequencies=frequencies)

        S = 0.5 * np.sqrt(psd.value * T)  # np.sqrt(N * N / 4 / (T) * psd.value)

        noise_r = S * (reals)
        noise_i = S * (imags)

        noise_f = noise_r + 1j * noise_i

        return TimeSeries(data=np.fft.irfft(noise_f, n=(N)), times=times)


class AdvancedLIGODesignSensitivity2018(LALSimulationPSD):
    psd_function = lalsimulation.SimNoisePSDaLIGODesignSensitivityT1800044


class AdvancedLIGO(AdvancedLIGODesignSensitivity2018):
    pass


KNOWN_PSDS = {"AdvancedLIGO": AdvancedLIGO}
