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
        df: float = 1,
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
        ts = np.fft.irfft(psd, n=(N)) * (N/2/dt)
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
        
        Normalization follows standard one-sided PSD conventions:
        - Interior frequency bins (k=1..N/2-1): X_k = sqrt(S1(f_k) * fs / 2) * (a_k + i b_k)
        - DC bin (k=0): X_0 = 0 (no offset)
        - Nyquist bin (k=N/2, when N even): X_{N/2} = sqrt(S1(f_{N/2}) * fs) * a_{N/2} (real)
        - Before applying the inverse real FFT, the frequency-domain noise is scaled by sqrt(N/2) to compensate for numpy's normalization.
        """

        times = np.asarray(times)
        N = len(times)
        dt = float(times[1] - times[0])
        sample_rate = 1.0 / dt
        
        df = sample_rate / N
        freqs = np.arange(0, N // 2 + 1) * df

        # One-sided PSD sampled on our frequency grid
        psd = np.asarray(self.frequency_domain(df=df, frequencies=freqs).data)
        if len(psd) > 1:
            psd[-1] = psd[-2]  # avoid edge artifacts at Nyquist

        # Random Gaussian draws
        reals = np.random.randn(len(freqs))
        imags = np.random.randn(len(freqs))

        # Construct one-sided frequency-domain noise
        noise_f = np.zeros(len(freqs), dtype=np.complex128)
        if len(freqs) > 2:
            amp_mid = np.sqrt(psd[1:-1] * sample_rate / 2.0)
            noise_f[1:-1] = amp_mid * (reals[1:-1] + 1j * imags[1:-1])

        # DC = 0 to avoid mean offset
        noise_f[0] = 0.0 + 0.0j

        # Nyquist bin is purely real with full factor (no 1/2)
        if N % 2 == 0:
            noise_f[-1] = np.sqrt(psd[-1] * sample_rate) * reals[-1]

        # Compensate for numpy's 1/N inverse FFT normalization
        noise_f *= np.sqrt(N / 2.0)

        # Inverse real FFT; numpy uses backward normalization so this is consistent
        data = np.fft.irfft(noise_f, n=N)
        return TimeSeries(data=data, times=times)


class AdvancedLIGODesignSensitivity2018(LALSimulationPSD):
    psd_function = lalsimulation.SimNoisePSDaLIGODesignSensitivityT1800044


class AdvancedLIGO(AdvancedLIGODesignSensitivity2018):
    pass

class ZeroNoise(AdvancedLIGO):
    def zeros(self, x, flow=None):
        return np.ones(len(x.data.data))
    psd_function = zeros

    def time_series(self, times):
        return np.zeros(len(times))


KNOWN_PSDS = {"AdvancedLIGO": AdvancedLIGO,
    "ZeroNoise": ZeroNoise,
}
