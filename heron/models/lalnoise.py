"""
Noise models from lalsimulation.
"""
import torch
import numpy as np

import lal
import lalsimulation

from ..types import PSD, TimeSeries
from . import PSDApproximant


class LALSimulationPSD(PSDApproximant):
    """A class to generate power spectral densities using LALSimulation."""
    def __init__(self):

        super().__init__()

    def frequency_domain(self,
                         df=1,
                         frequencies=None,
                         lower_frequency=20,
                         upper_frequency=1024,
                         mask_below=20):
        if frequencies is not None:
            upper_frequency = float(frequencies[-1])
            lower_frequency = float(frequencies[0])
            df = float(frequencies[1] - frequencies[0])
            
        N = int(1 + ((upper_frequency - lower_frequency)/df))
        
        psd_data = lal.CreateREAL8FrequencySeries(None,
                                                  lal.LIGOTimeGPS(0),
                                                  lower_frequency,
                                                  df,
                                                  lal.HertzUnit,
                                                  N)
        self.psd_function(psd_data, flow=lower_frequency)
        frequencies = torch.arange(lower_frequency, upper_frequency+df, df)
        psd_data = psd_data.data.data
        psd_data[frequencies < mask_below] = psd_data[frequencies > mask_below][0]
        psd = PSD(psd_data, frequencies=frequencies)
        return psd

    def time_domain(self, times):
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

        dt = times[1] - times[0]
        N = len(times)
        
        frequencies = torch.arange(0, int((len(times) + 1) // 2))# / (dt * len(times)))
        reals = np.random.randn(len(frequencies))
        imags = np.random.randn(len(frequencies))

        T = 1 / (frequencies[1] - frequencies[0])
        psd = self.frequency_domain(frequencies=frequencies)
        
        S = np.sqrt(N * N / 4 / (T) * psd.value)

        noise_r = S * (reals)
        noise_i = S * (imags)

        noise_f = noise_r + 1j * noise_i

        return TimeSeries(data=np.fft.irfft(noise_f, n=(N)),
                          times=times)


class AdvancedLIGODesignSensitivity2018(LALSimulationPSD):
    psd_function = lalsimulation.SimNoisePSDaLIGODesignSensitivityT1800044

class AdvancedLIGO(AdvancedLIGODesignSensitivity2018):
    pass

