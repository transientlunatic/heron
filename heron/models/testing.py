"""
Testing models
--------------

These models are designed purely for helping with unittests and other tests.
They aren't designed for use in production analyses!
"""

import numpy as np
import scipy.linalg
from astropy import units as u
from ..types import PSD, Waveform, WaveformDict
from . import PSDApproximant, WaveformApproximant

class FlatPSD(PSDApproximant):
    """
    Return a flat PSD.
    """
    
    def __init__(self):
        super().__init__()

    def psd_function(self, psd_data, flow):

        return np.ones_like(psd_data)
    
    def frequency_domain(
        self,
        df=1,
        frequencies=None,
        lower_frequency=20,
        upper_frequency=1024,
        mask_below=20,
    ):
        if frequencies is None:
            frequencies = np.arange(lower_frequency, upper_frequency + df, df)

        N = int(len(frequencies))
        df = float(frequencies[1] - frequencies[0])
        psd_data = np.ones(N)
        psd_data = self.psd_function(psd_data, flow=lower_frequency)
        psd_data[frequencies < mask_below] = psd_data[frequencies > mask_below][0]
        psd = PSD(psd_data, frequencies=frequencies)
        return psd

    def covariance_matrix(self, times):
        """
        Create the covariance matrix for this PSD.
        """
        N = int(len(times))
        autocovariance = np.exp(-np.arange(N)*0.1)
        return scipy.linalg.circulant(autocovariance)
    

class SineGaussianWaveform(WaveformApproximant):
    """
    A simple SineGaussian waveform for testing purposes.
    """

    def __init__(self):
        super().__init__()
        self._args = {
            "width": 0.02 * u.second,
            "frequency": 500 * u.Hertz,
            "segment length": 1 * u.second,
        }

    def time_domain(self, parameters, times=None, sample_rate=1024*u.Hertz):
        epoch = parameters.get("gpstime", parameters.get("epoch", 0))
        self._args.update(parameters)
        width = self._args['width']
        length = self._args['segment length']
        times = np.linspace(-length/2, length/2, int((length*sample_rate).value))
        envelope = np.exp(
            (- (times - epoch)**2/(2*width**2)).value
        ) / np.sqrt(2*np.pi*width**2)

        strain = np.sin((times*u.second * self._args['frequency']).value) * envelope
        hp_data = Waveform(data=strain,
                           times=times,
                           t0=epoch)
        hx_data = Waveform(data=strain,
                           times=times,
                           t0=epoch)
        return WaveformDict(parameters=self._args,
                            plus=hp_data,
                            cross=hx_data)
