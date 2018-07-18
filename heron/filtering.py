"""
Matched filtering functions.

░█░█░█▀▀░█▀▄░█▀█░█▀█
░█▀█░█▀▀░█▀▄░█░█░█░█
░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░▀

---------------------------------------------------
Heron is a matched filtering framework for Python.
---------------------------------------------------

--------------------------------------------------------------
Matched Filtering Routines
----
This code is designed for performing matched filtering using a
Gaussian Process Surrogate model.
---------------------------------------------------------------
"""

import numpy as np
from matplotlib.mlab import psd
from heron.sampling import draw_samples
from scipy import signal

def inner_product_noise(x, y, sigma, psd=None,  srate=16834):
    """
    Calculate the noise-weighted inner product of two random arrays.

    Parameters
    ----------
    x : `np.ndarray`
       The first data array
    y : `np.ndarray`
       The second data array
    sigma : `np.ndarray`
       The uncertainty to weight the inner product by.
    psd : `np.darray`
       The power spectral density to weight the inner product by.
    """
    nfft = 4*srate
    
    window = signal.get_window(('tukey', 0.1), len(x))
    fwindow = signal.get_window(('tukey', 0.1), nfft)

    xdata = x*window
    ydata = y*window

    noisefft = np.fft.rfft(sigma, nfft)*np.fft.rfft(sigma, nfft).conj()
    
    xy = np.fft.rfft(xdata, nfft)*np.fft.rfft(ydata, nfft).conj()
    if not psd:
        psd, pfreqs = psd(wdata, NFFT=nfft, Fs=srate, window=fwindow, noverlap=0)

    # The new weighting needs to be the sum of the sum of the PSD and
    # the sigma-weighting
    psd = psd + noisefft
    return 4*np.real(np.sum(xy/(psd)))

class Filter(object):
    """
    This class builds the filtering machinery from a provided surrogate
    model and noisy data.
    """

    def __init__(self, gp, data, times):
        """
        Construct a matched filter with a gaussian process regressor
        as the template bank, and noisy data to be filtered.

        Parameters
        ----------
        gp : `heron.regression`
           A trained Gaussian Process Regression model.
        data : `np.ndarray`
           A numpy array of data.
        """

        self.gp = gp
        self.data = data
        self.times = times

    def matched_likelihood(self, theta, psd=None, srate=16834):
        """
        Calculate the simple match of some data, given a template, and return its
        log-likelihood.

        Parameters
        ----------
        data : `np.ndarray`
           An array of data which is believed to contain a signal.
        theta : `np.ndarray`
           An array containing the location at which the template should be evaluated.
        """
        data = self.data
        gp = self.gp
        time = self.times
        
        cross = dict(zip(gp.training_object.target_names, theta))
        cross['t'] = [time[0], time[-1], len(time)]
        locs = draw_samples(gp, **cross)
        
        template, templatesigma = gp.prediction(locs)
        
        return -np.log(np.dot(templatesigma, templatesigma)) - 0.5* inner_product_noise(data-template, data-template, templatesigma, srate) 
