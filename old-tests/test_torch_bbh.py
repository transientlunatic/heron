import unittest

from heron.models.torchbased import HeronCUDA, train
from heron.likelihood import InnerProduct, Likelihood, CUDALikelihood
from heron.models.testing import TestModel, CUDATestModel
from elk.waveform import Timeseries

from torch import tensor
import torch
import gpytorch

import numpy as np
import numpy.testing as npt
np.random.seed(90)



class TestCUDALikelihood(unittest.TestCase):
    device = torch.device("cuda")

    @classmethod
    def setUpClass(cls):
        cls.generator = HeronCUDA(datafile="notebooks/new-data-interface/test_file_2.h5", datalabel="IMR training", 
                          device=cls.device, 
                          noise=[0.000001, 0.0001],
                          lengths={"mass ratio": [0.001, 0.25],
                                   "time": [0.001, 20]}
        )
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(1500):
            train(cls.generator, iterations=50)

    def setUp(self):

        data_length = 164
        fft_length = int(1+(data_length/2))
        
        generator = self.generator
        window = torch.blackman_window

        self.psd = torch.ones(fft_length, device=self.device)
        self.times = times = torch.linspace(0.05, 0.005, data_length)

        noise = 5e-20*torch.randn(data_length, device=self.device)

    def test_time_domain_variance_magnitude(self):
        """Check that variances have the correct general size."""
        signal = self.generator.time_domain_waveform(times=self.times, p={"mass ratio":0.7})

        self.assertLess(signal['plus'].variance.max(), 1e-20)
        self.assertGreater(signal['plus'].variance.max(), 1e-24)
        self.assertGreaterEqual(signal['plus'].variance.min(), 0.0)

    def test_freq_domain_variance_magnitude(self):
        """Check that variances have the correct general size."""
        signal = self.generator.frequency_domain_waveform(times=self.times,
                                                          window=torch.blackman_window,
                                                          p={"mass ratio":0.7})

        self.assertLess(signal['plus'].variance.abs().max(), 1e-20)
        self.assertGreater(signal['plus'].variance.abs().max(), 1e-24)
        self.assertGreaterEqual(signal['plus'].variance.abs().min(), 0.0)
        
    def test_time_domain_waveform_magnitude(self):
        """Check that waveforms have the correct general magnitude."""
        signal = self.generator.time_domain_waveform(times=self.times, p={"mass ratio":0.7})

        self.assertLess(signal['plus'].data.max(), 1e-20)
        self.assertGreater(signal['plus'].data.max(), 1e-24)

    def test_freq_domain_waveform_magnitude(self):
        """Check that waveforms have the correct general magnitude."""
        signal = self.generator.frequency_domain_waveform(times=self.times,
                                                          window=torch.blackman_window,
                                                          p={"mass ratio":0.7})

        self.assertLess(signal['plus'].data.abs().max(), 1e-20)
        self.assertGreater(signal['plus'].data.abs().max(), 1e-24)
