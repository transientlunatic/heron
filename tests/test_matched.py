"""
Tests for the matched filter likelihoods.
"""

import unittest
import numpy as np
import torch

from heron.models.torchbased import HeronCUDA
from heron.matched import CudaLikelihood

from elk.waveform import Timeseries

class TestCUDALikelihood(unittest.TestCase):
    """Test the CUDA-accelerated likelihood."""

    def setUp(self):
        self.model = HeronCUDA()
        noise = 5e-19 * torch.randn(100)
        times = np.linspace(-0.02, 0.02, 100)
        signal = self.model.time_domain_waveform({'mass ratio': 0.9}, times=times)
        self.data = Timeseries(data=(noise + signal[0].data), times=times)

        self.likelihood = CudaLikelihood(self.model,
                                         data=self.data,
                                         window=np.hanning(len(self.data)),
                                         detector_prefix='H1',
                                         psd=noise.cuda().rfft(1))


    def test_evaluation(self):
        """Check that evaluating the likelihood returns a float."""
        parameters = {"mass ratio": 0.9}
        self.assertIsInstance(self.likelihood(parameters).item(), np.float)
