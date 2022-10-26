"""
CUDA Waveform Tests with hypothesis
-----------------------------------

This test suite contains tests to ensure that the output of CUDA-based waveform models
both complies with the API and also behave as expected with various inputs.

These tests do not attempt to test the scientific accuracy of the waveforms.
"""

import unittest
from unittest import mock
from hypothesis import example, given, strategies as st

from heron.models import torchbased

import torch

models = {"HeronCUDA": torchbased.HeronCUDA}


class TestCUDAWaveformHypothesis(unittest.TestCase):

    def setUp(self):
        model = models['HeronCUDA']
        self.generative_model = model()
        times = torch.linspace(-0.01, 0.05, 100)
        p = {"mass ratio": 1.0}
        prediction_mean, prediction_var, prediction_covar = self.generative_model._predict(p=p, times=times)
    
    @given(st.floats(min_value=0.0,
                     max_value=1.0,
                     exclude_min=True))
    def test_mass_ratio(self, mass_ratio):

        times = torch.linspace(-0.01, 0.05, 100)
        p = {"mass ratio": mass_ratio}
        
        prediction_mean, prediction_var, prediction_covar = self.generative_model._predict(p=p, times=times)
            

        # A mean waveform should be produced.
        self.assertTrue(isinstance(prediction_mean, torch.Tensor))
        # The length of the produced waveform should be the same as the input times
        self.assertEqual(len(prediction_mean), len(times))
                
