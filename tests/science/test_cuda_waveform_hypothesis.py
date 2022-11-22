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

from pycbc.waveform import get_td_waveform
from heron.models import torchbased

import torch
import numpy as np

models = {"HeronCUDA": torchbased.HeronCUDA}


def generate_approximant(mass_ratio):
    q = mass_ratio
    apx = "IMRPhenomPv2"
    M = 20
    unaligned = {}
    m1 = M / (1+q)
    m2 = M / (1+1/q)
    assert ((m1 + m2) - M) < 1e-4

    unaligned['hp'], unaligned['hc'] = get_td_waveform(approximant=apx,
                                                       mass1=m1,
                                                       mass2=m2,
                                                       spin1z=0,
                                                       delta_t=1.0/4096,
                                                       f_lower=20)
    idx = (unaligned['hp'].sample_times > -0.05) & (unaligned['hp'].sample_times < 0.02)
    return unaligned['hp'].sample_times[idx], unaligned['hp'].data[idx], unaligned['hc'].data[idx]

def rough_match(a, b):
    return (np.inner(a, b) / np.inner(a, a))

class TestCUDAWaveformHypothesis(unittest.TestCase):

    def setUp(self):
        model = models['HeronCUDA']
        self.generative_model = model()
        times = torch.linspace(-0.05, 0.02, 286)
        p = {"mass ratio": 1.0}
        prediction_mean, prediction_var, prediction_covar = self.generative_model._predict(p=p, times=times)
        #torchbased.train(self.generative_model)
        
    @given(st.floats(min_value=0.005,
                     max_value=1.0,
                     exclude_min=True))
    def test_mass_ratio(self, mass_ratio):

        times = torch.linspace(-0.01, 0.05, 286)
        p = {"mass ratio": mass_ratio}

        approximant = generate_approximant(mass_ratio=mass_ratio)
        prediction_mean, prediction_var, prediction_covar = self.generative_model._predict(p=p, times=approximant[0])

        # Compare the output of the model with IMRPhenomPv2
        self.assertGreater(rough_match(approximant[1], prediction_mean.cpu().numpy()), 0.90)
