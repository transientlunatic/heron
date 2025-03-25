"""
Tests for the time-domain likelihood functions.
"""


import unittest

import numpy as np

from heron.models.testing import FlatPSD, SineGaussianWaveform
from heron import likelihood

class TestTDLikelihood(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 1024
        self.duration = 2 # seconds
        N = self.sample_rate * self.duration
        self.psd = FlatPSD()
        self.data = SineGaussianWaveform().time_domain(parameters={"width":0.05})['plus']

        self.likelihood = likelihood.TimeDomainLikelihood(
            data=self.data,
            psd=self.psd,
            detector="L1",
        )

    def test_evaluate(self):
        self.likelihood({"width": 0.01})
