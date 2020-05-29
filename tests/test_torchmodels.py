"""
Tests for the torch-based models.
"""

import unittest

from heron.models import torchbased

import numpy as np
import numpy.testing as npt
np.random.seed(90)


class TestHeronCUDA(unittest.TestCase):
    """
    Perform tests on the Torch model.
    """

    def setUp(self):

        self.model = torchbased.HeronCUDA()
        self.times = np.linspace(-0.02, 0.02, 1000)

        self.parameters_nonspin = {"mass ratio": 0.404}
        self.parameters_spin = {"mass ratio": 1.0, "spin 2z": 0.5}

    def test_nonspin_timedomain(self):
        """Test that the model can generate a non-spinning time domain waveform."""
        td_data = self.model.time_domain_waveform(times=self.times, p = self.parameters_nonspin)
        self.assertEqual(len(td_data[0]), len(self.times))
        self.assertEqual(len(td_data), 2)

    def test_nonspin_freqdomain(self):
        """Test that the model can generate a non-spinning frequency domain waveform."""
        fd_data = self.model.frequency_domain_waveform(times=self.times, p = self.parameters_nonspin)
        self.assertEqual(len(fd_data[0]), len(self.times)/2+1)
        self.assertEqual(len(fd_data), 2)

    def test_spin_timedomain(self):
        """Test that the model can generate a spinning time domain waveform."""
        td_data = self.model.time_domain_waveform(times=self.times, p = self.parameters_spin)
        self.assertEqual(len(td_data[0]), len(self.times))
        self.assertEqual(len(td_data), 2)

    def test_spin_freqdomain(self):
        """Test that the model can generate a spinning frequency domain waveform."""
        fd_data = self.model.frequency_domain_waveform(times=self.times, p = self.parameters_spin)
        self.assertEqual(len(fd_data[0]), len(self.times)/2+1)
        self.assertEqual(len(fd_data), 2)

    def test_mean_interface(self):
        """Test that the mean interface behaves correctly"""
        mean_data = self.model.mean(times=self.times, p=self.parameters_nonspin)
        self.assertEqual(len(mean_data), 2)
        self.assertEqual(len(mean_data[0]), len(self.times))

    def test_distribution_interface(self):
        """Test that the distribution interface behaves correctly"""
        wf_data = self.model.distribution(times=self.times, p=self.parameters_nonspin, samples=100)
        self.assertEqual(len(wf_data), 100)
        self.assertEqual(len(wf_data[0]), len(self.times))
