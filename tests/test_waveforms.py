"""
Code to test the correct performance of waveform approximants.
"""

import unittest

import heron.models.testing


class TestTestWaveforms(unittest.TestCase):

    def setUp(self):
        pass

    def test_plot_sg(self):
        waveform = heron.models.testing.SineGaussianWaveform()
        f = waveform.time_domain(parameters={"width":0.05})['plus'].plot()

        f.savefig("test_sg_waveform.png")

    def test_sg(self):
        waveform = heron.models.testing.SineGaussianWaveform()
        data = waveform.time_domain(parameters={"width":0.05})
        pass
        
