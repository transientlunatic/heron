"""
Tests for the science testing code.
"""

import unittest
import numpy as np

from heron import testing
from elk.waveform import NRWaveform, Timeseries
from elk.catalogue import NRCatalogue

from heron.models.georgebased import HeronHodlr

class MockWaveform(NRWaveform):
    def timeseries(self,
                   total_mass,
                   sample_rate=4096,
                   f_low=None,
                   distance=1,
                   coa_phase=0,
                   ma=None,
                   t_min=None,
                   t_max=None,
                   f_ref=None,
                   t_align=True):
        return (Timeseries(data=np.random.randn(1000)*1e-19, times=np.linspace(t_min, t_max, 1000)),
                Timeseries(data=np.random.randn(1000)*1e-19, times=np.linspace(t_min, t_max, 1000)))

class TestTests(unittest.TestCase):
    """
    Test the science testing code.
    """

    def setUp(self):
        self.model = HeronHodlr()
        self.samples_catalogue = NRCatalogue("GeorgiaTech")
        mock_waveforms = [
            MockWaveform("spam", {"q": 1.0,
                                  "tag": "test",
                                  "spin_1x": 0, "spin_1y": 0, "spin_1z": 0,
                                  "spin_2x": 0, "spin_2y": 0, "spin_2z": 0
            }),
            MockWaveform("eggs", {"q": 0.8,
                                  "tag": "test2",
                                  "spin_1x": 0, "spin_1y": 0, "spin_1z": 0,
                                  "spin_2x": 0, "spin_2y": 0, "spin_2z": 0
            })
            ]
        
        self.samples_catalogue.waveforms = mock_waveforms

    def test_nrcat_match(self):
        """Test the NR catalogue matcher."""
        matches = testing.nrcat_match(self.model, self.samples_catalogue)
        self.assertEqual(len(matches.values()), 2)
