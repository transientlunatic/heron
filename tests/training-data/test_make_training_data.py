import unittest
from astropy import units as u
from heron.training import makedata

class TestIMRPhenomPv2_Interface(unittest.TestCase):

    def setUp(self):
        self.waveform = makedata.IMRPhenomPv2()

    def test_values(self):
        self.assertTrue(min(self.waveform.time_domain({"m1": 100 * u.solMass})['plus'].times.value)<0.01)

    def test_hrss(self):
        self.assertTrue(max(self.waveform.time_domain({"m1": 100 * u.solMass}).hrss.value) < 1e5)
