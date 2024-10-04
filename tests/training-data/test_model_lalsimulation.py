import unittest
from astropy import units as u
from heron.models import lalsimulation

class TestIMRPhenomPv2_Interface(unittest.TestCase):

    def setUp(self):
        self.waveform = lalsimulation.IMRPhenomPv2()

    def test_values(self):
        self.assertTrue(min(self.waveform.time_domain({"m1": 100 * u.solMass, "m2": 50 * u.solMass})['plus'].times.value)<0.01)

    def test_hrss(self):
        self.assertTrue(max(self.waveform.time_domain({"m1": 100 * u.solMass, "m2": 50 * u.solMass}).hrss.value) < 1e5)

    def test_plot(self):
        data = self.waveform.time_domain({"m1": 100 * u.solMass, "m2": 50 * u.solMass})
        f = data['plus'].plot()
