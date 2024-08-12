import unittest

import numpy as np
from astropy import units as u

from heron.models.lalsimulation import SEOBNRv3, IMRPhenomPv2, IMRPhenomPv2_FakeUncertainty
from heron.models.lalnoise import AdvancedLIGO

import heron.filters

class Test_InnerProduct(unittest.TestCase):

    def setUp(self):
        self.waveform_model = IMRPhenomPv2()
        
    def test_inner_same_waveforms(self):
        parameters = {
            "mass_ratio": 0.6,
            "total_mass": 60 * u.solMass
        }
        times = np.linspace(-0.1, 0.05, 300)
        
        a = self.waveform_model.time_domain(parameters=parameters,
                                            times=times)['plus']
        b = self.waveform_model.time_domain(parameters=parameters,
                                            times=times)['plus']
        ip = heron.filters.InnerProduct()
        
        self.assertTrue(ip(a, b) > 0)

    def test_overlap_same_waveforms(self):
        parameters = {
            "mass_ratio": 0.6,
            "total_mass": 60 * u.solMass
        }
        times = np.linspace(-0.1, 0.05, 300)
        
        a = self.waveform_model.time_domain(parameters=parameters,
                                            times=times)['plus']
        b = self.waveform_model.time_domain(parameters=parameters,
                                            times=times)['plus']
        ip = heron.filters.Overlap()
        
        self.assertTrue(ip(a, b) == 1)

    def test_overlap_different_waveforms(self):
        parameters = {
            "mass_ratio": 0.6,
            "total_mass": 60 * u.solMass
        }
        times = np.linspace(-0.1, 0.05, 300)
        
        a = self.waveform_model.time_domain(parameters=parameters,
                                            times=times)['plus']

        parameters = {
            "mass_ratio": 0.7,
            "total_mass": 60 * u.solMass
        }
        b = self.waveform_model.time_domain(parameters=parameters,
                                            times=times)['plus']
        ip = heron.filters.Overlap()

        self.assertTrue(ip(a, b) < 1)
