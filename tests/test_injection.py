"""
Tests for the injection infrastructure in heron.
"""

import unittest

import numpy as np
import astropy.units as u

from heron.models.lalsimulation import SEOBNRv3, IMRPhenomPv2
from heron.models.lalnoise import AdvancedLIGO
from heron.injection import make_injection
from heron.detector import AdvancedLIGOHanford, AdvancedLIGOLivingston, AdvancedVirgo


class TestPSDGeneration(unittest.TestCase):
    """Check that heron can correctly generate PSDs using lalnoise."""

    def setUp(self):
        self.psd_model = AdvancedLIGO()

    def test_psd_creation(self):
        """Check that anything is produced when requesting a PSD"""
        psd = self.psd_model.frequency_domain(upper_frequency=1024,
                                              lower_frequency=20,
                                              df=1)
        #f = psd.plot()
        #f.savefig("psd.png")

    def test_timeseries_creation(self):
        """Check that a timeseries can be produced."""
        times = np.linspace(10, 11, 4096)
        data = self.psd_model.time_series(times)
        #f = data.plot()
        #f.savefig("noise.png")
        
class TestInjection(unittest.TestCase):

    def setUp(self):
        self.imr_waveform = IMRPhenomPv2()
        self.psd_model = AdvancedLIGO()

    def test_simple_inject(self):
        make_injection(waveform=IMRPhenomPv2,
                       injection_parameters={"mass_ratio": 0.6,
                                             "total_mass": 60 * u.solMass},
                       detectors={"AdvancedLIGOHanford": "AdvancedLIGO"}
                       )


    def test_multiple_inject(self):
        make_injection(waveform=IMRPhenomPv2,
                       injection_parameters={"mass_ratio": 0.6,
                                             "total_mass": 60 * u.solMass},
                       detectors={"AdvancedLIGOHanford": "AdvancedLIGO",
                                  "AdvancedLIGOLivingston": "AdvancedLIGO"},
                       framefile="test",
                       psdfile="psd"
                       )
