import unittest

import astropy.units as u
import numpy as np

from heron import detector
from heron.models.lalsimulation import SEOBNRv3, IMRPhenomPv2


class test_detectors(unittest.TestCase):

    def setUp(self):
        self.H1 = detector.AdvancedLIGOHanford()
        self.L1 = detector.AdvancedLIGOLivingston()
        self.V1 = detector.AdvancedVirgo()

    def test_antenna_response(self):
        self.assertEqual(self.H1.antenna_response(ra=0.5, dec=0.6, psi=0.9, time=100),
                         ((0.3814310282849612,  0.7325279875753569)))
        self.assertEqual(self.L1.antenna_response(ra=0.5, dec=0.6, psi=0.9, time=100),
                         ((-0.534420590624828, -0.8167918070770407)))
        self.assertEqual(self.V1.antenna_response(ra=0.5, dec=0.6, psi=0.9, time=100),
                         ((0.5814000629005196, -0.05146152057265055)))

class test_detector_operations(unittest.TestCase):
    """Check that operations related to detectors work."""

    def setUp(self):
        self.H1 = detector.AdvancedLIGOHanford()
        self.waveform = IMRPhenomPv2()

    def test_projection(self):
        times = np.linspace(4000, 4001, 4096)
        waveform_eval = self.waveform.time_domain(
            {"mass_ratio": 1.0,
             "total_mass": 30 * u.solMass,
             "luminosity_distance": 500 * u.Mpc},
            times=times,
        )
        projected_waveform = waveform_eval.project(detector=self.H1, ra=1.0, dec=1.0, psi=1.0, iota=0, phi_0=0)

        self.assertFalse(projected_waveform.times[0] == waveform_eval['plus'].times[0])

    def test_projection_in_dict(self):
        times = np.linspace(4000, 4001, 4096)
        waveform_eval = self.waveform.time_domain(
            {"mass_ratio": 1.0,
             "total_mass": 30 * u.solMass,
             "ra": 1.0,
             "dec": 1.0,
             "psi": 1.0,
             "theta_jn": 0,
             "phase": 0,
             "luminosity_distance": 500 * u.Mpc},
            times=times,
        )
        waveform_eval.project(detector=self.H1)

    def test_projection_in_dict_horizontal(self):
        times = np.linspace(4000, 4001, 4096)
        waveform_eval = self.waveform.time_domain(
            {"mass_ratio": 1.0,
             "total_mass": 30 * u.solMass,
             "zenith": 0,
             "azimuth": 1,
             "reference_frame": ["H1", "L1"],
             "psi": 1.0,
             "theta_jn": 0,
             "phase": 0,
             "luminosity_distance": 500 * u.Mpc},
            times=times,
        )
        waveform_eval.project(detector=self.H1)
