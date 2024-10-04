import unittest

import numpy as np
import astropy.units as u
import bilby.gw.prior

from heron.models.lalsimulation import SEOBNRv3, IMRPhenomPv2, IMRPhenomPv2_FakeUncertainty
from heron.models.lalnoise import AdvancedLIGO
from heron.injection import make_injection, make_injection_zero_noise
from heron.detector import (Detector,
                            AdvancedLIGOHanford,
                            AdvancedLIGOLivingston,
                            AdvancedVirgo)
from heron.likelihood import (MultiDetector,
                              TimeDomainLikelihood,
                              TimeDomainLikelihoodModelUncertainty,
                              TimeDomainLikelihoodPyTorch,
                              TimeDomainLikelihoodModelUncertaintyPyTorch)

from heron.inference import heron_inference, parse_dict, load_yaml

from torch.cuda import is_available

CUDA_NOT_AVAILABLE = not is_available()


class Test_Likelihood_ZeroNoise_With_Uncertainty(unittest.TestCase):
    """
    Test likelihoods on a zero noise injection.
    """
    
    def setUp(self):
        self.waveform = IMRPhenomPv2_FakeUncertainty()
        self.psd_model = AdvancedLIGO()

        self.injections = make_injection_zero_noise(waveform=IMRPhenomPv2,
                                         injection_parameters={"distance": 1000*u.megaparsec,
                                                               "mass_ratio": 0.6,
                                                               "gpstime": 0,
                                                               "total_mass": 60 * u.solMass},
                                         detectors={"AdvancedLIGOHanford": "AdvancedLIGO",
                                                    "AdvancedLIGOLivingston": "AdvancedLIGO"}
                                         )



    def test_likelihood_maximum_at_true_value_mass_ratio(self):
        
        data = self.injections['H1']

        likelihood = TimeDomainLikelihoodModelUncertainty(data, psd=self.psd_model)
        mass_ratios = np.linspace(0.1, 1.0, 100)

        log_likes = []
        for mass_ratio in mass_ratios:
        
            test_waveform = self.waveform.time_domain(parameters={"distance": 1000*u.megaparsec,
                                                                   "mass_ratio": mass_ratio,
                                                                  "gpstime": 0,
                                                                   "total_mass": 60 * u.solMass}, times=likelihood.times)
            projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                       ra=0, dec=0,
                                                       gpstime=0,
                                                       phi_0=0, psi=0,
                                                       iota=0)

            log_likes.append(likelihood.log_likelihood(projected_waveform, norm=False))

        self.assertTrue(np.abs(mass_ratios[np.argmax(log_likes)] - 0.6) < 0.1)

@unittest.skipIf(CUDA_NOT_AVAILABLE, "CUDA is not installed on this system")
class Test_Likelihood_ZeroNoise_With_Uncertainty_PyTorch(unittest.TestCase):
    """
    Test likelihoods on a zero noise injection.
    """
    
    def setUp(self):
        self.waveform = IMRPhenomPv2_FakeUncertainty()
        self.psd_model = AdvancedLIGO()

        self.injections = make_injection_zero_noise(waveform=IMRPhenomPv2,
                                         injection_parameters={"distance": 1000*u.megaparsec,
                                                               "mass_ratio": 0.6,
                                                               "gpstime": 0,
                                                               "total_mass": 60 * u.solMass},
                                         detectors={"AdvancedLIGOHanford": "AdvancedLIGO",
                                                    "AdvancedLIGOLivingston": "AdvancedLIGO"}
                                         )



    def test_likelihood_maximum_at_true_value_mass_ratio(self):
        
        data = self.injections['H1']

        likelihood = TimeDomainLikelihoodModelUncertaintyPyTorch(data, psd=self.psd_model)
        mass_ratios = np.linspace(0.1, 1.0, 100)

        log_likes = []
        for mass_ratio in mass_ratios:
        
            test_waveform = self.waveform.time_domain(parameters={"distance": 1000*u.megaparsec,
                                                                   "mass_ratio": mass_ratio,
                                                                  "gpstime": 0,
                                                                   "total_mass": 60 * u.solMass}, times=likelihood.times)
            projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                       ra=0, dec=0,
                                                       gpstime=0,
                                                       phi_0=0, psi=0,
                                                       iota=0)

            log_likes.append(likelihood.log_likelihood(projected_waveform, norm=False).cpu().numpy())

        self.assertTrue(np.abs(mass_ratios[np.argmax(log_likes)] - 0.6) < 0.1)
