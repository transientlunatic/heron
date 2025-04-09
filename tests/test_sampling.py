"""
Tests against the sampling interfaces
"""

import unittest

import numpy as np
import astropy.units as u

import bilby.gw.prior

from heron.models.lalsimulation import SEOBNRv3, IMRPhenomPv2, IMRPhenomPv2_FakeUncertainty
from heron.models.lalnoise import AdvancedLIGO
from heron.injection import make_injection
from heron.detector import AdvancedLIGOHanford, AdvancedLIGOLivingston, AdvancedVirgo
from heron.likelihood import MultiDetector, TimeDomainLikelihood, TimeDomainLikelihoodModelUncertainty

import heron.sampling
import heron.priors

class TestNessai(unittest.TestCase):

    def setUp(self):

        self.waveform = IMRPhenomPv2()
        self.psd_model = AdvancedLIGO()

        self.injection_parameters = {"luminosity_distance": 1000*u.megaparsec,
                                     "mass_ratio": 1.0,
                                     "total_mass": 40 * u.solMass,
                                     "ra": 1.0, "dec": 0, "psi": 0,
                                     "theta_jn": 0.1, "phase": 0,
                                     }
        
        self.injections = make_injection(waveform=IMRPhenomPv2,
                                         injection_parameters=self.injection_parameters,
                                         detectors={"AdvancedLIGOHanford": "AdvancedLIGO",
                                                    "AdvancedLIGOLivingston": "AdvancedLIGO"}
                                         )


        prior_dictionary = {
            "luminosity_distance": {
                "function": "UniformSourceFrame",
                "name": 'luminosity_distance',
                "minimum": 1e2,
                "maximum": 5e3
            },
            "mass_ratio": {
                "function": "UniformInComponentsMassRatio",
                "name": 'mass_ratio',
                "minimum": 0.125,
                "maximum": 1
                }
        }
        self.priors = heron.priors.PriorDict()
        self.priors.from_dictionary(prior_dictionary)
        data = self.injections['H1']

        self.likelihood = TimeDomainLikelihood(data,
                                               waveform=self.waveform,
                                               psd=self.psd_model,
                                               detector=AdvancedLIGOHanford())
        self.sampler = heron.sampling.NessaiSampler(self.likelihood,
                                                    self.priors,
                                                    self.injection_parameters)


    def test_sample(self):
        # Draw a test sample using Nessai
        
        self.assertTrue(isinstance(self.sampler.log_likelihood(self.injection_parameters), float))
