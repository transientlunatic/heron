"""

Tests of waveform inference
----------------------------

The tests in this file are intended to verify that the code is capable
of performing the various steps required in order to perform Bayesian
inference using a given signal model.

"""

import unittest

import numpy as np
import astropy.units as u

from heron.models.lalsimulation import SEOBNRv3, IMRPhenomPv2, IMRPhenomPv2_FakeUncertainty
from heron.models.lalnoise import AdvancedLIGO
from heron.injection import make_injection
from heron.detector import AdvancedLIGOHanford, AdvancedLIGOLivingston, AdvancedVirgo
from heron.likelihood import MultiDetector, TimeDomainLikelihood, TimeDomainLikelihoodModelUncertainty

class Test_Filter(unittest.TestCase):
    """Test that filters can be applied correctly to data."""

    def setUp(self):
        self.waveform = IMRPhenomPv2()
        self.psd_model = AdvancedLIGO()

        self.injections = make_injection(waveform=IMRPhenomPv2,
                                         injection_parameters={"distance": 1000*u.megaparsec},
                                         detectors={AdvancedLIGOHanford: AdvancedLIGO,
                                                    AdvancedLIGOLivingston: AdvancedLIGO}
                                         )

    def test_timedomain_psd(self):
        noise = self.psd_model.time_domain(times=self.injections['H1'].times)
        #print(noise)
        
    def test_snr(self):
        data = self.injections['H1']

        likelihood = TimeDomainLikelihood(data, psd=self.psd_model)
        
        test_waveform = self.waveform.time_domain(parameters={"m1": 35*u.solMass,
                                                              "m2": 30*u.solMass,
                                                              "distance": 410 * u.megaparsec}, times=data.times)
        
        snr = likelihood.snr(test_waveform.project(AdvancedLIGOHanford(),
                                                   ra=0, dec=0,
                                                   phi_0=0, psi=0,
                                                   iota=0))
        self.assertTrue(snr > 80 and snr < 90)

    # def test_snr_f(self):
    #     data = self.injections['H1']

    #     likelihood = TimeDomainLikelihood(data, psd=self.psd_model)
        
    #     test_waveform = self.waveform.time_domain(parameters={"m1": 35*u.solMass,
    #                                                           "m2": 30*u.solMass,
    #                                                           "distance": 410 * u.megaparsec}, times=data.times)
        
    #     snr = likelihood.snr_f(test_waveform.project(AdvancedLIGOHanford(),
    #                                                ra=0, dec=0,
    #                                                phi_0=0, psi=0,
    #                                                iota=0))
    #     self.assertTrue(snr > 80 and snr < 90)

        
        
    def test_likelihood(self):
        data = self.injections['H1']

        likelihood = TimeDomainLikelihood(data, psd=self.psd_model)
        
        test_waveform = self.waveform.time_domain(parameters={"m1": 40*u.solMass,
                                                              "m2": 50*u.solMass,
                                                              "distance": 200 * u.megaparsec}, times=data.times)

        projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                              ra=0, dec=0,
                                                              phi_0=0, psi=0,
                                                              iota=0)
        
        log_like = likelihood.log_likelihood(projected_waveform)

    def test_likelihood_with_uncertainty(self):
        data = self.injections['H1']

        likelihood = TimeDomainLikelihoodModelUncertainty(data, psd=self.psd_model)

        waveform = IMRPhenomPv2_FakeUncertainty()
        test_waveform = waveform.time_domain(parameters={"m1": 40*u.solMass,
                                                              "m2": 50*u.solMass,
                                                              "distance": 200 * u.megaparsec}, times=data.times)
        projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                              ra=0, dec=0,
                                                              phi_0=0, psi=0,
                                                              iota=0)
        log_like = likelihood.log_likelihood(projected_waveform)


    def test_sampling_with_uncertainty(self):
        waveform = IMRPhenomPv2_FakeUncertainty()
        likelihood = TimeDomainLikelihoodModelUncertainty(self.injections['H1'],
                                                          psd=self.psd_model,
                                                          waveform=waveform,
                                                          detector=AdvancedLIGOHanford())

        parameters = {"m1": 40*u.solMass,
                      "m2": 50*u.solMass,
                      "distance": 200 * u.megaparsec,
                      "ra": 0.1,
                      "dec": 0.1,
                      "theta_jn": 0.1,
                      "psi": 0,
                      "phase": 0,
                      }
        
        log_like = likelihood(parameters=parameters)

        self.assertTrue(-200000 < log_like < -100000)

    def test_sampling_with_uncertainty_multi(self):
        waveform = IMRPhenomPv2_FakeUncertainty()
        likelihood = MultiDetector(TimeDomainLikelihoodModelUncertainty(self.injections['H1'],
                                                                        psd=self.psd_model,
                                                                        waveform=waveform,
                                                                        detector=AdvancedLIGOHanford()),
                                   TimeDomainLikelihoodModelUncertainty(self.injections['L1'],
                                                                        psd=self.psd_model,
                                                                        waveform=waveform,
                                                                        detector=AdvancedLIGOLivingston())
                                   )

        parameters = {"m1": 40*u.solMass,
                      "m2": 50*u.solMass,
                      "distance": 200 * u.megaparsec,
                      "ra": 0.1,
                      "dec": 0.1,
                      "theta_jn": 0.1,
                      "psi": 0,
                      "phase": 0,
                      }
        
        log_like = likelihood(parameters=parameters)
        self.assertTrue(-400000 < log_like < -200000)
