"""

Tests of waveform inference
----------------------------

The tests in this file are intended to verify that the code is capable
of performing the various steps required in order to perform Bayesian
inference using a given signal model.

"""

import unittest
import logging

import numpy as np
import torch
import astropy.units as u
import bilby.gw.prior

from heron.models.lalsimulation import SEOBNRv3, IMRPhenomPv2, IMRPhenomPv2_FakeUncertainty
from heron.models.lalnoise import AdvancedLIGO
from heron.injection import make_injection, make_injection_zero_noise
from heron.detector import Detector, AdvancedLIGOHanford, AdvancedLIGOLivingston, AdvancedVirgo
from heron.likelihood import MultiDetector, TimeDomainLikelihood, TimeDomainLikelihoodModelUncertainty, TimeDomainLikelihoodPyTorch, TimeDomainLikelihoodModelUncertaintyPyTorch

from heron.inference import heron_inference, parse_dict, load_yaml

logging.basicConfig(level=logging.WARNING)

#@unittest.skip("Temp")
class Test_TimeDomain_Numpy_NoUncertainty(unittest.TestCase):
    """Test that filters can be applied correctly to data."""

    def setUp(self):
        self.waveform = IMRPhenomPv2()
        self.psd_model = AdvancedLIGO()

        self.injections = make_injection_zero_noise(waveform=IMRPhenomPv2,
                                         injection_parameters={"m1": 35*u.solMass,
                                                              "m2": 30*u.solMass,
                                                              "gpstime": 4000,
                                                              "distance": 410 * u.megaparsec},
                                         detectors={"AdvancedLIGOHanford": "AdvancedLIGO",
                                                    "AdvancedLIGOLivingston": "AdvancedLIGO"}
                                         )

    def test_timedomain_psd(self):
        noise = self.psd_model.time_domain(times=self.injections['H1'].times)
        #print(noise)
        
    def test_snr(self):
        data = self.injections['H1']

        likelihood = TimeDomainLikelihood(data, psd=self.psd_model)
        
        test_waveform = self.waveform.time_domain(parameters={"m1": 35*u.solMass,
                                                              "m2": 30*u.solMass,
                                                              "gpstime": 4000,
                                                              "distance": 410 * u.megaparsec}, times=data.times)
        
        snr = likelihood.snr(test_waveform.project(AdvancedLIGOHanford(),
                                                   ra=0, dec=0,
                                                   phi_0=0, psi=0,
                                                   iota=0))
        self.assertTrue(snr > 80 and snr < 90)

    def test_snr_f(self):
        data = self.injections['H1']

        likelihood = TimeDomainLikelihood(data, psd=self.psd_model)
        
        test_waveform = self.waveform.time_domain(parameters={"m1": 35*u.solMass,
                                                              "m2": 30*u.solMass,
                                                              "gpstime": 4000,
                                                              "distance": 410 * u.megaparsec}, times=data.times)
        
        snr = likelihood.snr_f(test_waveform.project(AdvancedLIGOHanford(),
                                                   ra=0, dec=0,
                                                   phi_0=0, psi=0,
                                                   iota=0))
        self.assertTrue(snr > 80 and snr < 90)

    def test_snr_equal_f_and_t(self):
        data = self.injections['H1']

        likelihood = TimeDomainLikelihood(data, psd=self.psd_model)
        
        test_waveform = self.waveform.time_domain(parameters={"m1": 35*u.solMass,
                                                              "m2": 30*u.solMass,
                                                              "gpstime": 4000,
                                                              "distance": 410 * u.megaparsec}, times=data.times)
        
        snr = likelihood.snr(test_waveform.project(AdvancedLIGOHanford(),
                                                   ra=0, dec=0,
                                                   phi_0=0, psi=0,
                                                   iota=0))
        snr_f = likelihood.snr_f(test_waveform.project(AdvancedLIGOHanford(),
                                                   ra=0, dec=0,
                                                   phi_0=0, psi=0,
                                                   iota=0))

        self.assertLessEqual(np.abs(snr-snr_f), 0.05)
        
    def test_likelihood(self):
        data = self.injections['H1']

        likelihood = TimeDomainLikelihood(data, psd=self.psd_model)
        
        test_waveform = self.waveform.time_domain(parameters={"m1": 35*u.solMass,
                                                              "m2": 30*u.solMass,
                                                              "gpstime": 4000,
                                                              "distance": 410 * u.megaparsec}, times=data.times)

        projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                              ra=0, dec=0,
                                                              phi_0=0, psi=0,
                                                              iota=0)
        
        log_like = likelihood.log_likelihood(projected_waveform)
        self.assertLessEqual(log_like, 1)

    def test_likelihood_no_norm(self):
        """Test that the maximum likelihood occurs at the injection."""
        
        data = self.injections['H1']

        likelihood = TimeDomainLikelihood(data, psd=self.psd_model)

        likes = []
        for m1 in np.linspace(34.9, 35.1, 21):
        
            test_waveform = self.waveform.time_domain(parameters={"m1": m1*u.solMass,
                                                                  "m2": 30*u.solMass,
                                                                  "gpstime": 4000,
                                                                  "distance": 410 * u.megaparsec}, times=data.times)

            projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                                  ra=0, dec=0,
                                                                  phi_0=0, psi=0,
                                                                  iota=0)

            log_like = likelihood.log_likelihood(projected_waveform, norm=False)
            likes.append(log_like)
            self.assertLessEqual(log_like, 1)
        self.assertEqual(likes[10], 0)
        self.assertEqual(np.max(likes), likes[10])
        self.assertFalse(likes[0] == likes[1])

    def test_likelihood_with_norm(self):
        """Test that the maximum likelihood occurs at the injection."""
        
        data = self.injections['H1']

        likelihood = TimeDomainLikelihood(data, psd=self.psd_model)

        likes = []
        for m1 in np.linspace(34.9, 35.1, 21):
        
            test_waveform = self.waveform.time_domain(parameters={"m1": m1*u.solMass,
                                                                  "m2": 30*u.solMass,
                                                                  "gpstime": 4000,
                                                                  "distance": 410 * u.megaparsec}, times=data.times)

            projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                                  ra=0, dec=0,
                                                                  phi_0=0, psi=0,
                                                                  iota=0)

            log_like = likelihood.log_likelihood(projected_waveform)
            likes.append(log_like)
            self.assertLessEqual(log_like, 1)
        self.assertEqual(np.max(likes), likes[10])
        self.assertFalse(likes[0] == likes[1])

#@unittest.skip("Temp")
class Test_TimeDomain_Numpy_NoUncertainty_Noise(Test_TimeDomain_Numpy_NoUncertainty):
    def setUp(self):
        self.waveform = IMRPhenomPv2()
        self.psd_model = AdvancedLIGO()

        self.injections = make_injection(waveform=IMRPhenomPv2,
                                         injection_parameters={"m1": 35*u.solMass,
                                                              "m2": 30*u.solMass,
                                                              "gpstime": 4000,
                                                              "distance": 410 * u.megaparsec},
                                         detectors={"AdvancedLIGOHanford": "AdvancedLIGO",
                                                    "AdvancedLIGOLivingston": "AdvancedLIGO"}
                                         )

    def test_likelihood_no_norm(self):
        pass


class Test_TimeDomain_Numpy_Uncertainty(unittest.TestCase):
    """Test that filters can be applied correctly to data."""

    def setUp(self):
        self.waveform = IMRPhenomPv2()
        self.psd_model = AdvancedLIGO()

        self.injection_p = {"m1": 35*u.solMass,
                            "m2": 30*u.solMass,
                            "gpstime": 4000,
                            "distance": 410 * u.megaparsec}
        
        self.injections = make_injection_zero_noise(waveform=IMRPhenomPv2,
                                         injection_parameters=self.injection_p,
                                         detectors={"AdvancedLIGOHanford": "AdvancedLIGO",
                                                    "AdvancedLIGOLivingston": "AdvancedLIGO"}
                                         )
    
    def test_likelihood_with_uncertainty(self):
        data = self.injections['H1']

        likelihood = TimeDomainLikelihoodModelUncertainty(data, psd=self.psd_model)
        likelihood_no = TimeDomainLikelihood(data, psd=self.psd_model)

        waveform = IMRPhenomPv2_FakeUncertainty()
        test_waveform = waveform.time_domain(parameters=self.injection_p, times=data.times, var=1e-80)
        projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                              ra=0, dec=0,
                                                              phi_0=0, psi=0,
                                                              iota=0)
        log_like = likelihood.log_likelihood(projected_waveform, norm=True)

        waveform = IMRPhenomPv2()
        test_waveform = waveform.time_domain(parameters=self.injection_p, times=data.times)
        projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                              ra=0, dec=0,
                                                              phi_0=0, psi=0,
                                                              iota=0)

        log_like_no = likelihood_no.log_likelihood(projected_waveform, norm=True)

        self.assertTrue(np.abs(log_like - log_like_no)<100)


#     def test_sampling_with_uncertainty(self):
#         waveform = IMRPhenomPv2_FakeUncertainty()
#         likelihood = TimeDomainLikelihoodModelUncertainty(self.injections['H1'],
#                                                           psd=self.psd_model,
#                                                           waveform=waveform,
#                                                           detector=AdvancedLIGOHanford())

#         parameters = {"m1": 40*u.solMass,
#                       "m2": 50*u.solMass,
#                       "distance": 200 * u.megaparsec,
#                       "ra": 0.1,
#                       "dec": 0.1,
#                       "theta_jn": 0.1,
#                       "gpstime": 4000,
#                       "psi": 0,
#                       "phase": 0,
#                       }
        
#         log_like = likelihood(parameters=parameters)
#         self.assertTrue(100000 < log_like < 10000)

#     def test_sampling_with_uncertainty_multi(self):
#         waveform = IMRPhenomPv2_FakeUncertainty()
#         likelihood = MultiDetector(TimeDomainLikelihoodModelUncertainty(self.injections['H1'],
#                                                                         psd=self.psd_model,
#                                                                         waveform=waveform,
#                                                                         detector=AdvancedLIGOHanford()),
#                                    TimeDomainLikelihoodModelUncertainty(self.injections['L1'],
#                                                                         psd=self.psd_model,
#                                                                         waveform=waveform,
#                                                                         detector=AdvancedLIGOLivingston())
#                                    )

#         parameters = {"m1": 40*u.solMass,
#                       "m2": 50*u.solMass,
#                       "distance": 200 * u.megaparsec,
#                       "ra": 0.1,
#                       "dec": 0.1,
#                       "theta_jn": 0.1,
#                       "psi": 0,
#                       "phase": 0,
#                       "gpstime": 4000,
#                       }
        
#         log_like = likelihood(parameters=parameters)
#         self.assertTrue(200000 < log_like < 10000)


# class TestInference(unittest.TestCase):

#     def setUp(self):
#         self.settings = load_yaml("tests/test_inference_config.yaml")

#     def test_parser(self):
#         outputs, _ = parse_dict(self.settings)
#         self.assertFalse("inference" in outputs)
#         self.assertTrue("psds" in outputs)
    
#     def test_parser_psds(self):
#         outputs, _ = parse_dict(self.settings)
#         self.assertTrue(isinstance(outputs["interferometers"]["H1"](), AdvancedLIGOHanford))
#         self.assertTrue(isinstance(outputs["psds"]["H1"](), AdvancedLIGO))


#     def test_sampler(self):
#         heron_inference("tests/test_inference_config.yaml")

class Test_PyTorch(unittest.TestCase):
    """Test that the pytorch likelihoods work."""

    def setUp(self):
        self.waveform = IMRPhenomPv2()
        self.psd_model = AdvancedLIGO()

        import numpy as np
        
        self.injections_zero = make_injection_zero_noise(waveform=IMRPhenomPv2,
                                         injection_parameters={"distance": 450*u.megaparsec,
                                                               "gpstime": 4000,
                                                               "total_mass": 60,
                                                               "mass_ratio": 0.9,
                                                               #"m1": 40*u.solMass,
                                                               #"m2": 50*u.solMass,
                                                               "ra": 1, "dec": 1, "phase": 0, "psi": 0, "theta_jn": 0,
                                                               },
                                         detectors={"AdvancedLIGOHanford": "AdvancedLIGO",
                                                    "AdvancedLIGOLivingston": "AdvancedLIGO"}
                                         )

        self.injections = make_injection(waveform=IMRPhenomPv2,
                                         injection_parameters={"distance": 450*u.megaparsec,
                                                               "gpstime": 4000,
                                                               #"total_mass": 60,
                                                               #"mass_ratio": 0.9,
                                                               "m1": 40*u.solMass,
                                                               "m2": 50*u.solMass,
                                                               "ra": 1, "dec": 1, "phase": 0, "psi": 0, "theta_jn": 0,
                                                               },
                                         detectors={"AdvancedLIGOHanford": "AdvancedLIGO",
                                                    "AdvancedLIGOLivingston": "AdvancedLIGO"}
                                         )

        
    # def test_timedomain_psd(self):
    #     noise = self.psd_model.time_domain(times=self.injections['H1'].times)
    #     #print(noise)
        
    # def test_snr(self):
    #     data = self.injections['H1']

    #     likelihood = TimeDomainLikelihoodPyTorch(data, psd=self.psd_model)
        
    #     test_waveform = self.waveform.time_domain(parameters={"distance": 450*u.megaparsec,
    #                                                            "gpstime": 4000,
    #                                                            "total_mass": 60,
    #                                                            "mass_ratio": 0.9,
    #                                                            "ra": 1, "dec": 1, "phase": 0, "psi": 0, "theta_jn": 0,}, times=data.times)
    #     projected = test_waveform.project(AdvancedLIGOHanford(),
    #                                       ra=1, dec=1,
    #                                       phi_0=0, psi=0,
    #                                       iota=0)
    #     snr = likelihood.snr(projected)
    #     snr_f = likelihood.snr_f(projected)
    #     self.assertLess(snr - snr_f, 0.1)
    #     self.assertTrue(snr > 150 and snr < 151)

    # def test_likelihood_no_normalisation(self):
    #     data = self.injections_zero['H1']

    #     likelihood = TimeDomainLikelihoodPyTorch(data, psd=self.psd_model)
        
    #     test_waveform = self.waveform.time_domain(parameters={"m1": 40*u.solMass,
    #                                                           "m2": 50*u.solMass,
    #                                                           "gpstime": 4000,
    #                                                           "distance": 450 * u.megaparsec}, times=data.times)

    #     projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
    #                                                           ra=0, dec=0,
    #                                                           phi_0=0, psi=0,
    #                                                           iota=0)
        
    #     log_like = likelihood.log_likelihood(projected_waveform, norm=False)
    #     print("log likelihood without normalisation, no unc", log_like)
    #     log_like = likelihood.log_likelihood(projected_waveform, norm=True)
    #     print("log likelihood with normalisation, no unc", log_like) 

    # def test_likelihood_with_uncertainty_no_normalisation(self):
    #     data = self.injections_zero['H1']

    #     likelihood = TimeDomainLikelihoodModelUncertaintyPyTorch(data, psd=self.psd_model)

    #     waveform = IMRPhenomPv2_FakeUncertainty()
    #     test_waveform = waveform.time_domain(parameters={"m1": 40*u.solMass,
    #                                                      "m2": 50*u.solMass,
    #                                                      "gpstime": 4000,
    #                                                      "distance": 450 * u.megaparsec},
    #                                          times=data.times,
    #                                          var=1e-48)
    #     projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
    #                                                           ra=0, dec=0,
    #                                                           phi_0=0, psi=0,
    #                                                           iota=0)
    #     log_like = likelihood.log_likelihood(projected_waveform, norm=False)
    #     log_like = likelihood.log_likelihood(projected_waveform, norm=True)

    # def test_likelihood_path_no_normalisation(self):
    #     data = self.injections_zero['H1']

    #     likelihood = TimeDomainLikelihood(data, psd=self.psd_model)

    #     log_like = []
    #     for m1 in np.linspace(59.99, 60.01, 5):
    #         waveform = IMRPhenomPv2()
    #         test_waveform = waveform.time_domain(parameters={"distance": 450*u.megaparsec,
    #                                                          "gpstime": 4000,
    #                                                          "total_mass": m1*u.solMass,
    #                                                          "mass_ratio": 0.9,
    #                                                          "ra": 1, "dec": 1, "phase": 0, "psi": 0, "theta_jn": 0,
    #                                                          }, times=data.times)
    #         projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
    #                                                    ra=1, dec=1,
    #                                                    phi_0=0, psi=0,
    #                                                    iota=0)
    #         log_like.append(likelihood.log_likelihood(projected_waveform, norm=False))


    #     self.assertEqual(log_like[2], 0)
    #     self.assertTrue(np.all(np.array(log_like) < 1e-4))


    def test_likelihood_single_no_normalisation_pytorch_uncert(self):
        """Test a series of likelihood evaluations in the region of the injected parameters including the likelihood normalisation.
        """
        data = self.injections_zero['H1']

        likelihood = TimeDomainLikelihoodModelUncertaintyPyTorch(data, psd=self.psd_model)

        waveform = IMRPhenomPv2_FakeUncertainty()

        log_like = []
        m1 = 60
        test_waveform = waveform.time_domain(parameters={"distance": 450*u.megaparsec,
                                                         "gpstime": 4000,
                                                         "total_mass": m1*u.solMass,
                                                           "mass_ratio": 0.9,
                                                           "ra": 1, "dec": 1, "phase": 0, "psi": 0, "theta_jn": 0,
                                                           },
                                             times=data.times,
                                             var=1e-50)
        projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                              ra=1, dec=1,
                                                              phi_0=0, psi=0,
                                                             iota=0)
        log_like = likelihood.log_likelihood(projected_waveform, norm=False).cpu().numpy()

        self.assertLessEqual(np.abs(log_like), 1e-3)              


    def test_likelihood_range_normalisation_pytorch_uncert(self):
        """Test a series of likelihood evaluations in the region of the injected parameters including the likelihood normalisation.
        """
        data = self.injections_zero['H1']

        likelihood = TimeDomainLikelihoodModelUncertaintyPyTorch(data, psd=self.psd_model)

        waveform = IMRPhenomPv2_FakeUncertainty()

        log_like = []
        for m1 in np.linspace(59.9, 60.1, 5):
            test_waveform = waveform.time_domain(parameters={"distance": 450*u.megaparsec,
                                                             "gpstime": 4000,
                                                             "total_mass": m1*u.solMass,
                                                               "mass_ratio": 0.9,
                                                               "ra": 1, "dec": 1, "phase": 0, "psi": 0, "theta_jn": 0,
                                                               }, times=data.times, var=1e-45)
            projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                                  ra=1, dec=1,
                                                                  phi_0=0, psi=0,
                                                                 iota=0)
            log_like.append(likelihood.log_likelihood(projected_waveform, norm=True).cpu().numpy())

        self.assertEqual(log_like[2], np.max(log_like))
