"""
Tests for the time-domain likelihood functions.
"""


import unittest

import numpy as np

from heron.likelihood import TimeDomainLikelihood, TimeDomainLikelihoodModelUncertainty
from heron.models.testing import SineGaussianWaveform, FlatPSD
from heron.detector import AdvancedLIGOHanford, AdvancedLIGOLivingston

from astropy import units as u
from astropy import coordinates as coords


from unittest.mock import Mock
from heron.likelihood import (
    TimeDomainLikelihood,
    TimeDomainLikelihoodModelUncertainty,
    MultiDetector,
    Likelihood
)

class TestTDLikelihood(unittest.TestCase):

    def setUp(self):
        test_waveform = SineGaussianWaveform()
        location = coords.SkyCoord(ra=10.625*u.degree, dec=41.2*u.degree, frame='icrs')
        position = {"ra": float(location.ra.to(u.radian).value),
            "dec": float(location.dec.to(u.radian).value),
            "psi": 0,
            "iota": 0,
            "phi_0": 0,
            "theta_jn": 0,
            "phase": 0,
        }
        self.inj_params = position.copy()
        self.inj_params.update({"width": 0.04 * u.second, "frequency": 500 * u.Hertz, 'amplitude':1.0})
        self.data = test_waveform.time_domain(
            self.inj_params
        )
        self.injection = self.data.project(detector=AdvancedLIGOHanford(), **self.inj_params)
        self.likelihood = TimeDomainLikelihood(data=self.injection,
                                  psd=FlatPSD(),
                                  detector=AdvancedLIGOHanford(),
                                  waveform=SineGaussianWaveform(),
        )

    def test_snr(self):
        snr_like = self.likelihood.snr(self.injection)
        snr_direct = np.sqrt(np.sum(self.injection**2)).value
        #print("SNR from likelihood: ", self.likelihood.snr(self.injection))
        #print("SNR from direct calc: ",np.sqrt(np.sum(self.injection**2)))
        self.assertAlmostEqual(snr_like,snr_direct, 2, "SNR from likelihood should match direct calculation, got {} vs {}".format(snr_like, snr_direct))
        
    def test_null_likelihood(self):
        logL_inj = self.likelihood(self.inj_params)
        empty_wf_params = self.inj_params.copy()
        empty_wf_params.update({"amplitude": 0.0})
        null_likelihood = self.likelihood(empty_wf_params)
        delta_logL= logL_inj - null_likelihood
        #print("null_likelihood = ", null_likelihood)
        #print("delta logL of injection ", delta_logL)
        #print("logL of injection ", logL_inj)
        snr_like = self.likelihood.snr(self.injection)

        self.assertAlmostEqual(delta_logL, 0.5*snr_like**2, 2, "delta log likelihood for injection parameters should be 0.5SNR^2, got {} vs {}".format(delta_logL, 0.5*snr_like**2))

class TestUncertainTDLikelihood(unittest.TestCase):

    def setUp(self):
        test_waveform = SineGaussianWaveform()
        location = coords.SkyCoord(ra=10.625*u.degree, dec=41.2*u.degree, frame='icrs')
        position = {"ra": float(location.ra.to(u.radian).value),
            "dec": float(location.dec.to(u.radian).value),
            "psi": 0,
            "iota": 0,
            "phi_0": 0,
            "theta_jn": 0,
            "phase": 0,
        }
        self.inj_params = position.copy()
        self.inj_params.update({"width": 0.04 * u.second, "frequency": 500 * u.Hertz, 'amplitude':1.0})
        self.data = test_waveform.time_domain(
            self.inj_params
        )
        self.injection = self.data.project(detector=AdvancedLIGOHanford(), **self.inj_params)
        self.likelihood = TimeDomainLikelihoodModelUncertainty(data=self.injection,
                                  psd=FlatPSD(),
                                  detector=AdvancedLIGOHanford(),
                                  waveform=SineGaussianWaveform(),
        )

    def test_snr(self):
        snr_like = self.likelihood.snr(self.injection)
        snr_direct = np.sqrt(np.sum(self.injection**2)).value
        #print("SNR from likelihood: ", self.likelihood.snr(self.injection))
        #print("SNR from direct calc: ",np.sqrt(np.sum(self.injection**2)))
        self.assertAlmostEqual(snr_like,snr_direct, 2, "SNR from likelihood should match direct calculation, got {} vs {}".format(snr_like, snr_direct))
        
    def test_null_likelihood(self):
        logL_inj = self.likelihood(self.inj_params)
        empty_wf_params = self.inj_params.copy()
        empty_wf_params.update({"amplitude": 0.0})
        null_likelihood = self.likelihood(empty_wf_params)
        delta_logL= logL_inj - null_likelihood
        #print("null_likelihood = ", null_likelihood)
        #print("delta logL of injection ", delta_logL)
        #print("logL of injection ", logL_inj)
        snr_like = self.likelihood.snr(self.injection)

        self.assertAlmostEqual(delta_logL, 0.5*snr_like**2, 2, "delta log likelihood for injection parameters should be 0.5SNR^2, got {} vs {}".format(delta_logL, 0.5*snr_like**2))
