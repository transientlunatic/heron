"""
Tests for the time-domain likelihood functions.
"""


import unittest

import numpy as np

from heron.models.testing import FlatPSD, SineGaussianWaveform
from heron import likelihood
from heron.detector import KNOWN_IFOS

class TestTDLikelihood(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 1024
        self.duration = 2 # seconds
        N = self.sample_rate * self.duration
        self.psd = FlatPSD()
        self.inject = 0.02
        self.data = SineGaussianWaveform().time_domain(
            parameters={"width":self.inject,
                        "ra": 1, "dec": 1,
                        "phase": 0, "psi": 0,
                        "theta_jn": 1}).project(
                            detector=KNOWN_IFOS["AdvancedLIGOLivingston"]())

        f = self.data.plot()
        f.savefig("test_data_plot.png")

        data = SineGaussianWaveform().time_domain(
            parameters={"width":self.inject,
                        "ra": 1, "dec": 1,
                        "phase": 0, "psi": 0,
                        "theta_jn": 1}).project(
                            detector=KNOWN_IFOS["AdvancedLIGOLivingston"]())

        f = data.plot()
        f.savefig("test_data_plot_2.png")
        
        self.likelihood = likelihood.TimeDomainLikelihood(
            data=self.data,
            psd=self.psd,
            detector=KNOWN_IFOS["AdvancedLIGOLivingston"](),
            waveform=SineGaussianWaveform(),
        )

    def test_evaluate(self):

        likelihoods = []
        for w in np.linspace(0.01, 0.19, 101):
        
            likelihoods.append(self.likelihood({"width": w,
                                                "ra": 1, "dec": 1,
                                                "phase": 0, "psi": 0,
                                                "theta_jn": 1}))
        likelihoods = np.array(likelihoods)
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1,1)
        ax.plot(np.linspace(0.01, 0.19, 101), likelihoods)
        f.savefig("test_likelihood_plot.png")
        self.assertTrue(
            np.abs(
                np.linspace(0.01, 0.19, 101)[np.argmax(likelihoods)] - self.inject)
            < (0.19-0.01)/100
            )


class TestTDLikelihoodUncertainty(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 1024
        self.duration = 2 # seconds
        N = self.sample_rate * self.duration
        self.psd = FlatPSD()
        self.inject = 0.02
        self.data = SineGaussianWaveform().time_domain(
            parameters={"width": self.inject,
                        "ra": 1, "dec": 1,
                        "phase": 0, "psi": 0,
                        "theta_jn": 1})

        f = self.data['plus'].plot()
        f.savefig("test_data_plot_unc.png")
        
        self.likelihood = likelihood.TimeDomainLikelihoodModelUncertainty(
            data=self.data.project(
                            detector=KNOWN_IFOS["AdvancedLIGOLivingston"]()),
            psd=self.psd,
            detector=KNOWN_IFOS["AdvancedLIGOLivingston"](),
            waveform=SineGaussianWaveform(),
        )

    def test_evaluate(self):

        likelihoods = []
        for w in np.linspace(0.01, 0.19, 100):
        
            likelihoods.append(self.likelihood({"width": w,
                                                "ra": 1, "dec": 1,
                                                "phase": 0, "psi": 0,
                                                "theta_jn": 1}))
        likelihoods = np.array(likelihoods)
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1,1)
        ax.plot(np.linspace(0.01, 0.19, 100), likelihoods)
        f.savefig("test_likelihood_unc_plot.png")
        self.assertTrue(np.abs(
            np.linspace(0.01, 0.19, 100)[np.argmax(likelihoods)]  - self.inject)
            < (0.19-0.01)/100
        )
