"""
Tests for the time-domain likelihood functions.
"""


import unittest

import numpy as np

from heron.models.testing import FlatPSD, SineGaussianWaveform
from heron.models.gpytorch import HeronNonSpinningApproximant
from heron import likelihood
from heron.detector import KNOWN_IFOS

from astropy import units as u

from unittest.mock import Mock
from heron.likelihood import (
    TimeDomainLikelihood,
    TimeDomainLikelihoodModelUncertainty,
    MultiDetector,
    Likelihood
)


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

class TestTDLikelihoodUncertaintyPyTorch(unittest.TestCase):

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
        
        self.likelihood = likelihood.TimeDomainLikelihoodModelUncertaintyGPU(
            data=self.data.project(
                            detector=KNOWN_IFOS["AdvancedLIGOLivingston"]()),
            psd=self.psd,
            detector=KNOWN_IFOS["AdvancedLIGOLivingston"](),
            waveform=SineGaussianWaveform(),
        )

        self.assertTrue(self.likelihood.device, "gpu")

    def test_evaluate(self):

        likelihoods = []
        for w in np.linspace(0.01, 0.15, 100):
        
            likelihoods.append(self.likelihood({"width": w,
                                                "ra": 1, "dec": 1,
                                                "phase": 0, "psi": 0,
                                                "theta_jn": 1}).to("cpu"))
        likelihoods = np.array(likelihoods)
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1,1)
        ax.plot(np.linspace(0.01, 0.15, 100), likelihoods)
        f.savefig("test_likelihood_unc_plot.png")
        print(np.linspace(0.01, 0.15, 100)[np.argmin(likelihoods)]  - self.inject)
        self.assertTrue(np.abs(
            np.linspace(0.01, 0.15, 100)[np.argmax(likelihoods)]  - self.inject)
            < (0.19-0.01)/100
        )


class TestTDLikelihood_GP(unittest.TestCase):

    @unittest.skip("HeronNonSpinningApproximant requires training data - test not properly configured")
    def setUp(self):
        self.sample_rate = 1024
        self.duration = 2 # seconds
        self.psd = FlatPSD()
        self.inject = 0.8
        self.data = HeronNonSpinningApproximant().time_domain(
            parameters={"mass ratio":self.inject,
                        "total mass": 50,
                        "ra": 1, "dec": 1,
                        "phase": 0, "psi": 0,
                        "theta_jn": 1}).project(
                            detector=KNOWN_IFOS["AdvancedLIGOLivingston"]())

        f = self.data.plot()
        f.savefig("test_data_plot_gp.png")

        data = SineGaussianWaveform().time_domain(
            parameters={
                "mass ratio":self.inject,
                "total mass": 50,
                "ra": 1, "dec": 1,
                "phase": 0, "psi": 0,
                "theta_jn": 1}).project(
                    detector=KNOWN_IFOS["AdvancedLIGOLivingston"]())

        f = data.plot()
        f.savefig("test_data_plot_2_gp.png")
        
        self.likelihood = likelihood.TimeDomainLikelihood(
            data=self.data,
            psd=self.psd,
            detector=KNOWN_IFOS["AdvancedLIGOLivingston"](),
            waveform=SineGaussianWaveform(),
        )

    def test_evaluate(self):

        likelihoods = []
        for _ in np.linspace(0.01, 1.0, 101):
        
            likelihoods.append(self.likelihood({"mass ratio":self.inject,
                                                "total mass": 50,
                                                "ra": 1, "dec": 1,
                                                "phase": 0, "psi": 0,
                                                "theta_jn": 1}))
        likelihoods = np.array(likelihoods)
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1,1)
        ax.plot(np.linspace(0.01, 1.0, 101), likelihoods)
        f.savefig("test_likelihood_plot_gp.png")
        self.assertTrue(
            np.abs(
                np.linspace(0.01, 1.0, 101)[np.argmax(likelihoods)] - self.inject)
            < (0.19-0.01)/100
            )


class TestLikelihoodBase(unittest.TestCase):
    """Test the base Likelihood class methods."""

    def setUp(self):
        self.lik = Likelihood()

    def test_logdet(self):
        A = np.array([[2, 1], [1, 2]])
        result = self.lik.logdet(A)
        expected = np.log(np.linalg.det(A))
        self.assertTrue(np.isclose(result, expected))

    def test_inverse(self):
        A = np.array([[4, 2], [1, 3]])
        result = self.lik.inverse(A)
        expected = np.linalg.inv(A)
        self.assertTrue(np.allclose(result, expected))

    def test_solve(self):
        A = np.array([[3, 1], [1, 2]])
        b = np.array([9, 8])
        result = self.lik.solve(A, b)
        expected = np.linalg.solve(A, b)
        self.assertTrue(np.allclose(result, expected))

    def test_eye(self):
        result = self.lik.eye(3)
        expected = np.eye(3)
        self.assertTrue(np.allclose(result, expected))


class TestTimeDomainLikelihood(unittest.TestCase):
    """Test the TimeDomainLikelihood class."""

    def setUp(self):
        """Create mock time series data and likelihood."""
        self.mock_data = Mock()
        self.mock_data.data = np.random.randn(100)
        self.mock_data.times = np.linspace(0, 1, 100)
        self.mock_data.determine_overlap = Mock(return_value=((0, 100), (0, 100)))
        
        self.mock_psd = Mock()
        self.mock_psd.covariance_matrix = Mock(return_value=np.eye(100))
        
        self.likelihood = TimeDomainLikelihood(self.mock_data, self.mock_psd)

    def test_initialization(self):
        """Test that likelihood initializes correctly."""
        self.assertEqual(self.likelihood.N, 100)
        self.assertEqual(len(self.likelihood.data), 100)
        self.assertEqual(self.likelihood.C.shape, (100, 100))
        self.assertTrue(np.allclose(self.likelihood.inverse_C, np.eye(100)))

    def test_log_likelihood_returns_finite(self):
        """Test that log_likelihood returns a finite value."""
        waveform = Mock()
        waveform.data = np.random.randn(100)
        
        result = self.likelihood.log_likelihood(waveform)
        self.assertTrue(np.isfinite(result))
        self.assertIsInstance(result, (float, np.floating))

    def test_log_likelihood_no_overlap(self):
        """Test that log_likelihood returns -inf when no overlap."""
        self.mock_data.determine_overlap = Mock(return_value=None)
        waveform = Mock()
        waveform.data = np.random.randn(100)
        
        result = self.likelihood.log_likelihood(waveform)
        self.assertEqual(result, -np.inf)

    def test_log_likelihood_sign(self):
        """Test that log_likelihood is negative for poor fits."""
        waveform = Mock()
        # Create a waveform very different from data (which is noise)
        waveform.data = np.ones(100) * 100
        
        result = self.likelihood.log_likelihood(waveform)
        self.assertLess(result, 0)

    def test_log_likelihood_normalization(self):
        """Test log_likelihood with and without normalization."""
        waveform = Mock()
        waveform.data = np.random.randn(100)
        
        result_with_norm = self.likelihood.log_likelihood(waveform, norm=True)
        result_without_norm = self.likelihood.log_likelihood(waveform, norm=False)
        
        self.assertNotEqual(result_with_norm, result_without_norm)
        self.assertLess(result_with_norm, result_without_norm)

    def test_snr_positive(self):
        """Test that SNR is positive."""
        waveform = Mock()
        waveform.data = np.random.randn(100)
        
        snr = self.likelihood.snr(waveform)
        self.assertGreaterEqual(snr, 0)

    def test_snr_zero_waveform(self):
        """Test SNR for zero waveform."""
        waveform = Mock()
        waveform.data = np.zeros(100)
        
        snr = self.likelihood.snr(waveform)
        self.assertEqual(snr, 0)

    def test_call_with_parameters(self):
        """Test calling likelihood with parameters."""
        # Setup mocks
        waveform = Mock()
        waveform._args = {'mass_1': None, 'mass_2': None}
        waveform.time_domain = Mock()
        
        mock_td_waveform = Mock()
        mock_td_waveform.data = np.random.randn(100)
        mock_projected = Mock()
        mock_projected.data = np.random.randn(100)
        mock_td_waveform.project = Mock(return_value=mock_projected)
        waveform.time_domain.return_value = mock_td_waveform
        
        detector = Mock()
        
        self.likelihood.waveform = waveform
        self.likelihood.detector = detector
        
        parameters = {'mass_1': 10.0, 'mass_2': 10.0}
        result = self.likelihood(parameters)
        
        self.assertTrue(np.isfinite(result))
        waveform.time_domain.assert_called_once()
        mock_td_waveform.project.assert_called_once_with(detector)


class TestTimeDomainLikelihoodModelUncertainty(unittest.TestCase):
    """Test the TimeDomainLikelihoodModelUncertainty class."""

    def setUp(self):
        self.mock_data = Mock()
        self.mock_data.data = np.random.randn(100)
        self.mock_data.times = np.linspace(0, 1, 100)
        self.mock_data.determine_overlap = Mock(return_value=((0, 100), (0, 100)))
        
        self.mock_psd = Mock()
        self.mock_psd.covariance_matrix = Mock(return_value=np.eye(100))
        
        self.likelihood = TimeDomainLikelihoodModelUncertainty(
            self.mock_data, self.mock_psd
        )

    def test_log_likelihood_with_covariance(self):
        """Test log_likelihood with waveform covariance."""
        waveform = Mock()
        waveform.data = np.random.randn(100)
        waveform.covariance = np.eye(100) * 0.01
        
        result = self.likelihood.log_likelihood(waveform)
        self.assertTrue(np.isfinite(result))
        self.assertIsInstance(result, (float, np.floating))

    def test_log_likelihood_accounts_for_model_uncertainty(self):
        """Test that model uncertainty affects likelihood."""
        waveform = Mock()
        waveform.data = np.random.randn(100)
        waveform.covariance = np.eye(100) * 0.01
        
        result_small_cov = self.likelihood.log_likelihood(waveform)
        
        # Larger model uncertainty should give larger (less negative) likelihood
        waveform.covariance = np.eye(100) * 1.0
        result_large_cov = self.likelihood.log_likelihood(waveform)
        
        # Larger covariance typically means less certain model,
        # which should affect the likelihood
        self.assertNotEqual(result_small_cov, result_large_cov)


class TestMultiDetector(unittest.TestCase):
    """Test the MultiDetector class."""

    def test_initialization_with_likelihoods(self):
        """Test MultiDetector initialization."""
        lik1 = Mock(spec=TimeDomainLikelihood)
        lik2 = Mock(spec=TimeDomainLikelihood)
        
        multi = MultiDetector(lik1, lik2)
        self.assertAlmostEqual(len(multi._likelihoods), 2)

    def test_empty_initialization(self):
        """Test MultiDetector with no likelihoods."""
        multi = MultiDetector()
        self.assertEqual(len(multi._likelihoods), 0)
        
        result = multi({'mass_1': 10.0})
        self.assertEqual(result, 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_singular_covariance_matrix(self):
        """Test behavior with singular covariance matrix."""
        data = Mock()
        data.data = np.random.randn(10)
        data.times = np.linspace(0, 1, 10)
        data.determine_overlap = Mock(return_value=((0, 10), (0, 10)))
        
        psd = Mock()
        # Singular matrix
        C = np.zeros((10, 10))
        C[0, 0] = 1
        psd.covariance_matrix = Mock(return_value=C)
        
        with self.assertRaises((np.linalg.LinAlgError, ZeroDivisionError)):
            TimeDomainLikelihood(data, psd)

    def test_mismatched_dimensions(self):
        """Test handling of mismatched array dimensions."""
        data = Mock()
        data.data = np.random.randn(100)
        data.times = np.linspace(0, 1, 100)*u.second
        
        psd = Mock()
        psd.covariance_matrix = Mock(return_value=np.eye(100))
        
        likelihood = TimeDomainLikelihood(data, psd)
        
        waveform = Mock()
        waveform.data = np.random.randn(50)  # Different size
        data.determine_overlap = Mock(return_value=((0, 50), (0, 50)))
        
        # Should handle gracefully
        result = likelihood.log_likelihood(waveform)
        self.assertTrue(np.isfinite(result))


if __name__ == '__main__':
    unittest.main()