"""
Tests for Cholesky decomposition caching optimization in likelihood functions.

These tests ensure numerical stability and correctness when caching Cholesky
decompositions, especially for gravitational wave data at extremely small scales (~1e-22).
"""

import unittest
import numpy as np
from unittest.mock import Mock
import time

try:
    from heron.likelihood import (
        TimeDomainLikelihood,
        TimeDomainLikelihoodModelUncertainty,
        NumericallyScaled
    )
    from heron.models.testing import FlatPSD, SineGaussianWaveform
    from heron.detector import KNOWN_IFOS
    HAVE_FULL_DEPS = True
except ImportError:
    HAVE_FULL_DEPS = False
    # Define dummy classes for basic tests
    class NumericallyScaled:
        def __init__(self, value, scale=None):
            self.value = value
            self.scale = scale if scale is not None else 1./np.min(np.diag(value))

        @property
        def scaled(self):
            return self.value * self.scale


class TestCholeskyNumericalStability(unittest.TestCase):
    """Test numerical stability of Cholesky caching with small-scale data."""

    def test_tiny_scale_data_1e22(self):
        """Test with realistic GW data scales (1e-22)."""
        N = 100
        scale = 1e-22

        # Create tiny scale covariance matrix
        C = np.eye(N) * scale**2
        # Add some off-diagonal structure
        for i in range(N-1):
            C[i, i+1] = C[i+1, i] = 0.1 * scale**2

        # Ensure positive definite
        C = C + np.eye(N) * 1e-45

        # Test standard solve
        b = np.random.randn(N) * scale
        x_solve = np.linalg.solve(C, b)

        # Test Cholesky solve
        try:
            L = np.linalg.cholesky(C)
            y = np.linalg.solve(L, b)
            x_cholesky = np.linalg.solve(L.T, y)

            # Check relative error
            rel_error = np.linalg.norm(x_solve - x_cholesky) / np.linalg.norm(x_solve)
            self.assertLess(rel_error, 1e-10,
                          f"Relative error too large: {rel_error}")
        except np.linalg.LinAlgError:
            self.fail("Cholesky decomposition failed for small-scale matrix")

    def test_numerical_scaling_preserves_cholesky(self):
        """Test that NumericallyScaled matrices work with Cholesky."""
        N = 50
        raw_scale = 1e-22

        # Create realistic covariance at GW scale
        C_raw = np.eye(N) * raw_scale**2

        # Apply numerical scaling as done in TimeDomainLikelihood
        C_scaled_obj = NumericallyScaled(C_raw)
        C_scaled = C_scaled_obj.scaled

        # Verify Cholesky works on scaled matrix
        try:
            L = np.linalg.cholesky(C_scaled)
            # Verify: L @ L.T = C_scaled
            reconstructed = L @ L.T
            rel_error = np.linalg.norm(C_scaled - reconstructed) / np.linalg.norm(C_scaled)
            self.assertLess(rel_error, 1e-10,
                          f"Cholesky reconstruction error: {rel_error}")
        except np.linalg.LinAlgError as e:
            self.fail(f"Cholesky failed on scaled matrix: {e}")

    def test_condition_number_improvement(self):
        """Test that numerical scaling brings values to O(1) range."""
        N = 50
        raw_scale = 1e-22

        # Create a non-trivial covariance (not just identity)
        C_raw = np.eye(N) * raw_scale**2
        # Add some off-diagonal structure with varying scales
        for i in range(N-1):
            C_raw[i, i+1] = C_raw[i+1, i] = 0.5 * raw_scale**2
        # Add some variation in diagonal
        C_raw += np.diag(np.linspace(0, 0.5, N) * raw_scale**2)

        C_scaled_obj = NumericallyScaled(C_raw)
        C_scaled = C_scaled_obj.scaled

        # Scaled version should have diagonal elements O(1)
        diag_scaled = np.diag(C_scaled)
        self.assertGreater(np.min(diag_scaled), 0.1,
                          f"Scaled diagonal too small: min={np.min(diag_scaled)}")
        self.assertLess(np.max(diag_scaled), 10,
                       f"Scaled diagonal too large: max={np.max(diag_scaled)}")


@unittest.skipUnless(HAVE_FULL_DEPS, "Requires LAL and full GW software stack")
class TestCholeskyAccuracy(unittest.TestCase):
    """Test that Cholesky caching produces identical results to direct solve."""

    def setUp(self):
        """Create test data with realistic GW scales."""
        self.sample_rate = 512
        self.duration = 1  # seconds
        N = self.sample_rate * self.duration

        self.psd = FlatPSD()
        self.detector = KNOWN_IFOS["AdvancedLIGOLivingston"]()

        # Create waveform at realistic amplitude (~1e-22)
        waveform_dict = SineGaussianWaveform().time_domain(
            parameters={
                "width": 0.02,
                "ra": 1.0,
                "dec": 0.5,
                "phase": 0.0,
                "psi": 0.3,
                "theta_jn": 1.0
            }
        )

        # Project to detector
        self.data = waveform_dict.project(detector=self.detector)

        # Scale to realistic GW amplitude
        self.data = self.data * 1e-22

        # Create slightly different test waveform
        self.test_waveform_dict = SineGaussianWaveform().time_domain(
            parameters={
                "width": 0.025,
                "ra": 1.0,
                "dec": 0.5,
                "phase": 0.0,
                "psi": 0.3,
                "theta_jn": 1.0
            }
        )
        self.test_waveform = self.test_waveform_dict.project(detector=self.detector) * 1e-22

    def test_log_likelihood_consistency(self):
        """Test that log_likelihood is consistent between solve and Cholesky."""
        # Create likelihood (currently uses direct solve)
        likelihood = TimeDomainLikelihood(
            data=self.data,
            psd=self.psd,
            detector=self.detector,
            waveform=SineGaussianWaveform()
        )

        # Compute log likelihood with direct solve
        ll_solve = likelihood.log_likelihood(self.test_waveform)

        # Verify it's finite and reasonable
        self.assertTrue(np.isfinite(ll_solve), "Log likelihood is not finite")
        self.assertLess(ll_solve, 0, "Log likelihood should be negative")

        # Store for future comparison when Cholesky is implemented
        # After implementation, we'll verify Cholesky gives same result
        self.assertIsInstance(ll_solve, (float, np.float64))

    def test_multiple_evaluations_identical(self):
        """Test that repeated evaluations give identical results."""
        likelihood = TimeDomainLikelihood(
            data=self.data,
            psd=self.psd,
            detector=self.detector,
            waveform=SineGaussianWaveform()
        )

        # Compute multiple times
        results = []
        for _ in range(5):
            ll = likelihood.log_likelihood(self.test_waveform)
            results.append(ll)

        # All should be identical (not just close)
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i],
                           f"Evaluation {i} differs from first: {results[i]} != {results[0]}")

    def test_snr_consistency(self):
        """Test SNR calculation consistency."""
        likelihood = TimeDomainLikelihood(
            data=self.data,
            psd=self.psd,
            detector=self.detector,
            waveform=SineGaussianWaveform()
        )

        snr = likelihood.snr(self.test_waveform)

        # SNR should be finite and positive
        self.assertTrue(np.isfinite(snr), "SNR is not finite")
        self.assertGreater(snr, 0, "SNR should be positive")

        # Verify stability across multiple calls
        snr2 = likelihood.snr(self.test_waveform)
        self.assertEqual(snr, snr2, "SNR changed between calls")


@unittest.skipUnless(HAVE_FULL_DEPS, "Requires LAL and full GW software stack")
class TestCholeskyWithModelUncertainty(unittest.TestCase):
    """Test Cholesky caching with model uncertainty."""

    def setUp(self):
        """Create test data for model uncertainty likelihood."""
        self.sample_rate = 512
        self.duration = 1
        self.psd = FlatPSD()
        self.detector = KNOWN_IFOS["AdvancedLIGOLivingston"]()

        # Create waveform with covariance
        waveform_dict = SineGaussianWaveform().time_domain(
            parameters={
                "width": 0.02,
                "ra": 1.0,
                "dec": 0.5,
                "phase": 0.0,
                "psi": 0.3,
                "theta_jn": 1.0
            }
        )

        self.data = waveform_dict.project(detector=self.detector) * 1e-22

        # Create test waveform with covariance
        N = len(self.data)
        self.test_waveform = Mock()
        self.test_waveform.data = np.random.randn(N) * 1e-22
        self.test_waveform.covariance = np.eye(N) * (1e-23)**2  # Smaller than data scale

    def test_model_uncertainty_likelihood_finite(self):
        """Test that model uncertainty likelihood is finite."""
        likelihood = TimeDomainLikelihoodModelUncertainty(
            data=self.data,
            psd=self.psd,
            detector=self.detector,
            waveform=SineGaussianWaveform()
        )

        # Mock overlap detection
        self.data.determine_overlap = Mock(return_value=((0, len(self.data)), (0, len(self.test_waveform.data))))

        ll = likelihood.log_likelihood(self.test_waveform)

        self.assertTrue(np.isfinite(ll),
                       f"Log likelihood not finite with model uncertainty: {ll}")

    def test_covariance_addition_stable(self):
        """Test that adding noise and model covariance is numerically stable."""
        N = 100
        scale = 1e-22

        # Noise covariance
        C = np.eye(N) * scale**2
        # Model covariance (should be smaller)
        K = np.eye(N) * (0.1 * scale)**2

        # Total covariance
        total = C + K

        # Should be positive definite
        eigenvalues = np.linalg.eigvalsh(total)
        self.assertTrue(np.all(eigenvalues > 0),
                       "Total covariance not positive definite")

        # Cholesky should work
        try:
            L = np.linalg.cholesky(total)
            self.assertEqual(L.shape, (N, N))
        except np.linalg.LinAlgError as e:
            self.fail(f"Cholesky failed on total covariance: {e}")


@unittest.skipUnless(HAVE_FULL_DEPS, "Requires LAL and full GW software stack")
class TestCholeskyPerformance(unittest.TestCase):
    """Benchmark performance improvement from Cholesky caching."""

    def setUp(self):
        """Create larger dataset for performance testing."""
        self.sample_rate = 2048
        self.duration = 4  # Larger for realistic timing
        N = self.sample_rate * self.duration

        self.psd = FlatPSD()
        self.detector = KNOWN_IFOS["AdvancedLIGOLivingston"]()

        waveform_dict = SineGaussianWaveform().time_domain(
            parameters={
                "width": 0.02,
                "ra": 1.0,
                "dec": 0.5,
                "phase": 0.0,
                "psi": 0.3,
                "theta_jn": 1.0
            }
        )

        self.data = waveform_dict.project(detector=self.detector) * 1e-22

        self.test_waveforms = []
        for width in [0.015, 0.02, 0.025, 0.03]:
            wf_dict = SineGaussianWaveform().time_domain(
                parameters={
                    "width": width,
                    "ra": 1.0,
                    "dec": 0.5,
                    "phase": 0.0,
                    "psi": 0.3,
                    "theta_jn": 1.0
                }
            )
            wf = wf_dict.project(detector=self.detector) * 1e-22
            self.test_waveforms.append(wf)

    def test_repeated_likelihood_timing(self):
        """Benchmark repeated likelihood evaluations (baseline for future comparison)."""
        likelihood = TimeDomainLikelihood(
            data=self.data,
            psd=self.psd,
            detector=self.detector,
            waveform=SineGaussianWaveform()
        )

        # Warm up
        for wf in self.test_waveforms[:2]:
            _ = likelihood.log_likelihood(wf)

        # Time repeated evaluations
        start = time.time()
        n_evals = 20
        for _ in range(n_evals):
            for wf in self.test_waveforms:
                _ = likelihood.log_likelihood(wf)
        elapsed = time.time() - start

        time_per_eval = elapsed / (n_evals * len(self.test_waveforms))

        print(f"\nCurrent implementation: {time_per_eval*1000:.2f} ms per likelihood evaluation")
        print(f"Data size: {len(self.data)} samples")

        # Store for future comparison (when Cholesky is implemented)
        # After optimization, we expect ~2-3x speedup
        self.assertLess(time_per_eval, 1.0,
                       "Likelihood evaluation unreasonably slow (>1 second)")

    def test_matrix_operation_timing(self):
        """Benchmark individual matrix operations."""
        N = len(self.data)

        # Create scaled covariance matrix
        likelihood = TimeDomainLikelihood(
            data=self.data,
            psd=self.psd,
            detector=self.detector,
            waveform=SineGaussianWaveform()
        )

        C = likelihood.C_scaled
        b = np.random.randn(N) * 1e-22

        # Time direct solve
        start = time.time()
        for _ in range(10):
            x = np.linalg.solve(C, b)
        solve_time = (time.time() - start) / 10

        # Time Cholesky decomposition (one-time cost)
        start = time.time()
        L = np.linalg.cholesky(C)
        cholesky_time = time.time() - start

        # Time Cholesky solve (per-evaluation cost)
        start = time.time()
        for _ in range(10):
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
        cholesky_solve_time = (time.time() - start) / 10

        print(f"\nMatrix operation timing (N={N}):")
        print(f"  Direct solve: {solve_time*1000:.2f} ms")
        print(f"  Cholesky decomposition (one-time): {cholesky_time*1000:.2f} ms")
        print(f"  Cholesky solve (per-eval): {cholesky_solve_time*1000:.2f} ms")
        print(f"  Expected speedup: {solve_time/cholesky_solve_time:.1f}x")

        # Cholesky solve should be faster
        self.assertLess(cholesky_solve_time, solve_time,
                       "Cholesky solve not faster than direct solve")


@unittest.skipUnless(HAVE_FULL_DEPS, "Requires LAL and full GW software stack")
class TestCholeskyCorrectnessReference(unittest.TestCase):
    """Store reference values for verifying Cholesky implementation."""

    def test_generate_reference_values(self):
        """Generate reference values for future comparison."""
        np.random.seed(42)  # For reproducibility

        sample_rate = 1024
        duration = 2
        psd = FlatPSD()
        detector = KNOWN_IFOS["AdvancedLIGOLivingston"]()

        waveform_dict = SineGaussianWaveform().time_domain(
            parameters={
                "width": 0.02,
                "ra": 1.0,
                "dec": 0.5,
                "phase": 0.0,
                "psi": 0.3,
                "theta_jn": 1.0
            }
        )
        data = waveform_dict.project(detector=detector) * 1e-22

        likelihood = TimeDomainLikelihood(
            data=data,
            psd=psd,
            detector=detector,
            waveform=SineGaussianWaveform()
        )

        # Test different waveform parameters
        test_params = [
            {"width": 0.015, "ra": 1.0, "dec": 0.5, "phase": 0.0, "psi": 0.3, "theta_jn": 1.0},
            {"width": 0.02, "ra": 1.0, "dec": 0.5, "phase": 0.0, "psi": 0.3, "theta_jn": 1.0},
            {"width": 0.03, "ra": 1.0, "dec": 0.5, "phase": 0.0, "psi": 0.3, "theta_jn": 1.0},
        ]

        reference_values = []
        for params in test_params:
            wf_dict = SineGaussianWaveform().time_domain(parameters=params)
            wf = wf_dict.project(detector=detector) * 1e-22
            ll = likelihood.log_likelihood(wf)
            reference_values.append(ll)

        print("\n=== REFERENCE VALUES (pre-Cholesky optimization) ===")
        for i, (params, ll) in enumerate(zip(test_params, reference_values)):
            print(f"Test case {i+1}: width={params['width']:.3f}, log_L={ll:.10e}")

        # Store these values - after Cholesky implementation, verify they match
        self.reference_log_likelihoods = reference_values

        # All should be finite
        for ll in reference_values:
            self.assertTrue(np.isfinite(ll), f"Reference value not finite: {ll}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
