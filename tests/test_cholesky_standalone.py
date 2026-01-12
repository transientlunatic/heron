"""
Standalone test to verify Cholesky caching implementation without LAL dependencies.
"""

import unittest
import numpy as np
from unittest.mock import Mock

# Test the likelihood module directly
from heron.likelihood import TimeDomainLikelihood, NumericallyScaled


class TestCholeskyImplementation(unittest.TestCase):
    """Test Cholesky caching implementation with mock data."""

    def setUp(self):
        """Create mock time series data and PSD."""
        np.random.seed(42)  # For reproducibility

        self.N = 512  # Number of samples
        self.scale = 1e-22  # GW data scale

        # Create mock data that mimics astropy quantities
        self.mock_data = Mock()
        self.mock_data.data = np.random.randn(self.N) * self.scale

        # Mock times with a .value attribute to handle astropy quantities
        times_array = np.linspace(0, 1, self.N)

        class MockScalar:
            """Mock for scalar time values with .value attribute."""
            def __init__(self, val):
                self.value = val
                self._val = val

            def __sub__(self, other):
                if hasattr(other, 'value'):
                    return MockScalar(self._val - other.value)
                return MockScalar(self._val - other)

        class MockTime:
            def __init__(self, array):
                self._array = array
                self.value = array

            def __getitem__(self, key):
                result = self._array[key]
                if np.isscalar(result):
                    return MockScalar(result)
                return result

            def __len__(self):
                return len(self._array)

        self.mock_data.times = MockTime(times_array)

        # Create mock PSD that returns a positive definite covariance
        self.mock_psd = Mock()
        # Simple diagonal covariance for testing
        C = np.eye(self.N) * (self.scale**2)
        # Add some off-diagonal structure
        for i in range(self.N-1):
            C[i, i+1] = C[i+1, i] = 0.3 * (self.scale**2)

        self.mock_psd.covariance_matrix = Mock(return_value=C)

    def test_cholesky_initialization(self):
        """Test that Cholesky decomposition is computed and cached."""
        likelihood = TimeDomainLikelihood(self.mock_data, self.mock_psd)

        # Check that Cholesky was computed
        self.assertTrue(likelihood._use_cholesky,
                       "Cholesky decomposition should be enabled")
        self.assertIsNotNone(likelihood.C_cholesky,
                            "Cholesky factor should be cached")

        # Verify it's a valid Cholesky decomposition: C = L @ L.T
        L = likelihood.C_cholesky
        C_reconstructed = L @ L.T
        C_original = likelihood.C_scaled

        rel_error = np.linalg.norm(C_reconstructed - C_original) / np.linalg.norm(C_original)
        self.assertLess(rel_error, 1e-10,
                       f"Cholesky reconstruction error too large: {rel_error}")

    def test_snr_with_cholesky(self):
        """Test SNR calculation with Cholesky decomposition."""
        likelihood = TimeDomainLikelihood(self.mock_data, self.mock_psd)

        # Create mock waveform
        waveform = Mock()
        waveform.data = np.random.randn(self.N) * self.scale

        # Calculate SNR with Cholesky
        snr_cholesky = likelihood.snr(waveform)

        # Verify SNR is finite and positive
        self.assertTrue(np.isfinite(snr_cholesky),
                       f"SNR not finite: {snr_cholesky}")
        self.assertGreater(snr_cholesky, 0,
                          f"SNR should be positive: {snr_cholesky}")

        # Disable Cholesky and compute SNR again
        likelihood._use_cholesky = False
        snr_direct = likelihood.snr(waveform)

        # Should give identical results
        rel_diff = abs(snr_cholesky - snr_direct) / snr_direct
        self.assertLess(rel_diff, 1e-10,
                       f"SNR differs between Cholesky and direct solve: {rel_diff}")

    def test_log_likelihood_full_overlap(self):
        """Test log_likelihood with full overlap using Cholesky."""
        # Setup overlap to return full range
        self.mock_data.determine_overlap = Mock(return_value=((0, self.N), (0, self.N)))

        likelihood = TimeDomainLikelihood(self.mock_data, self.mock_psd)

        # Create mock waveform with full overlap
        waveform = Mock()
        waveform.data = np.random.randn(self.N) * self.scale

        # Calculate with Cholesky (should use cached decomposition)
        self.assertTrue(likelihood._use_cholesky)
        ll_cholesky = likelihood.log_likelihood(waveform, norm=True)

        # Disable Cholesky and compute again
        likelihood._use_cholesky = False
        ll_direct = likelihood.log_likelihood(waveform, norm=True)

        # Re-enable for partial overlap test
        likelihood._use_cholesky = True

        # Results should be very close (within numerical precision)
        self.assertTrue(np.isfinite(ll_cholesky),
                       f"Cholesky log-likelihood not finite: {ll_cholesky}")
        self.assertTrue(np.isfinite(ll_direct),
                       f"Direct log-likelihood not finite: {ll_direct}")

        abs_diff = abs(ll_cholesky - ll_direct)
        rel_diff = abs_diff / abs(ll_direct) if ll_direct != 0 else abs_diff

        # Allow small numerical differences
        self.assertLess(rel_diff, 1e-8,
                       f"Log-likelihood differs: Cholesky={ll_cholesky}, Direct={ll_direct}, rel_diff={rel_diff}")

    def test_log_likelihood_partial_overlap(self):
        """Test log_likelihood with partial overlap."""
        # Setup overlap to return partial range
        start, end = 100, 400
        self.mock_data.determine_overlap = Mock(return_value=((start, end), (0, end-start)))

        likelihood = TimeDomainLikelihood(self.mock_data, self.mock_psd)

        # Create mock waveform with partial overlap
        waveform = Mock()
        waveform.data = np.random.randn(end-start) * self.scale

        # Calculate with Cholesky (should compute submatrix Cholesky)
        ll_partial = likelihood.log_likelihood(waveform, norm=True)

        # Should be finite
        self.assertTrue(np.isfinite(ll_partial),
                       f"Partial overlap log-likelihood not finite: {ll_partial}")

    def test_log_likelihood_no_overlap(self):
        """Test log_likelihood returns -inf when no overlap."""
        self.mock_data.determine_overlap = Mock(return_value=None)

        likelihood = TimeDomainLikelihood(self.mock_data, self.mock_psd)

        waveform = Mock()
        waveform.data = np.random.randn(self.N) * self.scale

        ll = likelihood.log_likelihood(waveform)

        self.assertEqual(ll, -np.inf,
                        "Should return -inf for no overlap")

    def test_multiple_likelihood_evaluations_consistent(self):
        """Test that repeated evaluations give consistent results."""
        self.mock_data.determine_overlap = Mock(return_value=((0, self.N), (0, self.N)))

        likelihood = TimeDomainLikelihood(self.mock_data, self.mock_psd)

        # Create fixed waveform
        np.random.seed(123)
        waveform = Mock()
        waveform.data = np.random.randn(self.N) * self.scale

        # Evaluate multiple times
        results = []
        for _ in range(5):
            ll = likelihood.log_likelihood(waveform, norm=True)
            results.append(ll)

        # All should be identical
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i],
                           f"Evaluation {i} differs: {results[i]} != {results[0]}")

    def test_normalization_flag(self):
        """Test that normalization flag works correctly."""
        self.mock_data.determine_overlap = Mock(return_value=((0, self.N), (0, self.N)))

        likelihood = TimeDomainLikelihood(self.mock_data, self.mock_psd)

        waveform = Mock()
        waveform.data = np.random.randn(self.N) * self.scale

        ll_with_norm = likelihood.log_likelihood(waveform, norm=True)
        ll_without_norm = likelihood.log_likelihood(waveform, norm=False)

        # Without normalization should be different
        self.assertNotEqual(ll_with_norm, ll_without_norm,
                           "Normalization should change result")
        # The difference should be significant (normalization constant)
        self.assertGreater(abs(ll_with_norm - ll_without_norm), 100,
                          "Normalization should have significant effect")


class TestCholeskyPerformanceBasic(unittest.TestCase):
    """Basic performance comparison."""

    def test_cholesky_faster_than_direct(self):
        """Test that Cholesky solve is faster than direct solve."""
        import time

        np.random.seed(42)
        N = 2048  # Larger for timing
        scale = 1e-22

        # Create test matrix
        C = np.eye(N) * (scale**2)
        for i in range(N-1):
            C[i, i+1] = C[i+1, i] = 0.3 * (scale**2)

        # Scale it
        C_scaled_obj = NumericallyScaled(C)
        C_scaled = C_scaled_obj.scaled

        # Create test vector
        b = np.random.randn(N) * scale * np.sqrt(C_scaled_obj.scale)

        # Time direct solve
        start = time.time()
        for _ in range(10):
            np.linalg.solve(C_scaled, b)
        direct_time = (time.time() - start) / 10

        # Time Cholesky
        L = np.linalg.cholesky(C_scaled)
        start = time.time()
        for _ in range(10):
            from scipy import linalg as scipy_linalg
            y = scipy_linalg.solve_triangular(L, b, lower=True)
            x = scipy_linalg.solve_triangular(L.T, y, lower=False)
        cholesky_time = (time.time() - start) / 10

        print(f"\nPerformance comparison (N={N}):")
        print(f"  Direct solve: {direct_time*1000:.2f} ms")
        print(f"  Cholesky solve: {cholesky_time*1000:.2f} ms")
        print(f"  Speedup: {direct_time/cholesky_time:.1f}x")

        # Cholesky should be faster
        self.assertLess(cholesky_time, direct_time,
                       "Cholesky solve should be faster than direct solve")


if __name__ == '__main__':
    unittest.main(verbosity=2)
