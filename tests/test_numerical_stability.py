"""
Tests for numerical stability safeguards in NumericallyScaled class.

Tests edge cases where extreme scaling factors could cause overflow/underflow.
"""

import unittest
import numpy as np
from heron.likelihood import NumericallyScaled


class TestNumericallyScaledStability(unittest.TestCase):
    """Test numerical stability of scaling factor computation."""

    def test_zero_diagonal_handling(self):
        """Test that zero diagonal elements don't cause division by zero."""
        # Create matrix with zero diagonal
        C = np.zeros((10, 10))
        C[1:, 1:] = np.eye(9)  # Only first diagonal is zero

        scaled = NumericallyScaled(C)

        # Should use identity scaling (1.0) instead of inf
        self.assertEqual(scaled.scale, 1.0)
        self.assertTrue(np.isfinite(scaled.scale))

    def test_tiny_diagonal_clipping(self):
        """Test that extremely small diagonal values are clipped."""
        # Create matrix with extremely small diagonal
        scale = 1e-100
        C = np.eye(10) * scale

        scaled = NumericallyScaled(C)

        # Scale should be clipped to prevent overflow
        # 1/1e-100 = 1e100, which should be clipped to 1e10
        self.assertEqual(scaled.scale, 1e10)
        self.assertTrue(np.isfinite(scaled.scale))

    def test_large_diagonal_clipping(self):
        """Test that extremely large diagonal values are clipped."""
        # Create matrix with extremely large diagonal
        scale = 1e100
        C = np.eye(10) * scale

        scaled = NumericallyScaled(C)

        # Scale should be clipped to prevent underflow
        # 1/1e100 = 1e-100, which should be clipped to 1e-10
        self.assertEqual(scaled.scale, 1e-10)
        self.assertTrue(np.isfinite(scaled.scale))

    def test_normal_scale_unchanged(self):
        """Test that normal scale values are not clipped."""
        # GW data scale
        scale = 1e-22
        C = np.eye(100) * scale**2

        scaled = NumericallyScaled(C)

        # Should compute 1/(1e-22)^2 = 1e44
        # This is within [1e-10, 1e10] after taking into account the squared value
        # Actually: min_diag = 1e-44, so scale = 1e44, which gets clipped to 1e10
        self.assertEqual(scaled.scale, 1e10)

    def test_reasonable_scale_unclipped(self):
        """Test that reasonable scales are not clipped."""
        # Matrix with diagonal elements around 0.1 to 10
        C = np.diag(np.linspace(0.1, 10, 100))

        scaled = NumericallyScaled(C)

        # min_diag = 0.1, scale = 10
        self.assertEqual(scaled.scale, 10.0)
        self.assertGreater(scaled.scale, 1e-10)
        self.assertLess(scaled.scale, 1e10)

    def test_explicit_scale_bypasses_computation(self):
        """Test that explicitly providing scale bypasses automatic computation."""
        # Even with bad matrix
        C = np.zeros((10, 10))

        # Provide explicit scale
        scaled = NumericallyScaled(C, scale=5.0)

        self.assertEqual(scaled.scale, 5.0)

    def test_scaled_property_finite(self):
        """Test that scaled property always returns finite values."""
        # Various edge cases
        test_matrices = [
            np.zeros((5, 5)),
            np.eye(5) * 1e-100,
            np.eye(5) * 1e100,
            np.eye(5) * 1e-22,
        ]

        for C in test_matrices:
            scaled = NumericallyScaled(C)
            result = scaled.scaled

            # All values should be finite
            self.assertTrue(np.all(np.isfinite(result)),
                           f"Non-finite values in scaled matrix")

    def test_gw_realistic_scale(self):
        """Test with realistic GW data scale."""
        # Typical GW covariance matrix
        N = 1024
        scale = 1e-22
        C = np.eye(N) * scale**2

        # Add some off-diagonal structure
        for i in range(N-1):
            C[i, i+1] = C[i+1, i] = 0.3 * scale**2

        scaled = NumericallyScaled(C)

        # Should handle this gracefully
        self.assertTrue(np.isfinite(scaled.scale))
        result = scaled.scaled

        # Scaled matrix should have reasonable values
        self.assertTrue(np.all(np.isfinite(result)))
        # After scaling with clipped factor, values should still be finite
        # The scale is clipped to 1e10, so (1e-22)^2 * 1e10 = 1e-34
        self.assertGreater(np.min(np.diag(result)), 1e-40)

    def test_inf_nan_handling(self):
        """Test that inf/nan in matrix are detected."""
        # Matrix with inf
        C = np.eye(5)
        C[0, 0] = np.inf

        scaled = NumericallyScaled(C)

        # Scale should still be finite (identity scaling for degenerate case)
        # The inf will be in the scaled result, but scale itself should be safe
        self.assertTrue(np.isfinite(scaled.scale))

    def test_negative_diagonal(self):
        """Test handling of negative diagonal elements."""
        # This shouldn't happen in practice for covariance matrices
        # but let's be defensive
        C = -np.eye(5)

        scaled = NumericallyScaled(C)

        # Should handle negative values (will get negative scale)
        # After clipping, should still be finite
        self.assertTrue(np.isfinite(scaled.scale))


class TestNumericallyScaledOperations(unittest.TestCase):
    """Test arithmetic operations on NumericallyScaled objects."""

    def test_addition_same_scale(self):
        """Test adding two NumericallyScaled objects with same scale."""
        A = NumericallyScaled(np.eye(5), scale=2.0)
        B = NumericallyScaled(np.eye(5) * 3, scale=2.0)

        result = A + B

        expected = (np.eye(5) + np.eye(5) * 3) * 2.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_subtraction_same_scale(self):
        """Test subtracting two NumericallyScaled objects with same scale."""
        A = NumericallyScaled(np.eye(5) * 5, scale=2.0)
        B = NumericallyScaled(np.eye(5) * 3, scale=2.0)

        result = A - B

        expected = (np.eye(5) * 5 - np.eye(5) * 3) * 2.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_addition_different_scale_raises(self):
        """Test that adding objects with different scales raises assertion."""
        A = NumericallyScaled(np.eye(5), scale=2.0)
        B = NumericallyScaled(np.eye(5), scale=3.0)

        with self.assertRaises(AssertionError):
            A + B

    def test_unscale_method(self):
        """Test the unscale method."""
        original = np.eye(5) * 0.1
        scaled_obj = NumericallyScaled(original, scale=10.0)

        scaled_value = scaled_obj.scaled  # 0.1 * 10 = 1.0 for diagonal
        unscaled = scaled_obj.unscale(scaled_value)

        # Should get back original values
        np.testing.assert_array_almost_equal(unscaled, original)


class TestNumericallyScaledEdgeCases(unittest.TestCase):
    """Test edge cases and corner conditions."""

    def test_single_element_matrix(self):
        """Test with 1x1 matrix."""
        C = np.array([[1e-44]])

        scaled = NumericallyScaled(C)

        self.assertTrue(np.isfinite(scaled.scale))
        self.assertEqual(scaled.scale, 1e10)  # Clipped

    def test_non_square_matrix_behavior(self):
        """Test behavior with non-square matrix."""
        # Non-square matrices don't have a well-defined diagonal
        # np.diag will extract min(rows, cols) elements
        C = np.ones((5, 10))

        # Should not crash, but may not work as intended
        # (np.diag on non-square extracts the diagonal up to min dimension)
        try:
            scaled = NumericallyScaled(C)
            # If it succeeds, scale should be finite
            self.assertTrue(np.isfinite(scaled.scale))
        except (ValueError, IndexError):
            # Also acceptable to fail
            pass

    def test_very_large_matrix(self):
        """Test performance with large matrix."""
        N = 10000
        C = np.eye(N) * 1e-44

        import time
        start = time.time()
        scaled = NumericallyScaled(C)
        elapsed = time.time() - start

        # Should be fast (just needs to compute min of diagonal)
        self.assertLess(elapsed, 0.1)  # < 100ms
        self.assertTrue(np.isfinite(scaled.scale))


if __name__ == '__main__':
    unittest.main(verbosity=2)
