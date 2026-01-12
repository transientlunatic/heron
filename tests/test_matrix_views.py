"""
Test to investigate and verify matrix view vs copy behavior.

This test analyzes whether NumPy slicing creates views or copies for different
matrix operations, and measures the performance impact.
"""

import unittest
import numpy as np
import time


class TestMatrixViewBehavior(unittest.TestCase):
    """Investigate NumPy's view vs copy behavior for matrix slicing."""

    def test_1d_slicing_creates_view(self):
        """Test that 1D array slicing creates a view."""
        arr = np.arange(1000)
        sliced = arr[100:200]

        # Check if it's a view
        self.assertTrue(np.shares_memory(arr, sliced),
                       "1D slicing should create a view")

    def test_2d_contiguous_row_slicing(self):
        """Test that contiguous row slicing creates a view."""
        mat = np.arange(1000000).reshape(1000, 1000)
        sliced = mat[100:200, :]

        # Check if it's a view
        self.assertTrue(np.shares_memory(mat, sliced),
                       "Contiguous row slicing should create a view")

    def test_2d_contiguous_column_slicing(self):
        """Test that contiguous column slicing creates a view."""
        mat = np.arange(1000000).reshape(1000, 1000)
        sliced = mat[:, 100:200]

        # Check if it's a view (this might be view but not contiguous)
        shares_mem = np.shares_memory(mat, sliced)
        # print(f"\nColumn slicing shares memory: {shares_mem}")
        # print(f"Column slice is C-contiguous: {sliced.flags['C_CONTIGUOUS']}")
        # print(f"Column slice is F-contiguous: {sliced.flags['F_CONTIGUOUS']}")

    def test_2d_submatrix_slicing(self):
        """Test that submatrix slicing (both dimensions) creates view."""
        mat = np.arange(1000000).reshape(1000, 1000)
        sliced = mat[100:200, 100:200]

        # Check if it's a view
        shares_mem = np.shares_memory(mat, sliced)
        is_view = sliced.base is not None

        # print(f"\nSubmatrix slicing:")
        # print(f"  Shares memory: {shares_mem}")
        # print(f"  Is view (has base): {is_view}")
        # print(f"  C-contiguous: {sliced.flags['C_CONTIGUOUS']}")
        # print(f"  F-contiguous: {sliced.flags['F_CONTIGUOUS']}")
        # print(f"  OWNDATA: {sliced.flags['OWNDATA']}")

        # In NumPy, submatrix slicing DOES create a view
        self.assertTrue(is_view, "Submatrix slicing should create a view")
        self.assertTrue(shares_mem, "Should share memory with parent")

    def test_symmetric_submatrix_slicing(self):
        """Test behavior for symmetric submatrix extraction (covariance case)."""
        # This is the pattern used in log_likelihood: C[a:b, a:b]
        mat = np.eye(1000)
        start, end = 100, 200

        submat = mat[start:end, start:end]

        is_view = submat.base is not None
        shares_mem = np.shares_memory(mat, submat)

        # print(f"\nSymmetric submatrix C[{start}:{end}, {start}:{end}]:")
        # print(f"  Is view: {is_view}")
        # print(f"  Shares memory: {shares_mem}")
        # print(f"  C-contiguous: {submat.flags['C_CONTIGUOUS']}")
        # print(f"  OWNDATA: {submat.flags['OWNDATA']}")

        # NumPy creates a view for this pattern
        self.assertTrue(is_view, "Symmetric submatrix should be a view")


class TestViewPerformance(unittest.TestCase):
    """Test performance implications of views vs copies."""

    def test_cholesky_on_view_vs_copy(self):
        """Compare Cholesky performance on view vs explicit copy."""
        # Create large positive definite matrix
        N = 2000
        mat = np.eye(N) + np.random.randn(N, N) * 0.01
        mat = mat @ mat.T  # Make positive definite

        # Extract submatrix as view
        start, end = 500, 1500
        sub_size = end - start

        # Warm up
        for _ in range(3):
            _ = np.linalg.cholesky(mat[start:end, start:end])
            _ = np.linalg.cholesky(mat[start:end, start:end].copy())

        # Time with view (implicit copy by Cholesky)
        n_iter = 20
        start_time = time.time()
        for _ in range(n_iter):
            submat_view = mat[start:end, start:end]
            np.linalg.cholesky(submat_view)
        view_time = (time.time() - start_time) / n_iter

        # Time with explicit copy
        start_time = time.time()
        for _ in range(n_iter):
            submat_copy = mat[start:end, start:end].copy()
            np.linalg.cholesky(submat_copy)
        copy_time = (time.time() - start_time) / n_iter

        # print(f"\n=== Cholesky on Submatrix Performance ===")
        # print(f"Submatrix size: {sub_size}x{sub_size}")
        # print(f"View (implicit copy): {view_time*1000:.3f} ms")
        # print(f"Explicit copy: {copy_time*1000:.3f} ms")
        # print(f"Difference: {abs(view_time - copy_time)*1000:.3f} ms")

        # They should be very similar
        rel_diff = abs(view_time - copy_time) / view_time
        self.assertLess(rel_diff, 0.2,
                       f"Performance difference too large: {rel_diff:.1%}")

    def test_solve_triangular_on_view_vs_copy(self):
        """Compare solve_triangular performance on view vs copy."""
        from scipy import linalg as scipy_linalg

        N = 2000
        L_full = np.tril(np.random.randn(N, N))
        np.fill_diagonal(L_full, np.abs(np.diag(L_full)) + 1)  # Ensure positive diagonal
        b_full = np.random.randn(N)

        start, end = 500, 1500

        # Warm up
        for _ in range(3):
            _ = scipy_linalg.solve_triangular(L_full[start:end, start:end],
                                             b_full[start:end], lower=True)
            _ = scipy_linalg.solve_triangular(L_full[start:end, start:end].copy(),
                                             b_full[start:end].copy(), lower=True)

        # Time with views
        n_iter = 100
        start_time = time.time()
        for _ in range(n_iter):
            L_sub = L_full[start:end, start:end]
            b_sub = b_full[start:end]
            scipy_linalg.solve_triangular(L_sub, b_sub, lower=True)
        view_time = (time.time() - start_time) / n_iter

        # Time with copies
        start_time = time.time()
        for _ in range(n_iter):
            L_sub = L_full[start:end, start:end].copy()
            b_sub = b_full[start:end].copy()
            _ = scipy_linalg.solve_triangular(L_sub, b_sub, lower=True)
        copy_time = (time.time() - start_time) / n_iter

        # print(f"\n=== solve_triangular Performance ===")
        # print(f"View: {view_time*1000:.3f} ms")
        # print(f"Copy: {copy_time*1000:.3f} ms")
        # print(f"Speedup with view: {copy_time/view_time:.2f}x")

        # Views should be faster (less memory allocation)
        self.assertLess(view_time, copy_time * 1.1,
                       "Views should not be significantly slower")


class TestLikelihoodSubmatrixPattern(unittest.TestCase):
    """Test the specific pattern used in likelihood computation."""

    def test_current_likelihood_pattern(self):
        """Analyze the exact pattern used in log_likelihood method."""
        # Simulate the likelihood scenario
        N_full = 2048
        C_scaled = np.eye(N_full) * 1e-44  # Typical GW scale after scaling

        # Partial overlap scenario
        a = (500, 1500)  # Data indices

        # Current code pattern (line 226 in likelihood.py):
        # C_scaled = self.C_scaled[a[0]:a[1], a[0]:a[1]]

        C_sub = C_scaled[a[0]:a[1], a[0]:a[1]]

        # print(f"\n=== Likelihood Submatrix Pattern Analysis ===")
        # print(f"Full matrix: {N_full}x{N_full}")
        # print(f"Submatrix: {a[1]-a[0]}x{a[1]-a[0]}")
        # print(f"Is view: {C_sub.base is not None}")
        # print(f"Shares memory: {np.shares_memory(C_scaled, C_sub)}")
        # print(f"Memory saved if view: {C_sub.nbytes / 1024:.1f} KB")

        # Verify correctness
        self.assertTrue(C_sub.base is not None, "Should be a view")

    def test_memory_allocation_overhead(self):
        """Measure pure memory allocation overhead for copies."""
        N_full = 4096
        mat = np.eye(N_full)

        sizes = [(100, 200), (500, 1000), (1000, 2000), (2000, 4000)]

        # print(f"\n=== Memory Allocation Overhead ===")
        # print(f"{'Range':<20} {'Copy Time':<15} {'Memory'}")

        for start, end in sizes:
            size = end - start

            # Time just the copy operation
            n_iter = 1000
            start_time = time.time()
            for _ in range(n_iter):
                _ = mat[start:end, start:end].copy()
            copy_time = (time.time() - start_time) / n_iter

            mem_kb = (size * size * 8) / 1024  # 8 bytes per float64

            # print(f"{str((start, end)):<20} {copy_time*1e6:>10.2f} µs   {mem_kb:>8.1f} KB")


class TestViewCorrectness(unittest.TestCase):
    """Verify that using views doesn't affect correctness."""

    def test_cholesky_correctness_with_view(self):
        """Verify Cholesky gives same result with view vs copy."""
        N = 1000
        mat = np.eye(N) + np.random.randn(N, N) * 0.01
        mat = mat @ mat.T

        start, end = 200, 600

        # Cholesky on view
        L_view = np.linalg.cholesky(mat[start:end, start:end])

        # Cholesky on copy
        L_copy = np.linalg.cholesky(mat[start:end, start:end].copy())

        # Should be identical
        np.testing.assert_array_almost_equal(L_view, L_copy,
                                            err_msg="View and copy should give same result")

    def test_operations_dont_modify_parent(self):
        """Verify that operations on views don't modify parent (when appropriate)."""
        mat = np.eye(10)
        mat_original = mat.copy()

        # Extract view
        submat = mat[2:5, 2:5]

        # Operations that create new arrays shouldn't affect parent
        np.linalg.cholesky(submat)

        np.testing.assert_array_equal(mat, mat_original,
                                     err_msg="Cholesky shouldn't modify parent through view")


if __name__ == '__main__':
    # Run with verbose output to see all print statements
    unittest.main(verbosity=2)
