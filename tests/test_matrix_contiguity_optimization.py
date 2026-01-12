"""
Test whether ensuring memory contiguity improves performance for linear algebra operations.

Many BLAS/LAPACK routines work more efficiently on contiguous arrays.
"""

import unittest
import numpy as np
import time
from scipy import linalg as scipy_linalg


class TestContiguityImpact(unittest.TestCase):
    """Test performance impact of memory contiguity."""

    def test_cholesky_contiguous_vs_noncontiguous(self):
        """Compare Cholesky on contiguous vs non-contiguous views."""
        N = 2000
        mat = np.eye(N) + np.random.randn(N, N) * 0.01
        mat = mat @ mat.T

        start, end = 500, 1500

        # Warm up
        for _ in range(3):
            submat_view = mat[start:end, start:end]
            _ = np.linalg.cholesky(submat_view)
            submat_contig = np.ascontiguousarray(mat[start:end, start:end])
            _ = np.linalg.cholesky(submat_contig)

        # Time with non-contiguous view
        n_iter = 30
        start_time = time.time()
        for _ in range(n_iter):
            submat_view = mat[start:end, start:end]
            np.linalg.cholesky(submat_view)
        view_time = (time.time() - start_time) / n_iter

        # Time with contiguous array
        start_time = time.time()
        for _ in range(n_iter):
            submat_contig = np.ascontiguousarray(mat[start:end, start:end])
            np.linalg.cholesky(submat_contig)
        contig_time = (time.time() - start_time) / n_iter

        # Time with plain copy
        start_time = time.time()
        for _ in range(n_iter):
            submat_copy = mat[start:end, start:end].copy()
            L = np.linalg.cholesky(submat_copy)
        copy_time = (time.time() - start_time) / n_iter

        # print(f"\n=== Cholesky Contiguity Impact ===")
        # print(f"Non-contiguous view: {view_time*1000:.3f} ms")
        # print(f"Contiguous (ascontiguousarray): {contig_time*1000:.3f} ms")
        # print(f"Plain copy: {copy_time*1000:.3f} ms")
        # print(f"View vs Contig speedup: {view_time/contig_time:.2f}x")
        # print(f"Contig vs Copy: {abs(contig_time-copy_time)*1000:.3f} ms diff")

    def test_solve_triangular_contiguity(self):
        """Test solve_triangular with different memory layouts."""
        N = 2000
        L_full = np.tril(np.random.randn(N, N))
        np.fill_diagonal(L_full, np.abs(np.diag(L_full)) + 1)
        b_full = np.random.randn(N)

        start, end = 500, 1500

        # Warm up
        for _ in range(3):
            L_sub = L_full[start:end, start:end]
            b_sub = b_full[start:end]
            _ = scipy_linalg.solve_triangular(L_sub, b_sub, lower=True)

            L_contig = np.ascontiguousarray(L_full[start:end, start:end])
            b_contig = np.ascontiguousarray(b_full[start:end])
            _ = scipy_linalg.solve_triangular(L_contig, b_contig, lower=True)

        # Time with views
        n_iter = 100
        start_time = time.time()
        for _ in range(n_iter):
            L_sub = L_full[start:end, start:end]
            b_sub = b_full[start:end]
            x = scipy_linalg.solve_triangular(L_sub, b_sub, lower=True)
        view_time = (time.time() - start_time) / n_iter

        # Time with contiguous
        start_time = time.time()
        for _ in range(n_iter):
            L_contig = np.ascontiguousarray(L_full[start:end, start:end])
            b_contig = np.ascontiguousarray(b_full[start:end])
            _ = scipy_linalg.solve_triangular(L_contig, b_contig, lower=True)
        contig_time = (time.time() - start_time) / n_iter

        # print(f"\n=== solve_triangular Contiguity Impact ===")
        # print(f"View: {view_time*1000:.3f} ms")
        # print(f"Contiguous: {contig_time*1000:.3f} ms")
        # print(f"Speedup: {view_time/contig_time:.2f}x")

    def test_realistic_likelihood_pattern(self):
        """Test the exact pattern used in likelihood with realistic sizes."""
        # Realistic GW parameters
        sample_rates = [512, 1024, 2048, 4096]
        duration = 4  # seconds
        overlap_fraction = 0.75  # Typical overlap

        # print(f"\n=== Realistic Likelihood Scenarios ===")
        # print(f"{'Sample Rate':<12} {'Matrix Size':<12} {'Overlap Size':<12} "
              f"{'View Time':<12} {'Contig Time':<12} {'Speedup'}")

        for sr in sample_rates:
            N = sr * duration
            # Create scaled covariance (typical after NumericallyScaled)
            C_scaled = np.eye(N) * 1.0  # O(1) after scaling

            # Add off-diagonal structure
            for i in range(min(N-1, 1000)):  # Limit for speed
                C_scaled[i, i+1] = C_scaled[i+1, i] = 0.1

            # Typical partial overlap
            overlap_size = int(N * overlap_fraction)
            start_idx = (N - overlap_size) // 2
            end_idx = start_idx + overlap_size

            # Warm up
            for _ in range(2):
                C_sub_view = C_scaled[start_idx:end_idx, start_idx:end_idx]
                try:
                    _ = np.linalg.cholesky(C_sub_view)
                except:
                    pass

                C_sub_contig = np.ascontiguousarray(C_scaled[start_idx:end_idx, start_idx:end_idx])
                try:
                    _ = np.linalg.cholesky(C_sub_contig)
                except:
                    pass

            # Time with view
            n_iter = 10
            start_time = time.time()
            for _ in range(n_iter):
                C_sub_view = C_scaled[start_idx:end_idx, start_idx:end_idx]
                try:
                    np.linalg.cholesky(C_sub_view)
                except np.linalg.LinAlgError:
                    # Expected for ill-conditioned matrices at this scale
                    pass
            view_time = (time.time() - start_time) / n_iter

            # Time with contiguous
            start_time = time.time()
            for _ in range(n_iter):
                C_sub_contig = np.ascontiguousarray(C_scaled[start_idx:end_idx, start_idx:end_idx])
                try:
                    np.linalg.cholesky(C_sub_contig)
                except np.linalg.LinAlgError:
                    # Expected for ill-conditioned matrices at this scale
                    pass
            contig_time = (time.time() - start_time) / n_iter

            speedup = view_time / contig_time if contig_time > 0 else 1.0

            # print(f"{sr:<12} {N:<12} {overlap_size:<12} "
                  f"{view_time*1000:>10.2f} ms {contig_time*1000:>10.2f} ms  {speedup:>6.2f}x")


class TestPracticalOptimization(unittest.TestCase):
    """Test whether optimization is worth it in practice."""

    def test_full_likelihood_evaluation_pattern(self):
        """Simulate a complete likelihood evaluation with partial overlap."""
        N_data = 2048
        N_overlap = 1500

        # Create mock data structures
        C_full = np.eye(N_data) * 1.0
        for i in range(min(N_data-1, 100)):
            C_full[i, i+1] = C_full[i+1, i] = 0.1

        _ = np.linalg.cholesky(C_full)

        start_idx, end_idx = 200, 200 + N_overlap
        data = np.random.randn(N_overlap)
        wf = np.random.randn(N_overlap) * 0.1

        residual = data - wf

        # Pattern 1: Current implementation (view-based)
        def current_pattern():
            C_sub = C_full[start_idx:end_idx, start_idx:end_idx]
            L_sub = np.linalg.cholesky(C_sub)
            y = scipy_linalg.solve_triangular(L_sub, residual, lower=True)
            x = scipy_linalg.solve_triangular(L_sub.T, y, lower=False)
            return residual @ x

        # Pattern 2: With explicit contiguous conversion
        def contiguous_pattern():
            C_sub = np.ascontiguousarray(C_full[start_idx:end_idx, start_idx:end_idx])
            L_sub = np.linalg.cholesky(C_sub)
            y = scipy_linalg.solve_triangular(L_sub, residual, lower=True)
            x = scipy_linalg.solve_triangular(L_sub.T, y, lower=False)
            return residual @ x

        # Pattern 3: Cache submatrix Cholesky if repeated
        def cached_submatrix_pattern():
            # This would be done once and cached
            C_sub = np.ascontiguousarray(C_full[start_idx:end_idx, start_idx:end_idx])
            _ = np.linalg.cholesky(C_sub)

            # This is done per evaluation
            y = scipy_linalg.solve_triangular(L_sub, residual, lower=True)
            x = scipy_linalg.solve_triangular(L_sub.T, y, lower=False)
            return residual @ x

        # Warm up
        for _ in range(5):
            _ = current_pattern()
            _ = contiguous_pattern()

        # Time current pattern
        n_iter = 100
        start_time = time.time()
        for _ in range(n_iter):
            current_pattern()
        current_time = (time.time() - start_time) / n_iter

        # Time contiguous pattern
        start_time = time.time()
        for _ in range(n_iter):
            contiguous_pattern()
        contig_time = (time.time() - start_time) / n_iter

        # Time just the solve part (if Cholesky was cached)
        C_sub_cached = np.ascontiguousarray(C_full[start_idx:end_idx, start_idx:end_idx])
        L_sub_cached = np.linalg.cholesky(C_sub_cached)

        start_time = time.time()
        for _ in range(n_iter):
            y = scipy_linalg.solve_triangular(L_sub_cached, residual, lower=True)
            x = scipy_linalg.solve_triangular(L_sub_cached.T, y, lower=False)
            # Basic sanity check to ensure computation produced a finite value
            self.assertTrue(np.isfinite(residual @ x))
        cached_time = (time.time() - start_time) / n_iter

        # print(f"\n=== Full Likelihood Pattern Comparison ===")
        # print(f"Data size: {N_data}, Overlap size: {N_overlap}")
        # print(f"Current (view): {current_time*1000:.3f} ms")
        # print(f"With ascontiguousarray: {contig_time*1000:.3f} ms")
        # print(f"Cached submatrix Cholesky: {cached_time*1000:.3f} ms")
        # print(f"Current vs Contig speedup: {current_time/contig_time:.2f}x")
        # print(f"Current vs Cached speedup: {current_time/cached_time:.2f}x")


if __name__ == '__main__':
    unittest.main(verbosity=2)
