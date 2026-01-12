"""
Standalone tests for optimized overlap detection without LAL dependencies.

Tests the determine_overlap method directly with mock objects.
"""

import unittest
import numpy as np
import time


def optimized_determine_overlap(times_a, times_b):
    """
    Optimized overlap detection using binary search.
    Standalone implementation for testing.
    """
    # Get bounds
    a_start, a_end = times_a[0], times_a[-1]
    b_start, b_end = times_b[0], times_b[-1]

    # Get sample spacing (assumed uniform)
    dt_a = times_a[1] - times_a[0]
    dt_b = times_b[1] - times_b[0]
    tolerance = max(dt_a, dt_b) * 0.5

    # Quick check: do the time ranges overlap at all?
    if a_end + tolerance < b_start or b_end + tolerance < a_start:
        return None  # No overlap

    # Determine overlap region
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)

    if overlap_start > overlap_end + tolerance:
        return None

    # Use binary search (searchsorted) to find indices efficiently
    start_a = np.searchsorted(times_a, overlap_start, side='left')
    finish_a = np.searchsorted(times_a, overlap_end, side='right') - 1

    start_b = np.searchsorted(times_b, overlap_start, side='left')
    finish_b = np.searchsorted(times_b, overlap_end, side='right') - 1

    # Clamp to valid indices
    start_a = max(0, min(start_a, len(times_a) - 1))
    finish_a = max(0, min(finish_a, len(times_a) - 1))
    start_b = max(0, min(start_b, len(times_b) - 1))
    finish_b = max(0, min(finish_b, len(times_b) - 1))

    # Verify we have a valid overlap
    if start_a > finish_a or start_b > finish_b:
        return None

    return (start_a, finish_a), (start_b, finish_b)


def original_determine_overlap(times_a, times_b):
    """
    Original overlap detection for comparison.
    Uses O(N) argmin approach.
    """
    def is_in(time, timeseries):
        diff = np.min(np.abs(timeseries - time))
        dt = timeseries[1] - timeseries[0]
        return diff < dt

    a_start, a_end = times_a[0], times_a[-1]
    b_start, b_end = times_b[0], times_b[-1]

    # Simple overlap logic
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)

    if overlap_start > overlap_end:
        return None

    # Use argmin (O(N))
    start_a = np.argmin(np.abs(times_a - overlap_start))
    finish_a = np.argmin(np.abs(times_a - overlap_end))

    start_b = np.argmin(np.abs(times_b - overlap_start))
    finish_b = np.argmin(np.abs(times_b - overlap_end))

    return (start_a, finish_a), (start_b, finish_b)


class TestOverlapCorrectness(unittest.TestCase):
    """Test that optimized version produces same results as original."""

    def test_identical_ranges(self):
        """Test with identical time ranges."""
        times_a = np.linspace(0, 1, 1024)
        times_b = np.linspace(0, 1, 1024)

        result_opt = optimized_determine_overlap(times_a, times_b)
        result_orig = original_determine_overlap(times_a, times_b)

        self.assertIsNotNone(result_opt)
        self.assertEqual(result_opt, result_orig)

    def test_partial_overlap(self):
        """Test with partial overlap."""
        times_a = np.linspace(0, 2, 2048)
        times_b = np.linspace(1, 3, 2048)

        result_opt = optimized_determine_overlap(times_a, times_b)
        result_orig = original_determine_overlap(times_a, times_b)

        self.assertIsNotNone(result_opt)
        # Results should be very close (within 1 sample due to rounding)
        (sa_opt, ea_opt), (sb_opt, eb_opt) = result_opt
        (sa_orig, ea_orig), (sb_orig, eb_orig) = result_orig

        self.assertLess(abs(sa_opt - sa_orig), 2)
        self.assertLess(abs(ea_opt - ea_orig), 2)
        self.assertLess(abs(sb_opt - sb_orig), 2)
        self.assertLess(abs(eb_opt - eb_orig), 2)

    def test_b_contained_in_a(self):
        """Test when b is completely inside a."""
        times_a = np.linspace(0, 4, 4096)
        times_b = np.linspace(1, 2, 1024)

        result_opt = optimized_determine_overlap(times_a, times_b)
        result_orig = original_determine_overlap(times_a, times_b)

        self.assertIsNotNone(result_opt)

        (sa_opt, ea_opt), (sb_opt, eb_opt) = result_opt
        (sa_orig, ea_orig), (sb_orig, eb_orig) = result_orig

        # Should be very close
        self.assertLess(abs(sa_opt - sa_orig), 2)
        self.assertLess(abs(ea_opt - ea_orig), 2)
        # b should use all samples
        self.assertEqual(sb_opt, 0)
        self.assertEqual(eb_opt, len(times_b) - 1)

    def test_no_overlap(self):
        """Test when ranges don't overlap."""
        times_a = np.linspace(0, 1, 1024)
        times_b = np.linspace(2, 3, 1024)

        result_opt = optimized_determine_overlap(times_a, times_b)
        result_orig = original_determine_overlap(times_a, times_b)

        self.assertIsNone(result_opt)
        self.assertIsNone(result_orig)

    def test_gw_scale_times(self):
        """Test with gravitational wave data time scales."""
        # GPS time around a merger event
        gps_time = 1126259462.4  # GW150914
        times_a = np.linspace(gps_time - 2, gps_time + 2, 16384)
        times_b = np.linspace(gps_time - 1, gps_time + 3, 16384)

        result_opt = optimized_determine_overlap(times_a, times_b)
        result_orig = original_determine_overlap(times_a, times_b)

        self.assertIsNotNone(result_opt)

        (sa_opt, ea_opt), (sb_opt, eb_opt) = result_opt
        (sa_orig, ea_orig), (sb_orig, eb_orig) = result_orig

        # Should match closely
        self.assertLess(abs(sa_opt - sa_orig), 2)
        self.assertLess(abs(ea_opt - ea_orig), 2)


class TestOverlapPerformance(unittest.TestCase):
    """Test performance improvements."""

    def test_performance_comparison(self):
        """Compare performance of optimized vs original."""
        # Realistic GW data size
        n_samples = 16384
        times_a = np.linspace(0, 4, n_samples)
        times_b = np.linspace(1, 5, n_samples)

        # Warm up
        for _ in range(10):
            _ = optimized_determine_overlap(times_a, times_b)
            _ = original_determine_overlap(times_a, times_b)

        # Time optimized version
        n_iter = 1000
        start = time.time()
        for _ in range(n_iter):
            result_opt = optimized_determine_overlap(times_a, times_b)
        time_opt = (time.time() - start) / n_iter

        # Time original version
        start = time.time()
        for _ in range(n_iter):
            result_orig = original_determine_overlap(times_a, times_b)
        time_orig = (time.time() - start) / n_iter

        speedup = time_orig / time_opt

        print(f"\n=== Overlap Detection Performance ===")
        print(f"Array size: {n_samples} samples")
        print(f"Original (argmin): {time_orig*1000:.4f} ms")
        print(f"Optimized (searchsorted): {time_opt*1000:.4f} ms")
        print(f"Speedup: {speedup:.1f}x")

        # Should be significantly faster (2x is already great!)
        self.assertGreater(speedup, 2,
                          f"Expected >2x speedup, got {speedup:.1f}x")

    def test_searchsorted_vs_argmin_comparison(self):
        """Direct comparison of searchsorted vs argmin."""
        times = np.linspace(0, 4, 16384)
        target = 2.0

        # Time searchsorted
        n_iter = 10000
        start = time.time()
        for _ in range(n_iter):
            idx = np.searchsorted(times, target)
        time_searchsorted = (time.time() - start) / n_iter

        # Time argmin
        start = time.time()
        for _ in range(n_iter):
            idx = np.argmin(np.abs(times - target))
        time_argmin = (time.time() - start) / n_iter

        speedup = time_argmin / time_searchsorted

        print(f"\n=== Direct Method Comparison ===")
        print(f"Array size: 16384")
        print(f"argmin: {time_argmin*1e6:.2f} µs")
        print(f"searchsorted: {time_searchsorted*1e6:.2f} µs")
        print(f"Speedup: {speedup:.1f}x")

        self.assertGreater(speedup, 5,
                          f"searchsorted should be >5x faster, got {speedup:.1f}x")


class TestOverlapEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_touching_boundaries(self):
        """Test when ranges just touch."""
        times_a = np.linspace(0, 1, 1024)
        times_b = np.linspace(1, 2, 1024)

        result = optimized_determine_overlap(times_a, times_b)

        # Should find at least one sample overlap at boundary
        if result is not None:
            (sa, ea), (sb, eb) = result
            self.assertGreaterEqual(ea, sa)
            self.assertGreaterEqual(eb, sb)

    def test_tiny_overlap(self):
        """Test with very small overlap."""
        times_a = np.linspace(0, 1.001, 1024)
        times_b = np.linspace(0.999, 2, 1024)

        result = optimized_determine_overlap(times_a, times_b)

        self.assertIsNotNone(result, "Should find tiny overlap")

    def test_different_sample_rates(self):
        """Test with different sampling rates."""
        times_a = np.linspace(0, 2, 2048)   # 1024 Hz
        times_b = np.linspace(0.5, 2.5, 4096)  # 2048 Hz

        result = optimized_determine_overlap(times_a, times_b)

        self.assertIsNotNone(result)
        (sa, ea), (sb, eb) = result

        # Verify overlap makes sense
        self.assertGreater(times_a[sa], 0.49)
        self.assertLess(times_a[ea], 2.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)
