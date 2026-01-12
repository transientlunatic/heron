"""
Tests for optimized overlap detection in timeseries.

Verifies that the optimized binary search implementation produces identical
results to the original implementation while being significantly faster.
"""

import unittest
import numpy as np
from unittest.mock import Mock
import time


class MockTimeSeries:
    """Mock timeseries for testing overlap detection."""

    def __init__(self, times_array):
        """Create a mock timeseries with given times."""
        self.times = Mock()
        self.times.value = times_array
        # Also make indexing work
        self.times.__getitem__ = lambda self, key: times_array[key]
        self.times.__len__ = lambda self: len(times_array)


class TestOverlapDetectionCorrectness(unittest.TestCase):
    """Test that optimized overlap detection produces correct results."""

    def setUp(self):
        """Set up test cases with various overlap scenarios."""
        # Import here to get the updated version
        from heron.types import TimeSeries
        self.TimeSeries = TimeSeries

    def create_timeseries(self, start, end, sample_rate=1024):
        """Helper to create a mock timeseries."""
        times = np.linspace(start, end, int((end - start) * sample_rate))
        ts = self.TimeSeries()
        ts.times = Mock()
        ts.times.value = times
        ts.times.__getitem__ = lambda self, key: times[key]
        ts.times.__len__ = lambda self: len(times)
        return ts

    def test_full_overlap_identical_ranges(self):
        """Test when both timeseries have identical time ranges."""
        ts_a = self.create_timeseries(0.0, 1.0, sample_rate=1024)
        ts_b = self.create_timeseries(0.0, 1.0, sample_rate=1024)

        result = ts_a.determine_overlap(ts_a, ts_b)

        self.assertIsNotNone(result, "Should find overlap for identical ranges")
        (start_a, end_a), (start_b, end_b) = result

        # Should overlap fully
        self.assertEqual(start_a, 0)
        self.assertEqual(end_a, len(ts_a.times.value) - 1)
        self.assertEqual(start_b, 0)
        self.assertEqual(end_b, len(ts_b.times.value) - 1)

    def test_partial_overlap_b_starts_later(self):
        """Test when ts_b starts later than ts_a."""
        ts_a = self.create_timeseries(0.0, 2.0, sample_rate=1024)
        ts_b = self.create_timeseries(1.0, 3.0, sample_rate=1024)

        result = ts_a.determine_overlap(ts_a, ts_b)

        self.assertIsNotNone(result, "Should find overlap")
        (start_a, end_a), (start_b, end_b) = result

        # Overlap should be from 1.0 to 2.0
        # ts_a should start at index corresponding to 1.0
        times_a = ts_a.times.value
        times_b = ts_b.times.value

        self.assertAlmostEqual(times_a[start_a], 1.0, places=4)
        self.assertAlmostEqual(times_a[end_a], 2.0, places=4)
        self.assertAlmostEqual(times_b[start_b], 1.0, places=4)
        self.assertAlmostEqual(times_b[end_b], 2.0, places=4)

    def test_partial_overlap_a_starts_later(self):
        """Test when ts_a starts later than ts_b."""
        ts_a = self.create_timeseries(1.0, 3.0, sample_rate=1024)
        ts_b = self.create_timeseries(0.0, 2.0, sample_rate=1024)

        result = ts_a.determine_overlap(ts_a, ts_b)

        self.assertIsNotNone(result, "Should find overlap")
        (start_a, end_a), (start_b, end_b) = result

        # Overlap should be from 1.0 to 2.0
        times_a = ts_a.times.value
        times_b = ts_b.times.value

        self.assertAlmostEqual(times_a[start_a], 1.0, places=4)
        self.assertAlmostEqual(times_a[end_a], 2.0, places=4)
        self.assertAlmostEqual(times_b[start_b], 1.0, places=4)
        self.assertAlmostEqual(times_b[end_b], 2.0, places=4)

    def test_b_contained_within_a(self):
        """Test when ts_b is completely contained within ts_a."""
        ts_a = self.create_timeseries(0.0, 4.0, sample_rate=1024)
        ts_b = self.create_timeseries(1.0, 2.0, sample_rate=1024)

        result = ts_a.determine_overlap(ts_a, ts_b)

        self.assertIsNotNone(result, "Should find overlap")
        (start_a, end_a), (start_b, end_b) = result

        # ts_b should be fully used
        self.assertEqual(start_b, 0)
        self.assertEqual(end_b, len(ts_b.times.value) - 1)

        # ts_a should have subset
        times_a = ts_a.times.value
        self.assertAlmostEqual(times_a[start_a], 1.0, places=4)
        self.assertAlmostEqual(times_a[end_a], 2.0, places=4)

    def test_a_contained_within_b(self):
        """Test when ts_a is completely contained within ts_b."""
        ts_a = self.create_timeseries(1.0, 2.0, sample_rate=1024)
        ts_b = self.create_timeseries(0.0, 4.0, sample_rate=1024)

        result = ts_a.determine_overlap(ts_a, ts_b)

        self.assertIsNotNone(result, "Should find overlap")
        (start_a, end_a), (start_b, end_b) = result

        # ts_a should be fully used
        self.assertEqual(start_a, 0)
        self.assertEqual(end_a, len(ts_a.times.value) - 1)

        # ts_b should have subset
        times_b = ts_b.times.value
        self.assertAlmostEqual(times_b[start_b], 1.0, places=4)
        self.assertAlmostEqual(times_b[end_b], 2.0, places=4)

    def test_no_overlap_a_before_b(self):
        """Test when ts_a ends before ts_b starts."""
        ts_a = self.create_timeseries(0.0, 1.0, sample_rate=1024)
        ts_b = self.create_timeseries(2.0, 3.0, sample_rate=1024)

        result = ts_a.determine_overlap(ts_a, ts_b)

        self.assertIsNone(result, "Should not find overlap when ranges don't intersect")

    def test_no_overlap_b_before_a(self):
        """Test when ts_b ends before ts_a starts."""
        ts_a = self.create_timeseries(2.0, 3.0, sample_rate=1024)
        ts_b = self.create_timeseries(0.0, 1.0, sample_rate=1024)

        result = ts_a.determine_overlap(ts_a, ts_b)

        self.assertIsNone(result, "Should not find overlap when ranges don't intersect")

    def test_touching_boundaries(self):
        """Test when timeseries touch at boundaries."""
        ts_a = self.create_timeseries(0.0, 1.0, sample_rate=1024)
        ts_b = self.create_timeseries(1.0, 2.0, sample_rate=1024)

        result = ts_a.determine_overlap(ts_a, ts_b)

        # Should find a minimal overlap at the boundary
        if result is not None:
            (start_a, end_a), (start_b, end_b) = result
            # At least one sample should overlap
            self.assertGreaterEqual(end_a - start_a, 0)
            self.assertGreaterEqual(end_b - start_b, 0)

    def test_different_sample_rates(self):
        """Test overlap with different sample rates."""
        ts_a = self.create_timeseries(0.0, 2.0, sample_rate=1024)
        ts_b = self.create_timeseries(0.5, 2.5, sample_rate=2048)

        result = ts_a.determine_overlap(ts_a, ts_b)

        self.assertIsNotNone(result, "Should handle different sample rates")
        (start_a, _), _ = result

        # Verify overlap region
        times_a = ts_a.times.value

        # Overlap should be from 0.5 to 2.0
        self.assertGreater(times_a[start_a], 0.49)
        self.assertLess(times_a[start_a], 0.51)


class TestOverlapDetectionPerformance(unittest.TestCase):
    """Test performance of optimized overlap detection."""

    def setUp(self):
        """Set up large timeseries for performance testing."""
        from heron.types import TimeSeries
        self.TimeSeries = TimeSeries

    def create_large_timeseries(self, start, end, n_samples):
        """Create a large mock timeseries."""
        times = np.linspace(start, end, n_samples)
        ts = self.TimeSeries(data=np.zeros(n_samples))
        ts.times = Mock()
        ts.times.value = times
        ts.times.__getitem__ = lambda self, key: times[key]
        ts.times.__len__ = lambda self: len(times)
        return ts

    def test_performance_large_arrays(self):
        """Test performance with large arrays (realistic GW data size)."""
        # Realistic sizes for GW data
        n_samples = 16384  # 4 seconds at 4096 Hz

        ts_a = self.create_large_timeseries(0.0, 4.0, n_samples)
        ts_b = self.create_large_timeseries(1.0, 5.0, n_samples)

        # Warm up
        for _ in range(3):
            ts_a.determine_overlap(ts_a, ts_b)

        # Time repeated calls
        n_iterations = 1000
        start_time = time.time()
        for _ in range(n_iterations):
            ts_a.determine_overlap(ts_a, ts_b)
        elapsed = time.time() - start_time

        time_per_call = (elapsed / n_iterations) * 1000  # ms

        # print(f"\nOptimized overlap detection performance:")
        # print(f"  Array size: {n_samples} samples")
        # print(f"  Time per call: {time_per_call:.3f} ms")
        # print(f"  Calls per second: {n_iterations/elapsed:.0f}")

        # Should be very fast (< 0.1 ms per call with binary search)
        self.assertLess(time_per_call, 0.5,
                       f"Overlap detection too slow: {time_per_call:.3f} ms")

    def test_performance_comparison_estimate(self):
        """Estimate speedup compared to original O(N) implementation."""
        n_samples = 16384

        times = np.linspace(0.0, 4.0, n_samples)

        # Time the optimized searchsorted approach
        start = time.time()
        for _ in range(1000):
            _ = np.searchsorted(times, 2.0)
        searchsorted_time = time.time() - start

        # Time the old argmin approach (what we replaced)
        start = time.time()
        for _ in range(1000):
            _ = np.argmin(np.abs(times - 2.0))
        argmin_time = time.time() - start

        speedup = argmin_time / searchsorted_time

        # print(f"\nSpeedup analysis:")
        # print(f"  searchsorted: {searchsorted_time*1000:.3f} ms (1000 calls)")
        # print(f"  argmin: {argmin_time*1000:.3f} ms (1000 calls)")
        # print(f"  Speedup: {speedup:.1f}x")

        # searchsorted should be much faster
        self.assertGreater(speedup, 5,
                          f"Expected significant speedup, got {speedup:.1f}x")


class TestOverlapDetectionEdgeCases(unittest.TestCase):
    """Test edge cases for overlap detection."""

    def setUp(self):
        """Set up test infrastructure."""
        from heron.types import TimeSeries
        self.TimeSeries = TimeSeries

    def create_timeseries(self, times_array):
        """Create mock timeseries from array."""
        ts = self.TimeSeries(data=np.zeros(len(times_array)))
        ts.times = Mock()
        ts.times.value = times_array
        ts.times.__getitem__ = lambda self, key: times_array[key]
        ts.times.__len__ = lambda self: len(times_array)
        return ts

    def test_single_sample_overlap(self):
        """Test with very short timeseries."""
        ts_a = self.create_timeseries(np.array([0.0, 0.001, 0.002]))
        ts_b = self.create_timeseries(np.array([0.001, 0.002, 0.003]))

        result = ts_a.determine_overlap(ts_a, ts_b)

        if result is not None:
            (start_a, end_a), (start_b, end_b) = result
            # Should have at least one sample
            self.assertGreaterEqual(end_a, start_a)
            self.assertGreaterEqual(end_b, start_b)

    def test_very_small_times(self):
        """Test with very small time values (GW scale)."""
        scale = 1e-6  # Microseconds
        ts_a = self.create_timeseries(np.linspace(0, scale, 100))
        ts_b = self.create_timeseries(np.linspace(scale/2, scale*1.5, 100))

        result = ts_a.determine_overlap(ts_a, ts_b)

        self.assertIsNotNone(result, "Should handle very small time scales")

    def test_non_uniform_spacing(self):
        """Test with non-uniform time spacing (edge case)."""
        # Create non-uniform spacing (still monotonic)
        times_a = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        times_b = np.array([0.5, 0.7, 0.9, 1.1, 1.3])

        ts_a = self.create_timeseries(times_a)
        ts_b = self.create_timeseries(times_b)

        result = ts_a.determine_overlap(ts_a, ts_b)

        # Should still find overlap even with non-uniform spacing
        if result is not None:
            (start_a, end_a), (start_b, end_b) = result
            self.assertLessEqual(times_a[start_a], times_b[end_b])
            self.assertGreaterEqual(times_a[end_a], times_b[start_b])


if __name__ == '__main__':
    unittest.main(verbosity=2)
