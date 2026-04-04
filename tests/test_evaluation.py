"""Tests for heron.evaluation — mismatch and calibration metrics."""

import numpy as np
import pytest

from heron.evaluation.mismatch import compute_overlap, compute_mismatch


class TestOverlap:

    def test_identical_waveforms(self):
        """Overlap of a waveform with itself should be ~1."""
        n = 1024
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h = np.sin(2 * np.pi * 100 * t) * np.exp(-50 * t)

        overlap = compute_overlap(h, h, dt)
        assert overlap == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_waveforms(self):
        """Sine and cosine at same frequency should have low overlap."""
        n = 4096
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h1 = np.sin(2 * np.pi * 100 * t)
        h2 = np.cos(2 * np.pi * 100 * t)

        overlap = compute_overlap(h1, h2, dt)
        # Not exactly zero due to finite duration, but should be small
        assert abs(overlap) < 0.1

    def test_mismatch_is_one_minus_overlap(self):
        n = 512
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h1 = np.sin(2 * np.pi * 50 * t)
        h2 = np.sin(2 * np.pi * 51 * t)

        overlap = compute_overlap(h1, h2, dt)
        mismatch = compute_mismatch(h1, h2, dt)
        assert mismatch == pytest.approx(1.0 - overlap, abs=1e-10)

    def test_with_psd(self):
        """Overlap with a PSD should not crash."""
        n = 512
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h = np.sin(2 * np.pi * 100 * t)

        freqs = np.fft.rfftfreq(n, d=dt)
        psd = np.ones_like(freqs)
        psd[0] = 1e10  # suppress DC

        overlap = compute_overlap(h, h, dt, psd=psd)
        assert overlap == pytest.approx(1.0, abs=1e-6)

    def test_zero_waveform(self):
        """Overlap with zero waveform should be 0."""
        n = 256
        h1 = np.zeros(n)
        h2 = np.sin(np.linspace(0, 10, n))
        assert compute_overlap(h1, h2, 1.0 / 4096) == 0.0

    def test_scaled_waveform_same_overlap(self):
        """Overlap is normalised, so scaling shouldn't change it."""
        n = 512
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h1 = np.sin(2 * np.pi * 100 * t)
        h2 = 1000 * h1  # same shape, different amplitude

        overlap = compute_overlap(h1, h2, dt)
        assert overlap == pytest.approx(1.0, abs=1e-10)
