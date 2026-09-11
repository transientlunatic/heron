"""Tests for heron.evaluation — mismatch, calibration, and PSD utilities."""

import numpy as np
import pytest

from heron.evaluation.mismatch import compute_overlap, compute_mismatch
from heron.evaluation.psd import aligo_design_psd, _analytic_aligo_psd


class TestOverlap:

    def test_identical_waveforms(self):
        """Overlap of a waveform with itself should be 1."""
        n = 1024
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h = np.sin(2 * np.pi * 100 * t) * np.exp(-50 * t)

        overlap = compute_overlap(h, h, dt)
        assert overlap == pytest.approx(1.0, abs=1e-6)

    def test_phase_shifted_waveforms(self):
        """Sine and cosine are the same waveform shifted by 90°.

        With phase maximisation (default), their overlap should be ~1.
        Without it, the overlap is near zero for a long signal.
        """
        n = 4096
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h1 = np.sin(2 * np.pi * 100 * t)
        h2 = np.cos(2 * np.pi * 100 * t)

        # Phase-maximised (default): h2 is h1 rotated by 90° → overlap ≈ 1
        assert compute_overlap(h1, h2, dt, maximize_phase=True, maximize_time=False) == pytest.approx(1.0, abs=0.01)

        # No maximisation: overlap is near zero for long monochromatic signals
        assert abs(compute_overlap(h1, h2, dt, maximize_phase=False, maximize_time=False)) < 0.05

    def test_time_shifted_waveform(self):
        """Overlap with time maximisation should recover a time-shifted copy."""
        n = 2048
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        # Compact waveform well within the signal
        h = np.sin(2 * np.pi * 80 * t) * np.exp(-200 * (t - 0.1) ** 2)

        # Shift by 5 ms (well within the 50 ms search window)
        shift_samples = 20  # 20/4096 ≈ 4.9 ms
        h_shifted = np.roll(h, shift_samples)

        overlap_with_max = compute_overlap(h, h_shifted, dt, maximize_time=True)
        overlap_no_max = compute_overlap(h, h_shifted, dt, maximize_time=False)

        # Time maximisation should recover the overlap; no-max should be lower
        assert overlap_with_max > overlap_no_max
        assert overlap_with_max == pytest.approx(1.0, abs=0.05)

    def test_mismatch_is_one_minus_overlap(self):
        n = 512
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h1 = np.sin(2 * np.pi * 50 * t)
        h2 = np.sin(2 * np.pi * 51 * t)

        overlap = compute_overlap(h1, h2, dt)
        mismatch = compute_mismatch(h1, h2, dt)
        assert mismatch == pytest.approx(1.0 - overlap, abs=1e-10)

    def test_with_flat_psd(self):
        """Overlap with a flat PSD (minus DC) should equal no-PSD overlap."""
        n = 512
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h = np.sin(2 * np.pi * 100 * t)

        freqs = np.fft.rfftfreq(n, d=dt)
        psd = np.ones_like(freqs)
        psd[0] = np.inf  # suppress DC

        overlap_psd = compute_overlap(h, h, dt, psd=psd)
        assert overlap_psd == pytest.approx(1.0, abs=1e-6)

    def test_zero_waveform(self):
        """Overlap with zero waveform should be 0."""
        n = 256
        h1 = np.zeros(n)
        h2 = np.sin(np.linspace(0, 10, n))
        assert compute_overlap(h1, h2, 1.0 / 4096) == 0.0

    def test_scaled_waveform_same_overlap(self):
        """Overlap is normalised — amplitude scaling must not change it."""
        n = 512
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h1 = np.sin(2 * np.pi * 100 * t)
        h2 = 1000 * h1

        overlap = compute_overlap(h1, h2, dt)
        assert overlap == pytest.approx(1.0, abs=1e-6)

    def test_negated_waveform_phase_max(self):
        """Negation is a 180° phase shift; phase-maximised overlap should be ~1."""
        n = 512
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h = np.sin(2 * np.pi * 100 * t) * np.exp(-10 * t)

        overlap = compute_overlap(h, -h, dt, maximize_phase=True, maximize_time=False)
        assert overlap == pytest.approx(1.0, abs=1e-6)


class TestALIGOPSD:

    def test_analytic_psd_positive(self):
        """Analytic PSD should be positive and finite between 10 Hz and 2 kHz."""
        freqs = np.linspace(20.0, 2000.0, 200)
        psd = _analytic_aligo_psd(freqs)
        assert np.all(np.isfinite(psd))
        assert np.all(psd > 0)

    def test_aligo_design_psd_low_freq_suppressed(self):
        """Frequencies below 10 Hz should be set to inf."""
        freqs = np.array([0.0, 5.0, 9.9, 10.0, 100.0])
        psd = aligo_design_psd(freqs)
        assert np.isinf(psd[0])  # DC
        assert np.isinf(psd[1])  # 5 Hz
        assert np.isinf(psd[2])  # 9.9 Hz
        assert np.isfinite(psd[3])  # 10 Hz — boundary
        assert np.isfinite(psd[4])  # 100 Hz

    def test_aligo_design_psd_shape(self):
        """PSD should match the input frequency array shape."""
        freqs = np.linspace(0.0, 2048.0, 1025)
        psd = aligo_design_psd(freqs)
        assert psd.shape == freqs.shape

    def test_mismatch_with_aligo_psd(self):
        """Mismatch of identical waveforms should be ~0 with ALIGO PSD."""
        n = 1024
        dt = 1.0 / 4096
        t = np.arange(n) * dt
        h = np.sin(2 * np.pi * 100 * t) * np.exp(-50 * (t - 0.1) ** 2)

        freqs = np.fft.rfftfreq(n, d=dt)
        psd = aligo_design_psd(freqs)

        mm = compute_mismatch(h, h, dt, psd=psd)
        assert mm == pytest.approx(0.0, abs=1e-6)
