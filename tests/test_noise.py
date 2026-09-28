"""Tests for heron.noise.noise_covariance."""
import numpy as np
import pytest
from numpy.linalg import eigvalsh

from heron.noise import noise_covariance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_psd(S0, f_low=0.0):
    """One-sided PSD that is S0 for f >= f_low, else 0."""
    def psd(freqs):
        out = np.full_like(freqs, 0.0)
        out[freqs >= f_low] = S0
        return out
    return psd


def _times(n, dt):
    return np.arange(n) * dt


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class TestStructure:

    def test_shape(self):
        C = noise_covariance(_times(50, 1/256.), _flat_psd(1.0, 20.0))
        assert C.shape == (50, 50)

    def test_symmetric(self):
        C = noise_covariance(_times(40, 1/512.), _flat_psd(1.0, 20.0))
        np.testing.assert_allclose(C, C.T, atol=1e-14)

    def test_positive_definite(self):
        C = noise_covariance(_times(30, 1/256.), _flat_psd(1.0, 20.0))
        assert eigvalsh(C).min() > 0.0

    def test_toeplitz_structure(self):
        n = 24
        C = noise_covariance(_times(n, 1/64.), _flat_psd(1.0, 5.0), jitter=0)
        # Every k-th super-diagonal should be constant.
        for k in range(1, n):
            diag_vals = np.diag(C, k)
            np.testing.assert_allclose(diag_vals, diag_vals[0], rtol=1e-10,
                                       err_msg=f"diagonal k={k} is not constant")


# ---------------------------------------------------------------------------
# Normalization: variance matches the integrated PSD
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_variance_matches_integrated_psd(self):
        # For flat S0 over [f_low, Nyquist], R(0) ≈ S0 * (Nyquist - f_low).
        S0 = 3.0
        f_low = 20.0
        dt = 1 / 512.0
        n = 1024
        nyquist = 0.5 / dt  # 256 Hz
        expected = S0 * (nyquist - f_low)

        C = noise_covariance(_times(n, dt), _flat_psd(S0, f_low),
                             f_low=f_low, jitter=0)
        # Discrete approximation error is O(df) relative to bandwidth.
        assert np.diag(C).mean() == pytest.approx(expected, rel=0.02)

    def test_scaling_linear_in_psd(self):
        times = _times(64, 1/128.)
        C1 = noise_covariance(times, _flat_psd(1.0, 20.0), jitter=0)
        C3 = noise_covariance(times, _flat_psd(3.0, 20.0), jitter=0)
        np.testing.assert_allclose(C3, 3.0 * C1, rtol=1e-12)

    def test_jitter_adds_to_diagonal_only(self):
        times = _times(16, 1/64.)
        eps = 5e-3
        C0 = noise_covariance(times, _flat_psd(1.0, 5.0), jitter=0)
        Cj = noise_covariance(times, _flat_psd(1.0, 5.0), jitter=eps)
        n = len(times)
        np.testing.assert_allclose(Cj - C0, eps * np.eye(n), atol=1e-15)


# ---------------------------------------------------------------------------
# Frequency cutoffs
# ---------------------------------------------------------------------------

class TestFrequencyCuts:

    def test_f_low_kills_contribution_below_cutoff(self):
        # A PSD that is non-zero only below f_low contributes nothing.
        f_low = 30.0
        times = _times(128, 1/256.)

        def below_only(freqs):
            return np.where((freqs > 0) & (freqs < f_low), 1.0, 0.0)

        C = noise_covariance(times, below_only, f_low=f_low, jitter=0)
        np.testing.assert_allclose(C, np.zeros_like(C), atol=1e-15)

    def test_f_high_removes_high_freq_contribution(self):
        # Restricted band [f_low, f_high] gives lower variance than full band.
        S0 = 1.0
        f_low = 20.0
        f_high = 60.0
        dt = 1 / 256.0
        times = _times(256, dt)

        C_full = noise_covariance(times, _flat_psd(S0, f_low),
                                  f_low=f_low, jitter=0)
        C_cut = noise_covariance(times, _flat_psd(S0, f_low),
                                 f_low=f_low, f_high=f_high, jitter=0)
        assert np.diag(C_cut).mean() < np.diag(C_full).mean()

    def test_infinite_psd_values_handled(self):
        # The aLIGO PSD sets S=inf below ~10 Hz; those bins must be zeroed.
        def psd_with_inf(freqs):
            out = np.where(freqs >= 20.0, 1.0, np.inf)
            out[freqs == 0.0] = np.inf
            return out

        C = noise_covariance(_times(64, 1/256.), psd_with_inf, f_low=20.0)
        assert np.all(np.isfinite(C))


# ---------------------------------------------------------------------------
# Correlation structure
# ---------------------------------------------------------------------------

class TestCorrelation:

    def test_flat_psd_gives_diagonal_covariance(self):
        # Flat PSD from f>0 to Nyquist (DC zeroed in implementation).
        # Trapezoidal quadrature gives: R(0) = S0 * (n//2 - 0.5) / (n * dt)
        # and R(l>0) ≈ 0 (off-diagonal fraction decays as 1/n).
        n = 1024
        dt = 1.0
        S0 = 4.0

        def flat_all(freqs):
            return np.where(freqs > 0, S0, 0.0)

        C = noise_covariance(_times(n, dt), flat_all, f_low=0.0, jitter=0, jitter_rel=0.0)
        # Exact trapezoidal result for this PSD (DC=0, interior×2, Nyquist×1).
        expected_var = S0 * (n // 2 - 0.5) / (n * dt)
        np.testing.assert_allclose(np.diag(C), expected_var, rtol=1e-10)
        # Off-diagonals decay as 1/n: for n=1024 the max should be negligible.
        max_offdiag = np.abs(np.triu(C, 1)).max()
        assert max_offdiag < 1e-3 * expected_var

    def test_colored_noise_is_correlated(self):
        # A steeply falling PSD produces a long-range autocorrelation.
        def steep_psd(freqs):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(freqs >= 20.0, 1.0 / freqs**3, 0.0)

        times = _times(64, 1/256.)
        C = noise_covariance(times, steep_psd, f_low=20.0, jitter=0)
        # lag-1 correlation coefficient should be significant (> 0.5).
        corr_coef = C[0, 1] / C[0, 0]
        assert corr_coef > 0.5


# ---------------------------------------------------------------------------
# Integration with the aLIGO design PSD
# ---------------------------------------------------------------------------

class TestAlIGOIntegration:

    def test_aligo_produces_finite_pd_matrix(self):
        from heron.evaluation.psd import aligo_design_psd

        n = 256
        dt = 1 / 512.0
        C = noise_covariance(_times(n, dt), aligo_design_psd, f_low=20.0)
        assert C.shape == (n, n)
        assert np.all(np.isfinite(C))
        assert eigvalsh(C).min() > 0.0

    def test_aligo_variance_physically_sensible(self):
        # aLIGO strain noise is ~1e-23 strain/rtHz; over ~200 Hz bandwidth,
        # variance ~ (1e-23)^2 * 200 ~ 2e-44.  Check order of magnitude.
        from heron.evaluation.psd import aligo_design_psd

        n = 512
        dt = 1 / 1024.0
        C = noise_covariance(_times(n, dt), aligo_design_psd, f_low=20.0, jitter=0)
        variance = np.diag(C).mean()
        assert 1e-46 < variance < 1e-42
