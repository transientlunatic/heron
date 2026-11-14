"""Unit tests for the shared benchmark-harness metric code (B0, #641)."""
import numpy as np
import pytest

from heron.benchmark import (
    calibration_by_group, calibration_summary, chi2_calibration,
    likelihood_agreement, make_test_set, subsample_mismatch,
    subsample_overlap, whitened_chi2,
)
from heron.benchmark.calibration import EXPECTED_MEDIAN_RATIO
from heron.evaluation.mismatch import compute_mismatch

N, DT = 1024, 1.0 / 1024


def _chirp():
    t = np.arange(N) * DT
    return np.hanning(N) * np.sin(2 * np.pi * (40 * t + 60 * t ** 2))


def _fourier_shift(x, samples):
    f = np.fft.rfftfreq(len(x), DT)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * samples * DT), len(x))


# --- 1. sub-sample overlap -------------------------------------------------

class TestSubsampleOverlap:

    @pytest.mark.parametrize("frac", [0.0, 0.1, 0.37, 0.5, 0.93])
    def test_fractionally_shifted_copy_has_zero_mismatch(self, frac):
        h = _chirp()
        assert subsample_mismatch(h, _fourier_shift(h, frac), DT) < 1e-10

    def test_recovers_the_shift(self):
        h = _chirp()
        _, shift = subsample_overlap(h, _fourier_shift(h, 0.37), DT, return_shift=True)
        assert shift / DT == pytest.approx(-0.37, abs=1e-4)

    def test_integer_lag_metric_cannot_do_this(self):
        """The reason this module exists: whole-sample lags leave residual mismatch."""
        h, h2 = _chirp(), _fourier_shift(_chirp(), 0.5)
        assert compute_mismatch(h, h2, DT) > 1e-4 > subsample_mismatch(h, h2, DT)

    def test_constant_phase_is_maximised(self):
        h = _chirp()
        analytic = np.fft.ifft(np.concatenate([np.fft.rfft(h)[:-1] * 2,
                                               np.zeros(N // 2)]))
        h_rot = (analytic * np.exp(1j * 1.1)).real
        assert subsample_mismatch(h, h_rot, DT) < 1e-6

    def test_distinct_waveforms_are_not_perfect(self):
        h = _chirp()
        t = np.arange(N) * DT
        other = np.hanning(N) * np.sin(2 * np.pi * 90 * t)
        assert subsample_mismatch(h, other, DT) > 0.1

    def test_psd_weighting_runs(self):
        h = _chirp()
        psd = 1.0 + 1e-3 * np.fft.rfftfreq(N, DT) ** 2
        assert subsample_mismatch(h, _fourier_shift(h, 0.2), DT, psd) < 1e-10


# --- 2. calibration --------------------------------------------------------

class TestCalibration:

    def test_known_gaussian_ratio_and_chi2(self):
        rng = np.random.default_rng(0)
        n, ndraw = 40, 4000
        A = rng.standard_normal((n, n))
        K = A @ A.T / n + 0.5 * np.eye(n)
        Lk = np.linalg.cholesky(K)
        r = rng.standard_normal((ndraw, n)) @ Lk.T
        var = np.diag(K)

        s = calibration_summary(np.tile(var, (ndraw, 1)), r)
        assert s.integrated_ratio == pytest.approx(1.0, abs=0.03)
        # per-sample median of var/err^2 is NOT 1 for a calibrated Gaussian:
        assert s.median_ratio == pytest.approx(EXPECTED_MEDIAN_RATIO, rel=0.05)
        assert s.frac_within_2sigma == pytest.approx(0.954, abs=0.01)

        c = chi2_calibration(r, K)          # chi2 on its expected law
        assert c["mean_chi2_per_dof"] == pytest.approx(1.0, abs=0.03)
        assert c["ks_pvalue"] > 0.01

    def test_overconfident_k_is_detected(self):
        rng = np.random.default_rng(1)
        r = rng.standard_normal((2000, 30))
        s = calibration_summary(np.full_like(r, 0.1), r)       # K 10x too small
        assert s.integrated_ratio == pytest.approx(0.1, rel=0.1)
        assert chi2_calibration(r, 0.1 * np.eye(30))["ks_pvalue"] < 1e-6

    def test_single_residual_p_value_uniform_under_null(self):
        rng = np.random.default_rng(2)
        K = np.eye(25) * 2.0
        p = [whitened_chi2(rng.standard_normal(25) * np.sqrt(2.0), K)[2]
             for _ in range(500)]
        assert 0.35 < np.mean(p) < 0.65

    def test_groups(self):
        var = np.array([1.0, 1.0, 4.0, 4.0])
        err = np.array([1.0, 1.0, 2.0, 2.0])
        out = calibration_by_group(var, err, np.array(["node", "node", "mid", "mid"]))
        assert out["node"].integrated_ratio == out["mid"].integrated_ratio == 1.0


# --- 3. likelihood agreement ----------------------------------------------

class TestLikelihoodAgreement:

    @pytest.mark.parametrize("n,m", [(256, 8), (512, 16)])
    def test_random_positive_definite_k(self, n, m):
        rng = np.random.default_rng(n)
        A = rng.standard_normal((n, n))
        C = A @ A.T / n + np.eye(n)
        U = 0.5 * rng.standard_normal((n, m))
        mu = rng.standard_normal(n)
        d = mu + rng.standard_normal(n)
        out = likelihood_agreement(C, mu, U, d)
        assert out["rel_diff"] < 1e-8

    def test_k_zero(self):
        rng = np.random.default_rng(5)
        n = 256
        A = rng.standard_normal((n, n))
        C = A @ A.T / n + np.eye(n)
        mu, d = rng.standard_normal(n), rng.standard_normal(n)
        assert likelihood_agreement(C, mu, None, d)["rel_diff"] < 1e-8

    def test_detects_a_wrong_likelihood(self):
        """The check must be able to fail: perturb the low-rank factor's partner."""
        rng = np.random.default_rng(6)
        n = 128
        C = np.eye(n)
        U = rng.standard_normal((n, 4))
        mu, d = np.zeros(n), rng.standard_normal(n)
        from heron.likelihood import MarginalLogLikelihood
        wrong = MarginalLogLikelihood(C, mu, 1.1 * (U @ U.T))(d)
        right = likelihood_agreement(C, mu, U, d)["dense"]
        assert abs(wrong - right) / abs(right) > 1e-6


# --- 4. test-set generator -------------------------------------------------

class TestTestSet:

    def test_reproducible_for_fixed_seed(self):
        a, b = make_test_set(seed=7), make_test_set(seed=7)
        assert a == b and a.digest() == b.digest()

    def test_seed_changes_random_parts_only(self):
        a, b = make_test_set(seed=1), make_test_set(seed=2)
        assert a.nodes == b.nodes and a.midpoints == b.midpoints
        assert a.off_grid != b.off_grid and a.digest() != b.digest()

    def test_structure_matches_spec(self):
        ts = make_test_set()
        assert len(ts.nodes) == 19 and len(ts.midpoints) == 18   # B0: ~18 + ~18
        assert min(ts.nodes) == 0.40 and max(ts.nodes) == 0.94
        assert all(0.4 <= q <= 0.95 for q in ts.off_grid)
        assert all(0.15 <= q < 0.4 for q in ts.low_q)
        # off-grid q's are genuinely off the training grid
        assert all(min(abs(q - n) for n in ts.nodes) > 0.002 for q in ts.off_grid)
