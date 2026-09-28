"""Tests for heron.detector — antenna patterns and waveform projection."""
import numpy as np
import pytest

from heron.detector import (
    antenna_patterns,
    detector_tensor,
    gmst_at_gps,
    project_waveform,
    _antenna_patterns_numpy,
    _DETECTORS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_waveform_dict(n=10, fp=1.0, fc=0.5):
    """Minimal WaveformDict-like object for projection tests."""
    from heron.types import Waveform, WaveformDict
    times = np.linspace(-0.5, 0.0, n)
    rng = np.random.default_rng(42)
    data_p = fp * rng.standard_normal(n)
    data_c = fc * rng.standard_normal(n)
    cov_p = np.eye(n) * fp**2
    cov_c = np.eye(n) * fc**2
    wf_p = Waveform(data=data_p, times=times, covariance=cov_p)
    wf_c = Waveform(data=data_c, times=times, covariance=cov_c)
    return WaveformDict(plus=wf_p, cross=wf_c)


# ---------------------------------------------------------------------------
# gmst_at_gps
# ---------------------------------------------------------------------------

class TestGMST:

    def test_returns_float_in_range(self):
        g = gmst_at_gps(1187008882.0)
        assert 0.0 <= g < 2.0 * np.pi

    def test_increases_with_time(self):
        t0 = 1187008882.0
        g0 = gmst_at_gps(t0)
        g1 = gmst_at_gps(t0 + 3600.0)
        # One hour later, GMST should be ~0.2617 rad larger (mod 2π).
        delta = (g1 - g0) % (2 * np.pi)
        expected = 7.2921150e-5 * 3600.0
        assert delta == pytest.approx(expected, rel=1e-4)

    def test_close_to_lal(self):
        lal = pytest.importorskip("lal")
        gps = 1187008882.0
        gmst_mine = gmst_at_gps(gps)
        gmst_lal = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(gps)) % (2 * np.pi)
        diff = abs((gmst_mine - gmst_lal + np.pi) % (2 * np.pi) - np.pi)
        assert diff < 0.01  # < 0.6 degrees


# ---------------------------------------------------------------------------
# detector_tensor
# ---------------------------------------------------------------------------

class TestDetectorTensor:

    @pytest.mark.parametrize("det", ["H1", "L1", "V1"])
    def test_symmetric(self, det):
        D = detector_tensor(det)
        np.testing.assert_allclose(D, D.T, atol=1e-15)

    @pytest.mark.parametrize("det", ["H1", "L1", "V1"])
    def test_traceless(self, det):
        # D = (x⊗x − y⊗y)/2; trace = (|x|² − |y|²)/2 ≈ 0 for unit arm vectors.
        x, y = _DETECTORS[det]
        D = detector_tensor(det)
        expected_trace = 0.5 * (np.dot(x, x) - np.dot(y, y))
        assert np.trace(D) == pytest.approx(expected_trace, abs=1e-14)

    @pytest.mark.parametrize("det", ["H1", "L1", "V1"])
    def test_matches_lal_response(self, det):
        lal = pytest.importorskip("lal")
        D_mine = detector_tensor(det)
        D_lal = np.array(lal.cached_detector_by_prefix[det].response)
        np.testing.assert_allclose(D_mine, D_lal, atol=1e-6)


# ---------------------------------------------------------------------------
# Numpy antenna pattern formula
# ---------------------------------------------------------------------------

class TestNumpyAntennaPatterns:
    """Test the pure-numpy implementation against known analytic results."""

    def test_overhead_source_ideal_detector(self):
        # Ideal L-detector: arms along x=[1,0,0], y=[0,1,0].
        # D = [[0.5,0,0],[0,-0.5,0],[0,0,0]].
        # For dec=0: q = (0, 0, -1) regardless of ha.
        # F+(psi=0) = p^T D p - q^T D q = 0.5*sin²ha - 0.5*cos²ha - 0
        #           = -0.5*cos(2*ha).
        # F×(psi=0) = 2*p^T D q = 2*(sin ha, cos ha, 0)·D·(0,0,-1) = 0.
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])

        gps_time = 0.0
        ha = gmst_at_gps(gps_time) - 0.0  # ra=0

        from heron import detector as det_mod
        orig = det_mod._DETECTORS.copy()
        det_mod._DETECTORS["TEST"] = (x, y)
        try:
            fp, fc = _antenna_patterns_numpy(0.0, 0.0, 0.0, gps_time, "TEST")
        finally:
            det_mod._DETECTORS = orig

        assert fp == pytest.approx(-0.5 * np.cos(2 * ha), abs=1e-12)
        assert fc == pytest.approx(0.0, abs=1e-12)

    def test_psi_rotation_quarter_turn(self):
        # F+(π/4) = F×(0), F×(π/4) = -F+(0) — rotation by 2ψ = π/2.
        fp0, fc0 = _antenna_patterns_numpy(0.5, 0.3, 0.0, 1.0, "H1")
        fp45, fc45 = _antenna_patterns_numpy(0.5, 0.3, np.pi / 4, 1.0, "H1")
        assert fp45 == pytest.approx(fc0, abs=1e-12)
        assert fc45 == pytest.approx(-fp0, abs=1e-12)

    def test_psi_rotation_half_turn(self):
        # F+(π/2) = -F+(0), F×(π/2) = -F×(0) — rotation by 2ψ = π.
        fp0, fc0 = _antenna_patterns_numpy(1.2, -0.4, 0.0, 2.5, "L1")
        fp90, fc90 = _antenna_patterns_numpy(1.2, -0.4, np.pi / 2, 2.5, "L1")
        assert fp90 == pytest.approx(-fp0, abs=1e-12)
        assert fc90 == pytest.approx(-fc0, abs=1e-12)

    def test_amplitude_invariant_under_psi(self):
        # |F+|² + |F×|² is constant as ψ varies.
        ra, dec, gmst = 0.8, 0.2, 1.5
        amp_sq = None
        for psi in np.linspace(0, np.pi, 20):
            fp, fc = _antenna_patterns_numpy(ra, dec, psi, 0.0, "H1")
            val = fp**2 + fc**2
            if amp_sq is None:
                amp_sq = val
            assert val == pytest.approx(amp_sq, rel=1e-12)

    @pytest.mark.parametrize("det", ["H1", "L1", "V1"])
    def test_matches_lal(self, det):
        lal = pytest.importorskip("lal")
        from heron.detector import gmst_at_gps
        gps = 1187008882.0
        gmst = gmst_at_gps(gps)
        D_lal = np.array(lal.cached_detector_by_prefix[det].response)
        for ra, dec, psi in [
            (0.0, 0.0, 0.0),
            (np.pi / 2, 0.0, 0.0),
            (0.0, np.pi / 4, 0.0),
            (1.0, 0.5, 0.3),
            (2.5, -0.8, 1.1),
        ]:
            fp_mine, fc_mine = _antenna_patterns_numpy(ra, dec, psi, gps, det)
            fp_lal, fc_lal = lal.ComputeDetAMResponse(D_lal, ra, dec, psi, gmst)
            assert fp_mine == pytest.approx(fp_lal, abs=1e-6), (
                f"F+ mismatch at ra={ra}, dec={dec}, psi={psi}: "
                f"mine={fp_mine:.6f}, lal={fp_lal:.6f}"
            )
            assert fc_mine == pytest.approx(fc_lal, abs=1e-6), (
                f"F× mismatch at ra={ra}, dec={dec}, psi={psi}: "
                f"mine={fc_mine:.6f}, lal={fc_lal:.6f}"
            )


# ---------------------------------------------------------------------------
# antenna_patterns dispatch
# ---------------------------------------------------------------------------

class TestAntennaPatterns:

    def test_returns_two_floats(self):
        fp, fc = antenna_patterns(1.0, 0.5, 0.3, 1187008882.0, "H1")
        assert isinstance(fp, float)
        assert isinstance(fc, float)

    def test_unknown_detector_raises(self):
        with pytest.raises((KeyError, Exception)):
            antenna_patterns(0.0, 0.0, 0.0, 0.0, "INVALID")

    def test_consistent_with_numpy(self):
        ra, dec, psi, gps = 0.8, -0.3, 0.5, 1187008882.0
        fp_dispatch, fc_dispatch = antenna_patterns(ra, dec, psi, gps, "H1")
        fp_np, fc_np = _antenna_patterns_numpy(ra, dec, psi, gps, "H1")
        # If LAL available they should match the numpy fallback to ~1e-5
        assert fp_dispatch == pytest.approx(fp_np, abs=1e-5)
        assert fc_dispatch == pytest.approx(fc_np, abs=1e-5)


# ---------------------------------------------------------------------------
# project_waveform
# ---------------------------------------------------------------------------

class TestProjectWaveform:

    def test_output_shapes(self):
        n = 20
        wfd = _make_waveform_dict(n=n)
        mu, K = project_waveform(wfd, f_plus=0.7, f_cross=-0.4)
        assert mu.shape == (n,)
        assert K.shape == (n, n)

    def test_zero_antenna_patterns_gives_zero(self):
        wfd = _make_waveform_dict(n=10)
        mu, K = project_waveform(wfd, f_plus=0.0, f_cross=0.0)
        np.testing.assert_allclose(mu, 0.0, atol=1e-15)
        np.testing.assert_allclose(K, 0.0, atol=1e-15)

    def test_plus_only(self):
        n = 8
        wfd = _make_waveform_dict(n=n)
        mu, K = project_waveform(wfd, f_plus=1.0, f_cross=0.0)
        np.testing.assert_allclose(mu, wfd["plus"].data)
        np.testing.assert_allclose(K, wfd["plus"].covariance)

    def test_cross_only(self):
        n = 8
        wfd = _make_waveform_dict(n=n)
        mu, K = project_waveform(wfd, f_plus=0.0, f_cross=1.0)
        np.testing.assert_allclose(mu, wfd["cross"].data)
        np.testing.assert_allclose(K, wfd["cross"].covariance)

    def test_linearity_in_f_plus(self):
        wfd = _make_waveform_dict(n=15)
        mu1, K1 = project_waveform(wfd, f_plus=2.0, f_cross=0.0)
        mu2, K2 = project_waveform(wfd, f_plus=1.0, f_cross=0.0)
        np.testing.assert_allclose(mu1, 2.0 * mu2, atol=1e-14)
        np.testing.assert_allclose(K1, 4.0 * K2, atol=1e-14)

    def test_covariance_adds_quadratically(self):
        # K = F+² K+ + F×² K× — verify quadratic scaling.
        n = 6
        wfd = _make_waveform_dict(n=n, fp=1.0, fc=1.0)
        fp, fc = 0.6, 0.8
        _, K = project_waveform(wfd, fp, fc)
        expected = fp**2 * wfd["plus"].covariance + fc**2 * wfd["cross"].covariance
        np.testing.assert_allclose(K, expected, atol=1e-14)

    def test_symmetry_of_covariance(self):
        wfd = _make_waveform_dict(n=12)
        _, K = project_waveform(wfd, 0.5, -0.3)
        np.testing.assert_allclose(K, K.T, atol=1e-14)
