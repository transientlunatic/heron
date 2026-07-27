"""Tests for heron.inference.projection — analytic extrinsic projection."""
import numpy as np
import pytest

from heron.detector import antenna_patterns, project_waveform
from heron.inference.projection import project_polarisations, project_variances
from heron.types import Waveform, WaveformDict


RA, DEC, PSI, GPS = 1.95, -1.27, 0.82, 1187008882.4


def _wf(n=64, var=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(-0.2, 0.02, n)
    hp = rng.standard_normal(n)
    hc = rng.standard_normal(n)
    cov_p = np.diag(np.full(n, var))
    cov_c = np.diag(np.full(n, 2 * var))  # deliberately unequal plus/cross var
    return WaveformDict(plus=Waveform(hp, t, cov_p), cross=Waveform(hc, t, cov_c))


class TestReducesToProjectWaveform:
    """At distance_ref/ι=0/φc=0, project_polarisations must reproduce
    heron.detector.project_waveform (mean + diagonal of K) exactly."""

    def test_exact_agreement(self):
        wf = _wf()
        fp, fc = antenna_patterns(RA, DEC, PSI, GPS, "H1")
        mu, k = project_polarisations(wf, f_plus=fp, f_cross=fc)
        mu_ref, K_ref = project_waveform(wf, fp, fc)
        assert np.array_equal(mu, mu_ref)
        assert np.array_equal(k, np.diag(K_ref))


class TestDistance:

    def test_mean_scales_inverse_distance(self):
        wf = _wf()
        fp, fc = 0.7, 0.3
        mu_ref, _ = project_polarisations(wf, f_plus=fp, f_cross=fc)
        mu, k = project_polarisations(
            wf, f_plus=fp, f_cross=fc, distance=400.0, distance_ref=100.0
        )
        assert np.allclose(mu, mu_ref * (100.0 / 400.0))

    def test_variance_scales_inverse_distance_squared(self):
        wf = _wf()
        fp, fc = 0.7, 0.3
        _, k_ref = project_polarisations(wf, f_plus=fp, f_cross=fc)
        _, k = project_polarisations(
            wf, f_plus=fp, f_cross=fc, distance=400.0, distance_ref=100.0
        )
        assert np.allclose(k, k_ref * (100.0 / 400.0) ** 2)


class TestInclination:

    def test_face_on_is_identity(self):
        wf = _wf()
        fp, fc = 0.7, 0.3
        mu0, k0 = project_polarisations(wf, f_plus=fp, f_cross=fc, inclination=0.0)
        mu, k = project_polarisations(wf, f_plus=fp, f_cross=fc)
        assert np.allclose(mu, mu0)
        assert np.allclose(k, k0)

    def test_edge_on_kills_cross(self):
        """At ι=π/2, cos ι = 0 so the cross quadrature vanishes; plus keeps
        (1+cos²ι)/2 = 1/2."""
        wf = _wf()
        fp, fc = 0.7, 0.3
        mu, _ = project_polarisations(wf, f_plus=fp, f_cross=fc, inclination=np.pi / 2)
        expected = fp * 0.5 * wf["plus"].data  # cross term zeroed
        assert np.allclose(mu, expected)


class TestCoalescencePhase:

    def test_power_preserved_for_equal_variance(self):
        """The φc rotation is orthogonal, so for equal plus/cross variance the
        projected variance is unchanged (before antenna weighting differences)."""
        n = 64
        t = np.linspace(-0.2, 0.02, n)
        v = 3e-4
        wf = WaveformDict(
            plus=Waveform(np.zeros(n), t, np.diag(np.full(n, v))),
            cross=Waveform(np.zeros(n), t, np.diag(np.full(n, v))),
        )
        k0 = project_variances(np.full(n, v), np.full(n, v), f_plus=1.0, f_cross=1.0)
        k = project_variances(
            np.full(n, v), np.full(n, v), f_plus=1.0, f_cross=1.0,
            coalescence_phase=0.7,
        )
        assert np.allclose(k, k0)

    def test_rotation_mixes_mean(self):
        wf = _wf()
        fp, fc = 1.0, 0.0  # isolate plus channel
        mu, _ = project_polarisations(wf, f_plus=fp, f_cross=fc, coalescence_phase=np.pi / 4)
        # h+' = cos(π/2) h+ - sin(π/2) h× = -h×
        assert np.allclose(mu, -wf["cross"].data, atol=1e-12)
