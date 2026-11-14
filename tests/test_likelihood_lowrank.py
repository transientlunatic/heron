"""Tests for the low-rank (Woodbury) waveform-covariance likelihood path."""
import numpy as np
import pytest
import torch

from heron.likelihood import LowRankMarginalLogLikelihood, MarginalLogLikelihood
from heron.inference.detectors import Detector
from heron.inference.network import NetworkLikelihood
from heron.units import MTSUN_SI
from heron.models.gp.demod_kron import (
    KroneckerDemodGPSurrogate,
    generate_kronecker_grid,
    warped_tau_grid,
)

from test_gp_delta import _make_models

TC = 1187008882.4
RA, DEC, PSI = 1.95, -1.27, 0.82


@pytest.fixture(scope="module")
def dense_problem():
    rng = np.random.default_rng(3)
    n, m = 120, 7
    A = rng.standard_normal((n, n))
    C = A @ A.T / n + np.eye(n)
    U = 0.5 * rng.standard_normal((n, m))
    mu = rng.standard_normal(n)
    d = mu + rng.standard_normal(n)
    L_C = torch.linalg.cholesky(torch.as_tensor(C, dtype=torch.float64))
    return C, U, mu, d, L_C


class TestLowRankMarginalLogLikelihood:

    def test_matches_dense(self, dense_problem):
        C, U, mu, d, L_C = dense_problem
        dense = MarginalLogLikelihood(C, mu, U @ U.T)
        low = LowRankMarginalLogLikelihood(L_C, mu, U)
        assert low(d) == pytest.approx(dense(d), rel=1e-12)
        assert low.log_det == pytest.approx(dense.log_det, rel=1e-12)

    def test_whiten_bilinear_form(self, dense_problem):
        C, U, mu, d, L_C = dense_problem
        low = LowRankMarginalLogLikelihood(L_C, mu, U)
        Sinv = np.linalg.inv(C + U @ U.T)
        rng = np.random.default_rng(0)
        v, w = rng.standard_normal(len(mu)), rng.standard_normal(len(mu))
        assert float(torch.dot(low.whiten(v), low.whiten(w))) == pytest.approx(
            v @ Sinv @ w, rel=1e-10)

    @pytest.mark.parametrize("U", [None, np.zeros((120, 0)), np.zeros((120, 3))])
    def test_zero_factor_is_matched_filter(self, dense_problem, U):
        C, _, mu, d, L_C = dense_problem
        assert LowRankMarginalLogLikelihood(L_C, mu, U)(d) == pytest.approx(
            MarginalLogLikelihood(C, mu, 0.0)(d), rel=1e-12)


@pytest.fixture(scope="module")
def surrogate():
    reference, oracle = _make_models()
    t_unit = 60.0 * MTSUN_SI
    tau = warped_tau_grid(-0.4 / t_unit, 0.02 / t_unit, 50)
    qs = np.linspace(0.3, 0.9, 7)
    yp, yc = generate_kronecker_grid(oracle, qs, tau, 60.0, 100.0, 20.0)
    return KroneckerDemodGPSurrogate(
        qs, tau, yp, yc, base_approximant=reference, oracle_approximant=None,
        # The toy residual is tiny next to C; inflate K so its effect on the
        # likelihood is well above roundoff.
        phase_correction=0.0, maxiter=60, covariance_inflation=1e9,
    )


@pytest.fixture
def network(surrogate, flat_psd):
    n, fs = 256, 512.0
    times = TC + (np.arange(n) - n // 2) / fs
    dets = [Detector.from_name(p, psd_fn=flat_psd) for p in ("H1", "L1")]
    rng = np.random.default_rng(5)
    data = {d.prefix: 0.1 * rng.standard_normal(n) for d in dets}
    return times, dets, data


PARAMS = {"mass_ratio": 0.55, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI,
          "inclination": 0.4, "luminosity_distance": 150.0}


def _dense_reference(like: NetworkLikelihood, params, marginal=False):
    """Recompute the low-rank likelihood with a dense K = U U^T per detector."""
    from heron.inference.projection import project_polarisations

    intrinsic, extr = like._split_params(params)
    total = 0.0
    for ch in like._channels:
        det = ch.detector
        dt = det.time_delay_from_geocentre(extr["ra"], extr["dec"], extr["tc"])
        wf, factors = like._predict_factor({**intrinsic, "times": like.times - (extr["tc"] + dt)})
        fp, fc = det.antenna_patterns(extr["ra"], extr["dec"], extr["psi"], extr["tc"])
        mu, _ = project_polarisations(
            wf, f_plus=fp, f_cross=fc, distance=extr["distance"],
            distance_ref=like._distance_ref, inclination=extr["inclination"],
            coalescence_phase=extr["coalescence_phase"])
        U = like._detector_factor(factors, fp, fc, extr, extr["coalescence_phase"])
        total += MarginalLogLikelihood(None, like._hp_filter(mu), U @ U.T,
                                       _L_C=ch.L_C)(ch.data)
    return total


class TestNetworkLowRank:

    def test_matches_dense_k_from_same_factor(self, surrogate, network):
        times, dets, data = network
        like = NetworkLikelihood(data, times, dets, surrogate,
                                 waveform_covariance="lowrank")
        params = {**PARAMS, "coalescence_phase": 0.7}
        assert like(params) == pytest.approx(_dense_reference(like, params), rel=1e-10)

    def test_k_actually_changes_likelihood(self, surrogate, network):
        times, dets, data = network
        params = {**PARAMS, "coalescence_phase": 0.7}
        with_k = NetworkLikelihood(data, times, dets, surrogate,
                                   waveform_covariance="lowrank")(params)
        no_k = NetworkLikelihood(data, times, dets, surrogate,
                                 use_waveform_uncertainty=False)(params)
        assert abs(with_k - no_k) > 1e-2

    def test_no_k_lowrank_flag_is_matched_filter(self, surrogate, network):
        times, dets, data = network
        params = {**PARAMS, "coalescence_phase": 0.7}
        a = NetworkLikelihood(data, times, dets, surrogate, use_waveform_uncertainty=False,
                              waveform_covariance="lowrank")(params)
        b = NetworkLikelihood(data, times, dets, surrogate,
                              use_waveform_uncertainty=False)(params)
        assert a == pytest.approx(b, rel=1e-12)

    @pytest.mark.parametrize("use_k, tol", [(False, 1e-3), (True, 5e-3)])
    def test_phase_marginalised_matches_numerical_integral(self, surrogate, network,
                                                           use_k, tol):
        """Analytic phase marginal vs a brute-force phase integral of the
        fixed-phase Gaussian, with K held at the phi_c=0 reference in both.

        The analytic form assumes A^T S^-1 A = B^T S^-1 B and A^T S^-1 B = 0.
        With K that needs a stationary S (documented approximation); even
        without K it needs A, B to be an exact quadrature pair, which holds
        only for an amplitude slowly varying against the carrier -- the toy
        chirplet's 0.15 s Gaussian envelope at 30 Hz leaves ~1e-4 nats. So
        tolerances, not equality."""
        times, dets, data = network
        kw = dict(waveform_covariance="lowrank", use_waveform_uncertainty=use_k)
        marg = NetworkLikelihood(data, times, dets, surrogate, marginalize_phase=True, **kw)
        fixed = NetworkLikelihood(data, times, dets, surrogate, **kw)
        # Freeze K at phi_c = 0 in the fixed-phase evaluator.
        orig = fixed._detector_factor
        fixed._detector_factor = lambda f, fp, fc, e, phi: orig(f, fp, fc, e, 0.0)
        phis = np.linspace(0, np.pi, 400, endpoint=False)
        lls = np.array([fixed({**PARAMS, "coalescence_phase": p}) for p in phis])
        from scipy.special import logsumexp
        brute = logsumexp(lls) - np.log(len(phis))
        assert marg(PARAMS) == pytest.approx(brute, abs=tol)

    def test_lowrank_rejects_k_smoothing(self, surrogate, network):
        times, dets, data = network
        with pytest.raises(ValueError, match="k_smoothing"):
            NetworkLikelihood(data, times, dets, surrogate, waveform_covariance="lowrank",
                              k_smoothing_offsets=[0.01])

    def test_rank_cap_is_respected(self, surrogate, network):
        times, dets, data = network
        like = NetworkLikelihood(data, times, dets, surrogate,
                                 waveform_covariance="lowrank", covariance_rank=4)
        _, factors = like._predict_factor({"mass_ratio": 0.55, "times": times - TC})
        assert factors["plus"].shape[1] <= 8
