"""Integration tests for GWLikelihood — injection recovery."""
import numpy as np
import pytest

from heron.detector import antenna_patterns, project_waveform
from heron.gw_likelihood import GWLikelihood
from heron.types import Waveform, WaveformDict


# ---------------------------------------------------------------------------
# Stub surrogate
# ---------------------------------------------------------------------------

class _SineSurrogate:
    """Deterministic surrogate for testing.

    h+(t) = A sin(2π f0 t),  h×(t) = A cos(2π f0 t)

    GP covariance is ε·I (near-perfect knowledge), so the marginal
    likelihood collapses to the standard N(d; μ, C) likelihood.
    The surrogate ignores all parameters except 'times'.
    """

    def __init__(self, f0: float = 50.0, amplitude: float = 1.0):
        self.f0 = f0
        self.amplitude = amplitude

    def predict(self, params: dict) -> WaveformDict:
        times = np.asarray(params["times"], dtype=float)
        n = len(times)
        phase = 2.0 * np.pi * self.f0 * times
        A = self.amplitude
        cov = np.eye(n) * (A * 1e-10) ** 2  # near-zero but positive-definite

        return WaveformDict(
            plus=Waveform(data=A * np.sin(phase), times=times, covariance=cov),
            cross=Waveform(data=A * np.cos(phase), times=times, covariance=cov),
        )


# Flat PSD in test units: S(f) = 1.0 for f ≥ 20 Hz.
# With dt=1/512, N=256: C_ii ≈ 1.0 × (256−20) ≈ 236.
# This keeps all numbers O(1) and avoids the aLIGO strain scale (~10⁻²³)
# which would be swamped by the default jitter (1e-10).
def _flat_psd(freqs):
    return np.where(freqs >= 20.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRUE_TC  = 1187008882.4
TRUE_RA  = 1.95
TRUE_DEC = -1.27
TRUE_PSI = 0.82
TRUE_Q   = 0.8
DETECTOR = "H1"


@pytest.fixture(scope="module")
def surrogate():
    return _SineSurrogate(f0=50.0, amplitude=1.0)


@pytest.fixture(scope="module")
def injection(surrogate):
    """Build a zero-noise injection and return (gw_ll, true_params, mu_true)."""
    dt = 1.0 / 512.0
    n = 256
    # Data grid centred on the merger time.
    times = TRUE_TC + (np.arange(n) - n // 2) * dt

    # True projected strain (no noise).
    fp_true, fc_true = antenna_patterns(TRUE_RA, TRUE_DEC, TRUE_PSI, TRUE_TC, DETECTOR)
    t_rel = times - TRUE_TC
    wf = surrogate.predict({"times": t_rel})
    mu_true, _ = project_waveform(wf, fp_true, fc_true)

    data = mu_true  # zero noise: residual at truth is exactly 0

    gw_ll = GWLikelihood(
        data=data,
        times=times,
        psd_fn=_flat_psd,
        surrogate=surrogate,
        detector=DETECTOR,
        f_low=20.0,
    )

    true_params = {
        "mass_ratio": TRUE_Q,
        "tc": TRUE_TC,
        "ra": TRUE_RA,
        "dec": TRUE_DEC,
        "psi": TRUE_PSI,
    }
    return gw_ll, true_params, mu_true


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------

class TestBasicSanity:

    def test_returns_finite_float(self, injection):
        gw_ll, true_params, _ = injection
        ll = gw_ll(true_params)
        assert np.isfinite(ll)
        assert isinstance(ll, float)

    def test_matches_oracle(self, injection):
        """GWLikelihood must agree with the reference numpy oracle."""
        from numpy.linalg import slogdet, solve
        gw_ll, true_params, _ = injection

        C = gw_ll.noise_covariance
        fp, fc = antenna_patterns(TRUE_RA, TRUE_DEC, TRUE_PSI, TRUE_TC, DETECTOR)
        t_rel = gw_ll.times - TRUE_TC
        wf = gw_ll.surrogate.predict({"times": t_rel})
        mu, K = project_waveform(wf, fp, fc)

        S = C + K
        r = gw_ll.data - mu
        n = len(r)
        _, logdet = slogdet(S)
        oracle = -0.5 * (n * np.log(2 * np.pi) + logdet + r @ solve(S, r))

        assert gw_ll(true_params) == pytest.approx(oracle, rel=1e-6)

    def test_residual_zero_at_truth(self, injection):
        """With d = mu_true, the quadratic term must vanish at the true parameters."""
        gw_ll, true_params, mu_true = injection
        assert np.allclose(gw_ll.data, mu_true)


# ---------------------------------------------------------------------------
# Injection recovery: true params maximise the likelihood
# ---------------------------------------------------------------------------

class TestInjectionRecovery:
    """With zero-noise injection (d = μ_true), the quadratic residual is 0 at
    truth and strictly positive for any perturbation, so the likelihood is
    guaranteed to peak at the injected parameters."""

    def test_tc_perturbation_lowers_likelihood(self, injection):
        gw_ll, true_params, _ = injection
        ll_true = gw_ll(true_params)

        # Shift tc by a quarter period of the 50 Hz signal (= 0.005 s).
        # This produces a 90° phase mismatch — maximal decorrelation.
        perturbed = {**true_params, "tc": TRUE_TC + 0.005}
        ll_perturbed = gw_ll(perturbed)
        assert ll_true > ll_perturbed, (
            f"Expected ll(truth)={ll_true:.3f} > ll(tc+Δt)={ll_perturbed:.3f}"
        )

    def test_ra_perturbation_lowers_likelihood(self, injection):
        gw_ll, true_params, _ = injection
        ll_true = gw_ll(true_params)

        perturbed = {**true_params, "ra": TRUE_RA + 1.0}
        ll_perturbed = gw_ll(perturbed)
        assert ll_true > ll_perturbed, (
            f"Expected ll(truth)={ll_true:.3f} > ll(ra+Δ)={ll_perturbed:.3f}"
        )

    def test_dec_perturbation_lowers_likelihood(self, injection):
        gw_ll, true_params, _ = injection
        ll_true = gw_ll(true_params)

        perturbed = {**true_params, "dec": TRUE_DEC + 0.5}
        ll_perturbed = gw_ll(perturbed)
        assert ll_true > ll_perturbed

    def test_psi_perturbation_lowers_likelihood(self, injection):
        gw_ll, true_params, _ = injection
        ll_true = gw_ll(true_params)

        # Shift psi by π/4 — swaps F+ and F×.
        perturbed = {**true_params, "psi": TRUE_PSI + np.pi / 4}
        ll_perturbed = gw_ll(perturbed)
        assert ll_true > ll_perturbed

    def test_likelihood_profile_monotone_near_truth(self, injection):
        """Log-likelihood decreases monotonically as |Δtc| grows."""
        gw_ll, true_params, _ = injection
        ll_true = gw_ll(true_params)

        shifts = [0.001, 0.002, 0.005, 0.01]
        ll_prev = ll_true
        for dt in shifts:
            p = {**true_params, "tc": TRUE_TC + dt}
            ll = gw_ll(p)
            assert ll < ll_prev, (
                f"Expected monotone decrease: ll(Δtc={dt})={ll:.2f} "
                f"should be < ll(Δtc={dt - shifts[0]})={ll_prev:.2f}"
            )
            ll_prev = ll
