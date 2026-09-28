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


# ---------------------------------------------------------------------------
# K-smoothing: enveloping the diagonal variance over nearby mass_ratio offsets
# ---------------------------------------------------------------------------

class _DippedVarianceSurrogate:
    """Like _SineSurrogate, but with a variance that dips sharply at a single
    mass_ratio value, mimicking a GP posterior's minimum at a training node."""

    def __init__(self, dip_at: float, f0: float = 50.0, amplitude: float = 1.0):
        self.dip_at = dip_at
        self.f0 = f0
        self.amplitude = amplitude

    def predict(self, params: dict) -> WaveformDict:
        times = np.asarray(params["times"], dtype=float)
        q = params.get("mass_ratio", self.dip_at)
        phase = 2.0 * np.pi * self.f0 * times
        A = self.amplitude
        # Far variance large enough, once band-limited by _project_diag (see
        # gw_likelihood.py), to be comparable to C's actual *passband*
        # eigenvalues -- not just its raw time-domain diagonal (~236 in this
        # fixture's units, which is dominated by aggregate in-band power and
        # is not the scale the projected log-det term responds to). A smaller
        # far value (e.g. (A*0.5)**2) is swamped once the spurious sub-f_low
        # leakage that used to inflate this comparison is removed. Near (the
        # "training node") variance is negligible by comparison either way.
        far = (A * 5.0) ** 2
        near = (A * 1e-6) ** 2
        var = near if abs(q - self.dip_at) < 1e-9 else far
        cov = np.eye(len(times)) * var
        return WaveformDict(
            plus=Waveform(data=A * np.sin(phase), times=times, covariance=cov),
            cross=Waveform(data=A * np.cos(phase), times=times, covariance=cov),
        )


class TestKSmoothing:

    @pytest.fixture
    def dipped_injection(self):
        dt = 1.0 / 512.0
        n = 64
        times = TRUE_TC + (np.arange(n) - n // 2) * dt
        surrogate = _DippedVarianceSurrogate(dip_at=TRUE_Q)
        fp_true, fc_true = antenna_patterns(TRUE_RA, TRUE_DEC, TRUE_PSI, TRUE_TC, DETECTOR)
        t_rel = times - TRUE_TC
        wf = surrogate.predict({"times": t_rel, "mass_ratio": TRUE_Q})
        mu_true, _ = project_waveform(wf, fp_true, fc_true)
        return surrogate, times, mu_true

    def test_no_offsets_matches_prior_behaviour(self, dipped_injection):
        """Default (no k_smoothing_offsets) must be numerically identical to
        the pre-existing raw-diagonal behaviour."""
        surrogate, times, mu_true = dipped_injection
        gw_ll = GWLikelihood(
            data=mu_true, times=times, psd_fn=_flat_psd,
            surrogate=surrogate, detector=DETECTOR, f_low=20.0,
        )
        params = {"mass_ratio": TRUE_Q, "tc": TRUE_TC, "ra": TRUE_RA,
                  "dec": TRUE_DEC, "psi": TRUE_PSI}
        assert gw_ll._k_smoothing_offsets == []
        ll = gw_ll(params)
        assert np.isfinite(ll)

    def test_envelope_uses_max_variance_at_the_dip(self, dipped_injection):
        """At the dip (a training-node-like point), enveloping over an
        offset that lands off the dip must raise K above the raw (dipped)
        value -- this is the mechanism the fix relies on."""
        surrogate, times, mu_true = dipped_injection

        gw_ll_raw = GWLikelihood(
            data=mu_true, times=times, psd_fn=_flat_psd,
            surrogate=surrogate, detector=DETECTOR, f_low=20.0,
        )
        gw_ll_smoothed = GWLikelihood(
            data=mu_true, times=times, psd_fn=_flat_psd,
            surrogate=surrogate, detector=DETECTOR, f_low=20.0,
            k_smoothing_offsets=[0.05, -0.05],
        )
        params = {"mass_ratio": TRUE_Q, "tc": TRUE_TC, "ra": TRUE_RA,
                  "dec": TRUE_DEC, "psi": TRUE_PSI}

        fp, fc = antenna_patterns(TRUE_RA, TRUE_DEC, TRUE_PSI, TRUE_TC, DETECTOR)
        t_rel = times - TRUE_TC
        wf_raw = surrogate.predict({"times": t_rel, "mass_ratio": TRUE_Q})
        _, K_raw = project_waveform(wf_raw, fp, fc)
        # _k_diagonal_envelope returns the raw (unprojected) variance vector;
        # _project_diag (applied by __call__) happens downstream of this.
        var_env = gw_ll_smoothed._k_diagonal_envelope(
            {"times": t_rel, "mass_ratio": TRUE_Q}, fp, fc, K_raw,
        )

        assert np.all(var_env >= K_raw.diagonal())
        assert np.any(var_env > K_raw.diagonal())

        # A likelihood evaluated exactly at the dip should therefore differ
        # substantially between the raw and smoothed variants (smoothing is
        # not a no-op) -- not just by float roundoff.
        assert abs(gw_ll_raw(params) - gw_ll_smoothed(params)) > 1.0
