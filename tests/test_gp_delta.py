"""Tests for heron.models.gp.delta — DeltaGPSurrogate.

Uses a pair of analytic "toy chirp" models (a base and an oracle differing
by small, smooth, q-dependent log-amplitude and phase perturbations) so the
whole delta pipeline — residual computation, alignment, support/amplitude
cuts, training, exact base evaluation at predict time — is exercised
without lalsuite.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from heron.types import Waveform, WaveformDict
from heron.models.gp.delta import (
    DeltaGPSurrogate,
    _ApproximantEvaluator,
    compute_delta_targets,
)


def _zero_delta(q, t):
    return np.zeros_like(t)


class ToyChirpModel:
    """Analytic chirplet approximant with injectable amplitude/phase deltas.

    Base waveform: A(q,t) = (0.5+q) * exp(-(t/0.15)^2),
    Phi(q,t) = 2*pi*f0*t + 30*q*t^2. The convention matches the delta
    module's decomposition: h_plus = A*cos(Phi), h_cross = A*sin(Phi).
    """

    def __init__(self, dlogA=None, dphi=None, phase_offset=0.0, f0=30.0):
        # Module-level defaults (not lambdas) so the stub stays picklable —
        # DemodGPSurrogate/DeltaGPSurrogate carry a non-registry reference
        # instance through pickle (see their __getstate__).
        self.dlogA = dlogA if dlogA is not None else _zero_delta
        self.dphi = dphi if dphi is not None else _zero_delta
        self.phase_offset = phase_offset
        self.f0 = f0

    def amplitude_phase(self, q, times):
        t = np.asarray(times, dtype=np.float64)
        log_amp = np.log((0.5 + q) * np.exp(-((t / 0.15) ** 2))) + self.dlogA(q, t)
        phase = 2 * np.pi * self.f0 * t + 30.0 * q * t**2 + self.phase_offset \
            + self.dphi(q, t)
        return np.exp(log_amp), phase

    def strain(self, q, times):
        amp, phase = self.amplitude_phase(q, times)
        return amp * np.cos(phase), amp * np.sin(phase)

    def time_domain(self, parameters, times=None):
        q = float(parameters["mass_ratio"])
        if times is None:
            times = np.linspace(-0.5, 0.05, 600)
        times = np.asarray(times, dtype=np.float64)
        h_plus, h_cross = self.strain(q, times)
        return WaveformDict(
            parameters={"mass_ratio": q},
            plus=Waveform(data=h_plus, times=times),
            cross=Waveform(data=h_cross, times=times),
        )


# True deltas: smooth, q-dependent, and zero at the earliest training time
# (t=-0.4) so "anchor" alignment is a no-op on the genuine signal and
# strain-level comparisons against the oracle remain meaningful.
def _true_dlogA(q, t):
    return 0.15 * q * (t + 0.4)


def _true_dphi(q, t):
    return 0.8 * q * (t + 0.4)


def _make_models(f0=30.0):
    base = ToyChirpModel(f0=f0)
    oracle = ToyChirpModel(dlogA=_true_dlogA, dphi=_true_dphi, f0=f0)
    return base, oracle


def _make_training_data(oracle, mass_ratios=(0.3, 0.5, 0.7, 0.9), n_per_q=50):
    all_x, all_plus, all_cross = [], [], []
    for q in mass_ratios:
        times = np.linspace(-0.4, 0.02, n_per_q)
        h_plus, h_cross = oracle.strain(q, times)
        all_x.append(np.column_stack([np.full(n_per_q, q), times]))
        all_plus.append(h_plus)
        all_cross.append(h_cross)
    return (
        torch.tensor(np.vstack(all_x), dtype=torch.float32),
        torch.tensor(np.concatenate(all_plus), dtype=torch.float32),
        torch.tensor(np.concatenate(all_cross), dtype=torch.float32),
    )


def _evaluator(model):
    return _ApproximantEvaluator(model, total_mass=60.0, distance=100.0, f_low=20.0)


class TestComputeDeltaTargets:

    def test_recovers_known_deltas_sparse_fallback(self):
        """No oracle evaluator: the oracle side is decomposed from the
        (well-sampled) training strain itself."""
        base, oracle = _make_models()
        train_x, y_plus, y_cross = _make_training_data(oracle)

        x_sorted, dlogA, dphi = compute_delta_targets(
            train_x, y_plus, y_cross, _evaluator(base), phase_alignment="anchor"
        )

        q_col = x_sorted[:, 0].numpy()
        t_col = x_sorted[:, -1].numpy()
        # float32 training data limits agreement to ~1e-5 relative.
        np.testing.assert_allclose(dlogA.numpy(), _true_dlogA(q_col, t_col), atol=1e-4)
        np.testing.assert_allclose(dphi.numpy(), _true_dphi(q_col, t_col), atol=1e-4)

    def test_dense_oracle_path_matches_fallback(self):
        """When the training grid is dense enough that the fallback unwrap
        is safe, the two paths must produce the same targets and rows."""
        base, oracle = _make_models()
        train_x, y_plus, y_cross = _make_training_data(oracle)

        x_a, dlogA_a, dphi_a = compute_delta_targets(
            train_x, y_plus, y_cross, _evaluator(base)
        )
        x_b, dlogA_b, dphi_b = compute_delta_targets(
            train_x, y_plus, y_cross, _evaluator(base),
            oracle_evaluator=_evaluator(oracle),
        )

        np.testing.assert_allclose(x_a.numpy(), x_b.numpy(), atol=1e-6)
        np.testing.assert_allclose(dlogA_a.numpy(), dlogA_b.numpy(), atol=1e-4)
        np.testing.assert_allclose(dphi_a.numpy(), dphi_b.numpy(), atol=1e-4)

    def test_dense_oracle_path_defeats_unwrap_aliasing(self):
        """Training samples too sparse for phase unwrapping (steps >> pi):
        the sparse fallback aliases and gets the phase delta badly wrong,
        while the dense-native-grid oracle path recovers the truth. This is
        the failure observed on real IMRPhenomXAS/IMRPhenomD data."""
        base, oracle = _make_models(f0=80.0)
        # dt = 0.42/15 = 0.028 s -> phase steps ~ 2*pi*80*0.028 = 14 rad >> pi
        train_x, y_plus, y_cross = _make_training_data(oracle, n_per_q=16)

        x_dense, _, dphi_dense = compute_delta_targets(
            train_x, y_plus, y_cross, _evaluator(base),
            oracle_evaluator=_evaluator(oracle),
        )
        x_sparse, _, dphi_sparse = compute_delta_targets(
            train_x, y_plus, y_cross, _evaluator(base)
        )

        true_dense = _true_dphi(x_dense[:, 0].numpy(), x_dense[:, -1].numpy())
        np.testing.assert_allclose(dphi_dense.numpy(), true_dense, atol=1e-3)

        true_sparse = _true_dphi(x_sparse[:, 0].numpy(), x_sparse[:, -1].numpy())
        err_sparse = np.abs(dphi_sparse.numpy() - true_sparse).max()
        assert err_sparse > 1.0, (
            f"expected the sparse unwrap to alias here (err {err_sparse:.3f})"
        )

    def test_rows_outside_base_support_are_dropped(self):
        """Training times extending past the base model's native support
        (t > 0.05 for the toy) must be cut, not clamped into garbage."""
        base, oracle = _make_models()
        n = 60
        times = np.linspace(-0.4, 0.2, n)  # last ~37% beyond base support
        h_plus, h_cross = oracle.strain(0.5, times)
        train_x = torch.tensor(
            np.column_stack([np.full(n, 0.5), times]), dtype=torch.float32
        )
        x_out, _, _ = compute_delta_targets(
            train_x,
            torch.tensor(h_plus, dtype=torch.float32),
            torch.tensor(h_cross, dtype=torch.float32),
            _evaluator(base),
            oracle_evaluator=_evaluator(oracle),
        )
        assert x_out.shape[0] < n
        assert float(x_out[:, -1].max()) <= 0.05 + 1e-9

    def test_amplitude_floor_drops_taper_region(self):
        """Rows where the amplitude has decayed below amp_floor_rel of peak
        (here the far tail of the Gaussian envelope) are dropped."""
        base, oracle = _make_models()
        n = 80
        times = np.linspace(-0.49, 0.02, n)  # A(-0.49)/A(0) ~ 2e-5 < 1e-4
        h_plus, h_cross = oracle.strain(0.5, times)
        train_x = torch.tensor(
            np.column_stack([np.full(n, 0.5), times]), dtype=torch.float32
        )
        x_out, _, _ = compute_delta_targets(
            train_x,
            torch.tensor(h_plus, dtype=torch.float32),
            torch.tensor(h_cross, dtype=torch.float32),
            _evaluator(base),
            oracle_evaluator=_evaluator(oracle),
            amp_floor_rel=1e-4,
        )
        assert x_out.shape[0] < n
        assert float(x_out[:, -1].min()) > -0.49

    def test_branch_alignment_keeps_sub_2pi_offset(self):
        """An oracle with a genuine constant reference-phase offset: branch
        alignment must remove only whole 2*pi multiples, preserving the
        physical sub-2*pi offset in the residual."""
        offset = 0.3
        base = ToyChirpModel()
        oracle = ToyChirpModel(phase_offset=offset)
        train_x, y_plus, y_cross = _make_training_data(oracle)

        _, _, dphi = compute_delta_targets(
            train_x, y_plus, y_cross, _evaluator(base),
            phase_alignment="branch", oracle_evaluator=_evaluator(oracle),
        )

        np.testing.assert_allclose(dphi.numpy(), np.full(len(dphi), offset), atol=1e-4)

    def test_anchor_alignment_zeroes_earliest_sample(self):
        base = ToyChirpModel()
        oracle = ToyChirpModel(phase_offset=0.3, dphi=_true_dphi)
        train_x, y_plus, y_cross = _make_training_data(oracle)

        x_sorted, _, dphi = compute_delta_targets(
            train_x, y_plus, y_cross, _evaluator(base),
            phase_alignment="anchor", oracle_evaluator=_evaluator(oracle),
        )

        q_col = x_sorted[:, 0].numpy()
        dphi_np = dphi.numpy()
        for q in np.unique(q_col):
            first = np.where(q_col == q)[0][0]
            assert abs(dphi_np[first]) < 1e-6

    def test_invalid_alignment_raises(self):
        base, oracle = _make_models()
        train_x, y_plus, y_cross = _make_training_data(oracle)
        with pytest.raises(ValueError, match="phase_alignment"):
            compute_delta_targets(
                train_x, y_plus, y_cross, _evaluator(base), phase_alignment="nope"
            )


class TestDeltaGPSurrogate:

    @pytest.fixture(scope="class")
    def models_and_surrogate(self):
        base, oracle = _make_models()
        train_x, y_plus, y_cross = _make_training_data(oracle)
        surrogate = DeltaGPSurrogate(
            train_x=train_x,
            train_y_plus=y_plus,
            train_y_cross=y_cross,
            base_approximant=base,
            oracle_approximant=oracle,
            warping="chirp",
            nu=2.5,
            device="cpu",
            total_mass=60.0,
            distance=100.0,
            training_iterations=20,
        )
        return base, oracle, surrogate

    def test_predict_returns_waveform_dict(self, models_and_surrogate):
        _, _, surrogate = models_and_surrogate
        wf = surrogate.predict({
            "mass_ratio": 0.6,
            "time": {"lower": -0.3, "upper": 0.02, "number": 100},
        })
        assert isinstance(wf, WaveformDict)
        assert "plus" in wf and "cross" in wf

    def test_predict_has_correct_shapes(self, models_and_surrogate):
        _, _, surrogate = models_and_surrogate
        n = 80
        wf = surrogate.predict({
            "mass_ratio": 0.6,
            "time": {"lower": -0.3, "upper": 0.02, "number": n},
        })
        assert wf["plus"].data.shape == (n,)
        assert wf["plus"].times.shape == (n,)
        assert wf["plus"].covariance.shape == (n, n)
        assert wf["cross"].covariance.shape == (n, n)

    def test_covariance_is_positive_semidefinite(self, models_and_surrogate):
        _, _, surrogate = models_and_surrogate
        wf = surrogate.predict({
            "mass_ratio": 0.6,
            "time": {"lower": -0.3, "upper": 0.02, "number": 50},
        })
        for pol in ("plus", "cross"):
            eigvals = np.linalg.eigvalsh(wf[pol].covariance)
            assert (eigvals >= -1e-6).all(), f"{pol}: negative eigenvalues {eigvals.min()}"

    def test_beats_base_at_held_out_mass_ratio(self, models_and_surrogate):
        """The core value test: at a q between training nodes, the delta
        surrogate must reproduce the oracle substantially better than the
        raw base approximant does."""
        base, oracle, surrogate = models_and_surrogate
        q = 0.6
        times = np.linspace(-0.35, 0.02, 120)

        wf = surrogate.predict({"mass_ratio": q, "times": times})
        oracle_plus, oracle_cross = oracle.strain(q, times)
        base_plus, base_cross = base.strain(q, times)

        for pred, oracle_h, base_h in (
            (wf["plus"].data, oracle_plus, base_plus),
            (wf["cross"].data, oracle_cross, base_cross),
        ):
            err_pred = np.max(np.abs(pred - oracle_h))
            err_base = np.max(np.abs(base_h - oracle_h))
            assert err_pred < 0.5 * err_base, (
                f"delta surrogate ({err_pred:.4g}) should beat "
                f"raw base ({err_base:.4g})"
            )

    def test_identical_models_predict_base(self):
        """Oracle == base: deltas vanish, prediction degrades gracefully to
        the exact base waveform."""
        base = ToyChirpModel()
        train_x, y_plus, y_cross = _make_training_data(base, mass_ratios=(0.4, 0.8))
        surrogate = DeltaGPSurrogate(
            train_x=train_x,
            train_y_plus=y_plus,
            train_y_cross=y_cross,
            base_approximant=base,
            oracle_approximant=base,
            training_iterations=10,
        )
        times = np.linspace(-0.3, 0.01, 60)
        wf = surrogate.predict({"mass_ratio": 0.6, "times": times})
        base_plus, _ = base.strain(0.6, times)
        peak = np.max(np.abs(base_plus))
        np.testing.assert_allclose(wf["plus"].data, base_plus, atol=2e-3 * peak)

    def test_save_load_roundtrip(self, models_and_surrogate):
        base, oracle, surrogate = models_and_surrogate
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            surrogate.save(path)

            # The toy models are not in heron.models.lalsimulation, so they
            # must be supplied explicitly on load.
            loaded = DeltaGPSurrogate.load(
                path, device="cpu",
                base_approximant=base, oracle_approximant=oracle,
            )

            params = {
                "mass_ratio": 0.6,
                "time": {"lower": -0.2, "upper": 0.01, "number": 50},
            }
            wf_orig = surrogate.predict(params)
            wf_loaded = loaded.predict(params)

            np.testing.assert_allclose(
                wf_orig["plus"].data, wf_loaded["plus"].data, atol=1e-5
            )
            np.testing.assert_allclose(
                wf_orig["cross"].data, wf_loaded["cross"].data, atol=1e-5
            )

    def test_parameter_names(self, models_and_surrogate):
        _, _, surrogate = models_and_surrogate
        assert "mass_ratio" in surrogate.parameter_names

    def test_parameter_bounds(self, models_and_surrogate):
        _, _, surrogate = models_and_surrogate
        lo, hi = surrogate.parameter_bounds["mass_ratio"]
        assert lo == pytest.approx(0.3)
        assert hi == pytest.approx(0.9)

    def test_predict_with_times_array(self, models_and_surrogate):
        _, _, surrogate = models_and_surrogate
        wf = surrogate.predict({
            "mass_ratio": 0.6,
            "times": np.linspace(-0.2, 0.01, 60),
        })
        assert wf["plus"].data.shape == (60,)
