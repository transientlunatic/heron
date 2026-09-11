"""Tests for heron.models.gp.demod — DemodGPSurrogate.

Reuses the analytic "toy chirp" base/oracle pair from the delta tests (a
reference and an oracle differing by small, smooth, q-dependent log-amplitude
and phase perturbations) so the whole demod pipeline — heterodyned residual
targets, training, exact reference evaluation and exact-linear covariance
reconstruction at predict time — is exercised without lalsuite.

The toy strain is O(1), so ``output_scale=1.0`` here (unlike the 1e27 the real
~1e-21-strain training uses) and ``phase_correction=0.0`` (the toy reference
and oracle share a phase convention).
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from heron.types import WaveformDict
from heron.models.gp.demod import DemodGPSurrogate

from test_gp_delta import ToyChirpModel, _make_models, _make_training_data


def _make_surrogate(iterations=25):
    reference, oracle = _make_models()
    train_x, y_plus, y_cross = _make_training_data(oracle)
    surrogate = DemodGPSurrogate(
        train_x=train_x,
        train_y_plus=y_plus,
        train_y_cross=y_cross,
        base_approximant=reference,
        oracle_approximant=None,
        phase_correction=0.0,
        warping="chirp",
        nu=2.5,
        output_scale=1.0,
        device="cpu",
        total_mass=60.0,
        distance=100.0,
        training_iterations=iterations,
    )
    return reference, oracle, surrogate


class TestDemodGPSurrogate:

    @pytest.fixture(scope="class")
    def models_and_surrogate(self):
        return _make_surrogate()

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
            assert (eigvals >= -1e-6 * eigvals.max()).all(), (
                f"{pol}: negative eigenvalues {eigvals.min()}"
            )

    def test_reconstruction_is_exact_inverse_of_demodulation(self, models_and_surrogate):
        """The demod-specific correctness claim: h = h_ref + Re*cos + Im*sin
        is the exact inverse of z = (h - h_ref) * e^{+i Phi_ref}. Demodulating
        the *predicted* residual back must return the inner Re/Im GP means
        exactly, and the plus/cross covariance must equal the exact linear
        congruence of the Re/Im covariances (no delta-method approximation)."""
        _, _, surrogate = models_and_surrogate
        times = np.linspace(-0.3, 0.02, 70)
        params = {"mass_ratio": 0.6, "times": times}

        wf = surrogate.predict(params)
        inner = surrogate._gp.predict(params)
        re_z = inner["plus"].data
        im_z = inner["cross"].data

        q_arr = np.full(len(times), 0.6)
        hXp, hXc, cosP, sinP = surrogate._reference(q_arr, times)

        # Residual (predicted strain minus exact reference), demodulated back.
        rp = wf["plus"].data - hXp
        rc = wf["cross"].data - hXc
        re_back = rp * cosP + rc * sinP
        im_back = rp * sinP - rc * cosP
        np.testing.assert_allclose(re_back, re_z, atol=1e-10, rtol=1e-6)
        np.testing.assert_allclose(im_back, im_z, atol=1e-10, rtol=1e-6)

        # Covariance congruence.
        cov_re = inner["plus"].covariance
        cov_im = inner["cross"].covariance
        cc = np.outer(cosP, cosP)
        ss = np.outer(sinP, sinP)
        np.testing.assert_allclose(
            wf["plus"].covariance, cc * cov_re + ss * cov_im, atol=1e-12, rtol=1e-6
        )
        np.testing.assert_allclose(
            wf["cross"].covariance, ss * cov_re + cc * cov_im, atol=1e-12, rtol=1e-6
        )

    def test_diagonal_mode_matches_full_diagonal(self, models_and_surrogate):
        """covariance='diagonal' returns exactly the diagonal of the full
        covariance (and no N×N matrix), and the mean is unchanged."""
        _, _, surrogate = models_and_surrogate
        params = {"mass_ratio": 0.6, "times": np.linspace(-0.3, 0.02, 90)}
        full = surrogate.predict(params, covariance="full")
        diag = surrogate.predict(params, covariance="diagonal")
        for pol in ("plus", "cross"):
            assert diag[pol].covariance is None
            assert diag[pol].variance is not None
            np.testing.assert_array_equal(diag[pol].data, full[pol].data)
            np.testing.assert_allclose(
                diag[pol].variance, np.diag(full[pol].covariance), rtol=1e-9, atol=1e-30
            )

    def test_none_mode_is_mean_only(self, models_and_surrogate):
        """covariance='none' returns the same mean with no covariance work."""
        _, _, surrogate = models_and_surrogate
        params = {"mass_ratio": 0.6, "times": np.linspace(-0.3, 0.02, 90)}
        full = surrogate.predict(params, covariance="full")
        none = surrogate.predict(params, covariance="none")
        for pol in ("plus", "cross"):
            assert none[pol].covariance is None and none[pol].variance is None
            np.testing.assert_array_equal(none[pol].data, full[pol].data)

    def test_covariance_diagonal_matches_predict(self, models_and_surrogate):
        """covariance_diagonal() equals the diagonal predict variance at the
        reference distance."""
        _, _, surrogate = models_and_surrogate
        params = {"mass_ratio": 0.6, "times": np.linspace(-0.3, 0.02, 90)}
        cd = surrogate.covariance_diagonal(params)
        diag = surrogate.predict(params, covariance="diagonal")
        for pol in ("plus", "cross"):
            np.testing.assert_allclose(cd[pol], diag[pol].variance, rtol=1e-9, atol=1e-30)

    def test_envelope_covariance_diagonal_empty_offsets(self, models_and_surrogate):
        """With no offsets the envelope equals the plain diagonal variance."""
        _, _, surrogate = models_and_surrogate
        params = {"mass_ratio": 0.6, "times": np.linspace(-0.3, 0.02, 60)}
        env = surrogate.envelope_covariance_diagonal(params, [], "mass_ratio")
        cd = surrogate.covariance_diagonal(params)
        for pol in ("plus", "cross"):
            np.testing.assert_allclose(env[pol], cd[pol], rtol=1e-9, atol=1e-30)

    def test_envelope_covariance_diagonal_is_upper_bound(self, models_and_surrogate):
        """The enveloped variance dominates the base-point variance pointwise
        (it is an elementwise max over the offsets)."""
        _, _, surrogate = models_and_surrogate
        params = {"mass_ratio": 0.6, "times": np.linspace(-0.3, 0.02, 60)}
        base = surrogate.covariance_diagonal(params)
        env = surrogate.envelope_covariance_diagonal(
            params, [-0.05, 0.05], "mass_ratio"
        )
        for pol in ("plus", "cross"):
            assert (env[pol] >= base[pol] - 1e-30).all()

    def test_pickle_roundtrip(self, models_and_surrogate):
        """The surrogate must pickle (for nessai/bilby n_pool workers) even
        with a non-registry reference approximant and after predict() has
        materialised its caches."""
        import pickle

        _, _, surrogate = models_and_surrogate
        params = {"mass_ratio": 0.6, "times": np.linspace(-0.3, 0.02, 70)}
        wf = surrogate.predict(params)  # materialise caches
        restored = pickle.loads(pickle.dumps(surrogate))
        wf2 = restored.predict(params)
        for pol in ("plus", "cross"):
            np.testing.assert_array_equal(wf[pol].data, wf2[pol].data)
            np.testing.assert_array_equal(wf[pol].covariance, wf2[pol].covariance)

    def test_beats_reference_at_held_out_mass_ratio(self, models_and_surrogate):
        """At a q between training nodes, the demod surrogate must reproduce
        the oracle substantially better than the raw reference approximant."""
        reference, oracle, surrogate = models_and_surrogate
        q = 0.6
        times = np.linspace(-0.35, 0.02, 120)

        wf = surrogate.predict({"mass_ratio": q, "times": times})
        oracle_plus, oracle_cross = oracle.strain(q, times)
        ref_plus, ref_cross = reference.strain(q, times)

        for pred, oracle_h, ref_h in (
            (wf["plus"].data, oracle_plus, ref_plus),
            (wf["cross"].data, oracle_cross, ref_cross),
        ):
            err_pred = np.max(np.abs(pred - oracle_h))
            err_ref = np.max(np.abs(ref_h - oracle_h))
            assert err_pred < 0.5 * err_ref, (
                f"demod surrogate ({err_pred:.4g}) should beat "
                f"raw reference ({err_ref:.4g})"
            )

    def test_identical_models_predict_reference(self):
        """Oracle == reference: the heterodyned residual vanishes, so the
        prediction degrades gracefully to the exact reference waveform."""
        reference = ToyChirpModel()
        train_x, y_plus, y_cross = _make_training_data(reference, mass_ratios=(0.4, 0.8))
        surrogate = DemodGPSurrogate(
            train_x=train_x,
            train_y_plus=y_plus,
            train_y_cross=y_cross,
            base_approximant=reference,
            oracle_approximant=None,
            phase_correction=0.0,
            output_scale=1.0,
            training_iterations=10,
        )
        times = np.linspace(-0.3, 0.01, 60)
        wf = surrogate.predict({"mass_ratio": 0.6, "times": times})
        ref_plus, ref_cross = reference.strain(0.6, times)
        peak = np.max(np.abs(ref_plus))
        np.testing.assert_allclose(wf["plus"].data, ref_plus, atol=5e-3 * peak)
        np.testing.assert_allclose(wf["cross"].data, ref_cross, atol=5e-3 * peak)

    def test_save_load_roundtrip(self, models_and_surrogate):
        reference, _, surrogate = models_and_surrogate
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            surrogate.save(path)

            # The toy reference is not in heron.models.lalsimulation, so it
            # must be supplied explicitly on load.
            loaded = DemodGPSurrogate.load(
                path, device="cpu", base_approximant=reference,
            )

            params = {
                "mass_ratio": 0.6,
                "time": {"lower": -0.2, "upper": 0.01, "number": 50},
            }
            wf_orig = surrogate.predict(params)
            wf_loaded = loaded.predict(params)

            np.testing.assert_allclose(
                wf_orig["plus"].data, wf_loaded["plus"].data, atol=1e-6
            )
            np.testing.assert_allclose(
                wf_orig["cross"].data, wf_loaded["cross"].data, atol=1e-6
            )
            np.testing.assert_allclose(
                wf_orig["plus"].covariance, wf_loaded["plus"].covariance, atol=1e-8
            )

    def test_distance_scaling(self, models_and_surrogate):
        """Doubling luminosity_distance halves the strain and quarters the
        covariance (both reference and residual scale as 1/distance)."""
        _, _, surrogate = models_and_surrogate
        times = np.linspace(-0.2, 0.01, 40)
        wf_100 = surrogate.predict({"mass_ratio": 0.6, "times": times,
                                    "luminosity_distance": 100.0})
        wf_200 = surrogate.predict({"mass_ratio": 0.6, "times": times,
                                    "luminosity_distance": 200.0})
        np.testing.assert_allclose(wf_200["plus"].data, wf_100["plus"].data / 2.0,
                                   atol=1e-9, rtol=1e-6)
        np.testing.assert_allclose(
            wf_200["plus"].covariance, wf_100["plus"].covariance / 4.0,
            atol=1e-12, rtol=1e-6,
        )

    def test_covariance_inflation_scales_covariance_not_mean(self):
        """covariance_inflation multiplies the returned covariance by the given
        scalar and leaves the mean untouched; it survives save/load."""
        reference, oracle = _make_models()
        train_x, y_plus, y_cross = _make_training_data(oracle)
        common = dict(
            train_x=train_x, train_y_plus=y_plus, train_y_cross=y_cross,
            base_approximant=reference, oracle_approximant=None,
            phase_correction=0.0, output_scale=1.0, training_iterations=15,
        )
        base = DemodGPSurrogate(**common, covariance_inflation=1.0)
        infl = DemodGPSurrogate(**common, covariance_inflation=9.0)

        params = {"mass_ratio": 0.6, "times": np.linspace(-0.25, 0.01, 40)}
        wf0 = base.predict(params)
        wf9 = infl.predict(params)
        # Mean identical, covariance scaled by exactly 9.
        np.testing.assert_allclose(wf9["plus"].data, wf0["plus"].data, atol=1e-12)
        np.testing.assert_allclose(
            wf9["plus"].covariance, 9.0 * wf0["plus"].covariance, rtol=1e-9, atol=1e-30
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "infl.pt"
            infl.save(path)
            loaded = DemodGPSurrogate.load(path, device="cpu", base_approximant=reference)
            assert loaded.covariance_inflation == pytest.approx(9.0)

    def test_parameter_names(self, models_and_surrogate):
        _, _, surrogate = models_and_surrogate
        assert "mass_ratio" in surrogate.parameter_names

    def test_parameter_bounds(self, models_and_surrogate):
        _, _, surrogate = models_and_surrogate
        lo, hi = surrogate.parameter_bounds["mass_ratio"]
        assert lo == pytest.approx(0.3)
        assert hi == pytest.approx(0.9)
