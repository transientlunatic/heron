"""Tests for heron.models.gp.checkpoint — version-robust GP state loading."""

import copy
import logging
import re

import numpy as np
import pytest
import torch

from heron.models.gp.checkpoint import (
    check_hyperparameters,
    hyperparameter_summary,
    is_prior_key,
    load_gp_state,
)
from heron.models.gp.exact import ExactGPSurrogate

from test_gp_exact import _make_synthetic_training_data


def _rename_prior_buffers(state):
    """Simulate the other GPyTorch release's prior-buffer naming."""
    swap = {"_transformed_": "_buffered_", "_buffered_": "_transformed_"}

    def rename(key):
        return re.sub(r"\._(transformed|buffered)_",
                      lambda m: "." + swap[f"_{m.group(1)}_"], key)

    return {rename(k): v for k, v in state.items()}


@pytest.fixture(scope="module")
def model():
    train_x, y_plus, y_cross = _make_synthetic_training_data(
        n_per_q=30, mass_ratios=(0.5, 1.0)
    )
    return ExactGPSurrogate(
        train_x=train_x, train_y_plus=y_plus, train_y_cross=y_cross,
        warping="chirp", output_scale=1.0, device="cpu",
        total_mass=60.0, distance=100.0, training_iterations=10,
    )


PARAMS = {"mass_ratio": 0.7, "times": np.linspace(-0.4, 0.01, 40)}


def _predict(surrogate):
    wf = surrogate.predict(dict(PARAMS), covariance="diagonal")
    return wf["plus"].data, wf["plus"].variance


def _saved(model, tmp_path):
    path = tmp_path / "model.pt"
    model.save(path)
    return path, torch.load(path, weights_only=False)


class TestPriorKeys:

    def test_prior_keys_detected(self):
        assert is_prior_key("covar_module.base_kernel.kernels.0."
                            "lengthscale_prior._transformed_loc")
        assert is_prior_key("a.lengthscale_prior._buffered_scale")
        assert not is_prior_key("covar_module.base_kernel.kernels.0."
                                "raw_lengthscale")
        assert not is_prior_key("likelihood.noise_covar.raw_noise")

    def test_model_has_prior_buffers(self, model):
        # Guard: if the model ever stops registering priors these tests
        # stop exercising the rename path.
        assert any(is_prior_key(k) for k in model.models["plus"].state_dict())


class TestRoundTrip:

    def test_save_records_hyperparameters_and_versions(self, model, tmp_path):
        _, ck = _saved(model, tmp_path)
        assert set(ck["hyperparameters"]) == {"plus", "cross"}
        hp = ck["hyperparameters"]["plus"]
        assert any(k.endswith("lengthscale") for k in hp)
        assert "covar_module.outputscale" in hp
        assert set(ck["environment"]) == {"torch", "gpytorch"}

    def test_roundtrip_is_exact(self, model, tmp_path):
        path, _ = _saved(model, tmp_path)
        loaded = ExactGPSurrogate.load(path)
        for a, b in zip(_predict(model), _predict(loaded)):
            np.testing.assert_array_equal(a, b)

    def test_other_gpytorch_prior_naming_loads(self, model, tmp_path, caplog):
        path, ck = _saved(model, tmp_path)
        ck["model_states"] = {n: _rename_prior_buffers(s)
                              for n, s in ck["model_states"].items()}
        torch.save(ck, path)
        with caplog.at_level(logging.INFO, "heron.models.gp.checkpoint"):
            loaded = ExactGPSurrogate.load(path)
        assert "different GPyTorch naming" in caplog.text
        assert "differs from the one rebuilt" not in caplog.text
        for a, b in zip(_predict(model), _predict(loaded)):
            np.testing.assert_array_equal(a, b)

    def test_checkpoint_without_record_still_loads(self, model, tmp_path):
        path, ck = _saved(model, tmp_path)
        del ck["hyperparameters"], ck["environment"]
        torch.save(ck, path)
        loaded = ExactGPSurrogate.load(path)
        np.testing.assert_array_equal(_predict(model)[0], _predict(loaded)[0])


class TestStrictness:

    def test_missing_trained_parameter_raises(self, model, tmp_path):
        path, ck = _saved(model, tmp_path)
        del ck["model_states"]["plus"]["covar_module.raw_outputscale"]
        torch.save(ck, path)
        with pytest.raises(RuntimeError, match="raw_outputscale"):
            ExactGPSurrogate.load(path)

    def test_unexpected_parameter_raises(self, model, tmp_path):
        path, ck = _saved(model, tmp_path)
        ck["model_states"]["plus"]["covar_module.raw_extra"] = torch.zeros(1)
        torch.save(ck, path)
        with pytest.raises(RuntimeError, match="raw_extra"):
            ExactGPSurrogate.load(path)

    def test_hyperparameter_record_mismatch_raises(self, model, tmp_path):
        path, ck = _saved(model, tmp_path)
        rec = ck["hyperparameters"]["plus"]
        key = next(k for k in rec if k.endswith("lengthscale"))
        rec[key] = [v * 1.5 for v in rec[key]]
        torch.save(ck, path)
        with pytest.raises(RuntimeError, match="does not reproduce"):
            ExactGPSurrogate.load(path)

    def test_changed_prior_value_warns(self, model, tmp_path, caplog):
        gp = model.models["plus"]
        state = {k: v.clone() for k, v in gp.state_dict().items()}
        key = next(k for k in state if is_prior_key(k) and k.endswith("loc"))
        state[key] = state[key] + 1.0
        clone = copy.deepcopy(gp)
        with caplog.at_level(logging.WARNING, "heron.models.gp.checkpoint"):
            load_gp_state(clone, state, name="clone")
        assert "differs from the one rebuilt" in caplog.text


def test_summary_matches_constrained_values(model):
    gp = model.models["plus"]
    summary = hyperparameter_summary(gp)
    np.testing.assert_allclose(
        summary["covar_module.outputscale"],
        gp.covar_module.outputscale.detach().double().flatten().numpy(),
    )
    check_hyperparameters(gp, summary)  # self-consistent: no raise
    check_hyperparameters(gp, None)  # absent record: no-op
