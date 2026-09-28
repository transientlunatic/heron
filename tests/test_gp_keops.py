"""Tests for heron.models.gp.keops — KeOps-backed exact-GP prediction.

The KeOps path must reproduce the dense gpytorch prediction exactly (same
hyperparameters, warping and float64 arithmetic), for every kernel
configuration `_ExactGPModel` can build. The kernel re-expression itself is
checked against gpytorch through the dense backend (no pykeops needed); the
end-to-end `predict(..., use_keops=True)` comparisons are skipped when
pykeops is unavailable.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch
import gpytorch
import pytest

from heron.models.gp import keops as gp_keops
from heron.models.gp.exact import ExactGPSurrogate
from heron.models.gp.demod import DemodGPSurrogate

from test_gp_exact import _make_synthetic_training_data
from test_gp_delta import _make_models, _make_training_data

requires_keops = pytest.mark.skipif(
    not gp_keops.KEOPS_AVAILABLE, reason="pykeops is not available"
)

# Every kernel structure _ExactGPModel can build: plain Matern x Matern,
# nonstationary (merger) time kernel, warped mass-ratio kernel, and the
# additive q-floor kernel (nested ScaleKernel inside an AdditiveKernel).
KERNEL_CONFIGS = {
    "matern": {},
    "merger": {"merger_kernel": True, "ls_min_time_merger": 0.001},
    "q_warp": {"q_warping": "eta"},
    "q_floor": {"q_floor_kernel": True},
}


def _make_surrogate(**kwargs):
    train_x, y_plus, y_cross = _make_synthetic_training_data(
        n_per_q=30, mass_ratios=(0.3, 0.6, 1.0)
    )
    return ExactGPSurrogate(
        train_x=train_x,
        train_y_plus=y_plus,
        train_y_cross=y_cross,
        warping="chirp",
        nu=2.5,
        output_scale=1.0,
        device="cpu",
        training_iterations=3,
        **kwargs,
    )


@pytest.fixture(scope="module", params=list(KERNEL_CONFIGS))
def surrogate(request):
    return _make_surrogate(**KERNEL_CONFIGS[request.param])


def _params(n, q=0.45):
    return {"mass_ratio": q, "times": np.linspace(-0.45, 0.02, n)}


def _max_rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# -- kernel re-expression (dense backend, no pykeops needed) ---------------


def test_dense_formula_matches_gpytorch_kernel(surrogate):
    model = surrogate._get_predict_models()["plus"]
    train_x = model.train_inputs[0]
    x, _, _ = surrogate._build_eval_points(_params(57))
    with torch.no_grad():
        expected = model.covar_module(x, train_x).to_dense()
        got = gp_keops.dense_cross_covariance(model.covar_module, x, train_x)
    assert got.dtype == torch.float64
    assert got.shape == expected.shape
    assert _max_rel(got, expected) < 1e-12


def test_unsupported_kernel_raises():
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    x = torch.rand(5, 1, dtype=torch.float64)
    with pytest.raises(gp_keops.UnsupportedKernelError):
        gp_keops.dense_cross_covariance(kernel, x, x)


def test_missing_pykeops_raises_import_error(monkeypatch):
    s = _make_surrogate()
    monkeypatch.setattr(gp_keops, "KEOPS_AVAILABLE", False)
    with pytest.raises(ImportError, match="pykeops"):
        s.predict(_params(10), covariance="none", use_keops=True)
    # The default path is unaffected.
    s.predict(_params(10), covariance="none")


def test_flag_is_runtime_only_and_not_saved():
    s = _make_surrogate(use_keops=True)
    assert s.use_keops is True
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.pt"
        s.save(path)
        ckpt = torch.load(path, weights_only=False)
        assert not any("keops" in k for k in ckpt)
        assert ExactGPSurrogate.load(path).use_keops is False
        assert ExactGPSurrogate.load(path, use_keops=True).use_keops is True


def test_pickle_preserves_flag():
    s = _make_surrogate(use_keops=True)
    assert pickle.loads(pickle.dumps(s)).use_keops is True
    s.use_keops = False
    assert pickle.loads(pickle.dumps(s)).use_keops is False


# -- end-to-end prediction (needs pykeops) ----------------------------------


@requires_keops
@pytest.mark.parametrize("n", [64, 700])  # below/above gpytorch's eager-kernel size
def test_predict_mean_and_variance_match_dense(surrogate, n):
    p = _params(n)
    for cov in ("none", "diagonal"):
        dense = surrogate.predict(p, covariance=cov, use_keops=False)
        fast = surrogate.predict(p, covariance=cov, use_keops=True)
        for pol in ("plus", "cross"):
            assert _max_rel(fast[pol].data, dense[pol].data) < 1e-9
            if cov == "diagonal":
                assert fast[pol].covariance is None
                assert _max_rel(fast[pol].variance, dense[pol].variance) < 1e-9
            else:
                assert fast[pol].variance is None


@requires_keops
def test_attribute_default_and_full_covariance_unchanged(surrogate):
    p = _params(40)
    dense = surrogate.predict(p, covariance="full")
    surrogate.use_keops = True
    try:
        fast_diag = surrogate.predict(p, covariance="diagonal")
        fast_full = surrogate.predict(p, covariance="full")  # dense path by design
    finally:
        surrogate.use_keops = False
    for pol in ("plus", "cross"):
        np.testing.assert_array_equal(fast_full[pol].covariance, dense[pol].covariance)
        assert _max_rel(fast_diag[pol].variance, np.diag(dense[pol].covariance)) < 1e-9


@requires_keops
def test_covariance_diag_helpers_match(surrogate):
    p = _params(50)
    dense = surrogate.envelope_covariance_diagonal(p, offsets=[-0.01, 0.01])
    surrogate.use_keops = True
    try:
        fast = surrogate.envelope_covariance_diagonal(p, offsets=[-0.01, 0.01])
    finally:
        surrogate.use_keops = False
    for pol in ("plus", "cross"):
        assert _max_rel(fast[pol], dense[pol]) < 1e-9


@requires_keops
def test_variance_chunking_is_invariant():
    s = _make_surrogate()
    model = s._get_predict_models()["plus"]
    x, _, _ = s._build_eval_points(_params(100))
    _, v_one = gp_keops.latent_posterior(
        model, x, cholesky_size=s.cholesky_size, mean=False, variance=True,
    )
    _, v_chunked = gp_keops.latent_posterior(
        model, x, cholesky_size=s.cholesky_size, mean=False, variance=True,
        chunk_elements=7 * model.train_inputs[0].shape[0],  # 7-row chunks
    )
    assert _max_rel(v_chunked, v_one) < 1e-12


@requires_keops
def test_demod_predict_matches_dense():
    reference, oracle = _make_models()
    train_x, y_plus, y_cross = _make_training_data(oracle)
    s = DemodGPSurrogate(
        train_x=train_x, train_y_plus=y_plus, train_y_cross=y_cross,
        base_approximant=reference, oracle_approximant=None,
        phase_correction=0.0, output_scale=1.0, device="cpu",
        training_iterations=5,
    )
    assert s.use_keops is False
    p = {"mass_ratio": 0.6, "times": np.linspace(-0.35, 0.015, 300)}
    for cov in ("none", "diagonal"):
        dense = s.predict(p, covariance=cov)
        fast = s.predict(p, covariance=cov, use_keops=True)
        for pol in ("plus", "cross"):
            assert _max_rel(fast[pol].data, dense[pol].data) < 1e-9
            if cov == "diagonal":
                assert _max_rel(fast[pol].variance, dense[pol].variance) < 1e-9

    dense_diag = s.covariance_diagonal(p)
    s.use_keops = True
    assert s._gp.use_keops is True
    fast_diag = s.covariance_diagonal(p)
    for pol in ("plus", "cross"):
        assert _max_rel(fast_diag[pol], dense_diag[pol]) < 1e-9
    assert pickle.loads(pickle.dumps(s)).use_keops is True
