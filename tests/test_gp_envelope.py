"""Tests for the learned time envelope (TimeEnvelope / EnvelopeKernel / EnvelopeNoise)."""

import numpy as np
import pytest
import torch

from heron.models.gp.demod import DemodGPSurrogate
from heron.models.gp.kernels import TimeEnvelope

from test_gp_delta import _make_models, _make_training_data


def test_envelope_is_piecewise_linear_in_log_and_flat_outside():
    t = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    env = TimeEnvelope(t, n_knots=3).double()
    with torch.no_grad():
        env.log_s.copy_(torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64))
    eff = env.effective_log_s().detach().numpy()
    assert abs(eff.mean()) < 1e-12 and np.all(np.diff(eff) > 0)
    got = env(torch.tensor([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0], dtype=torch.float64))
    expect = np.exp([eff[0], eff[0], (eff[0] + eff[1]) / 2, eff[1], (eff[1] + eff[2]) / 2, eff[2], eff[2]])
    np.testing.assert_allclose(got.detach().numpy(), expect, rtol=1e-10)


def test_envelope_gradient_reaches_knots():
    t = torch.linspace(0.0, 1.0, 50)
    env = TimeEnvelope(t, n_knots=4)
    env(torch.tensor([0.1, 0.6, 0.9])).sum().backward()
    assert env.log_s.grad is not None and float(env.log_s.grad.abs().sum()) > 0


def _make(knots, iterations=25):
    reference, oracle = _make_models()
    train_x, y_plus, y_cross = _make_training_data(oracle)
    surrogate = DemodGPSurrogate(
        train_x=train_x, train_y_plus=y_plus, train_y_cross=y_cross,
        base_approximant=reference, oracle_approximant=None, phase_correction=0.0,
        warping="chirp", nu=2.5, output_scale=1.0, device="cpu",
        total_mass=60.0, distance=100.0, training_iterations=iterations,
        amplitude_normalise=True, envelope_knots=knots,
    )
    return reference, surrogate


@pytest.fixture(scope="module")
def trained():
    return _make(knots=5)


def _kernel_envelope(model):
    return model.covar_module.base_kernel.envelope


def test_kernel_and_noise_share_one_envelope(trained):
    _, s = trained
    m = s._gp.models["plus"]
    assert _kernel_envelope(m) is m.likelihood.noise_covar.envelope
    assert len({id(p) for p in m.parameters() if p.shape == _kernel_envelope(m).log_s.shape}) >= 1


def test_kernel_and_noise_scale_with_envelope(trained):
    _, s = trained
    m = s._gp.models["plus"]
    env = _kernel_envelope(m)
    x = m.train_x[::40].double()
    m = m.double().eval()
    n = x.shape[0]
    with torch.no_grad():
        saved = env.log_s.clone()
        env.log_s.zero_()
        k0 = m.covar_module(x).to_dense()
        n0 = m.likelihood.noise_covar(x, shape=torch.Size([n])).diagonal()
        env.log_s.copy_(torch.linspace(-1.0, 1.0, len(saved), dtype=env.log_s.dtype))
        sv = env(x[:, -1])
        k1 = m.covar_module(x).to_dense()
        n1 = m.likelihood.noise_covar(x, shape=torch.Size([n])).diagonal()
        env.log_s.copy_(saved)
    np.testing.assert_allclose(k1.numpy(), (sv[:, None] * k0 * sv[None, :]).numpy(), rtol=1e-8)
    np.testing.assert_allclose(n1.numpy(), (n0 * sv**2).numpy(), rtol=1e-8)


def test_training_moves_envelope_and_roundtrips(trained, tmp_path):
    reference, s = trained
    env = _kernel_envelope(s._gp.models["plus"])
    assert float(env.log_s.detach().abs().max()) > 0
    s.save(tmp_path / "e.pt")
    loaded = DemodGPSurrogate.load(tmp_path / "e.pt", device="cpu", base_approximant=reference)
    assert loaded._gp.envelope_knots == 5
    p = {"mass_ratio": 0.6, "times": np.linspace(-0.3, 0.02, 60)}
    a, b = s.predict(p, covariance="diagonal"), loaded.predict(p, covariance="diagonal")
    np.testing.assert_allclose(a["plus"].data, b["plus"].data, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(a["plus"].variance, b["plus"].variance, rtol=1e-6, atol=1e-14)


def test_full_and_diagonal_variance_agree(trained):
    _, s = trained
    p = {"mass_ratio": 0.6, "times": np.linspace(-0.3, 0.02, 70)}
    full = s.predict(p, covariance="full")
    diag = s.predict(p, covariance="diagonal")
    v = np.diag(full["plus"].covariance)
    np.testing.assert_allclose(diag["plus"].variance, v, rtol=1e-6, atol=1e-12 * v.max())
    assert np.all(np.isfinite(v)) and np.all(v >= -1e-12 * v.max())


def test_envelope_off_is_unchanged_structure():
    _, s = _make(knots=0, iterations=3)
    m = s._gp.models["plus"]
    assert not hasattr(m.covar_module.base_kernel, "envelope")


def test_roughness_zero_for_linear_log_envelope_and_positive_otherwise():
    t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    env = TimeEnvelope(t, n_knots=6).double()
    with torch.no_grad():
        env.log_s.copy_(torch.linspace(-0.5, 0.5, 6, dtype=torch.float64))
    assert float(env.roughness()) < 1e-4
    with torch.no_grad():
        env.log_s[3] += 0.5
    assert float(env.roughness()) > 0.01


def test_block_loo_matches_brute_force_and_trains():
    from heron.models.gp.exact import ExactGPSurrogate

    _, s = _make(knots=5, iterations=3)
    model = s._gp.models["plus"]
    model.train()
    model.likelihood.train()
    x, y = model.train_x, model.train_y
    with torch.no_grad():
        got = ExactGPSurrogate._block_loo_nll(model, model(x))
        Ky = model.likelihood(model(x), *model.train_inputs).covariance_matrix.double()
        Ky = Ky + 1e-6 * torch.diagonal(Ky).mean() * torch.eye(Ky.shape[0], dtype=Ky.dtype)
        y = y.double()
        _, inv = torch.unique(x[:, :-1], dim=0, return_inverse=True)
        total = 0.0
        for g in range(int(inv.max()) + 1):
            b = inv == g
            Kbb, Kbo, Koo = Ky[b][:, b], Ky[b][:, ~b], Ky[~b][:, ~b]
            mu = Kbo @ torch.linalg.solve(Koo, y[~b])
            cov = Kbb - Kbo @ torch.linalg.solve(Koo, Kbo.T)
            d = torch.distributions.MultivariateNormal(mu, covariance_matrix=(cov + cov.T) / 2)
            total -= float(d.log_prob(y[b]))
    n = y.numel()
    np.testing.assert_allclose(float(got), (total - 0.5 * n * np.log(2 * np.pi)) / n, rtol=1e-4)

    s2 = _make(knots=5, iterations=3)[1]
    assert s2._gp.objective == "mll"


def test_q_envelope_scales_kernel_by_parameter_and_roundtrips(tmp_path):
    reference, oracle = _make_models()
    train_x, y_plus, y_cross = _make_training_data(oracle)
    s = DemodGPSurrogate(
        train_x=train_x, train_y_plus=y_plus, train_y_cross=y_cross,
        base_approximant=reference, oracle_approximant=None, phase_correction=0.0,
        warping="chirp", nu=2.5, output_scale=1.0, device="cpu",
        total_mass=60.0, distance=100.0, training_iterations=5,
        amplitude_normalise=True, envelope_knots=5, q_envelope_knots=4,
    )
    m = s._gp.models["plus"]
    kern = m.covar_module.base_kernel
    assert kern.q_envelope is m.likelihood.noise_covar.q_envelope
    assert float(kern.q_envelope.log_s.abs().sum()) > 0
    x = train_x[:6].to(torch.float64)
    with torch.no_grad():
        full = kern(x).to_dense()
        base = kern.base_kernel(x).to_dense()
        sc = kern.envelope(x[:, -1]) * kern.q_envelope(x[:, 0])
    np.testing.assert_allclose(full.numpy(), (sc[:, None] * base * sc[None]).numpy(), rtol=1e-5)
    path = tmp_path / "q_env.pt"
    s.save(str(path))
    assert DemodGPSurrogate.load(str(path), device="cpu", base_approximant=reference)._gp.q_envelope_knots == 4


def test_demod_with_eta_q_warping_trains_and_roundtrips(tmp_path):
    reference, oracle = _make_models()
    train_x, y_plus, y_cross = _make_training_data(oracle)
    s = DemodGPSurrogate(
        train_x=train_x, train_y_plus=y_plus, train_y_cross=y_cross,
        base_approximant=reference, oracle_approximant=None, phase_correction=0.0,
        warping="chirp", nu=2.5, output_scale=1.0, device="cpu",
        total_mass=60.0, distance=100.0, training_iterations=3,
        amplitude_normalise=True, envelope_knots=5, q_warping="eta", ls_min_q=0.01,
    )
    s.save(tmp_path / "w.pt")
    loaded = DemodGPSurrogate.load(tmp_path / "w.pt", device="cpu", base_approximant=reference)
    assert loaded._gp.q_warping == "eta"
    p = {"mass_ratio": float(train_x[0, 0]), "times": np.linspace(-0.2, 0.02, 20)}
    np.testing.assert_allclose(
        s.predict(p, covariance="diagonal")["plus"].data,
        loaded.predict(p, covariance="diagonal")["plus"].data, rtol=1e-4, atol=1e-12)


def test_loo_scale_field_fit_lowers_loo_and_leaves_lengthscales():
    from heron.models.gp.exact import ExactGPSurrogate

    _, s = _make(knots=5, iterations=3)
    gp = s._gp
    model = gp.models["plus"]
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if "envelope" not in n and "raw_outputscale" not in n}
    hist = gp.fit_scale_fields_loo(iterations=3)
    assert hist["plus"][-1] <= hist["plus"][0] + 1e-9
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p.detach(), before[n])
        assert p.requires_grad
