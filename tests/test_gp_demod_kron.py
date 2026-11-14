"""Tests for heron.models.gp.demod_kron.KroneckerDemodGPSurrogate.

Uses the analytic toy chirp reference/oracle pair from the delta tests
(no lalsuite). The stubs ignore total mass and distance, so in this model they
*are* "the waveform at reference_mass / reference_distance"; geometric
rescaling to any other (M, D) is then checked for exact self-similarity.
"""
import pickle

import numpy as np
import pytest

from heron.models.gp.demod_kron import (
    KroneckerDemodGPSurrogate,
    generate_kronecker_grid,
    warped_tau_grid,
)
from heron.inference.projection import project_factors
from heron.units import MTSUN_SI

from test_gp_delta import _make_models

M_REF, D_REF = 60.0, 100.0
T_UNIT = M_REF * MTSUN_SI
TIMES = np.linspace(-0.38, 0.015, 200)


@pytest.fixture(scope="module")
def trained():
    reference, oracle = _make_models()
    tau = warped_tau_grid(-0.4 / T_UNIT, 0.02 / T_UNIT, 60)
    qs = np.linspace(0.3, 0.9, 7)
    yp, yc = generate_kronecker_grid(oracle, qs, tau, M_REF, D_REF, 20.0)
    surrogate = KroneckerDemodGPSurrogate(
        qs, tau, yp, yc, base_approximant=reference, oracle_approximant=None,
        phase_correction=0.0, reference_mass=M_REF, reference_distance=D_REF,
    )
    return reference, oracle, surrogate


def test_grid_generation_rejects_window_before_oracle_start():
    _, oracle = _make_models()
    # Toy oracle's native grid starts at -0.5 s.
    tau = np.linspace(-0.6 / T_UNIT, 0.0, 10)
    with pytest.raises(ValueError, match="precedes"):
        generate_kronecker_grid(oracle, [0.5], tau, M_REF, D_REF, 20.0)


def test_off_grid_prediction_matches_oracle(trained):
    _, oracle, s = trained
    for q in (0.45, 0.62, 0.85):
        wf = s.predict({"mass_ratio": q, "times": TIMES}, covariance="diagonal")
        hp, hc = oracle.strain(q, TIMES)
        scale = np.abs(hp).max()
        assert np.abs(wf["plus"].data - hp).max() < 1e-3 * scale
        assert np.abs(wf["cross"].data - hc).max() < 1e-3 * scale


def test_total_mass_and_distance_are_exact_rescalings(trained):
    _, _, s = trained
    base = s.predict({"mass_ratio": 0.62, "times": TIMES}, covariance="diagonal")
    # Doubling M doubles the timescale and the amplitude; tripling D divides it.
    scaled = s.predict({"mass_ratio": 0.62, "times": 2 * TIMES, "total_mass": 2 * M_REF,
                        "luminosity_distance": 3 * D_REF}, covariance="diagonal")
    np.testing.assert_allclose(scaled["plus"].data, base["plus"].data * 2 / 3,
                               rtol=1e-12, atol=1e-15 * np.abs(base["plus"].data).max())
    np.testing.assert_allclose(scaled["plus"].variance, base["plus"].variance * (2 / 3) ** 2,
                               rtol=1e-12, atol=1e-30)


def test_defaults_are_reference_mass_and_distance(trained):
    _, _, s = trained
    a = s.predict({"mass_ratio": 0.5, "times": TIMES}, covariance="none")
    b = s.predict({"mass_ratio": 0.5, "times": TIMES, "total_mass": M_REF,
                   "luminosity_distance": D_REF}, covariance="none")
    np.testing.assert_array_equal(a["plus"].data, b["plus"].data)
    assert s.distance_factor == D_REF


def test_full_covariance_is_psd_and_matches_diagonal(trained):
    _, _, s = trained
    full = s.predict({"mass_ratio": 0.62, "times": TIMES}, covariance="full")
    diag = s.predict({"mass_ratio": 0.62, "times": TIMES}, covariance="diagonal")
    for pol in ("plus", "cross"):
        C = full[pol].covariance
        np.testing.assert_allclose(np.diag(C), diag[pol].variance,
                                   rtol=1e-8, atol=1e-12 * np.diag(C).max())
        w = np.linalg.eigvalsh(C)
        assert w.min() > -1e-8 * w.max()


def test_factor_reproduces_plus_cross_and_cross_covariance(trained):
    _, _, s = trained
    full = s.predict({"mass_ratio": 0.62, "times": TIMES}, covariance="full")
    wf, U = s.predict_factor({"mass_ratio": 0.62, "times": TIMES})
    np.testing.assert_array_equal(wf["plus"].data, full["plus"].data)
    ref = np.abs(full["plus"].covariance).max()
    for pol in ("plus", "cross"):
        assert np.abs(U[pol] @ U[pol].T - full[pol].covariance).max() < 0.02 * ref
    assert np.abs(U["plus"] @ U["cross"].T - full.cross_covariance).max() < 0.02 * ref


def test_projected_factor_matches_projected_joint_covariance(trained):
    """project_factors must propagate h+/hx cross-covariance exactly."""
    _, _, s = trained
    _, U = s.predict_factor({"mass_ratio": 0.62, "times": TIMES})
    fp, fc, inc, phi = 0.4, -0.7, 0.6, 0.9
    U_det = project_factors(U["plus"], U["cross"], f_plus=fp, f_cross=fc,
                            inclination=inc, coalescence_phase=phi)
    # Same linear map applied to the factor-implied joint covariance.
    Cpp, Ccc, Cpc = (U["plus"] @ U["plus"].T, U["cross"] @ U["cross"].T,
                     U["plus"] @ U["cross"].T)
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    a = (0.5 * (1 + np.cos(inc) ** 2)) * np.array([c2, -s2])
    b = np.cos(inc) * np.array([s2, c2])
    w = fp * a + fc * b  # h_det = w[0] h+ + w[1] hx
    K = w[0] ** 2 * Cpp + w[1] ** 2 * Ccc + w[0] * w[1] * (Cpc + Cpc.T)
    np.testing.assert_allclose(U_det @ U_det.T, K, atol=1e-10 * np.abs(K).max())
    assert np.abs(Cpc).max() > 1e-3 * np.abs(Cpp).max()  # cross-cov is non-trivial


def test_outside_training_window_returns_bare_reference(trained):
    reference, _, s = trained
    t = np.linspace(-0.48, -0.42, 20)  # before the -0.4 s training window
    wf = s.predict({"mass_ratio": 0.62, "times": t}, covariance="diagonal")
    hp, _ = reference.strain(0.62, t)
    np.testing.assert_allclose(wf["plus"].data, hp, rtol=1e-10)
    assert np.all(wf["plus"].variance == 0.0)
    _, U = s.predict_factor({"mass_ratio": 0.62, "times": t})
    assert np.all(U["plus"] == 0.0)


def test_save_load_roundtrip(trained, tmp_path):
    reference, _, s = trained
    path = tmp_path / "kron.pt"
    s.save(path)
    s2 = KroneckerDemodGPSurrogate.load(path, base_approximant=reference)
    a = s.predict({"mass_ratio": 0.62, "times": TIMES}, covariance="full")
    b = s2.predict({"mass_ratio": 0.62, "times": TIMES}, covariance="full")
    np.testing.assert_array_equal(a["plus"].data, b["plus"].data)
    np.testing.assert_array_equal(a["cross"].covariance, b["cross"].covariance)


def test_pickle_roundtrip(trained):
    _, _, s = trained
    s2 = pickle.loads(pickle.dumps(s))
    a = s.predict({"mass_ratio": 0.62, "times": TIMES}, covariance="diagonal")
    b = s2.predict({"mass_ratio": 0.62, "times": TIMES}, covariance="diagonal")
    np.testing.assert_array_equal(a["plus"].variance, b["plus"].variance)


def test_eta_q_warping_trains(trained):
    reference, oracle, _ = trained
    tau = warped_tau_grid(-0.4 / T_UNIT, 0.02 / T_UNIT, 40)
    qs = np.linspace(0.3, 0.9, 7)
    yp, yc = generate_kronecker_grid(oracle, qs, tau, M_REF, D_REF, 20.0)
    s = KroneckerDemodGPSurrogate(qs, tau, yp, yc, base_approximant=reference,
                                  oracle_approximant=None, phase_correction=0.0,
                                  q_warping="eta", maxiter=50)
    hp, _ = oracle.strain(0.55, TIMES)
    wf = s.predict({"mass_ratio": 0.55, "times": TIMES}, covariance="none")
    assert np.abs(wf["plus"].data - hp).max() < 1e-2 * np.abs(hp).max()
