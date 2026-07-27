"""Shared pytest fixtures for heron tests."""
import numpy as np
import pytest
import torch

from heron.models.gp.exact import ExactGPSurrogate


@pytest.fixture(scope="session")
def tiny_gp():
    """Minimal trained ExactGPSurrogate for tests that need a live GP.

    10 time samples × 2 mass ratios = 20 training points, 3 Adam steps.
    Built once per session; fast enough to not slow down the suite.
    """
    n_per_q = 10
    all_x, all_plus, all_cross = [], [], []
    for q in (0.5, 1.0):
        times = np.linspace(-0.3, 0.02, n_per_q)
        envelope = np.exp(-50 * times**2)
        all_x.append(np.column_stack([np.full(n_per_q, q), times]))
        all_plus.append(np.sin(2 * np.pi * 30 * times) * envelope * q)
        all_cross.append(np.cos(2 * np.pi * 30 * times) * envelope * q)

    return ExactGPSurrogate(
        train_x=torch.tensor(np.vstack(all_x), dtype=torch.float32),
        train_y_plus=torch.tensor(np.concatenate(all_plus), dtype=torch.float32),
        train_y_cross=torch.tensor(np.concatenate(all_cross), dtype=torch.float32),
        warping="chirp",
        nu=2.5,
        output_scale=1.0,
        device="cpu",
        total_mass=60.0,
        distance=100.0,
        training_iterations=3,
        optimizer="adam",
    )


# ---------------------------------------------------------------------------
# Shared stubs for the inference (PE) layer tests
# ---------------------------------------------------------------------------

class StubSurrogate:
    """Deterministic, parameter-light surrogate for PE-layer tests.

    ``h+(t) = A sin(2π f0 t)``, ``h×(t) = A cos(2π f0 t)`` (face-on quadratures),
    with a constant diagonal covariance ``var·I``.  Exposes ``distance_factor``
    so the projection/likelihood distance scaling is exercised.  Ignores all
    parameters except ``times`` (and, optionally, a variance that dips at a given
    mass ratio — see ``dip_at``).
    """

    distance_factor = 100.0

    def __init__(self, f0=50.0, amplitude=1.0, var=1e-6, dip_at=None, dip_var=1e-12):
        self.f0 = f0
        self.amplitude = amplitude
        self.var = var
        self.dip_at = dip_at
        self.dip_var = dip_var

    def predict(self, params):
        from heron.types import Waveform, WaveformDict

        t = np.asarray(params["times"], dtype=float)
        n = len(t)
        phase = 2.0 * np.pi * self.f0 * t
        A = self.amplitude
        var = self.var
        if self.dip_at is not None and abs(params.get("mass_ratio", self.dip_at) - self.dip_at) < 1e-9:
            var = self.dip_var
        cov = np.eye(n) * var
        return WaveformDict(
            plus=Waveform(A * np.sin(phase), t, cov),
            cross=Waveform(A * np.cos(phase), t, cov),
        )


@pytest.fixture
def stub_surrogate():
    return StubSurrogate()


@pytest.fixture
def flat_psd():
    """Flat PSD S(f)=1 for f>=20 Hz (test units keep numbers O(1))."""
    def _psd(freqs):
        return np.where(np.asarray(freqs, dtype=float) >= 20.0, 1.0, 0.0)
    return _psd
