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
