"""
Parameter space sampling strategies for training data generation.

Provides space-filling designs (Sobol, Latin Hypercube) that give
better coverage than grid sampling, especially in higher dimensions.
"""

from __future__ import annotations

import numpy as np
from scipy.stats.qmc import Sobol, LatinHypercube


def sobol_sample(
    bounds: dict[str, tuple[float, float]],
    n_samples: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate Sobol sequence samples in a parameter space.

    Parameters
    ----------
    bounds : dict
        Parameter names → (lower, upper) bounds.
    n_samples : int
        Number of samples. Rounded up to next power of 2 for Sobol.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Parameter names → arrays of sample values.
    """
    names = list(bounds.keys())
    d = len(names)
    lower = np.array([bounds[n][0] for n in names])
    upper = np.array([bounds[n][1] for n in names])

    # Sobol requires power-of-2 samples; generate at least n_samples
    sampler = Sobol(d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n_samples, 2))))
    unit_samples = sampler.random_base2(m)[:n_samples]

    # Scale to parameter bounds
    samples = lower + unit_samples * (upper - lower)

    return {name: samples[:, i] for i, name in enumerate(names)}


def latin_hypercube_sample(
    bounds: dict[str, tuple[float, float]],
    n_samples: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate Latin Hypercube samples in a parameter space.

    Parameters
    ----------
    bounds : dict
        Parameter names → (lower, upper) bounds.
    n_samples : int
        Number of samples.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Parameter names → arrays of sample values.
    """
    names = list(bounds.keys())
    d = len(names)
    lower = np.array([bounds[n][0] for n in names])
    upper = np.array([bounds[n][1] for n in names])

    sampler = LatinHypercube(d, seed=seed)
    unit_samples = sampler.random(n_samples)

    samples = lower + unit_samples * (upper - lower)

    return {name: samples[:, i] for i, name in enumerate(names)}
