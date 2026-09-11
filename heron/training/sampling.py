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


def jittered_grid_sample(
    bounds: dict[str, tuple[float, float]],
    n_samples: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate a stratified ("jittered grid") sample in a parameter space.

    Each dimension independently is split into ``n_samples`` equal-width
    cells and one uniform-random point is drawn per cell. This guarantees
    no clumps and no voids (unlike pure random sampling) while remaining
    irregular (unlike a fixed grid) — the important property when the aim
    is to break a tensor-product training grid without introducing new,
    *irregular* posterior-variance spikes where the sampler happens to
    leave a gap.

    For a single dimension this is exactly one-dimensional stratified
    sampling; for several dimensions the per-dimension marginals are each
    stratified (the cells are shuffled independently per dimension, so the
    joint design is Latin-hypercube-like rather than a full tensor grid).

    Parameters
    ----------
    bounds : dict
        Parameter names → (lower, upper) bounds.
    n_samples : int
        Number of samples (= number of strata per dimension).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Parameter names → arrays of sample values.
    """
    names = list(bounds.keys())
    rng = np.random.default_rng(seed)

    out = {}
    for name in names:
        lower, upper = bounds[name]
        edges = np.linspace(lower, upper, n_samples + 1)
        # One uniform draw inside each [edges[i], edges[i+1]) cell.
        jitter = rng.uniform(size=n_samples)
        values = edges[:-1] + jitter * (edges[1:] - edges[:-1])
        # Independent per-dimension shuffle so that, in >1D, dimensions do
        # not stay co-sorted (which would collapse the design back onto the
        # grid diagonal).
        rng.shuffle(values)
        out[name] = values
    return out
