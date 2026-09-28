"""Priors for parameter estimation.

A small hierarchy of 1-D priors, each defined by its inverse-CDF (``rescale``)
so it plugs straight into nested sampling's unit-hypercube transform, plus a
:class:`PriorDict` that composes them.

``PriorDict`` deliberately exposes the *same* interface
(:attr:`~PriorDict.ndim`, :attr:`~PriorDict.parameter_names`,
:meth:`~PriorDict.transform`, :meth:`~PriorDict.to_dict`,
:meth:`~PriorDict.log_prior`) as :class:`heron.sampling.UniformPrior`, so it is
a drop-in replacement in :class:`heron.sampling.DynestySampler` and
:class:`heron.inference.sampler.NessaiSampler` with no sampler changes.

The concrete priors cover the extrinsic GW set:

- :class:`Uniform`   — mass_ratio, tc, psi, coalescence phase (periodic where set)
- :class:`Sine`      — inclination / theta_jn (p ∝ sin θ on [0, π])
- :class:`Cosine`    — declination (p ∝ cos δ on [−π/2, π/2])
- :class:`PowerLaw`  — luminosity distance (α = 2 ≈ uniform in Euclidean volume)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Prior(ABC):
    """Base class for a 1-D prior on ``[minimum, maximum]``.

    Parameters
    ----------
    minimum, maximum : float
        Support bounds.
    periodic : bool
        Whether the parameter is periodic on its range (e.g. polarisation psi
        on [0, π], coalescence phase on [0, 2π]).  Used by samplers that want
        periodic reparameterisations (nessai).
    latex_label : str or None
        Optional display label.
    """

    def __init__(self, minimum, maximum, periodic=False, latex_label=None):
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.periodic = bool(periodic)
        self.latex_label = latex_label

    @abstractmethod
    def rescale(self, u: float) -> float:
        """Inverse CDF: map ``u`` in [0, 1] to a value in the support."""

    @abstractmethod
    def ln_prob(self, value: float) -> float:
        """Log prior density at *value* (−inf outside the support)."""

    def _outside(self, value: float) -> bool:
        return not (self.minimum <= value <= self.maximum)

    def sample(self, rng=None):
        """Draw a single sample."""
        rng = np.random.default_rng() if rng is None else rng
        return self.rescale(float(rng.random()))


class Uniform(Prior):
    """Uniform prior on ``[minimum, maximum]``."""

    def rescale(self, u: float) -> float:
        return self.minimum + u * (self.maximum - self.minimum)

    def ln_prob(self, value: float) -> float:
        if self._outside(value):
            return -np.inf
        return -np.log(self.maximum - self.minimum)


class Sine(Prior):
    """Prior with density ``p(θ) ∝ sin θ`` on ``[minimum, maximum] ⊆ [0, π]``.

    The natural prior for inclination / theta_jn (isotropic orientation).
    """

    def __init__(self, minimum=0.0, maximum=np.pi, **kwargs):
        super().__init__(minimum, maximum, **kwargs)
        self._c_min = np.cos(self.minimum)
        self._norm = self._c_min - np.cos(self.maximum)  # ∫ sin θ dθ

    def rescale(self, u: float) -> float:
        return float(np.arccos(self._c_min - u * self._norm))

    def ln_prob(self, value: float) -> float:
        if self._outside(value):
            return -np.inf
        return float(np.log(np.sin(value)) - np.log(self._norm))


class Cosine(Prior):
    """Prior with density ``p(δ) ∝ cos δ`` on ``[minimum, maximum] ⊆ [−π/2, π/2]``.

    The natural prior for declination (isotropic sky).
    """

    def __init__(self, minimum=-np.pi / 2, maximum=np.pi / 2, **kwargs):
        super().__init__(minimum, maximum, **kwargs)
        self._s_min = np.sin(self.minimum)
        self._norm = np.sin(self.maximum) - self._s_min  # ∫ cos δ dδ

    def rescale(self, u: float) -> float:
        return float(np.arcsin(self._s_min + u * self._norm))

    def ln_prob(self, value: float) -> float:
        if self._outside(value):
            return -np.inf
        return float(np.log(np.cos(value)) - np.log(self._norm))


class PowerLaw(Prior):
    """Prior with density ``p(x) ∝ x^α`` on ``[minimum, maximum]``.

    ``α = 2`` gives the Euclidean uniform-in-volume distance prior; ``α = −1``
    is the log-uniform prior (handled as a special case).
    """

    def __init__(self, alpha, minimum, maximum, **kwargs):
        super().__init__(minimum, maximum, **kwargs)
        self.alpha = float(alpha)
        if self.minimum <= 0.0:
            raise ValueError("PowerLaw requires minimum > 0")
        if self.alpha == -1.0:
            self._log_ratio = np.log(self.maximum / self.minimum)
        else:
            b = self.alpha + 1.0
            self._b = b
            self._min_b = self.minimum**b
            self._span_b = self.maximum**b - self._min_b
            self._log_norm = np.log(self._span_b / b)

    def rescale(self, u: float) -> float:
        if self.alpha == -1.0:
            return float(self.minimum * np.exp(u * self._log_ratio))
        return float((self._min_b + u * self._span_b) ** (1.0 / self._b))

    def ln_prob(self, value: float) -> float:
        if self._outside(value):
            return -np.inf
        if self.alpha == -1.0:
            return float(-np.log(value) - np.log(self._log_ratio))
        return float(self.alpha * np.log(value) - self._log_norm)


class PriorDict:
    """An ordered collection of named 1-D priors.

    Parameters
    ----------
    priors : dict[str, Prior]
        Ordered mapping of parameter name to its prior.  Iteration order sets
        the column order used by :meth:`transform` / :meth:`to_dict`.
    """

    def __init__(self, priors: dict[str, Prior]):
        self.priors: dict[str, Prior] = dict(priors)

    @property
    def parameter_names(self) -> list[str]:
        return list(self.priors.keys())

    @property
    def ndim(self) -> int:
        return len(self.priors)

    @property
    def periodic_parameters(self) -> list[str]:
        return [n for n, p in self.priors.items() if p.periodic]

    def bounds(self) -> dict[str, tuple[float, float]]:
        return {n: (p.minimum, p.maximum) for n, p in self.priors.items()}

    def transform(self, u: np.ndarray) -> np.ndarray:
        """Map the unit hypercube to physical parameter values."""
        u = np.asarray(u, dtype=float)
        return np.array([p.rescale(u[i]) for i, p in enumerate(self.priors.values())])

    def to_dict(self, theta: np.ndarray) -> dict:
        """Convert a parameter array to a name → value dict."""
        return {n: float(theta[i]) for i, n in enumerate(self.parameter_names)}

    def log_prior(self, theta: np.ndarray) -> float:
        """Total log prior density; −inf if any component is out of support."""
        lp = 0.0
        for i, p in enumerate(self.priors.values()):
            lp += p.ln_prob(theta[i])
            if not np.isfinite(lp):
                return -np.inf
        return float(lp)

    def sample(self, rng=None) -> dict:
        """Draw one sample as a name → value dict."""
        rng = np.random.default_rng() if rng is None else rng
        return {n: p.sample(rng) for n, p in self.priors.items()}
