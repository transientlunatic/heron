"""Generic sampler interface for GW parameter estimation.

The design separates three concerns:

  Prior        — maps the unit hypercube to physical parameters (and back).
  BaseSampler  — thin protocol: __init__(log_likelihood, prior) + run() → SamplerResult.
  SamplerResult— uniform wrapper around sampler output.

Adding a new sampler (bilby, nessai, nautilus, …) means subclassing BaseSampler
and implementing run().  Nothing else in the pipeline changes.

Usage::

    prior = UniformPrior([
        Parameter("mass_ratio", 0.5, 1.0),
        Parameter("tc",         tc_true - 0.05, tc_true + 0.05),
        Parameter("ra",         0.0, 2 * np.pi),
        Parameter("dec",       -np.pi / 2, np.pi / 2),
        Parameter("psi",        0.0, np.pi),
    ])

    sampler = DynestySampler(gw_ll, prior, nlive=500)
    result  = sampler.run()
    post    = result.posterior_dict()   # dict of equal-weight 1-D arrays
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

@dataclass
class Parameter:
    """A single parameter with uniform prior bounds."""
    name: str
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


class UniformPrior:
    """Independent uniform prior over all parameters.

    Parameters
    ----------
    parameters : list[Parameter]
        Ordered list of parameter definitions.
    """

    def __init__(self, parameters: list[Parameter]):
        self.parameters = list(parameters)

    @property
    def ndim(self) -> int:
        return len(self.parameters)

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def transform(self, u: np.ndarray) -> np.ndarray:
        """Map the unit hypercube to physical parameter space.

        Parameters
        ----------
        u : ndarray, shape (ndim,)
            Points in [0, 1]^ndim.

        Returns
        -------
        theta : ndarray, shape (ndim,)
            Physical parameter values.
        """
        theta = np.empty(self.ndim)
        for i, p in enumerate(self.parameters):
            theta[i] = p.lower + u[i] * p.width
        return theta

    def to_dict(self, theta: np.ndarray) -> dict:
        """Convert a parameter array to a dict suitable for GWLikelihood."""
        return {p.name: float(theta[i]) for i, p in enumerate(self.parameters)}

    def log_prior(self, theta: np.ndarray) -> float:
        """Log of the (unnormalised) uniform prior density — 0 inside, -inf outside."""
        for i, p in enumerate(self.parameters):
            if not (p.lower <= theta[i] <= p.upper):
                return -np.inf
        return 0.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class SamplerResult:
    """Sampler-agnostic container for nested-sampling output.

    Parameters
    ----------
    samples : ndarray, shape (N, ndim)
        Live + dead points with associated log-weights.
    log_weights : ndarray, shape (N,)
        Normalised log-weights: log(w_i / Σ w_j).
    log_evidence : float
        log Z estimate.
    log_evidence_err : float
        1σ uncertainty on log Z.
    parameter_names : list[str]
        Names matching columns of *samples*.
    raw : object
        The underlying sampler result object (e.g. ``dynesty.results.Results``).
    """

    def __init__(
        self,
        samples: np.ndarray,
        log_weights: np.ndarray,
        log_evidence: float,
        log_evidence_err: float,
        parameter_names: list[str],
        raw: Any = None,
    ):
        self.samples = np.asarray(samples)
        self.log_weights = np.asarray(log_weights)
        self.log_evidence = float(log_evidence)
        self.log_evidence_err = float(log_evidence_err)
        self.parameter_names = list(parameter_names)
        self.raw = raw

    @classmethod
    def from_dynesty(cls, dynesty_result, prior: UniformPrior) -> "SamplerResult":
        """Construct from a dynesty ``Results`` object."""
        res = dynesty_result
        # Normalised log-weights: log w_i - log Z.
        log_weights = res.logwt - res.logz[-1]
        return cls(
            samples=res.samples,
            log_weights=log_weights,
            log_evidence=float(res.logz[-1]),
            log_evidence_err=float(res.logzerr[-1]),
            parameter_names=prior.parameter_names,
            raw=res,
        )

    def posterior_samples(self, n: int | None = None) -> np.ndarray:
        """Return equal-weight posterior samples drawn from the weighted set.

        Parameters
        ----------
        n : int or None
            Number of samples to draw.  Defaults to the effective sample size.

        Returns
        -------
        ndarray, shape (n, ndim)
        """
        weights = np.exp(self.log_weights)
        weights /= weights.sum()
        rng = np.random.default_rng()
        n_eff = int(1.0 / (weights**2).sum()) if n is None else n
        idx = rng.choice(len(weights), size=n_eff, p=weights)
        return self.samples[idx]

    def posterior_dict(self, n: int | None = None) -> dict[str, np.ndarray]:
        """Return equal-weight posterior samples as a name → 1-D array dict."""
        s = self.posterior_samples(n=n)
        return {name: s[:, i] for i, name in enumerate(self.parameter_names)}

    def posterior_median(self) -> dict[str, float]:
        """Weighted median for each parameter."""
        weights = np.exp(self.log_weights)
        weights /= weights.sum()
        order = np.argsort(self.samples, axis=0)
        cumw = np.take_along_axis(
            np.broadcast_to(weights[:, None], self.samples.shape),
            order, axis=0,
        ).cumsum(axis=0)
        median_idx = np.argmax(cumw >= 0.5, axis=0)
        medians = self.samples[order[median_idx, np.arange(self.samples.shape[1])],
                               np.arange(self.samples.shape[1])]
        return {name: float(medians[i]) for i, name in enumerate(self.parameter_names)}


# ---------------------------------------------------------------------------
# Sampler base class
# ---------------------------------------------------------------------------

class BaseSampler(ABC):
    """Protocol for all samplers.

    Subclasses must implement :meth:`run` and may override :meth:`loglike_array`.

    Parameters
    ----------
    log_likelihood : callable
        A callable that accepts a parameter dict and returns a log-likelihood scalar.
        Typically a :class:`~heron.gw_likelihood.GWLikelihood` instance.
    prior : UniformPrior
        Defines the parameter space and unit-hypercube transform.
    """

    def __init__(self, log_likelihood, prior: UniformPrior):
        self._log_likelihood = log_likelihood
        self.prior = prior

    def loglike_array(self, theta: np.ndarray) -> float:
        """Evaluate the log-likelihood from a parameter array."""
        return float(self._log_likelihood(self.prior.to_dict(theta)))

    @abstractmethod
    def run(self, **kwargs) -> SamplerResult:
        """Run the sampler and return the result."""


# ---------------------------------------------------------------------------
# Dynesty
# ---------------------------------------------------------------------------

class DynestySampler(BaseSampler):
    """Nested sampler backed by dynesty.

    Parameters
    ----------
    log_likelihood : callable
        Accepts a parameter dict, returns a float log-likelihood.
    prior : UniformPrior
        Parameter space definition and prior transform.
    nlive : int
        Number of live points.  Higher → more accurate evidence but slower.
    dynamic : bool
        If True, use dynesty's dynamic nested sampler.
    sampler_kwargs : dict
        Extra keyword arguments forwarded to ``dynesty.[Dynamic]NestedSampler``.
    """

    def __init__(
        self,
        log_likelihood,
        prior: UniformPrior,
        nlive: int = 500,
        dynamic: bool = False,
        **sampler_kwargs,
    ):
        super().__init__(log_likelihood, prior)
        self.nlive = nlive
        self.dynamic = dynamic
        self.sampler_kwargs = sampler_kwargs

    def run(self, **run_kwargs) -> SamplerResult:
        """Run dynesty and return a :class:`SamplerResult`.

        Parameters
        ----------
        **run_kwargs
            Forwarded to ``sampler.run_nested()``.  Useful options include
            ``dlogz`` (stopping criterion) and ``print_progress``.
        """
        import dynesty

        cls = dynesty.DynamicNestedSampler if self.dynamic else dynesty.NestedSampler
        sampler = cls(
            self.loglike_array,
            self.prior.transform,
            self.prior.ndim,
            nlive=self.nlive,
            **self.sampler_kwargs,
        )
        sampler.run_nested(**run_kwargs)
        return SamplerResult.from_dynesty(sampler.results, self.prior)
