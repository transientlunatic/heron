"""nessai (flow-accelerated nested sampling) backend.

:class:`NessaiSampler` subclasses :class:`heron.sampling.BaseSampler`, so it
drops into the existing PE pipeline exactly like
:class:`heron.sampling.DynestySampler` — same ``(log_likelihood, prior)``
constructor and ``run() -> SamplerResult`` contract.  nessai wins over dynesty
as the dimensionality grows (full extrinsic ≈ 7–8 D); at very low dimension the
two are comparable.

The prior (a :class:`heron.inference.prior.PriorDict` or
:class:`heron.sampling.UniformPrior`) supplies ``names``, ``bounds``,
``log_prior`` and — where declared — periodic parameters, which are handed to
nessai as reparameterisations.
"""
from __future__ import annotations

import numpy as np

from heron.sampling import BaseSampler, SamplerResult

# nessai is an optional dependency. Import its Model at module scope (guarded) so
# our subclass is defined at module level -- a *local* class can never be
# pickled, and pickling the model to worker processes is exactly what n_pool
# parallelism requires. When nessai is absent the base is a plain object and
# NessaiSampler.run() raises a clear error at call time.
try:
    from nessai.model import Model as _NessaiModel
    _HAVE_NESSAI = True
except ImportError:  # pragma: no cover - exercised only without nessai
    _NessaiModel = object
    _HAVE_NESSAI = False


class _HeronNessaiModel(_NessaiModel):
    """Module-level (hence picklable) nessai model wrapping a heron prior + LL.

    Stores the prior and the user log-likelihood callable directly, so an
    instance pickles cleanly to ``n_pool`` workers *provided* both of those are
    picklable — the heron ``PriorDict`` is, and the surrogate-backed likelihood
    is once the surrogate/approximant define ``__getstate__`` (see
    ``ExactGPSurrogate``/``DemodGPSurrogate``/``LALSimulationApproximant``).
    """

    def __init__(self, names, bounds, periodic, prior, loglike_fn):
        self.names = list(names)
        self.bounds = {n: list(bounds[n]) for n in self.names}
        # Declare periodic parameters for nessai's reparameterisation.
        self.reparameterisations = {
            n: {"reparameterisation": "periodic"} for n in periodic
        } or None
        self._prior = prior
        self._loglike_fn = loglike_fn

    def log_prior(self, x):
        x = np.atleast_1d(x)
        out = np.empty(len(x))
        for i, row in enumerate(x):
            theta = np.array([row[n] for n in self.names])
            out[i] = self._prior.log_prior(theta)
        return out

    def log_likelihood(self, x):
        x = np.atleast_1d(x)
        out = np.empty(len(x))
        for i, row in enumerate(x):
            theta = np.array([row[n] for n in self.names])
            out[i] = self._loglike_fn(self._prior.to_dict(theta))
        return out


class NessaiSampler(BaseSampler):
    """Nested sampler backed by nessai's :class:`~nessai.flowsampler.FlowSampler`.

    Parameters
    ----------
    log_likelihood : callable
        Accepts a parameter dict, returns a float log-likelihood.
    prior : PriorDict or UniformPrior
        Parameter space definition (must expose ``parameter_names``, ``ndim``,
        ``to_dict`` and ``log_prior``; ``PriorDict`` additionally provides
        ``bounds`` and ``periodic_parameters``).
    """

    def _bounds(self) -> dict[str, tuple[float, float]]:
        prior = self.prior
        if hasattr(prior, "bounds"):
            return prior.bounds()
        # Fall back to heron.sampling.UniformPrior's Parameter list.
        return {p.name: (p.lower, p.upper) for p in prior.parameters}

    def _periodic(self) -> list[str]:
        return list(getattr(self.prior, "periodic_parameters", []))

    def _build_model(self):
        if not _HAVE_NESSAI:
            raise ImportError(
                "nessai is required for NessaiSampler; install it to run this "
                "backend (the heron.inference PE layer otherwise works without it)."
            )
        # `_log_likelihood` is the raw user callable (BaseSampler stores it);
        # passing it plus the prior directly -- rather than the bound
        # `self.loglike_array` -- keeps the picklable model free of any
        # reference to this sampler instance (which, post-run, holds the
        # unpicklable nessai FlowSampler).
        return _HeronNessaiModel(
            names=self.prior.parameter_names,
            bounds=self._bounds(),
            periodic=self._periodic(),
            prior=self.prior,
            loglike_fn=self._log_likelihood,
        )

    def run(
        self,
        output: str = "nessai_output",
        nlive: int = 1000,
        seed: int | None = None,
        resume: bool = False,
        plot: bool = False,
        **kwargs,
    ) -> SamplerResult:
        """Run nessai and return a :class:`~heron.sampling.SamplerResult`.

        Parameters
        ----------
        output : str
            Directory nessai writes its state / diagnostics to.
        nlive : int
            Number of live points.
        seed : int or None
            Random seed.
        resume : bool
            Resume from a previous run in *output* if present.
        plot : bool
            Produce nessai's diagnostic plots.
        **kwargs
            Forwarded to ``FlowSampler``.
        """
        from nessai.flowsampler import FlowSampler

        model = self._build_model()
        sampler = FlowSampler(
            model, output=output, nlive=nlive, seed=seed, resume=resume, **kwargs,
        )
        sampler.run(plot=plot)
        return self._to_result(sampler)

    def _to_result(self, sampler) -> SamplerResult:
        """Adapt a completed nessai FlowSampler into a SamplerResult."""
        names = self.prior.parameter_names
        posterior = sampler.posterior_samples
        samples = np.column_stack([np.asarray(posterior[n], dtype=float) for n in names])
        n = len(samples)
        # nessai returns equal-weight posterior samples.
        log_weights = np.full(n, -np.log(max(n, 1)))

        log_z = float(
            getattr(sampler, "log_evidence",
                    getattr(sampler.ns, "log_evidence", np.nan))
        )
        log_z_err = float(
            getattr(sampler, "log_evidence_error",
                    getattr(sampler.ns, "log_evidence_error", np.nan))
        )
        return SamplerResult(
            samples=samples,
            log_weights=log_weights,
            log_evidence=log_z,
            log_evidence_err=log_z_err,
            parameter_names=list(names),
            raw=sampler,
        )
