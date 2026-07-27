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
        from nessai.model import Model

        names = self.prior.parameter_names
        bounds = self._bounds()
        periodic = self._periodic()
        prior = self.prior
        loglike_array = self.loglike_array

        class _HeronModel(Model):
            def __init__(self):
                self.names = list(names)
                self.bounds = {n: list(bounds[n]) for n in names}
                # Declare periodic parameters for nessai's reparameterisation.
                self.reparameterisations = {
                    n: {"reparameterisation": "periodic"} for n in periodic
                } or None

            def log_prior(self, x):
                x = np.atleast_1d(x)
                out = np.empty(len(x))
                for i, row in enumerate(x):
                    theta = np.array([row[n] for n in self.names])
                    out[i] = prior.log_prior(theta)
                return out

            def log_likelihood(self, x):
                x = np.atleast_1d(x)
                out = np.empty(len(x))
                for i, row in enumerate(x):
                    theta = np.array([row[n] for n in self.names])
                    out[i] = loglike_array(theta)
                return out

        return _HeronModel()

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

        log_z = float(getattr(sampler, "log_evidence", getattr(sampler.ns, "log_evidence", np.nan)))
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
