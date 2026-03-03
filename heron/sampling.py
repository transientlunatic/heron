"""
This module contains interfaces to heron to allow straight-forward sampling to be performed.
"""

from astropy import units as u

import nessai.model
from nessai.utils import setup_logger
import torch
import numpy as np

try:
    from aspire import Aspire
    from aspire.samples import Samples as AspireSamples
    _ASPIRE_AVAILABLE = True
except ImportError:
    _ASPIRE_AVAILABLE = False


class SamplerBase:
    pass


class NessaiSampler(SamplerBase, nessai.model.Model):
    """Nessai model for Heron Likelihoods.

    This simple model uses uniform priors on all parameters.

    Parameters
    ----------
    heron_likelihood
        Instance of heron likelihood.
    priors
        Prior dictionary.
    """

    allow_vectorised = True

    def __init__(self, likelihood, priors, base_p):
        # Names of parameters to sample
        self.priors = priors
        self.names = priors.names

        self.likelihood = likelihood

        self.base_p = self._convert_units(base_p)

        self._update_bounds()

    def _convert_units(self, p):
        # Only convert dictionaries
        if isinstance(p, dict):
            # Units
            units = {"luminosity_distance": u.megaparsec}  #

            base_p = {}
            for name, base in p.items():
                if name in units and isinstance(base, u.Quantity):
                    base_p[name] = base.to(units[name]).value
                else:
                    base_p[name] = base
        else:
            base_p = p
        return base_p

    def _update_bounds(self):
        self.bounds = {
            key: [self.priors[key].minimum, self.priors[key].maximum]
            for key in self.names
        }

    def log_prior(self, x):
        if isinstance(x, np.ndarray):
            x = x[0]
        return self.priors.ln_prob(dict(zip(self.names, x)))

    def log_likelihood(self, x):
        # Convert everything into python scalars
        with torch.inference_mode():
            # Need to convert from numpy floats to python floats
            x = self._convert_units(x)

            # Check if x contains arrays (vectorized) or scalars
            # x is a dict where values can be either scalars or arrays
            first_value = x[self.names[0]]
            is_vectorized = isinstance(first_value, (np.ndarray, list))

            if is_vectorized:
                # Vectorized case: x[n] is an array of values
                # Evaluate likelihood for each sample
                n_samples = len(first_value)
                likelihoods = np.zeros(n_samples)
                for i in range(n_samples):
                    sample_params = self.base_p.copy()
                    sample_params.update({n: float(x[n][i]) for n in self.names})
                    likelihoods[i] = self.likelihood(sample_params)
                return likelihoods
            else:
                # Non-vectorized case: x[n] is a scalar
                self.base_p.update({n: float(x[n]) for n in self.names})
                likelihood = self.likelihood(self.base_p)
                return likelihood


def load_bilby_result_as_samples(path, parameter_names):
    """Load a bilby result file and return an aspire Samples object.

    Parameters
    ----------
    path : str
        Path to the bilby result HDF5 or JSON file.
    parameter_names : list of str
        Ordered list of parameter names to extract from the posterior.
        Must match the order used by the aspire sampler's ``dims``.

    Returns
    -------
    AspireSamples
        Samples object with ``.x`` shape ``(n_samples, n_dims)``,
        ``.log_likelihood``, and ``.log_prior``.
    """
    if not _ASPIRE_AVAILABLE:
        raise ImportError("aspire is not installed; cannot load bilby result as aspire Samples.")
    import bilby
    result = bilby.core.result.read_in_result(path)
    df = result.posterior
    x = df[parameter_names].values
    log_L = df["log_likelihood"].values if "log_likelihood" in df.columns else np.zeros(len(df))
    log_prior = np.array([
        result.priors.ln_prob(dict(zip(parameter_names, row))) for row in x
    ])
    return AspireSamples(x=x, log_likelihood=log_L, log_prior=log_prior)


class AspireSampler(SamplerBase):
    """Aspire SMC sampler for Heron likelihoods.

    Optionally seeded from a pre-existing bilby result to use the FD posterior
    as the SMC proposal, correcting it toward the Heron GPR posterior via
    sequential tempering.

    Parameters
    ----------
    likelihood : callable
        Heron likelihood (e.g. ``MultiDetector``).
    priors : heron.priors.PriorDict
        Prior dict (bilby-compatible).
    base_p : dict
        Fixed parameters not sampled (e.g. ``total_mass``, ``gpstime``).
    initial_result : str, optional
        Path to a bilby HDF5/JSON result file to use as the initial SMC
        proposal.  If ``None``, the prior is used.
    """

    def __init__(self, likelihood, priors, base_p, initial_result=None):
        if not _ASPIRE_AVAILABLE:
            raise ImportError("aspire is not installed; cannot use AspireSampler.")
        self.likelihood = likelihood
        self.priors = priors
        self.names = priors.names
        self.base_p = self._convert_units(base_p)
        self.initial_result = initial_result

    def _convert_units(self, p):
        if isinstance(p, dict):
            units = {"luminosity_distance": u.megaparsec}
            base_p = {}
            for name, base in p.items():
                if name in units and isinstance(base, u.Quantity):
                    base_p[name] = base.to(units[name]).value
                else:
                    base_p[name] = base
        else:
            base_p = p
        return base_p

    def _log_likelihood(self, samples):
        log_L = np.zeros(len(samples))
        with torch.inference_mode():
            for i, row in enumerate(samples.x):
                params = self.base_p.copy()
                params.update({n: float(row[j]) for j, n in enumerate(self.names)})
                log_L[i] = self.likelihood(params)
        return log_L

    def _log_prior(self, samples):
        return np.array([
            self.priors.ln_prob(dict(zip(self.names, row)))
            for row in samples.x
        ])

    def _draw_from_prior(self, n):
        samples_list = [self.priors.sample() for _ in range(n)]
        x = np.array([[s[name] for name in self.names] for s in samples_list])
        log_prior = np.array([self.priors.ln_prob(s) for s in samples_list])
        return AspireSamples(x=x, log_likelihood=np.zeros(n), log_prior=log_prior)

    def sample(self, n_samples=500, n_epochs=30, sampler="smc", sampler_kwargs=None):
        """Run aspire SMC and return the posterior.

        Parameters
        ----------
        n_samples : int
            Number of SMC particles.
        n_epochs : int
            Epochs to train the normalising flow on the initial samples.
        sampler : str
            Aspire sampler name (``"smc"``, ``"minipcn_smc"``, etc.).
        sampler_kwargs : dict, optional
            Extra keyword arguments passed to ``aspire.sample_posterior``.

        Returns
        -------
        posterior : AspireSamples
        history : aspire SMCHistory
        """
        aspire_model = Aspire(
            log_likelihood=self._log_likelihood,
            log_prior=self._log_prior,
            dims=len(self.names),
            parameters=self.names,
        )
        if self.initial_result is not None:
            initial = load_bilby_result_as_samples(self.initial_result, self.names)
        else:
            initial = self._draw_from_prior(5000)
        aspire_model.fit(initial, n_epochs=n_epochs)
        return aspire_model.sample_posterior(
            sampler=sampler,
            n_samples=n_samples,
            return_history=True,
            sampler_kwargs=sampler_kwargs or {},
        )
