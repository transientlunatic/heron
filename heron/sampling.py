"""
This module contains interfaces to heron to allow straight-forward sampling to be performed.
"""
import logging

from astropy import units as u

import nessai.model
from nessai.utils import setup_logger
import torch
import numpy as np


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

    allow_vectorised = False

    def __init__(self, likelihood, priors, base_p):

        self.logger = logger = logging.getLogger(
            "heron.sampling.NessaiSampler"
        )

        
        # Names of parameters to sample
        self.priors = priors
        self.names = priors.names

        self.likelihood = likelihood

        self.base_p = self._convert_units(base_p)

        self._update_bounds()

    def _convert_units(self, p):
        # Only convert dictionaries
        # Units
        units = {"luminosity_distance": u.megaparsec}  #
        if isinstance(p, dict):
            base_p = {}
            for name, base in p.items():
                if name in units and isinstance(base, u.Quantity):
                    base_p[name] = base.to(units[name]).value
                else:
                    base_p[name] = base
        elif isinstance(p, np.ndarray) and hasattr(p, "names"):
            if not (isinstance(p.luminosity_distance, u.Quantity)):
                p.luminosity_distance = p.luminosity_distance * u.megaparsec
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
        lnp = self.priors.ln_prob(dict(zip(self.names, x)))
        self.logger.info(f"Log prior: {lnp}")
        return lnp

    def log_likelihood(self, x):
        # Convert everything into python scalars
        with torch.inference_mode():
            # Need to convert from numpy floats to python floats
            x = self._convert_units(x)
            self.base_p.update({n: float(x[n]) for n in self.names})
            likelihood = self.likelihood(self.base_p)

            return likelihood
