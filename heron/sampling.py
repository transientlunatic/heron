from nessai.model import Model
from nessai.utils import setup_logger

import torch
import numpy as np


class HeronSampler(Model):
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

    def __init__(self, heron_likelihood, priors, base_p, uncertainty=True):
        # Names of parameters to sample
        self.names = list(priors.keys())
        self.bounds = priors
        self.heron_likelihood = heron_likelihood
        self.waveform_uncertainty = uncertainty
        self.base_p = base_p

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x))
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0], dtype=float)
        return log_p

    def log_likelihood(self, x):
        with torch.inference_mode():
            # Need to convert from numpy floats to python floats
            self.base_p.update({n: float(x[n]) for n in self.names})
            return (
                self.heron_likelihood(self.base_p, model_var=self.waveform_uncertainty)
                .cpu()
                .numpy()
            )
