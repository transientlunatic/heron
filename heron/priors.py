"""
Prior distributions for GP hyperpriors.
"""

from scipy import stats
from scipy.special import ndtri

class Prior(object):
    """
    A prior probability distribution.
    """
    pass

class Normal(Prior):
    """
    A normal prior probability distribution.
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.distro = stats.norm
        
    def logp(self, x):
        return self.distro.logpdf(x, loc = self.mean, scale = self.std)

    def transform(self, x):
        """
        Transform from unit normalisation to this prior.

        Parameters
        ----------
        x : float
           The position in the normalised hyperparameter space
        """
        
        return self.mean + self.srd * ndtri(x)
