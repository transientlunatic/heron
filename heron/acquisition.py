"""
Acquisition functions for Bayesian optimisation.
"""


import numpy as np
from scipy.stats import norm

def expected_improvement(mean, err, **kwargs):
    xi = kwargs.get('xi', 1)
    mean = mean / np.max(mean)
    err = err / np.max(err)
    y_max = mean.max()
    err = np.maximum(err, 1e-9)
    z = (mean - y_max - xi)/err
    return (mean - y_max - xi) * norm.cdf(z) + err * norm.pdf(z)

def pure_exploitation(mean,err, args):
    return err

def lower_bound(mean, err, **kwargs):
    scale = kwargs.get('scale', 1)
    mean = mean / np.max(mean)
    err = err / np.max(err)
    print kwargs
    return mean - scale * err

def probable_improvement(mean, err, xi=1):
    from scipy.stats import norm
    mean = mean / np.max(mean)
    err = err / np.max(err)
    y_max = mean.max()
    # Avoid points with zero variance
    #err = np.maximum(err, 1e-9)
    z = (mean - y_max - xi)/err
    return norm.cdf(z)
