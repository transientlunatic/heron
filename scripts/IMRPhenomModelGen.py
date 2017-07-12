"""
This script is designed to produce and train a Gaussian
"""

import numpy as np
from heron import data
from heron import regression
from george import kernels
import emcee
from scipy import stats
import scipy

import matplotlib.pyplot as plt

training_data = np.loadtxt("/home/daniel/data/heron/IMRPhenomPv2_nonspinning_q1to10.dat")

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
        distro = stats.norm(self.mean, self.std)
        
    def logp(self, x):
        return distro.logpdf(x)

def get_starting(data):
    values = []
    for ax in xrange(data.targets.shape[1]):
        print ax
        values.append(np.median(np.unique(np.diff(data.targets[:, ax])))/2)
    return np.array(values)

data = data.Data(training_data[:2,::5].T, training_data[2,::5],
                 label_sigma = [0.1],
                 target_names = ["t", "q"],
                 label_names = ["hx"],)

sep = get_starting(data)# np.array([1/0.21, 1/1.3,])# 1/3., 0.001, 0.001, 0.001, 0.001, 0.001])
hyper_priors = [Normal(hyper, 1) for hyper in sep]
k3 = np.std(data.labels) * kernels.Matern52Kernel(sep, ndim=len(sep))
kernel = k3
gp = regression.Regressor(data, kernel = kernel, hyperpriors = hyper_priors)

from heron import training
#print training.ln_likelihood(sep, gp)

samples, burn = gp.train("MCMC")

import corner

corner.corner(samples, labels=gp.gp.kernel.get_parameter_names())
plt.savefig("corner_test.png")

import pickle
with open("IMRPhenomPv2.gp", "wb") as filedump:
    pickle.dump(gp, filedump)

    
gp.kernel.set_vector(np.max(samples, axis=0))

gp.save("IMRPhenomPv2.gp")
