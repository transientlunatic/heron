"""
This script is designed to produce and train a Gaussian
"""

import numpy as np
from heron import data, regression
from heron import priors
from george import kernels
import scipy

import matplotlib.pyplot as plt

training_data = np.loadtxt("/home/daniel/data/heron/ns-polytrope.txt")


data = data.Data(training_data[:5].T, training_data[5],
                 label_sigma = [0.1],
                 target_names = ["t", "logP1", "gamma1", "gamma2", "gamma3"],
                 label_names = ["amp"],)

sep = np.abs(data.get_starting())
hyper_priors = [priors.Normal(hyper, 1) for hyper in sep]
k3 = np.std(data.labels) * kernels.Matern52Kernel(sep, ndim=len(sep))
kernel = k3
gp = regression.Regressor(data, kernel = kernel, hyperpriors = hyper_priors)

from heron import training
#print training.ln_likelihood(sep, gp)

samples, burn = gp.train("MCMC")

import corner

np.savetxt("ns-samples.txt", samples)

corner.corner(samples, labels=gp.gp.kernel.get_parameter_names())
plt.savefig("corner_NS.png")
    
#gp.kernel.set_vector(np.max(samples, axis=0))

gp.save("NSPolytrope.gp")
