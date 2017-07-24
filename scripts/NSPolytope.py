"""
This script is designed to produce and train a Gaussian
"""

import numpy as np
from heron import data, regression, corner
from heron import priors
from george import kernels
import scipy

import matplotlib.pyplot as plt

training_data = np.loadtxt("/home/daniel/data/heron/ns-polytrope.txt")

print len(training_data)

data_amp = data.Data(training_data[::5,(0,1,2,3)], #:4
                 training_data[::5,5],
                 test_size=0.1,
                 label_sigma = [0.1],
                 target_names = ["t", "logP1", "gamma1", "gamma2",],# "gamma3"],
                 label_names = ["amp"],)

# interesting observation, the Data object doesn't check that the x and y sets are the same length...

data_phase = data.Data(training_data[::5,(0,1,2,3)], #:4
                 training_data[::5,6],
                 test_size=0.1,
                 #label_sigma = [0.1],
                 target_names = ["t", "logP1", "gamma1", "gamma2",],# "gamma3"],
                 label_names = ["phase"],)

data_density = corner.corner(data_amp)
data_density.savefig("ns_cornerplot.pdf")

sep = np.abs(data_amp.get_starting())+0.001
hyper_priors = [priors.Normal(hyper, 2) for hyper in sep]
k3 = np.std(data_amp.labels)**2 * kernels.Matern52Kernel(sep, ndim=len(sep))
kernel = k3
gp = regression.Regressor(data_amp, kernel = kernel, hyperpriors = hyper_priors)
gp_phase = regression.Regressor(data_phase, kernel = kernel, hyperpriors = hyper_priors)

from heron import training
print training.ln_likelihood(sep, gp)

samples, burn = gp.train("MCMC")

#MAP = gp.train("MAP")
#MAP = gp_phase.train("MAP")

# import corner

# np.savetxt("ns-samples.txt", samples)

# corner.corner(samples, labels=gp.gp.kernel.get_parameter_names())
# plt.savefig("corner_NS.png")
    
# #gp.kernel.set_vector(np.max(samples, axis=0))

gp.save("NSPolytrope_amp.gp")
gp_phase.save("NSPolytrope_phase.gp")
