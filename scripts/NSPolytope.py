"""
This script is designed to produce and train a Gaussian
"""

import matplotlib
matplotlib.use("agg")
import numpy as np
from heron import data, regression, corner
from heron import priors
from george import kernels
import scipy
import matplotlib.pyplot as plt

training_data = np.loadtxt("/home/daniel/data/heron/ns-polytrope.txt")

data = data.Data(training_data[:,(0,1,2,3)], #:4
                 training_data[:,(5,6)],
                 test_size=0,
                 #label_sigma = [0.01, 0.01],
                 target_names = ["t", "logP1", "gamma1", "gamma2",],# "gamma3"],
                 label_names = ["amp", "phase"],)

data_density = corner.corner(data)
data_density.savefig("ns_cornerplot.pdf")

sep = np.abs(data.get_starting())+0.001
hyper_priors = [priors.Normal(1.0/hyper, 2) for hyper in sep]
k3 = np.std(data.labels[0,:])**2 * kernels.Matern52Kernel(sep, ndim=len(sep))

kernel = k3
gp = regression.MultiTaskGP(data, kernel = kernel, hyperpriors = hyper_priors)
#gp_phase = regression.Regressor(data_phase, kernel = kernel, hyperpriors = hyper_priors)


from heron import training
#print training.ln_likelihood(sep, gp)

#samples, burn = gp.train("MCMC")
#samples_p, burn_p = gp_phase.train('MCMC')
#gp_phase.kernel.set_vector(gp.kernel.get_vector())
#np.savetxt("ns-samples.txt", samples)
#np.savetxt("ns-phase-samples.txt", samples_p)

MAP = gp.train("MAP")
#MAP = gp_phase.train("MAP")

# import corner


# corner.corner(samples, labels=gp.gp.kernel.get_parameter_names())
# plt.savefig("corner_NS.png")
    
# #gp.kernel.set_vector(np.max(samples, axis=0))

#gp.save("NSPolytrope_amp.gp")
#gp_phase.save("NSPolytrope_phase.gp")
