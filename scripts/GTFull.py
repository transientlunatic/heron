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

training_data_x = np.loadtxt("/home/daniel/data/heron/GT_training_x.txt")
training_data_y = np.loadtxt("/home/daniel/data/heron/GT_training_y.txt")

#training = training_data_x[2:,0]## == [0,0,0,0,0,0]
#print "training", training

data = data.Data(training_data_x.T, #:4
                 np.atleast_2d(training_data_y),
                 label_sigma = 5e-26,
                 test_size=0.25,
                 #label_sigma = [0.01, 0.01],
                 target_names = ["t", "q", "S1x", "S1y", "S1z", "S2x", "S2y", "S2z"],# "gamma3"],
                 label_names = ["h+"],)

#data_density = corner.corner(data)
#data_density.savefig("GT_cornerplot.pdf")

sep = np.abs(data.get_starting()) + 0.00001
print "sep", sep
hyper_priors = [priors.Normal(1.0/hyper, 2) for hyper in sep]
k3 = np.std(data.labels[0,:])**2 * kernels.Matern52Kernel(sep**2, ndim=len(sep))

kernel = k3
gp = regression.SingleTaskGP(data, kernel = kernel, hyperpriors = hyper_priors)

from heron import training

#MAP = gp.train("MAP")
training.train_cv(gp)
gp.save("GT_BBH.gp")

#MAP = gp_phase.train("MAP")

# import corner


# corner.corner(samples, labels=gp.gp.kernel.get_parameter_names())
# plt.savefig("corner_NS.png")
    
# #gp.kernel.set_vector(np.max(samples, axis=0))

#gp.save("NSPolytrope_amp.gp")
#gp_phase.save("NSPolytrope_phase.gp")
