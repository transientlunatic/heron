import numpy as np
from heron import data
from heron import regression
from george import kernels
import emcee

training_x = np.loadtxt("/home/daniel/data/heron/GT_training_x.txt")
training_y = np.loadtxt("/home/daniel/data/heron/GT_training_y.txt")


data = data.Data(training_x.reshape(-1, 8)[:10000], training_y[:10000],
                 label_sigma = [0.1],
                 target_names = ["t", "q", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z"],
                 label_names = ["hx"],)

sep = np.array([1/0.21, 1/1.3, 1/3., 0.001, 0.001, 0.001, 0.001, 0.001])
k3 = kernels.Matern52Kernel(sep**2, ndim=len(sep))
kernel = k3
gp = regression.Regressor(data, kernel = kernel)

from heron import training
print gp.training_y.shape
print gp.training_data.shape
print training.ln_likelihood(sep, gp)

samples, burn = gp.train("MCMC", sampler="pt")

import corner
import matplotlib.pyplot as plt

corner.corner(samples, labels=gp.gp.kernel.get_parameter_names())
plt.savefig("corner_test.png")

gp.save("test.gp")
