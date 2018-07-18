import matplotlib.pyplot as plt

import astropy
from astropy.table import Table
from heron import data, regression, corner, priors, sampling
import os
import heron
from glob import glob
import numpy as np

from scipy.signal import decimate

# Keep track of the various times that things happen at
ptimes = {}

from gtdata import get_dataset, t, columns

from george import kernels
import george
import scipy.optimize


def neglk(x, gp, training):
    if np.any(x<0): return np.inf
    x = np.log(x)
    if not np.all(np.isfinite(x)): return np.inf
    gp.set_parameter_vector(x)
    return -gp.log_likelihood(training.labels, quiet=True)#,# -gp.grad_log_likelihood(training.labels, quiet=True)




def main():

    from awesome_codename import generate_codename
    codename = generate_codename()


    #### The monster

    query = (    (t["$a_{1x}$"]>=-100)  )

    #
    # Building the Kernel
    #
    k1 = kernels.Matern52Kernel(0.001, ndim=len(columns), axes=0)
    k2 = kernels.Matern52Kernel(0.05, ndim=len(columns), axes=0)
    k_massr = kernels.ExpKernel((.15), ndim=len(columns), axes=(1))
    k_spinx = kernels.ExpKernel((0.125, 0.125, 0.125), ndim=len(columns), axes=(2,3,4))
    k_spiny = kernels.ExpKernel((0.125, 0.125, 0.125), ndim=len(columns), axes=(5,6,7))
    kL = kernels.ExpKernel((.01, .01, .01), ndim=len(columns), axes=(8,9,10))
    kernel = 3.5 * k2 * (0.5* k_massr) * (1*k_spinx) * (1*k_spiny) * (1 * kL)
    # 0.1 * k1 +

    #
    # First, generate a model without the timeseries, so that we can train
    # on something small, then introduce the times.
    #

    training_x, training_y, test_x, test_y, test_waveforms, test_pars, train_wave, train_pars, train_table, test_table = get_dataset(t, query = query, waveforms = 500, inspiral=50, ringdown=50, skip=1)
    print("Shape of training data: {}".format(training_x.shape))

    print "Training data assembled. {} training points.".format(len(training_y))

    training_spin_monster = training = heron.data.Data(targets=training_x.T, labels=np.array(training_y),
                               label_sigma = 1e-6,
                              target_names = columns, label_names = ["h+"] )
    print "Data object created."

    gp_spin_monster = gp = regression.SingleTaskGP(training_spin_monster, kernel = kernel, solver=george.HODLRSolver, tikh=0.000000001)

    saveout = "models/{}.gp".format(codename)

    print "Model Created."
    print "Saving model as {}".format(saveout)
    gp.save(saveout)


if __name__ == "__main__":
    main()
