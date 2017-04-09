import matplotlib as mpl
mpl.use('Agg')

import george
from heron import data
import matplotlib.pyplot as plt
import numpy as np
import scipy

import lalsimulation, lal

def rprop(function, dfunction, theta0 = 1.0, grow = 1.2, shrink = 0.8, step_min = 1e-5, step_max = 0.1, updates=50):
    """
    Calculate the optimum of a function using the RPROP algorithm.
    
    Parameters
    ----------
    function : python callable
        The function to be optimised.
    dfunction : python callable
        The derivative of the function to be optimised.
    grow : float
        The factor by which the step size can grow during an update.
    shrink : float
        The factor by which the step size can shrink during an update.
    step_min : float
        The smallest possible size of step.
    step_max : float
        The largest possible size of step.
    """
    if isinstance(theta0, np.ndarray):
        # We're optimising a multidimensional function, so we'll need to make
        # sure that our step sizes come as a vector too.
        step = np.array([0.05]*len(theta0))
    else:
        step = 0.05
    thetas = [theta0]
    for i in xrange(updates):
        theta1 = theta0 - np.sign(dfunction(theta0)) * step
        test = (dfunction(theta0) * dfunction(theta1))
        #if (dfunction(theta0) * dfunction(theta1)) > 0:
        step[test > 0] *= grow
        step[test < 0] *= shrink
        #    step *= grow
        #elif (dfunction(theta0) * dfunction(theta1)) < 0:
        #    step *= shrink
        #else:
        #    pass
        thetas.append(theta1)
        theta0 = theta1
    return theta0, thetas

def generate_new_points(massratio, spin1x, npoints = 300, tstart = -0.1, tend = 0.005):
    """
    
    Parameters
    ----------
    massratio : float
        The ratio of the two component masses.
    spin1x : array
        The vector of spin components for mass 1.
    spin2x : array
        The vector of spin components for mass 2.
    npoints : int
        The desired number of points in the output waveform
    tstart : float
        The start time of the output waveform.
    tend : float
        the end time of the output waveform.
    
    Outputs
    -------
    data : array
        An array of data in the format expected by the heron data object.
    
    """
    chi1_l = 0
    chi2_l = 0
    chip = 0
    thetaJ = 0
    mass2 = 15 * lal.MSUN_SI
    mass1 = mass2 / massratio
    #print mass1 / lal.MSUN_SI
    distance = 5 * lal.PC_SI
    alpha0 = 0
    phic = 0
    f_ref = 100 * lal.HertzUnit,
    waveflags = lalsimulation.SimInspiralCreateWaveformFlags()
    approximant = lalsimulation.SimInspiralGetApproximantFromString("IMRPhenomP")
    #phorder = lalsimulation.SimInspiralGetOrderFromString("IMRPhenomP0")
    #f, ax = plt.subplots(10, sharey=True, figsize=(5,15))
    data = np.zeros((npoints, 4))
    #m1 = mass1 / lal.MSUN_SI
    coaphase = 0
    
    spin1x, spin1y, spin1z = spin1x,0,0
    spin2x, spin2y, spin2z = spin1x,0,0
    flower = 10 
    fref = 10
    distance = 400 *1e6 * lal.PC_SI
    z = 0.0
    inclination = 0.0
    lambda1 = lambda2 = 0.0
    amporder = 0
    phorder = 0
    dt = (tend-tstart)/npoints
    hp, hc = lalsimulation.SimInspiralTD(coaphase, dt, 
                                         mass1, mass2,
                                         spin1x, spin1y, spin1z,
                                         spin2x, spin2y, spin2z,
                                         flower, fref, distance, z,
                                         inclination, lambda1, lambda2,
                                         waveflags, None, amporder, phorder,
                                         approximant
                                        )
    times = np.linspace(0, len(hp.data.data)*hp.deltaT, len(hp.data.data)) + hp.epoch
    data[:,-1] =  hp.data.data[(tstart<times) & (times<tend)]
    data[:,0] = times[(tstart<times) & (times<tend)]
    data[:,1] = massratio * np.ones(npoints)
    data[:,2] = spin1x * np.ones(npoints)
    return data

from george import kernels
from heron import regression

# # Georgia Tech Mass ratios
ratios = [ 0.06649688,  0.07314657,  0.07979625,  0.0831211 ,  0.08644594,
        0.09309563,  0.09974532,  0.106395  ,  0.11304469,  0.11636954,
        0.11969438,  0.12634407,  0.13299376,  0.13964344,  0.14629313,
        0.14961798,  0.15294282,  0.15959251,  0.16624219,  0.17289188,
        0.17954157,  0.18619126,  0.19284095,  0.19949063,  0.20614099,
        0.21279001,  0.2194397 ,  0.22608872,  0.23273907,  0.23938943,
        0.24603845,  0.2526888 ,  0.25933716,  0.26598751,  0.27263786,
        0.27928622,  0.28593724,  0.29258693,  0.29923662,  0.33248372,
        0.39898193,  0.46547815,  0.66472407,  1.        ]

test_ratios = ratios[:-1] + np.diff(ratios)/2


bbh_data_2 = np.vstack([generate_new_points(ratio, 0, 250, tstart=-0.10, tend=0.05) for ratio in ratios[:10:2]])
bbh_test_2 = np.vstack([generate_new_points(ratio, 0, 250, tstart=-0.10, tend=0.05) for ratio in test_ratios[1:10:2]])

bbh_2 = data.Data(bbh_data_2[:,:2], bbh_data_2[:,-1],
                    target_names = ["Times", "Mass"],
                    label_names = ["hp strain"],
                    test_size = 0,
                    test_targets = bbh_test_2[:,:2],
                    test_labels = bbh_test_2[:,-1]
                    )

# Gaussian Process Kernel Model
k0 =  np.std(bbh_2.labels)
k1 =  kernels.ExpSquaredKernel(0.00015, axes = 0, ndim=2)
k2 =  kernels.ExpSquaredKernel((1./39.9)**2, axes = 1, ndim=2)
k3 = kernels.Matern52Kernel(0.00015, axes=0, ndim=2)
kernel = k0 * k3 * k2
gp = regression.Regressor(bbh_2, kernel=kernel, yerror = 0, tikh=1e-6)

# Optimise the hyperparameters of the model
# This is the learning step, in effect.
#gp.optimise()
def gll(theta):
    gp.kernel.set_vector(theta)
    return -gp.grad_loglikelihood()
def ll(theta):
    gp.kernel.set_vector(theta)
    return -gp.loglikelihood()
result, trace = rprop(ll, gll, gp.kernel.get_vector(), updates = 100)

print trace

print gp.kernel.get_vector()

# Make a sample waveform to examine the optimisation
f, ax = plt.subplots(1,1, figsize=(15,5))
n = 300
t = np.linspace(-0.1,0.05,n)
new_point = np.zeros((n,2))
new_point[:,1] = 0.17
new_point[:,0] = t
pred = gp.prediction(new_point)
ax.plot(t, pred[0])
err =pred[1]
err2 = err*1.96
ax.fill_between(t, pred[0] - err, pred[0] + err , alpha = 0.2)
ax.fill_between(t, pred[0] - err2, pred[0] + err2 , alpha = 0.1)
plt.savefig("heron2-samplewave.png")

# Perform the battery of standardised tests against the test data
print "Finding RMSE of HERON-2 model"
gp.test_predict()
print gp.rmse()
print "Finding correlation of HERON-2 model"
print gp.correlation() 

# Pickle the model
import pickle
with open("heron2.gp", "wb") as savefile:
    pickle.dump(gp, savefile)

