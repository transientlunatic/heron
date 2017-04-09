import matplotlib as mpl
mpl.use('Agg')

import emcee
import george
from heron import data
import matplotlib.pyplot as plt
import numpy as np
import scipy

import lalsimulation, lal

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

waveforms = []
test_waveforms = []

print "Generating Waveforms..."

for spin in [0.0, 0.05, 0.1]:
    for ratio in ratios[:40]:
        waveforms.append(generate_new_points(ratio, spin, 250, tstart=-0.10, tend=0.05))
for spin in [0.025, 0.075]:
    for ratio in test_ratios[:40]:
        test_waveforms.append(generate_new_points(ratio, spin, 250, tstart=-0.10, tend=0.05))      

bbh_data_3 = np.vstack(waveforms)
bbh_test_3 = np.vstack(test_waveforms)

print "Done Generating Waveforms..."
print "Making the training object..."

bbh_3 = data.Data(bbh_data_3[:,:3], bbh_data_3[:,-1],
                    target_names = ["Times", "Mass", "Spin"],
                    label_names = ["hp strain"],
                    test_size = 0,
                    test_targets = bbh_test_3[:,:3],
                    test_labels = bbh_test_3[:,-1]
                    )

print "Assembling the kernel..."

k0 =  np.std(bbh_3.labels)#0.02418475
k1 =  kernels.ExpSquaredKernel(0.00186233, axes = 0, ndim=3)
k2 =  kernels.ExpSquaredKernel(0.00105753, axes = 1, ndim=3)
k4 =  kernels.ExpSquaredKernel((1./3.)**2, axes = 2, ndim=3)
k3 =  kernels.Matern52Kernel(0.00186233,   axes = 0, ndim=3)
kernel = k0 + (k1*k2*k4) #k4 * k3 * k2

print "Making the Regressor..."
gp = regression.Regressor(bbh_3, kernel=kernel, yerror = 0, tikh=1e-5)


gp.optimise()

print "Saving the model..."

import pickle
with open("spinning.gp", "wb") as savefile:
    pickle.dump(gp, savefile)

f, ax = plt.subplots(2,1, figsize=(15,5))
n = 300
t = np.linspace(-0.1,0.05,n)
new_point = np.zeros((n,3))
new_point[:,1] = 0.15
new_point[:,2] = 0
new_point[:,0] = t
pred = gp.prediction(new_point)
ax[0].plot(t, pred[0])
err =pred[1]
err2 = err*1.96
ax[0].fill_between(t, pred[0] - err, pred[0] + err , alpha = 0.2)
ax[0].fill_between(t, pred[0] - err2, pred[0] + err2 , alpha = 0.1)
plt.savefig("spin-test.png")


times = np.linspace(-0.1,0.05,100)
ratios = np.linspace(0.07, 0.7, 100)
pdata = np.zeros((100,100))
udata = np.zeros((100,100))
xv, yv = np.meshgrid(times, ratios, sparse=False, indexing='ij')
for i in xrange(100):
    for j in xrange(100):
        pdata[i][j], udata[i][j] = gp.prediction([[xv[i,j], yv[i,j], 0.0]])
plt.figure(figsize=(10,10))
plt.imshow(pdata, extent=(0.07,0.7, -0.1, 0.05), interpolation="none", aspect=7, origin='lower')
plt.colorbar()
plt.savefig("ratios-surface-spin0.png")
plt.imshow(udata, extent=(0.07,0.7, -0.1, 0.05), interpolation="none", aspect=7, origin='lower')
plt.colorbar()
plt.savefig("ratios-surface-spin0-uncer.png")

times = np.linspace(-0.1,0.05,100)
ratios = np.linspace(0.00, 0.1, 100)
pdata = np.zeros((100,100))
udata = np.zeros((100,100))
xv, yv = np.meshgrid(times, ratios, sparse=False, indexing='ij')
for i in xrange(100):
    for j in xrange(100):
        pdata[i][j], udata[i][j] = gp.prediction([[xv[i,j], 0.17,  yv[i,j]]])
plt.figure(figsize=(10,10))
plt.imshow(pdata, extent=(0.0,0.7, -0.1, 0.05), interpolation="none", aspect=7, origin='lower')
plt.colorbar()
plt.savefig("spin-surface-ratio0d17.png")
plt.imshow(udata, extent=(0.0,0.7, -0.1, 0.05), interpolation="none", aspect=7, origin='lower')
plt.colorbar()
plt.savefig("spin-surface-ratio0d17-uncer.png")
