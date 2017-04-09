from heron import data
import matplotlib.pyplot as plt
import numpy as np
from heron import regression
from heron import kernels
from heron import acquisition
from scipy.optimize import minimize

bbh_text = np.genfromtxt("/home/daniel/repositories/heron/data/bbh_mass1_only_hp_3D.txt", delimiter=" ")
bbh_test = np.genfromtxt("/home/daniel/repositories/heron/data/bbh_mass1_only_hp_3D_TEST.txt", delimiter=" ")

bbh = data.Data(bbh_text[:,:3], bbh_text[:,-1],              
                    target_names = ["Times", "Mass", "Spin"],
                    label_names = ["hp strain"],
                    test_size = 0,
                    test_targets = bbh_test[:,:3],
                    test_labels = bbh_test[:,-1]
                    )

x0 = [(1/np.std(bbh.labels))**2, 50, 10, 10]
gp = regression.Regressor(bbh, kernel=kernels.SquaredExponential(ndim=3), yerror = 1e-23, tikh=1e-6)
gp.set_hyperparameters(x0)
#gp.optimise()
x0 = [ 24.95395191,  21.95756258,  -2.07481906, -13.40978644]
#x0 = [0.029974254529549851,  101.2117402 ,    0.84988828]
gp.set_hyperparameters(x0)

import lalsimulation, lal

def generate_new_points(mass1, spin1x):

    chi1_l = 0
    chi2_l = 0
    chip = 0
    thetaJ = 0
    m1 = mass1
    distance = 5 * lal.PC_SI
    alpha0 = 0
    phic = 0
    f_ref = 100 * lal.HertzUnit,
    waveflags = lalsimulation.SimInspiralCreateWaveformFlags()
    approximant = lalsimulation.SimInspiralGetApproximantFromString("IMRPhenomP")
    #phorder = lalsimulation.SimInspiralGetOrderFromString("IMRPhenomP0")
    #f, ax = plt.subplots(10, sharey=True, figsize=(5,15))
    data = np.zeros((150, 4))
    mass1 = mass1 * lal.MSUN_SI
    coaphase = 0
    mass2 = 6 * lal.MSUN_SI
    spin1x, spin1y, spin1z = spin1x,0,0
    spin2x, spin2y, spin2z = 1,0,0
    flower = 10 
    fref = 10
    distance = 400 *1e6 * lal.PC_SI
    z = 0.0
    inclination = 0.0
    lambda1 = lambda2 = 0.0
    amporder = 0
    phorder = 0
    dt = 0.0002
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
    data[:,-1] =  hp.data.data[(-0.02<times) & (times<0.01)]
    data[:,0] = times[(-0.02<times) & (times<0.01)]
    data[:,1] = m1 * np.ones(150)
    data[:,2] = spin1x * np.ones(150)
    return data

import scipy
from scipy.optimize import minimize

class MyBounds(object):
     def __init__(self, xmax=[15,1], xmin=[5,0] ):
         self.xmax = np.array(xmax)
         self.xmin = np.array(xmin)
     def __call__(self, **kwargs):
         x = kwargs["x_new"]
         tmax = bool(np.all(x <= self.xmax))
         tmin = bool(np.all(x >= self.xmin))
         return tmax and tmin
def nei(m):
    return -gp.expected_improvement([0, m[0], m[1]])
def infill(number, gp):
    new_points = []
    for i in range(number):
        mybounds = MyBounds()
        x0 = [10, 0]
        new = scipy.optimize.basinhopping(nei, x0, niter=100, accept_test=mybounds)
        #target = bbh.denormalise(new.x, "target")
        target = new.x
        new_data = generate_new_points(target[0], target[1])
        gp.add_data(new_data[0,:3], new_data[0,-1])
        gp.optimise()
        print target, gp.loglikelihood(), gp.correlation(), gp.rmse()
        new_points.append(np.append(target, [gp.loglikelihood(), gp.correlation(), gp.rmse()]))

new_points = np.asarray(infill(100, gp))
np.savetxt("new_points.txt", new_points)

