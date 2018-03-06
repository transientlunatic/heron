import otter
import otter.bootstrap as bt
import otter.plot as op


report = otter.Otter("/home/daniel/data/reports/heron/report.html",
                     title = "HERON BBH Surrogate Model",
                     author = "Daniel Williams")


import astropy
from astropy.table import Table
import os
from glob import glob
import numpy as np
import george
from george import kernels
import emcee
import corner
import matplotlib.pyplot as plt

#
# TRAINING DATA
#

def find_data(tag, path = "/home/daniel/data/gravitational-waves/gt-old/"):
    """
    Find the data files which contain the NR data for a given tag.
    """
    result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*{}*.asc'.format(tag)))]
    return result

headers = ['Index', 'Name', 'tag', '$q$', '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$', '$L_x$', '$L_y$', '$L_z$', 'mf', 'af', 'mW']
t = Table.read('/home/daniel/data/gravitational-waves/gt-new/GT_CATALOG_TABLE.txt', format="ascii", names=headers)

columns = ['t', '$q$', '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$']
training_x = []
training_y = []
for j,row in enumerate(t.to_pandas().iterrows()):
    waveform_file = find_data(row[1]['tag'])
    if len(waveform_file)!=1:
        continue
    data = np.loadtxt(waveform_file[0])[::15]
    
    hrss = np.sqrt(data[:,1]**2 + data[:,2]**2)
    
    data[:,0] = data[:,0] - data[np.argmax(hrss),0]
    times = data[:,0][hrss.argmax()-60:hrss.argmax() + 40]
    # Use ~250 times from each waveform
    if len(times)==0: continue
    rowdata = np.zeros((len(columns), len(times)))
    for i, col in enumerate(columns):
        if i == 0: 
            rowdata[i,:] = data[:,0][hrss.argmax()-60:hrss.argmax() + 40]
        else:
            rowdata[i,:] = np.tile(row[1][col], len(times))
    training_y.append(data[:,2][hrss.argmax() - 60:hrss.argmax() + 40])
    training_x.append(np.atleast_2d(rowdata))


training_y = np.hstack(training_y)
training_x = np.hstack(training_x)



row = bt.Row(1)
row[0] + "# Available Training Data"
report + row

from scipy.stats import gaussian_kde
import numpy as np


plt.style.use("/home/daniel/papers/thesis/thesis-style.mpl")

figsize = (7.5,7.5)
colnames = ['t', '$q$', '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$',]
f, ax = plt.subplots(len(colnames),len(colnames),figsize=figsize)
#yvalues = data_object.denormalise(data_object.labels, "label")

for i, xcol in enumerate(colnames):
    for j, ycol in enumerate(colnames):   

        if not i==len(colnames)-1:
            ax[i,j].set_xticklabels([])
        else:
            ax[i,j].set_xlabel(colnames[j])
        if not j==0: 
            ax[i,j].set_yticklabels([])
        else:
            ax[i,j].set_ylabel(colnames[i])
        if j>i: 
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['bottom'].set_visible(False)
            ax[i,j].spines['left'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].grid(False)
            ax[i,j].yaxis.set_ticks_position('none')
            ax[i,j].xaxis.set_ticks_position('none')
            continue


        if i == j:
            ax[i,j]
            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].grid(False)
            ax[i,j].yaxis.set_ticks_position('left')
            ax[i,j].xaxis.set_ticks_position('bottom')
            n, bins, rectangles = ax[i,j].hist(training_x[i], normed=True,alpha=0.6)
            #try:
            kernel = gaussian_kde(training_x[i].T)
            positions = np.linspace(training_x[i].min(), training_x[i].max(), 100)
            ax[i,j].plot(positions, n.max()*kernel(positions)/kernel(positions).max())


            continue
        #ax[i,j].set_xlim([data[:,i].min(), data[:,i].max()])
        #ax[i,j].set_ylim([data[:,j].min(), data[:,j].max()])   
        #ax[i,j].plot(t[ycol], t[xcol], '.')
        hexes = ax[i,j].hexbin(training_x[j], training_x[i], gridsize=15, cmap="Reds", bins='log', vmin=0, vmax=2)

f.subplots_adjust(wspace=0.05, hspace=0.05)
#cb = f.colorbar(ax[2,3], cax = ax[4,5]) 
cbar = f.colorbar(hexes, ax = ax[4,5], orientation="vertical")
cbar.set_label("Number density")

row = bt.Row(1)
row[0] + "There are {} samples available.".format(len(training_y))
row + op.Figure(report, f)
report + row

row = bt.Row(1)
row[0] + "# Training the GP"
report + row

training_x_batch, training_y_batch = training_x, training_y
rng = np.random.RandomState(0)
ixs = rng.randint(len(data), size=10000)
training_x_batch, training_y_batch = training_x[:,ixs].T, training_y[ixs]
k0 =  np.std(training_y_batch)**2
sep = np.array([0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
sep = np.random.rand(len(sep))
k3 = kernels.Matern52Kernel(sep**2, ndim=len(sep))
kernel = k0 + k3
gp = george.GP(kernel, tol=1e-6, solver=george.HODLRSolver)
gp.compute(training_x_batch, yerr=0.001)

from scipy.optimize import minimize
def neg_ln_like(p):
    gp.set_vector(p)
    return -gp.lnlikelihood(training_y_batch)

def ln_like(p):
    if np.any(p[2:]>15): return -np.inf
    if np.any(p[2:]<-15): return -np.inf
    #if p[0]<2: return -np.inf
    if p[0]<-20: return -np.inf
    if -12>p[1]>0: return -np.inf
    #if 0>p[1]>15: return -np.inf
    gp.set_vector(p)
    return gp.lnlikelihood(training_y_batch)

def grad_neg_ln_like(p):
    gp.set_vector(p)
    return gp.grad_lnlikelihood(training_y_batch)

# First find the MAP estimate
MAP = minimize(neg_ln_like, gp.get_vector(), method="BFGS", )

row = bt.Row(1)
row[0] + "## MAP estimate of GP Parameters"
row[0] + "{}".format(MAP.x)
report+row

row = bt.Row(1)
row[0] + "## MCMC Burn-in"

from IPython.core.display import clear_output
def run_sampler(sampler, initial, iterations):
    """
    Run the MCMC sampler for some number of iterations, 
    but output a progress bar so you can keep track of what's going on
    """
    sampler.run_mcmc(initial, 1)
    for iteration in xrange(iterations/10-1):
        sampler.run_mcmc(None, 10)
        clear_output()
        print "{}% \r".format(1000*float(iteration+1) / iterations)
    return sampler


ndim, nwalkers = len(sep)+1, 100
p0 = [MAP.x for i in range(nwalkers)]
#p0 = [MAP.x for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_like, threads=4)
burn = run_sampler(sampler, p0, 1000)

samples = burn.chain[:, :, :].reshape((-1, ndim))
fig = corner.corner(samples, labels=gp.kernel.get_parameter_names() ,lines=np.median(samples, axis=0))

row + op.Figure(report, fig)

report + row

sampler.reset()

row = bs.Row(1)
row + "## Production sampling"

sampler.reset()
sampler = run_sampler(sampler, p0, 2000)
samples = sampler.chain[:, :, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=gp.kernel.get_parameter_names() ,lines=np.median(samples, axis=0))

row + "MCMC Parameter estimates: {}".format(np.median(samples[900:], axis=0))
row + op.Figure(report, fig)

import pickle
with open("samples.dat", "wb") as tracefile:
    pickle.dump(samples, tracefile)
with open("gp.dat", "wb") as gpfile:
    pickle.dump(gp, gpfile)

report + row

gp.set_vector(np.median(samples[900:], axis=0))

row = bt.Row(1)
row + "# Grid spacings"
spacings = np.exp(gp.get_vector()[1:])

row + spacings

axes = []
for axis in spacings:
    axes.append(np.arange(0, 1, axis))

grid = np.array(np.meshgrid(*axes))
points = grid.T.reshape(-1, 8)

row + "There are {} grid points".format(len(points))

values = []
for i in xrange(len(points)/1000):
    values += list(gp.predict(training_y, points[1000*i:1000*(i+1)], return_cov=False, return_var=True)[1])#[1][0]

with open("testpoints.dat", "wb") as tracefile:
    pickle.dump(values, tracefile)
    
report + row

report.show()
