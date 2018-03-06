"""

This script trains the full HERON model on all numerical
relativity parameters using the numerical relativity data from Georgia
Tech.

"""

import os
from astropy.io import ascii
import numpy.ma as ma
import numpy as np
import math
from astropy.table import Table, vstack
from cStringIO import StringIO
import base64
from IPython.display import display, HTML
import pandas as pd
import emcee
# Turn off the max column width so the HTML 
# image tags don't get truncated 
pd.set_option('display.max_colwidth', -1)

# Turning off the max column will display all the data in
# our arrays so limit the number of element to display
pd.set_option('display.max_seq_items', 2)


# Load all of the NR data

main_dir = '/scratch/aries/gt_bbh/'

fieldnames = ['No.', 'RunDir', 'D', 'q', 'a1', 'a2', 'th1L', 'th2L', 
                  'ph1', 'ph2', 'th12', 'thSL', 'thJL', 'Mmin(f=30Hz)', 'Mmin(f=10Hz)']

waveforms = Table()

for series_dir in os.walk(main_dir).next()[1]:
    par_file = main_dir+series_dir+'/README_'+series_dir+'.txt'
    par_dic = ascii.read(par_file,names=fieldnames)
    parameters = Table(par_dic)
    # parameters.add_columns(('time', 'h+', 'hx'))
    pathnames, times, hp, hx = [], [], [], []
    for i in xrange(len(parameters)):
        os.path.join(main_dir,series_dir,parameters[i]['RunDir'])
        try:
            for filename in os.listdir(os.path.join(main_dir,series_dir,parameters[i]['RunDir'])):
                    if filename.endswith('.asc'):
                        pathname = os.path.join(main_dir,series_dir,parameters[i]['RunDir'],filename)
                        pathnames.append(pathname)
                        data = ascii.read(pathname)
                        #for name in fieldnames:
                        #    wf_dic[name] = par_dic[name][wf_num]
                        times.append(np.array(data['t_sim']))
                        hp.append(data['h_+'])
                        hx.append(data['h_x'])
        except:
            pathnames.append('')
            times.append([])
            hp.append([])
            hx.append([])
            
    parameters['times'], parameters['h+'], parameters['hx'] = times, hp, hx
    waveforms = vstack([waveforms, parameters])

import gwgpr
import gwgpr.nr as nr

cat = nr.NRCatalogue('/scratch/aries/gt_bbh/')
cat_f = nr.NRCatalogue('/scratch/aries/gt_bbh/')
#cat.waveforms = cat.waveforms[cat.waveforms['series']=='S-series-v2']
cols = ['q', 'a1','a2', 'th1L', 'ph1', 'th12', 'thSL', 'thJL']


SKIP = 5
import numpy as np
training_x = []
training_y = []
for waveform in cat.waveforms[cols].iterrows():
    try:
        wave = cat.load(waveform[0])
        times = wave.times
        hp = wave.data[0]
        hc = wave.data[1]
        # Select a limited time span from -200 to 100
        #locs = times== 0 
        #locs = np.roll(locs, 100)
        locs = [(times>-50) & (times<10)]
        N = len(times[locs][::SKIP])
        data = hp[locs][::SKIP]
        params = [list(waveform[1])]*N
        times = times[locs]
        if np.isnan(params).any():
            print "{} contains nan".format(waveform[0])
            continue
        for i in xrange(N):
            out = []
            out.append(times[i])
            out.extend(params[i])
            training_x.append(out)
        training_y.extend(list(data))
        #training.append( times[locs][::4], params, data) 
        #print zip(cols, list(waveform[1]))
    except:
        print "{} unvailable".format(waveform[0])

times = np.linspace(-199.85, 47, 300)
N = 300
#data = hp[locs][::4]
params =  [[1.0,  0.81,  0.8, 30.0, 180.0, 60.0, 60.0, 17.0]]*N
#times = times[locs][::4]
eval_x = []
for i in xrange(N):
    out = []
    #out.append(times[i])
    out.append(params[i])
    eval_x.append(out)


import george
from george import kernels

k0 =  np.std(training_y)**2
#sep = np.array([0.83, 0.0001, 0.2, 0.2, 20, 90, 50, 25,  10])
sep = np.array([0.01, 0.1, 0.2, 0.2, 20, 90, 50, 25,  10])
k3 = kernels.Matern52Kernel(sep**2, ndim=9)
kernel = k0+k3 
gp = george.GP(kernel, tol=1e-10, solver=george.HODLRSolver, mean=0, nleaf=100, )
gp.compute(training_x, sort=False)

from scipy.optimize import minimize

def ln_prior(p):
    """Set some flat priors to make sure that the optimisation isn't
    completely stupid; might want to put a bit more thought into this
    further down the line.

    """
    if np.all(p>0) and np.all(p<100):
        return 0
    return - np.inf



def lnprob(p):
    gp.set_vector(p)
    return ln_prior(p) + gp.lnlikelihood(training_y)

#minimize(neg_ln_like, gp.get_vector(), method="BFGS")



ndim, nwalkers = len(gp.get_vector()), 100

p0 = [gp.get_vector() for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,)
#burn-in
pos, prob, state = sampler.run_mcmc(p0, 1000)
a = sampler.reset()
a = sampler.run_mcmc(p0, 10000)

import corner
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
fig = corner.corner(samples, labels=cols)

fig.savefig("corner-hyper.pdf")

print gp.get_vector()


import pickle
with open("full.gp", "wb") as savefile:
    pickle.dump([gp, training_y], savefile)

with open("full-mcmc.gp", "wb") as savefile:
    pickle.dump(sampler.chain, savefile)

resolution = 100
cols_axis = {
    "times": np.linspace(-100, 100, resolution),
    "q": np.linspace(0, 10, resolution),
    "a1" : np.linspace(0, 1, resolution),
    "a2": np.linspace(0, 1, resolution),
    "th1L": np.linspace(0, 180, resolution),
    "ph1": np.linspace(-180, 180, resolution),
    "th12": np.linspace(0, 180, resolution),
    "thSL": np.linspace(0, 180, resolution),
    "thJL": np.linspace(0, 45, resolution)
}
cols = ['times', 'q', 'a1', 'a2', 'th1L', 'ph1', 'th12', 'thSL', 'thJL'] 

def gen2plane(col1, col2, intersept = [ 0,  1.5,    0.8,    0.8,   60. ,  180. ,   30. ,   75. ,   22. ], resolution = 100):
    
    pdata = np.zeros((100,100))
    udata = np.zeros((100,100))
    
    col1_ax = cols_axis[col1]
    col2_ax = cols_axis[col2]
    #
    col1_loc = cols.index(col1)
    col2_loc = cols.index(col2)
    #
    xv, yv = np.meshgrid(col1_ax, col2_ax, sparse=False, indexing='xy')
    for i in xrange(100):
        for j in xrange(100):
            new_vec = np.copy(intersept)
            new_vec[col1_loc] = xv[i,j]
            new_vec[col2_loc] = yv[i,j]
            # Calculate the spin/mass surface for time = 0.00
            
            pdata[i][j], udata[i][j] = gp.predict(training_y, [new_vec])
    return pdata, udata, [col1_ax.min(), col1_ax.max(), col2_ax.min(), col2_ax.max()]

spacings = np.exp(gp.kernel.get_vector())**0.5
spacings = spacings[1:]
samp_cols_axis = {
    #"times"
    "q": np.arange(0+10%spacings[0]/2, 10, spacings[0]),
    "a1" : np.arange(0+1%spacings[1]/2, 1, spacings[1]),
    "a2": np.arange(0+1%spacings[2]/2, 1, spacings[2]),
    "th1L": np.arange(0+180%spacings[3]/2, 180, spacings[3]),
    "ph1": np.arange(-180+360%spacings[4]/2, 180, spacings[4]),
    "th12": np.arange(0+180%spacings[5]/2, 180, spacings[5]),
    "thSL": np.arange(0+180%spacings[6]/2, 180, spacings[6]),
    "thJL": np.arange(0+45%spacings[7]/2, 45, spacings[7])
}
def sample_grid(col1, col2):
    
    resolution = 100

    col1_ax = samp_cols_axis[col1]
    col2_ax = samp_cols_axis[col2]
    #
    col1_loc = cols.index(col1)
    col2_loc = cols.index(col2)
    #
    xv, yv = np.meshgrid(col1_ax, col2_ax, sparse=False, indexing='xy')
    return xv, yv
    

f, ax = plt.subplots(len(cols), len(cols), figsize = (20,20))
for i in range(len(cols)):
    for j in range(len(cols)):
        print i,j
        if j<i: 
            ax[j,i].axis('off')
            continue
        elif i == j:
            ax[j,i].axis("off")
            wv = training_x
            #pars = [   1.5,    0.8,    0.8,   60. ,  180. ,   30. ,   75. ,   22. ]
            pars = [  0,  1.5,    0.8,    0.8,   60. ,  180. ,   30. ,   75. ,   22. ]
            diffs = np.array(wv / wv.max()) - pars/np.array(wv.max())
            ax[j,i].hist2d(wv[:,i], np.sqrt((diffs**2).sum(axis=1)), bins=20, cmap='Greys');
            
        else:
            
            plt.setp(ax[j,i].get_xticklabels(), visible=False, rotation='vertical');
            plt.setp(ax[j,i].get_yticklabels(), visible=False, rotation='vertical');
            pdata, udata, extent = gen2plane(cols[i], cols[j])
            ax[j,i].imshow(udata, extent = extent, aspect = (extent[1] - extent[0]) / (extent[3] - extent[2]), origin='lower')
            ax[j,i].plot(pars[i], pars[j], 'o', c='red')
            if (cols[i] != "times") and (cols[j] != "times"):
                xv, yv = sample_grid(cols[i], cols[j])
                for a in xrange(xv.shape[0]):
                    for b in xrange(yv.shape[1]):
                        ax[j,i].plot(xv[a,b], yv[a,b], '+', c='white')
        plt.savefig("spacings.pdf")
for i,val in enumerate(cols):
    ax[-1,i].set_xlabel(val);
    plt.setp(ax[-1,i].get_xticklabels(), visible=True, rotation='vertical');
    ax[i, 0].set_ylabel(val);
    plt.setp(ax[i, 0].get_yticklabels(), visible=True)
    
plt.savefig("spacings.pdf")
