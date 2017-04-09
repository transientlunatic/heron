"""This script trains the full HERON model on a different number of
numerical relativity simulations in order to verify that the
hyperparameters converge. All relativity parameters are trained using
the numerical relativity data from Georgia Tech.

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

number_waveforms = [10, 20, 50, 100, 200, 400]

SKIP = 5
import numpy as np
training_x = []
training_y = []
included_waveforms = 0
for number_waveform in number_waveforms:
    for waveform in cat.waveforms[:number_waveform][cols].iterrows():
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
            # Acknowledge that this was added
            included_waveforms += 1
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
    def neg_ln_like(p):
        gp.set_vector(p)
        return -gp.lnlikelihood(training_y)

    minimize(neg_ln_like, gp.get_vector(), method="BFGS")

    print included_waveforms, gp.get_vector()


    import pickle
    with open("full-{}.gp".format(included_waveforms), "wb") as savefile:
        pickle.dump([gp, training_y], savefile)
