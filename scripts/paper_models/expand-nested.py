"""This is a script which is designed to incrementally extend the
Gaussian process to determine what the largest model we can reasonably
run tests on.

"""

import otter
import otter.bootstrap as bt
import otter.plot as op


import astropy
from astropy.table import Table
import os
from glob import glob
import numpy as np
import george
from george import kernels
import emcee
#import corner

from heron import data, regression, corner, priors, sampling
from heron import data
import heron

import matplotlib.pyplot as plt
import tabulate 
import sys

#

headers = ['Index', 'Name', 'tag', '$q$', '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$', '$L_x$', '$L_y$', '$L_z$', 'mf', 'af', 'mW']
t = Table.read('/home/daniel/data/gravitational-waves/gt-new/GT_CATALOG_TABLE.txt', format="ascii", names=headers)

def find_data(tag, path = "/home/daniel/data/gravitational-waves/gt-test/"):
    """
    Find the data files which contain the NR data for a given tag.
    """
    result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*{}*.asc'.format(tag)))]
    return result

#

columns = ['t', '$q$', '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$']


total_waveforms = int(sys.argv[1])

report = otter.Otter("/home/daniel/www/reports/heron/expansion/{}.html".format(total_waveforms),
                     "/home/daniel/.otter",
                     author="Daniel Williams",
                     title="Test report for Heron BBH",
                     subtitle="{} Waveforms".format(total_waveforms)
)

#

training_x = []
training_y = []
waveformsinc = 0
waveform_table = []
for j,row in enumerate(t.to_pandas().iterrows()):
    if waveformsinc >= total_waveforms:
        break

    
    waveform_file = find_data(row[1]['tag'])
    print waveform_file
    if len(waveform_file)!=1:
        continue
    waveform_table.append(j)
    waveformsinc += 1
    data = np.loadtxt(waveform_file[0])[::5]
    
    hrss = np.sqrt(data[:,1]**2 + data[:,2]**2)
    
    data[:,0] = data[:,0] - data[np.argmax(hrss),0]
    times = data[:,0][hrss.argmax()-500:hrss.argmax() + 100]
    # Use ~250 times from each waveform
    if len(times)==0: continue
    rowdata = np.zeros((len(columns), len(times)))
    for i, col in enumerate(columns):
        if i == 0: 
            rowdata[i,:] = data[:,0][hrss.argmax()-500:hrss.argmax() + 100]
        else:
            rowdata[i,:] = np.tile(row[1][col], len(times))
    training_y.append(data[:,2][hrss.argmax() - 500:hrss.argmax() + 100])
    training_x.append(np.atleast_2d(rowdata))

table_row = bt.Row()
table_row + tabulate.tabulate(t.to_pandas().ix[waveform_table], tablefmt="simple")
report + table_row
training_y = np.hstack(training_y)
training_x = np.hstack(training_x)


#np.savetxt("data/GT_training_x_{}.txt".format(j), training_x)
#np.savetxt("data/GT_training_y_{}.txt".format(j), training_y)


training_data_x = training_x # np.loadtxt("/home/daniel/data/heron/GT_training_x.txt")
training_data_y = training_y #np.loadtxt("/home/daniel/data/heron/GT_training_y.txt")

data = heron.data.Data(
    training_data_x.T, #:4
    training_data_y,
    label_sigma = 5e-5,
    test_size=0,
    target_names = columns, 
    label_names = ["h+"])

#print("Label dimensions: {}".format(data.labels.ndim))

# report_row = bt.Row()

#data_density = corner.corner(data)

#cornerplot = op.Figure(report, data_density)

#report_row + cornerplot
#report + report_row

model_row = bt.Row()


sep = data.get_starting() + 0.01
#print(sep)

#model_row + "Rough estimate of initial values for hyperparameters: {}\n".format(sep**2)
#model_row + "Data standard deviation squared: {}\n".format(np.std(data.labels[0,:])**2)
hyper_priors = [priors.Normal(1.0/hyper**2, 2) for hyper in sep]
k3 = kernels.Matern52Kernel(sep**2, ndim=len(sep))

report + model_row

kernel = k3
import george
gp = regression.SingleTaskGP(data, kernel = kernel, hyperpriors = hyper_priors)#, solver=george.BasicSolver)

#data_density.savefig("data/GT_cornerplot.pdf")

#gp.save("models/GT_{}_untrained.gp".format(total_waveforms))
# MAP = gp.train("MAP")
# gp.save("GT_{}_MAP.gp".format(j))
# training.train_cv(gp)
# gp.save("models/GT_{}_CV.gp".format(j))
#

# Some code to make nice plots of the GPR model


#MAP = gp.train("MAP", basinhoppping=True)

#---

print("Starting sep: {}".format(sep))

import nestle
from scipy.special import ndtri
def prior_transform(x): 
    #scales =  np.array([25, 6, 4, 1, 1, 1, 1, 1, 1])
    #offsets = np.array([20, 3, 2, 0, 0, 0, 0, 0, 0])
    #scales = np.array([2, 20, 2, 2, 2, 2, 2, 2])
    #offsets = np.array([ 9, 10, 1, 1, 1, 1, 1, 1])
    sigma = 2
    return sep + sigma * ndtri(x)
    
    #return scales*x - offsets
ndim = len(gp.gp.get_vector())
nest = nestle.sample(gp.ln_likelihood, prior_transform, ndim, method='multi', npoints=500)
nest_max = np.argmin(nest['logl'])
gp.gp.set_vector(nest['samples'][nest_max])
print("Trained vector: {}".format(gp.gp.get_vector()))
import corner
fig = corner.corner(nest.samples, weights=-nest.weights)
fig.set_size_inches(8., 8.)


report_row = bt.Row()
dataplot = op.Figure(report, fig)
report_row + dataplot
report + report_row


#nest = gp.train("nested")
print("Nested results: {}".format(nest))
#gp.save("models/GT_{}_nested.gp".format(j))
#---

pdata, udata = gp.prediction(training_data_x[:,:1000].T)
f,ax = plt.subplots(1,1)
ax.plot(pdata)

ax.plot(training_data_y[:1000])

#MAP_row = bt.Row()
#MAP_row + "# MAP Estimate\n"
#MAP_row + MAP
#report + MAP_row

report_row = bt.Row()
dataplot = op.Figure(report, f)
report_row + dataplot
report + report_row



def gen2plane(col1, col2, data, intersept = None, resolution = 100):
    vals = data.targets[0] #np.random.choice(data.targets, size = (1, data.targets.shape[1]))

    intersept = dict(zip(columns, vals))

    #pdata = np.zeros((100,100))
    #udata = np.zeros((100,100))
    res = resolution
    col1_ax = np.linspace(-1,1, res)#cols_axis[col1]
    col2_ax =  np.linspace(-1,1, res)#cols_axis[col2]
    #
    col1_loc = cols.index(col1)
    col2_loc = cols.index(col2)
    #

    loc_dict = {col1: [-1, 1, res], col2: [-1, 1, res]}

    for col in cols:
        
        if col in loc_dict.keys():
            continue
        else:
            loc_dict[col] = intersept[col]

    samplelocs = sampling.draw_samples(gp, **loc_dict)
    pdata, udata = gp.prediction(samplelocs, normalised=True)
    return pdata, udata, [col1_ax.min(), col1_ax.max(), col2_ax.min(), col2_ax.max()]


cols = columns
f, ax = plt.subplots(len(cols), len(cols), figsize = (13,13))
for i in range(0,len(cols)):
    for j in range(0,len(cols)):
        #print "Producing plot at {} {}".format(i, j)
        if j<i: 
            ax[j,i].axis('off')
            continue
        elif i == j:
            ax[j,i].axis("off")
            #plt.setp(ax[j,i].get_yticklabels(), visible=True)
            #plt.setp(ax[j,i].get_xticklabels(), visible=False)
            wv = np.array(training_data_x)
            #pars = [  0,  1.5,    0.8,    0.8,   60. ,  180. ,   30. ,   75. ,   22. ]
            pars = training_data_x.T[:,0]
            diffs = np.array(wv / wv.max()) - pars/np.array(wv.max())
            #ax[j,i].hist2d(wv[:,i], np.sqrt((diffs**2).sum(axis=1)), bins=20, cmap='Greys');
            
        else:
            
            plt.setp(ax[j,i].get_xticklabels(), visible=False, rotation='vertical');
            plt.setp(ax[j,i].get_yticklabels(), visible=False, rotation='vertical');
            pdata, udata, extent = gen2plane(cols[i], cols[j], data,  pars)
            res = int(np.sqrt(len(udata)))
            #print "res: {}".format(res)
            #extent = [0,1,0,1]
            ax[j,i].imshow(udata.reshape(res,res), extent = extent, aspect = (extent[1] - extent[0]) / (extent[3] - extent[2]), origin='lower', cmap="magma_r", alpha=0.8)
            #ax[j,i].plot(pars[i], pars[j], 'o', c='red')
        
for i,val in enumerate(cols):
    ax[-1,i].set_xlabel(val);
    plt.setp(ax[-1,i].get_xticklabels(), visible=True, rotation='vertical');
    ax[i, 0].set_ylabel(val);
    plt.setp(ax[i, 0].get_yticklabels(), visible=True)


report_row = bt.Row()

dataplot = op.Figure(report, f)

report_row + dataplot
report + report_row

report.show()
