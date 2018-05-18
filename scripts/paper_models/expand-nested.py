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


from awesome_codename import generate_codename

import time

plt.style.use("/home/daniel/papers/thesis/thesis-style.mpl")

columns = ['t', '$q$', '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$', '$L_x$', '$L_y$', '$L_z$']

def get_dataset(t, query, waveforms = 40, inspiral = 250, ringdown = 50, skip = 10,
                data_path  = "/home/daniel/data/gravitational-waves/gt-old/",
                test_path = "/home/daniel/data-nosync/GW_Waveforms-master/Waveform_txtFiles/GT/"
):
    def find_data(tag, path = data_path):
        """
        Find the data files which contain the NR data for a given tag.
        """
        result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*{}*.asc'.format(tag)))]
        return result

    #
    t = t[query]
    inspiral = -1 * inspiral
    columns = ['t', '$q$', '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$', '$L_x$', '$L_y$', '$L_z$']


    total_waveforms = waveforms
    test_waveforms = []
    test_parameters = []
    training_x = []
    training_y = []
    missing_x = []
    missing_y = []
    waveformsinc = 0
    waveform_table = []
    for j,row in enumerate(t):
        if waveformsinc >= total_waveforms:  break
        waveform_file = find_data(row['tag'])
        #print waveform_file
        #print waveform_file
        if len(waveform_file)!=1:
            #print "{} missing.".format(row['Name'])
            

            try:
                data = np.loadtxt(test_path+row['Name']+".txt")[::skip]
                #print "It will be added to the test data."
                test_waveforms.append(row['Name'])
                rowdata = np.zeros(len(columns))
                for i, col in enumerate(columns[1:]):
                    rowdata[i+1] = row[col]
                #print(rowdata)
                test_parameters.append(rowdata)
            except IOError:
                continue
            
            hrss = np.sqrt(data[:,1]**2 + data[:,2]**2)

            data[:,0] = data[:,0] - data[np.argmax(data[:,2]),0]
            times = data[:,0]
            if len(times)==0:
                #print "{} missing.".format(row['Name'])
                continue
        
            ix_selection = (times>=inspiral) & (times<=ringdown)
            times = times[ix_selection]
        
        
            rowdata = np.zeros((len(columns), len(times)))
            for i, col in enumerate(columns):
                if i == 0: 
                    rowdata[i,:] = data[:,0][ix_selection]
                else:
                    rowdata[i,:] = np.tile(row[col], len(times))
                    missing_y.append(data[:,2][ix_selection])
                    missing_x.append(np.atleast_2d(rowdata))

            
            continue
        waveform_table.append(j)
        waveformsinc += 1
        data = np.loadtxt(waveform_file[0])[::skip]
        #try:
        #    data = np.loadtxt(test_path+row['Name']+".txt")[::skip]
        #except IOError:
        #    print "-- {}".format(row['Name'])
        #    continue
        hrss = np.sqrt(data[:,1]**2 + data[:,2]**2)

        data[:,0] = data[:,0] - data[np.argmax(data[:,2]),0]
        times = data[:,0]
        if len(times)==0:
            print "{} missing.".format(row['Name'])
            continue
        
        ix_selection = (times>=inspiral) & (times<=ringdown)
        times = times[ix_selection]
       
        
        rowdata = np.zeros((len(columns), len(times)))
        for i, col in enumerate(columns):
            if i == 0: 
                rowdata[i,:] = data[:,0][ix_selection]
            else:
                rowdata[i,:] = np.tile(row[col], len(times))
        training_y.append(data[:,2][ix_selection])
        training_x.append(np.atleast_2d(rowdata))
    training_y = np.hstack(training_y)
    training_x = np.hstack(training_x)
    return training_x, training_y, missing_x, missing_y, test_waveforms, test_parameters, waveform_table





#

#
# Manage the command-line options which are supplied to the script
#

import click

@click.command()
@click.option('--waveforms', default=1, help = "The number of waveforms to include in the model's training data.")
@click.option('--optimizer', default="nestle", help = "The optimisation strategy to be used.")
@click.option('--livepoints', default=100, help="Number of nested sampling live points.")
@click.option('--label', default=None, help = "A specific label for this model.")
@click.option('--maxiter', default = 10000000, help = "The maximum number of nested iterations.")
@click.option('--data', default="/home/daniel/data/gravitational-waves/gt-test/",
              help="A directory containing the waveforms which should be loaded to form the model.")
@click.option('--inspiral', default=250,
              help = "The number of time units of inspiral to be included from each waveform")
@click.option('--ringdown', default=50,
              help = "The number of time units of ringdown to be included from each waveform")

def main(waveforms, optimizer, livepoints, label, maxiter, data, inspiral, ringdown):

    if not label:
        label = generate_codename()
    print "HERON Model {}".format(label)

    report = otter.Otter("/home/daniel/www/reports/heron/expansion/{}.html".format(label),
                     "/home/daniel/.otter",
                     author="Daniel Williams",
                     title="Test report for Heron BBH",
                     subtitle="{}".format(label)
    )
    
    ptimes = {}

    headers = ['Index', 'Name', 'tag', '$q$',
               '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$',
               '$L_x$', '$L_y$', '$L_z$', 'mf', 'af', 'mW']
    t = Table.read('/home/daniel/data/gravitational-waves/gt-new/training_waveforms.txt',
                   format="ascii", names=headers)
    
    query = (    (t["$a_{1x}$"]>=-100) )
    
    training_x, training_y, test_x, test_y, test_waveforms, test_pars, waveform_table = get_dataset(t, query = query, waveforms = 490, inspiral=inspiral, ringdown=ringdown, skip=20)
    

    export_table = tabulate.tabulate(t.to_pandas().ix[waveform_table], headers = t.to_pandas( ).columns.values, tablefmt="pipe")

    print(export_table)

    table_row = bt.Row()
    table_row + "## Training waveforms"
    table_row + "Model contains {} waveforms".format(waveforms)
    table_row + "<pre>{}</pre>".format(export_table)
    report + table_row

    np.savetxt("data/GT_training_x_{}.txt".format(label), training_x)
    np.savetxt("data/GT_training_y_{}.txt".format(label), training_y)

    data = heron.data.Data(
        training_x.T, #:4
        training_y,
        label_sigma = 0,
        test_size=0,
        target_names = columns, 
        label_names = ["h+"])


    model_row = bt.Row()


    sep = data.get_starting() + 0.01
    #print(sep)

    #model_row + "Rough estimate of initial values for hyperparameters: {}\n".format(sep**2)
    #model_row + "Data standard deviation squared: {}\n".format(np.std(data.labels[0,:])**2)
    #hyper_priors = [priors.Normal(25, 5)]
    #for hyper in sep:
    #    hyper_priors.append( priors.Normal(hyper**2, 5)  )
    report + model_row

    #kernel = k4


    k1 = kernels.Matern52Kernel(0.001, ndim=len(columns), axes=0)
    k2 = kernels.Matern52Kernel(0.05, ndim=len(columns), axes=0)
    k_massr = kernels.ExpKernel((.15), ndim=len(columns), axes=(1))
    k_spinx = kernels.ExpKernel((0.125, 0.125, 0.125), ndim=len(columns), axes=(2,3,4))
    k_spiny = kernels.ExpKernel((0.125, 0.125, 0.125), ndim=len(columns), axes=(5,6,7))
    kL = kernels.ExpKernel((.01, .01, .01), ndim=len(columns), axes=(8,9,10))
    kernel = 3.5 * k2 * (1.0* k_massr) * (1 * kL) * (1*k_spinx) * (1*k_spiny)


    
    import george

    ptimes['build start'] = time.time()
    gp = regression.SingleTaskGP(data, kernel = kernel,
                                 tikh=0.000000001,
                                 )
    ptimes['build end'] = time.time()


    ptimes['optim start'] = time.time()
    if optimizer == "nestle":

        import nestle
        from scipy.special import ndtri
        
        def prior_transform(x):
            return 1 * x - .5# + sep2

        ndim = len(gp.gp.get_parameter_vector())
        nest = nestle.sample(gp.neg_ln_likelihood,
                             prior_transform,
                             ndim,
                             method='multi',
                             callback=nestle.print_progress,
                             maxiter=maxiter,
                             npoints=livepoints)
                            #decline_factor = 0.5)
        nest_max = np.argmax(nest['logl'])
        gp.gp.set_parameter_vector(nest['samples'][nest_max])


        
    ptimes['optim end'] = time.time()
    
    import corner
    fig = corner.corner(nest.samples, weights=nest.weights, truths = nest['samples'][nest_max])
    fig.set_size_inches(8., 8.)


    report_row = bt.Row()
    report_row + "##Performance Information"
    report_row + "Model build time: {}s".format(ptimes['build end'] - ptimes['build start'])
    report_row + "Model training time: {}s".format(ptimes['optim end'] - ptimes['optim start'])
    report_row + "## Covariance information"
    report_row + "HERON Monster Kernel"
    report_row + "##Training Information"
    report_row + "Optimisation strategy: {}".format(optimizer)
    report_row + "Nested sampling live points: {}".format(livepoints)
    report_row + "Nested sampling iter-limit: {}".format(maxiter)
    report_row + "Trained vector: {}".format(gp.gp.get_parameter_vector())
    report_row + "Training iterations: {}".format(nest['niter'])
    report_row + "log(z): {}".format(nest['logz'])
    report_row + "max(l): {}".format(np.max(nest['logl']))
    report + report_row

    report_row = bt.Row()
    dataplot = op.Figure(report, fig)
    report_row + dataplot
    report + report_row

    gp.save("models/{}.gp".format(label))
    #---

    pdata, udata = gp.prediction(training_x[:,:5000].T)
    f,ax = plt.subplots(2,1, sharey=True, sharex=True)
    ax[1].plot(pdata, label="Prediction")
    ax[1].legend()
    ax[0].plot(training_y[:5000], label="Training data")
    ax[0].legend()


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
    f2, ax2 = plt.subplots(len(cols), len(cols), figsize = (13,13))
    for i in range(0,len(cols)):
        for j in range(0,len(cols)):
            #print "Producing plot at {} {}".format(i, j)
            if j<i: 
                ax[j,i].axis('off')
                ax2[j,i].axis('off')
                continue
            elif i == j:
                ax[j,i].axis("off")
                ax2[j,i].axis("off")
                #plt.setp(ax[j,i].get5H_yticklabels(), visible=True)
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
                ax2[j,i].imshow(pdata.reshape(res,res), extent = extent, aspect = (extent[1] - extent[0]) / (extent[3] - extent[2]), origin='lower', cmap="magma_r", alpha=0.8)
                #ax[j,i].plot(pars[i], pars[j], 'o', c='red')

    for i,val in enumerate(cols):
        ax[-1,i].set_xlabel(val);
        plt.setp(ax[-1,i].get_xticklabels(), visible=True, rotation='vertical');
        ax[i, 0].set_ylabel(val);
        plt.setp(ax[i, 0].get_yticklabels(), visible=True)

    report_row = bt.Row()

    report_row + "## Model value plot"
    dataplot = op.Figure(report, f2)

    report_row + dataplot
    report + report_row

    report_row = bt.Row()

    report_row + "## Model uncertainty plot"
    dataplot = op.Figure(report, f)

    report_row + dataplot
    report + report_row

    report_row = bt.Row()

    from heron import corner

    f = corner.corner(data)

    trainingplot = op.Figure(report, f)

    report_row + "## Training data density plot"
    report_row + trainingplot
    report + report_row

    report.show()

if __name__ == "__main__":
    main()

