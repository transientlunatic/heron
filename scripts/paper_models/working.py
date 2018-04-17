import matplotlib.pyplot as plt

import astropy
from astropy.table import Table
from heron import data, regression, corner, priors, sampling
import os
import heron
from glob import glob
import numpy as np

# Keep track of the various times that things happen at
ptimes = {}

headers = ['Index', 'Name', 'tag', '$q$', '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$', '$L_x$', '$L_y$', '$L_z$', 'mf', 'af', 'mW']
t = Table.read('/home/daniel/data/gravitational-waves/gt-new/training_waveforms.txt', format="ascii", names=headers)

columns = ['t', '$q$', '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$', '$L_z$']

def get_dataset(t, query, waveforms = 40, inspiral = 250, ringdown = 50, skip = 10):
    default_path = "/home/daniel/data/gravitational-waves/gt-old/"
    test_path = "/home/daniel/data-nosync/GW_Waveforms-master/Waveform_txtFiles/GT/"
    def find_data(tag, path = default_path):
        """
        Find the data files which contain the NR data for a given tag.
        """
        result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*{}*.asc'.format(tag)))]
        return result

    #
    t = t[query]
    inspiral = -1 * inspiral
    columns = ['t', '$q$', '$a_{1x}$', '$a_{1y}$', '$a_{1z}$', '$a_{2x}$', '$a_{2y}$', '$a_{2z}$']


    total_waveforms = waveforms

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
            print "{} missing.".format(row['Name'])
            print "It will be added to the test data."

            data = np.loadtxt(test_path+row['Name']+".txt")[::skip]
            
            hrss = np.sqrt(data[:,1]**2 + data[:,2]**2)

            data[:,0] = data[:,0] - data[np.argmax(data[:,1]),0]
            times = data[:,0]#[hrss.argmax()-200:hrss.argmax() + 50]
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
                    missing_y.append(data[:,2][ix_selection])
                    missing_x.append(np.atleast_2d(rowdata))

            
            continue
        waveform_table.append(j)
        waveformsinc += 1
        data = np.loadtxt(waveform_file[0])[::skip]

        hrss = np.sqrt(data[:,1]**2 + data[:,2]**2)

        data[:,0] = data[:,0] - data[np.argmax(hrss),0]
        times = data[:,0]#[hrss.argmax()-200:hrss.argmax() + 50]
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
    return training_x, training_y, missing_x, missing_y



from george import kernels
import george
import scipy.optimize


def neglk(x, gp, training):
    if np.any(x<0): return np.inf
    x = np.log(x)
    if not np.all(np.isfinite(x)): return np.inf
    gp.set_parameter_vector(x)
    return -gp.log_likelihood(training.labels, quiet=True)#,# -gp.grad_log_likelihood(training.labels, quiet=True)



def predict_plane(gp, training, i, j, res=(100, 100), intersept=None):
    if intersept==None:
        intersept = [0] * training.targets.shape[1]
        intersept = training.normalise(intersept, "target")
    else:
        intersept = training.normalise(intersept, "target")
        
    gridpoints = np.meshgrid(np.linspace(0,1,res[0]), np.linspace(0,1,res[1]))
    points = np.dstack(gridpoints).reshape(-1, 2)
    points2 = np.ones((points.shape[0], training.targets.shape[1]))
    points2 *= intersept
    
    points2[:,i] = points[:,0]
    points2[:,j] = points[:,1]
    
    prediction = gp.predict(training.labels, points2, return_var=True)
    
    return prediction[0].reshape(res), prediction[1].reshape(res), points2

def plot_plane(gp, training, i, j, intersept=None, figsize=(5,10)):
    plane, uplane, points = predict_plane(gp, training, i, j, intersept = intersept)

    f, ax = plt.subplots(1,2, figsize=figsize, sharey=True, sharex=True)
    xmin = training.denormalise(points[points[:,i].argmin()], "target")[i]
    xmax = training.denormalise(points[points[:,i].argmax()], "target")[i]

    ymin = training.denormalise(points[points[:,j].argmin()], "target")[j]
    ymax = training.denormalise(points[points[:,j].argmax()], "target")[j]

    aspect = np.abs((xmax-xmin)/(ymax-ymin))

    ax[0].imshow(plane, extent=(xmin,xmax,ymin,ymax), aspect=aspect, origin='lower')
    ax[1].imshow(uplane, extent=(xmin,xmax,ymin,ymax), aspect=aspect, origin='lower')

    

    ax[0].grid(color='w', linestyle='dotted', linewidth=1)
    ax[1].grid(color='w', linestyle='dotted', linewidth=1)

    
    ax[0].set_xlabel(columns[i])
    ax[1].set_xlabel(columns[i])
    ax[0].set_ylabel(columns[j])
    return f





#### The monster

query = (    (t["$a_{1x}$"]==0)
           & (t["$a_{1y}$"]==0)
           & (t["$a_{1z}$"]==0)
           & (t["$a_{2x}$"]==0)
             & (t["$a_{2y}$"]==0)
             & (t["$a_{2z}$"]==0)
)




training_x, training_y, test_x, test_y = get_dataset(t, query = query, waveforms = 49, inspiral=150, ringdown=50, skip=15)

print "Training data assembled. {} training points.".format(len(training_y))


training_spin_monster = heron.data.Data(targets=training_x.T, labels=np.array(training_y),
                           label_sigma = 0,
                          target_names = columns, label_names = ["h+"] )
print "Data object created."

k1 = kernels.Matern52Kernel(0.01, ndim=len(training_x.T[0]), axes=0)
k2 = kernels.Matern52Kernel(0.1, ndim=len(training_x.T[0]), axes=0)
k3 = kernels.ExpKernel((.2), ndim=len(training_x.T[0]), axes=(1))
k4 = kernels.ExpKernel((0.125, 0.125, 0.125), ndim=len(training_x.T[0]), axes=(2,3,4))
k5 = kernels.ExpKernel((0.125, 0.125, 0.125), ndim=len(training_x.T[0]), axes=(5,6,7))
kernel = 1.0 * ( 0.1 * k1 + 1.0 * k2) * (1.0* k3) # * (1*k4) * (1*k5)
gp_spin_monster = george.GP(kernel, mean = 0.5, solver=george.HODLRSolver, seed=1, tol=0.000000001, min_size=100) 
gp_spin_monster.compute(x=training_spin_monster.targets, yerr=training_spin_monster.label_sigma)

gp = gp_spin_monster
training = training_spin_monster

print "Model Created."

# f = plot_plane(gp_spin_monster, training_spin_monster, i=0, j=1, figsize=(10,10),
#            intersept=[0,0,0,0,0.0,0,0,0,]);

# f.savefig("monster_qplane_untrained.pdf")

intersept = [0, 3 , 0, 0,0,0,0,0]
points = np.ones((250, training.targets.shape[1]))

intersept = training.normalise(intersept, "target")

points *= intersept
points[:,0] = np.linspace(0, 1, 250)

prediction = gp.predict(training.labels, points, return_var=True)
f, ax = plt.subplots(1,1)
#xaxis = np.linspace(training.denormalise([0], "target"),training.denormalise([1], "target"), 250)
xaxis = np.linspace(-150,50,250)
ax.plot(xaxis, prediction[0]-0.5, label="Prediction")
ax.fill_between(xaxis, prediction[0]-0.5-prediction[1], prediction[0]-0.5+prediction[1], alpha = 0.5)


data = np.genfromtxt("/home/daniel/data-nosync/GW_Waveforms-master/Waveform_txtFiles/GT/GT0453.txt")

hrss = np.sqrt(data[:,1]**2 + data[:,2]**2)

data[:,0] = data[:,0] - data[np.argmax(data[:,1]),0]
      

ax.plot(data[:,0], -data[:,1], label="Test data")
ax.legend()

ax.set_xlim([-150,50])

f.savefig("test.png")

# print "Training."

# x = scipy.optimize.minimize(neglk, np.exp(gp.get_parameter_vector()), args=(gp, training_spin_monster), method = "L-BFGS-B")
# gp.set_parameter_vector(np.log(x.x))

# print x.x

# gp.compute(x=training.targets, yerr = training.label_sigma)

# f = plot_plane(gp_spin_monster, training_spin_monster, i=0, j=1, figsize=(10,10),
#            intersept=[0,0,0,0,0.0,0,0,0,]);

# f.savefig("monster_qplane.pdf")


