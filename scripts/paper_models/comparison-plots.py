import matplotlib.pyplot as plt
plt.style.use("/home/daniel/thesis/thesis-style.mpl")

import astropy
from astropy.table import Table
from heron import data, regression, corner, priors, sampling
import sys
import heron
import numpy as np

import gtdata

codename = sys.argv[1].split(".")[0].split("/")[1].replace(" ", "-")
gp = regression.load(sys.argv[1])
training = gp.training_object

query = (    (gtdata.t["$a_{1x}$"]>=-100)  )

inspiral = 50
ringdown = 50

training_x, training_y, test_x, test_y, test_waveforms, test_pars, train_wave, train_pars, train_table, test_table = gtdata.get_dataset(gtdata.t, query = query, waveforms = 500, inspiral=50, ringdown=50, skip=25)

print train_table

def unfaithfulness(a,b):
    top = np.dot(a,b)
    aa = np.sqrt(np.dot(a,a))
    bb = np.sqrt(np.dot(b,b))

    return 1 - (top)/(aa * bb)

def pfaithfulness(a,b, var):
    top = np.sum((a * b)/var)
    aa = np.sqrt(np.dot(a,a))
    bb = np.sqrt(np.dot(b,b))

    return (top)/(aa * bb)

for params, waveform, row in zip(test_pars, test_waveforms, test_table):

    data = np.genfromtxt(waveform)
    times = data[:,0] = data[:,0] - data[np.argmax(data[:,1]),0]
    ix_selection = (times>=-inspiral) & (times<=ringdown)
    times = times[ix_selection]
    wave = data[:,1][ix_selection]

    intersept = training.normalise(params, "target")
    points = np.ones((len(times), training.targets.shape[1]))

    points *= intersept
    points[:,0] =  np.linspace(0,1,len(wave)) 

    prediction = gp.prediction(points, normalised = True)
    
    width = 3.487 #* 2
    height = width / 1.618
    
    f, ax = plt.subplots(1,1, sharex = True, figsize=(width, height));
    

    ax.plot(times, wave);

    np.savetxt("{}-prediction.dat".format(row['Name']), prediction[0])

    ax.plot(times, prediction[0], '--', color="#348ABD")
    ax.fill_between(times, prediction[0]-prediction[1], prediction[0]+prediction[1], alpha = 0.3, color="#348ABD")
    
    print row['Name'], unfaithfulness(wave, prediction[0]), pfaithfulness(wave, prediction[0], prediction[1]),

    ax.set_xlabel("Time [$(t-t_0) / M$]")
    ax.set_ylabel("Strain [$R/M$]")

    f.tight_layout()
    f.savefig("test_plots/{}/{}.pdf".format(codename, row['Name']))

    plt.close()

