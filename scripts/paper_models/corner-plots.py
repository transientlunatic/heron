import matplotlib.pyplot as plt
plt.style.use("/home/daniel/thesis/thesis-style.mpl")

import astropy
from astropy.table import Table
from heron import data, regression, corner, priors, sampling
import sys
import heron
import numpy as np

import gtdata, planeplot


gp = regression.load(sys.argv[1])
training = gp.training_object

width = 3.487 #* 2
height = width / 1.618


#print training.targets[0]

print training.targets[0]
intersept = training.normalise([1.45, 0, 0, 0, 0, 0, 0, 0], "targets")
print intersept
f = planeplot.plot_plane(gp, training, 0, 1, intersept, figsize=(width, height))


f.tight_layout()
f.savefig("{}-time-mass.pdf".format(sys.argv[1].split(".")[0]))
