import matplotlib as mpl
mpl.use('Agg')

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("/home/daniel/papers/thesis/thesis-style.mpl")

with open(sys.argv[1]) as data:
    gp, training_y = pickle.load(data)


times = np.linspace(-0.1,0.05,100)
ratios = np.linspace(0.07, 0.7, 100)
spins = np.linspace(0, 0.01, 100)
pdata = np.zeros((100,100))
udata = np.zeros((100,100))
#xv, yv = np.meshgrid(times, ratios, sparse=False, indexing='ij')
#for i in xrange(100):
#    for j in xrange(100):
#        # Calculate the time/mass surface for spin = 0.
#        pdata[i][j], udata[i][j] = gp.prediction([[xv[i,j], yv[i,j], 0.0037]])

#np.save("heron3-masstime-prediction.dat", pdata)
#np.save("heron3-masstime-uncertainty.dat", udata)

#xv, yv = np.meshgrid(times, spins, sparse=False, indexing='ij')
#for i in xrange(100):
#    for j in xrange(100):
#        # Calculate the time/mass surface for ratio = 0.2
#        pdata[i][j], udata[i][j] = gp.prediction([[xv[i,j], 0.2, yv[i,j]]])

#np.save("heron3-spintime-prediction.dat", pdata)
#np.save("heron3-spintime-uncertainty.dat", udata)



xv, yv = np.meshgrid(ratios, spins, sparse=False, indexing='ij')
for i in xrange(100):
    for j in xrange(100):
        # Calculate the spin/mass surface for time = 0.00
        pdata[i][j], udata[i][j] = gp.predict(training_y, [0.00, xv[i,j], yv[i,j]])

np.save("heron3-spinmass-prediction.dat", pdata)
np.save("heron3-spinmass-uncertainty.dat", udata)



# plt.figure(figsize=(10,10))
# plt.imshow(pdata, extent=(0.07,0.7, -0.1, 0.05), interpolation="none", aspect=5, origin='lower')
# plt.colorbar()

# plt.savefig("heron2-modelsurface.png")

# plt.figure(figsize=(10,10))
# plt.imshow(udata, extent=(0.07,0.7, -0.1, 0.05), interpolation="none", aspect=5, origin='lower')
# plt.colorbar()

# plt.savefig("heron2-uncertaintysurface.png")
