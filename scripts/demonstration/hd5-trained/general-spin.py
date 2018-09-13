import matplotlib.pyplot as plt
from george import kernels
import numpy as np
import elk.catalogue
from heron import data, regression, corner, priors, sampling

import otter
from otter import bootstrap as bt

report = otter.Otter(filename="nonspinning.html",
                     title="Heron non-spinning BBH Model",
                     subtitle="Trained off the GT HDF5 files."
)

root = "/home/daniel/data/gravitational-waves/heron/training/"
#data = np.genfromtxt(root+"training_data_GT_4096Hz_1MPc_MTOT20.dat")
#data_sxs = np.genfromtxt(root+"training_data_SXS_4096Hz_1MPc_MTOT20.dat")
#data_imr = np.genfromtxt(root+"training_data_IMR_GT_4096Hz_1MPc_MTOT20.dat")

catalogue = elk.catalogue(origin="GeorgiaTech")
non_spinning_catalogue = catalogue.spin_free()

data = catalogue.create_training_data(100, fmin=90).T


columns = {0:  "time",
           1:  "mass ratio",
           2:  "spin 1x",
           3:  "spin 1y",
           4:  "spin 1z",
           5:  "spin 2x",
           6:  "spin 2y",
           7:  "spin 2z",
           8: "h+",
           9: "hx"
}
c_ind = {j:i for i,j in columns.items()}


data[c_ind['time']] *= 10000

problem_dims = 2#8 #len(columns.keys())

with report:
    report += "# Training Data"

    report += """
Non-spinning waveforms only.
"""

with report:

    report += "## Mass ratio against time"

    report += "### Georgia Tech NR Waveforms"
    
    f, ax = plt.subplots(1,1)
    ax.scatter(data[c_ind['time']], data[c_ind['mass ratio']],
               s = 1,
               c = data[c_ind['h+']], cmap="RdBu",)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mass ratio")
    f.tight_layout()
    report + f


    report += "### SXS Waveforms"
    
    f, ax = plt.subplots(1,1)
    
    cax = f.add_axes([0.27, 0.95, 0.5, 0.05])
    sca = ax.scatter(data_sxs[c_ind['time']], data_sxs[c_ind['mass ratio']],
               s = 1,
               c = data_sxs[c_ind['h+']], cmap="RdBu",)
    
    f.colorbar(sca, cax=cax, orientation='horizontal')
    ax.set_xlabel("Time")
    ax.set_ylabel("Mass ratio")
    f.tight_layout()
    report + f
    
    report += "### IMRPhenomPv2 Waveforms"
    
    f, ax = plt.subplots(1,1)
    cax = f.add_axes([0.27, 0.95, 0.5, 0.05])

    sca = ax.scatter(data_imr[c_ind['time']], data_imr[c_ind['mass ratio']],
               s = 1,
               c = data_imr[c_ind['h+']], cmap="RdBu",)

    
    f.colorbar(sca, cax=cax, orientation='horizontal')
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Mass ratio")
    f.tight_layout()
    report + f


with report:
    report + "# Gaussian process"

with report:
    report + "## GT waveform data"
    
time_covariance = kernels.ExpSquaredKernel(50,
                                           ndim=problem_dims,
                                           axes=c_ind['time'],)
##kernels.RationalQuadraticKernel(0.5, 0.75,
mass_covariance = kernels.ExpSquaredKernel(0.5, #0.09
                                                  ndim=problem_dims,
                                           axes=c_ind['mass ratio'])
#[.003, 0.48, 0.48, 0.48, 0.48, 0.48]
#spin_covariance = kernels.ExpSquaredKernel([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], ndim=problem_dims, axes=[2,3,4,5,6,7])
#L_covariance = kernels.ExpSquaredKernel(.005, ndim=problem_dims, axes=[8])

covariance = 1.1 * mass_covariance * time_covariance  #* spin_covariance #* L_covariance


import george
gp = george.GP(covariance)#, solver = george.HODLRSolver, tol=1e-6)
yerr = np.ones(len(data.T))*1e-7


import scipy.optimize as op

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    print(np.exp(p))
    ll = gp.log_likelihood(data[c_ind['h+']]*1e19, quiet=True)
    print(-ll)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    print(np.exp(p))
    gp.set_parameter_vector(p)
    print(gp.log_likelihood(data[c_ind['h+']]*1e19, quiet=True))
    return -gp.grad_log_likelihood(data[c_ind['h+']]*1e19, quiet=True)

# You need to compute the GP once before starting the optimization.

gp.compute(data[:problem_dims].T, yerr)

# Print the initial ln-likelihood.
print(gp.log_likelihood(data[c_ind['h+']]*1e19))

# # # Run the optimization routine.
# p0 = gp.get_parameter_vector()

# import climin
# import climin.adam
# import cPickle
# opt = climin.Adadelta(p0, grad_nll, step_rate=0.1, momentum=0.9)

# for i in opt:
#     #with open('state.pkl', 'w') as fp:
#     #    cPickle.dump(i, fp)
#     if i['n_iter'] > 100: break
    
# results = op.minimize(nll, p0, jac=grad_nll, method="BFGS")

#Update the kernel and print the final log-likelihood.
#gp.set_parameter_vector(results.x)

# gp.set_parameter_vector(np.log([9.90155239e-08, 1.42572275e-03, 3.06422789e+01, 8.63975784e-04,
#                                 8.99052567e-02, 8.76127346e-02, 3.10985928e-02, 9.96816887e-02,
#                                 4.15039126e-03]))

print(gp.log_likelihood(data[c_ind['h+']]*1e19))


x = np.linspace(-50, 50, 100)
y = np.linspace(0.2, 1., 50)
gridpoints = np.meshgrid(x,y)
points = np.zeros((100*50, problem_dims))
#points[:,[2]] *= 1.0
points[:,[c_ind['time'],c_ind['mass ratio']]] = np.dstack(gridpoints).reshape(-1,2)

mean_gt, var_gt = gp.predict(data[c_ind['h+']]*1e19, points)

f, ax = plt.subplots(1,1)
im = ax.imshow(mean_gt.T.reshape(50,100), origin="lower",
               cmap = "magma",
               extent=(points[:,0].min(),
                       points[:,0].max(),
                       points[:,1].min(),
                       points[:,1].max()),
               aspect=50,
)
cax = f.add_axes([0.27, 0.8, 0.5, 0.05])
f.colorbar(im, cax=cax, orientation='horizontal')

#ax.set_ylim([0,5])
g, ax = plt.subplots(1,1)
im = ax.imshow(np.diag(var_gt).T.reshape(50,100), origin="lower",
          cmap = "magma",
          extent=(points[:,0].min(),
                  points[:,0].max(),
                  points[:,1].min(),
                  points[:,1].max()),
          aspect=50
)

with report:
    report += f
    report += g

# x = np.linspace(-50, 50, 100)
# y = np.linspace(-1, 1, 50)
# gridpoints = np.meshgrid(x,y)
# points = np.zeros((100*50, problem_dims))
# #points[:,[2]] *= 1.0
# points[:,[c_ind['time'],c_ind['spin 1x']]] = np.dstack(gridpoints).reshape(-1,2)

# points[:,c_ind['mass ratio']] = np.ones(50*100)

# mean_gt, var_gt = gp.predict(data[c_ind['h+']]*1e19, points)

# f, ax = plt.subplots(1,1)
# im = ax.imshow(mean_gt.T.reshape(50,100), origin="lower",
#                cmap = "magma_r",
#                extent=(points[:,0].min(),
#                        points[:,0].max(),
#                        points[:,c_ind['spin 1x']].min(),
#                        points[:,c_ind['spin 1x']].max()),
#                aspect=150,
# )
# cax = f.add_axes([0.27, 0.8, 0.5, 0.05])
# f.colorbar(im, cax=cax, orientation='horizontal')

# #ax.set_ylim([0,5])
# g, ax = plt.subplots(1,1)
# im = ax.imshow(np.diag(var_gt).T.reshape(50,100), origin="lower",
#           cmap = "magma_r",
#           extent=(points[:,0].min(),
#                   points[:,0].max(),
#                   points[:,c_ind['spin 1x']].min(),
#                   points[:,c_ind['spin 1x']].max()),
#           aspect=150
# )

# with report:
#     report += "###Spin 1x"
#     report += f
#     report += g
    
with report:
    report + "## Evaluated at IMR locations"
    points = data_imr[:problem_dims, ::2].T
    mean, var = gp.predict(data[c_ind['h+']], points)

    diff = (data_imr[c_ind['h+'],::2] - mean)**2/np.diag(var)
    print(diff)

    f, ax = plt.subplots(1,1)
    cax = f.add_axes([0.27, 0.8, 0.5, 0.05])
    sca = ax.scatter(data_imr[c_ind['time'],::2], data_imr[c_ind['mass ratio'],::2],
               s = 1,
               c = diff, cmap="viridis",)
    f.colorbar(sca, cax=cax, orientation='horizontal')
    ax.set_xlabel("Time")
    ax.set_ylabel("Mass ratio")

    report += f


with report:
    report + "## Waveforms at GT Locations"
    mean, var = gp.predict(data[c_ind['h+']], data[:problem_dims,:1000].T)

    f, ax = plt.subplots(1,1)
    ax.plot(mean, alpha=0.5)
    ax.plot(data[c_ind['h+'],:1000], alpha=0.5)
    #ax.plot(data_imr[c_ind['h+'],:1000]*1e19)
    ax.set_xlabel("Time")
    plt.close()

    report += f

with report:
    report + "## Waveforms at IMR Locations"
    mean, var = gp.predict(data[c_ind['h+']], data_imr[:problem_dims,:1000].T)

    f, ax = plt.subplots(1,1)
    ax.plot(mean, alpha=0.5)
    ax.plot(data_imr[c_ind['h+'],:1000], alpha=0.5)
    ax.set_ylim([-3e-19, 3e-19]);
    ax.set_xlabel("Time")
    plt.close()

    report += f
    
    

# with report:
#     report + "## Evaluated at SXS locations"
#     points = data_sxs[:problem_dims, ::2].T
#     mean, var = gp.predict(data[c_ind['h+']], points)

#     diff = (data_sxs[c_ind['h+'],::2] - mean)**2 / np.diag(var)
#     print(diff)
#     f, ax = plt.subplots(1,1)
#     cax = f.add_axes([0.27, 0.8, 0.5, 0.05])
#     sca = ax.scatter(data_sxs[c_ind['time'],::2], data_sxs[c_ind['mass ratio'],::2],
#                s = 1,
#                c = diff, cmap="viridis",)
#     f.colorbar(sca, cax=cax, orientation='horizontal')
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Mass ratio")

#     report += f
    
# with report:
#     report + "## SXS waveform data"


# import george
# gp = george.GP(covariance)#, solver = george.HODLRSolver, tol=0)
# yerr = np.ones(len(data_sxs.T))*1e-7

# print(np.mean(data_sxs[c_ind['h+']]*1e19))

# gp.compute(data_sxs[:problem_dims].T, yerr)


# x = np.linspace(-50, 50, 100)
# y = np.linspace(0.1,  0.9, 50)
# gridpoints = np.meshgrid(x,y)
# points = np.zeros((100*50, problem_dims))
# #points[:,[2]] *= 1.0
# points[:,[c_ind['time'],c_ind['mass ratio']]] = np.dstack(gridpoints).reshape(-1,2)

# mean_sxs, var_sxs = gp.predict(data_sxs[c_ind['h+']]*1e19, points)

# f, ax = plt.subplots(1,1)
# im = ax.imshow(mean_sxs.T.reshape(50,100), origin="lower",
#                cmap = "magma",
#                extent=(points[:,0].min(),
#                        points[:,0].max(),
#                        points[:,1].min(),
#                        points[:,1].max()),
#                aspect=50,
# )
# cax = f.add_axes([0.27, 0.8, 0.5, 0.05])
# f.colorbar(im, cax=cax, orientation='horizontal')

# #ax.set_ylim([0,5])
# g, ax = plt.subplots(1,1)
# im = ax.imshow(np.diag(var_sxs).T.reshape(50,100), origin="lower",
#           cmap = "magma",
#           extent=(points[:,0].min(),
#                   points[:,0].max(),
#                   points[:,1].min(),
#                   points[:,1].max()),
#           aspect=150
# )

# with report:
#     report += f
#     report += g

    

with report:
    report + "## IMR Data"

import george
gp = george.GP(covariance)#, solver = george.HODLRSolver)
yerr = np.ones(len(data_imr.T))*1e-7


gp.compute(data_imr[:problem_dims].T, yerr)


x = np.linspace(-50, 50, 100)
y = np.linspace(0.1, 0.9, 50)
gridpoints = np.meshgrid(x,y)
points = np.zeros((100*50, problem_dims))
#points[:,[2]] *= 1.0
points[:,[c_ind['time'],c_ind['mass ratio']]] = np.dstack(gridpoints).reshape(-1,2)

mean_imr, var_imr = gp.predict(data_imr[c_ind['h+']]*1e19, points)

f, ax = plt.subplots(1,1)
im = ax.imshow(mean_imr.T.reshape(50,100), origin="lower",
               cmap = "magma",
               extent=(points[:,0].min(),
                       points[:,0].max(),
                       points[:,1].min(),
                       points[:,1].max()),
               aspect=150,
)
cax = f.add_axes([0.27, 0.8, 0.5, 0.05])
f.colorbar(im, cax=cax, orientation='horizontal')

#ax.set_ylim([0,5])
g, ax = plt.subplots(1,1)
im = ax.imshow(np.diag(var_imr).T.reshape(50,100), origin="lower",
          cmap = "magma",
          extent=(points[:,0].min(),
                  points[:,0].max(),
                  points[:,1].min(),
                  points[:,1].max()),
          aspect=150
)

with report:
    report += f


# f, ax = plt.subplots(1,1)
# im = ax.imshow(mean_imr.T.reshape(50,100) - mean_sxs.T.reshape(50,100), origin="lower",
#                cmap = "magma",
#                extent=(points[:,0].min(),
#                        points[:,0].max(),
#                        points[:,1].min(),
#                        points[:,1].max()),
#                aspect=150,
# )
# cax = f.add_axes([0.27, 0.8, 0.5, 0.05])
# f.colorbar(im, cax=cax, orientation='horizontal')

# with report:
#     report + "# Difference"
#     report + "## IMRPhenomP and SXS"
#     report + f

f, ax = plt.subplots(1,1)
im = ax.imshow(mean_imr.T.reshape(50,100) - mean_gt.T.reshape(50,100), origin="lower",
               cmap = "magma",
               extent=(points[:,0].min(),
                       points[:,0].max(),
                       points[:,1].min(),
                       points[:,1].max()),
               aspect=150,
)
cax = f.add_axes([0.27, 0.8, 0.5, 0.05])
f.colorbar(im, cax=cax, orientation='horizontal')

with report:
    report + "# Difference"
    report + "## IMRPhenomP and GT"
    report + f
