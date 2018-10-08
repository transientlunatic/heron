import matplotlib.pyplot as plt
from george import kernels
import numpy as np
import elk.catalogue
from heron import data, regression, corner, priors, sampling
from heron import waveform
import otter
from otter import bootstrap as bt

from collections import OrderedDict

import yaml

with open("general.yaml") as f:
    config = yaml.safe_load(f)

print(config["report"]["title"])

report = otter.Otter(filename=config["report"]["path"],
                     title=config["report"]["title"],
                     subtitle=config["report"]["subtitle"]
)

catalogue = elk.catalogue.NRCatalogue(origin=config["training_data"]["catalogue"])
#catalogue = catalogue.query("spin_1x == 0 & spin_1y == 0 & spin_2x == 0 & spin_2y ==0")
catalogue_test = elk.catalogue.NRCatalogue(origin=config["test_data"]["catalogue"])
# catalogue_test = catalogue_test.spin_free()

c_ind = catalogue.c_ind

total_mass = float(config["training_data"]["total mass"])

problem_dims = 8
# kernels_list = []


time_covariance = kernels.RationalQuadraticKernel(.05, 400,
#time_covariance = kernels.ExpSquaredKernel(450,
                                           ndim=problem_dims,
                                           axes=c_ind['time'],)

#mass_covariance = kernels.RationalQuadraticKernel(0.5, 0.5,
mass_covariance = kernels.ExpSquaredKernel(0.005, #0.09
                                           ndim=problem_dims,
                                           axes=c_ind['mass ratio'])
#[.003, 0.48, 0.48, 0.48, 0.48, 0.48]
spin_covariance = kernels.ExpSquaredKernel([0.005, 0.005, 0.005, 0.005, 0.005, 0.005], ndim=problem_dims, axes=[2,3,4,5,6,7])

covariance =  1e1 * mass_covariance * time_covariance  * spin_covariance #* L_covariance

gp = gp_cat = waveform.GPCatalogue(catalogue, covariance,
                                   total_mass=total_mass, fsample=4*1024,
                                   solver=None,
                                   mean=0.0,
                                   white_noise=1e-1,
                                   fmin=95,)
imr_cat = elk.catalogue.PPCatalogue("IMRPhenomPv2", total_mass=total_mass)

#gp_cat.optimise(max_iter=1000)
# So these are the parameters which are *easiest* to find, which give a really boring and
# fairly unpredictive model.
# gp_cat.gp.set_parameter_vector(np.log([2.07429046e+00, 5.76056375e-05, 1.49288141e+00, 2.26176799e+02,
#                                        5.75947665e-01, 3.17516612e1, 7.22846480e-01, 1.05775480e-01,
#                                        1.39420724e-01, 1.57279701e-01]
# ))

#gp_cat.gp.set_parameter_vector(np.log([1.43070199e+00,
#                                       4.53999298e-04,
#                                       2.20264658e+00, 4.53999298e+02,
#                                       1e-4,1e-4,1e-4,#2.05790712e-04, 4.11457123e-04, 1.21732870e-04,
#                                       1e-4,1e-4,1e-4#1.61567205e-04, 1.69408025e-04, 4.53999298e-05
#]))

# After 1000 training iterations with ADMA
gp_cat.gp.set_parameter_vector(np.log([1.72104338e+00,
                                       3.75605402e-04,
                                       1.85927706e+00, 3.79218372e+02,
                                       8.53034888e-05, 8.48911219e-05, 8.38426260e-05,
                                       8.53171760e-05, 8.48904881e-05, 8.24766499e-05,]))

#gp_cat.optimise("adam", max_iter=1000)



with report:
    report += "# Training Data"

    report += """
    Non-spinning waveforms only.
    """

with report:
    report += "## Data Coverage"
    report += catalogue.coverage_plot()

with report:

    report += "## Mass ratio against time"

    report += "### Georgia Tech NR Waveforms"
    
    f, ax = plt.subplots(1,1)
    ax.scatter(gp.training_data[:,c_ind['time']], gp.training_data[:,c_ind['mass ratio']],
               s = 1,
               c = gp.training_data[:,c_ind['h+']], cmap="RdBu",)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mass ratio")
    f.tight_layout()
    report + f

    

with report:
    report + "# Gaussian process"

with report:
    report + "## GT waveform data"


mean_gt, var_gt = gp_cat.mean({"time": [-150, 50, 200], "mass ratio": [0.2, 1.0, 50]},
                         fixed = {"spin 1x": 0,
                                  "spin 1y": 0,
                                  "spin 1z": 0,
                                  "spin 2x": 0,
                                  "spin 2y": 0,
                                  "spin 2z": 0,})

f, ax = plt.subplots(1,1)
im = ax.imshow(mean_gt, origin="lower", cmap = "magma", vmin=-3, vmax=3,
               extent = (-150, 50, 0.2, 1.0),
               aspect = (200 / 0.8))
cax = f.add_axes([0.9, 0.1, 0.02, 0.8])
f.colorbar(im, cax=cax, orientation='vertical')

ax.set_xlabel("Time [s * 1e4]")
ax.set_ylabel("Mass Ratio")

g, ax = plt.subplots(1,1)
im = ax.imshow(var_gt, origin="lower",
               cmap = "magma",
               extent = (-150, 50, 0.2, 1.0),
               aspect = (200 / 0.8))

cax = g.add_axes([0.8, 0.1, 0.02, 0.8])
g.colorbar(im, cax=cax, orientation='vertical')

ax.set_xlabel("Time [s * 1e4]")
ax.set_ylabel("Mass Ratio")

with report:
    report += f
    report += g


with report:
    report += "## GPR surrogate comparisons"

    report += "### GT Comparisons"

for waveform in catalogue.waveforms:
    f, ax = plt.subplots(1,1)
    test_wave = waveform

    with report:
        report += "Waveform {}".format(waveform.tag)
        report += "Mass ratio: {}".format(waveform.mass_ratio)
        report += "Spin: {}".format(waveform.spins)
    
    samples = gp_cat.waveform_samples(p={"mass ratio": test_wave.mass_ratio,
                                         "spin 1x": test_wave.spin_1x,
                                         "spin 1y": test_wave.spin_1y,
                                         "spin 1z": test_wave.spin_1z,
                                         "spin 2x": test_wave.spin_2x,
                                         "spin 2y": test_wave.spin_2y,
                                         "spin 2z": test_wave.spin_2z,},
                                      time_range=[-150, 100, 400])
    ax.plot(np.linspace(-15, 10, 400), samples.T/1e19, alpha=0.01, color='k', linewidth=1)

    try:
        hp, hx = test_wave.timeseries(total_mass=total_mass,
                                      sample_rate=4096,
                                      flow=93.0,
                                      distance=1)


        ax.plot(hp.times*1e3, -hp.data);
        ax.plot(hp.times*1e3,  hp.data);
    except:
        pass
    
with report:
    report += "SXS Comparisons"
        
for waveform in catalogue_test.waveforms:
    f, ax = plt.subplots(1,1)
    test_wave = waveform

    with report:
        report += "Waveform {}".format(waveform.tag)
        report += "Mass ratio: {}".format(waveform.mass_ratio)
        report += "Spin: {}".format(waveform.spins)
    
    samples = gp_cat.waveform_samples(p={"mass ratio": test_wave.mass_ratio,
                                         "spin 1x": test_wave.spin_1x,
                                         "spin 1y": test_wave.spin_1y,
                                         "spin 1z": test_wave.spin_1z,
                                         "spin 2x": test_wave.spin_2x,
                                         "spin 2y": test_wave.spin_2y,
                                         "spin 2z": test_wave.spin_2z,},
                                      time_range=[-150, 100, 400])
    ax.plot(np.linspace(-15, 10, 400), samples.T/1e19, alpha=0.01, color='k', linewidth=1)

    #nearest_nr = catalogue.find_closest([0.5,0,0,0,0,0,0])

    # imr = imr_cat.waveform(p={"mass ratio": test_wave.mass_ratio,
    #                           "spin 1x": test_wave.spin_1x,
    #                           "spin 1y": test_wave.spin_1y,
    #                           "spin 1z": test_wave.spin_1z,
    #                           "spin 2x": test_wave.spin_2x,
    #                           "spin 2y": test_wave.spin_2y,
    #                           "spin 2z": test_wave.spin_2z,},
    #                        time_range=[-150, 100, 8000])

    # ax.plot(imr[0].times*1e3, imr[0].data);
    # ax.plot(imr[0].times*1e3, -imr[0].data);

    try:
        hp, hx = test_wave.timeseries(total_mass=total_mass,
                                      sample_rate=4096,
                                      flow=93.0,
                                      distance=1)


        ax.plot(hp.times*1e3, -hp.data);
        ax.plot(hp.times*1e3,  hp.data);

    except:
        pass

    ax.set_ylabel("Strain at 1Mpc")
    ax.set_xlabel("Time from merger at 60 solMass [ms]")

    ax.set_xlim([-15, 10]);

    f.tight_layout();

    def faithfulness(a,b):
        top = np.dot(a,b)
        aa = np.sqrt(np.dot(a,a))
        bb = np.sqrt(np.dot(b,b))

        return (top)/(aa * bb)

    #faith = [faithfulness(

    
    with report:

        
        report += f

