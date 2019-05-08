import heron
from heron import waveform
from george import kernels
import numpy as np

import elk
import elk.catalogue

total_mass = 60

catalogue = elk.catalogue.NRCatalogue(origin="GeorgiaTech")

problem_dims = 8
c_ind = catalogue.c_ind
time_covariance = kernels.RationalQuadraticKernel(.05, 400,
                                           ndim=problem_dims,
                                           axes=c_ind['time'],)
mass_covariance = kernels.ExpSquaredKernel(0.005,
                                           ndim=problem_dims,
                                           axes=c_ind['mass ratio'])
spin_covariance = kernels.ExpSquaredKernel([0.005, 0.005, 0.005, 
                                            0.005, 0.005, 0.005], 
                                           ndim=problem_dims, 
                                           axes=[2,3,4,5,6,7])

covariance =  1e1 * mass_covariance * time_covariance * spin_covariance

gp = gp_cat = waveform.GPCatalogue(catalogue, covariance,
                                   total_mass=total_mass, fsample=100,
                                   mean=0.0,
                                   ma=[(2,2)],
                                   solver="hodlr",
                                   tmax=0.02,
                                   white_noise=1e-6,)


# After 2000 training iterations with ADAM
#gp_cat.gp.set_parameter_vector(np.log([1.96628473e+00,
 #                                      3.33975754e-04,
 #                                      1.64245209e+00, 3.29722921e+02,
 #                                      8.78725810e-04, 7.28872939e-04, 6.98418530e-04,
 #                                      8.91739716e-04, 7.28836193e-04, 7.05170046e-04]))

vector = """3.76709468e-01 4.41674434e-05 5.03683736e+00 3.83973183e+01
 1.87667157e-04 1.95780210e-04 1.20018476e-02 2.51931077e-03
 1.03071882e-04 3.26014002e-02"""

vector = """3.64080766e-01 1.71199754e-04 1.97539594e+01 2.20940436e+00
 4.15865573e-02 1.61720345e-02 2.35052834e-01 1.85571791e-02
 6.98060731e-03 3.27260094e-01"""


vector = vector.split()
vector = map(float, vector)
gp_cat.gp.set_parameter_vector(np.log(vector))

gp_cat.optimise("adam", max_iter=10000, step_rate=0.2, momentum=0.1)

print(np.exp(gp_cat.gp.get_parameter_vector()))
