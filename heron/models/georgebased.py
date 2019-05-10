from . import Model
from .gw import BBHSurrogate, BBHNonSpinSurrogate, HofTSurrogate

import numpy as np
import george
from george import HODLRSolver
import elk

from elk.catalogue import Catalogue

import scipy.optimize as op

def train(model):
    """
    Train a george-based Gaussian process model.
    """

    def callback(p):
        print '{}\t{}'.format(np.exp(p),  model.log_evidence(p)[0])

    def nll(k):
        ll = model.log_evidence(k)[0]
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(k):
        return - model.log_evidence(k)[1]

    model.gp.white_noise.set_parameter_vector(0.1)
    p0 = model.gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B", callback=callback)
    model.gp.set_parameter_vector(results.x)
    model.gp.white_noise.set_parameter_vector(0.0)

    return results

class HodlrGPR(Model):
    """
    A GPR model using the hierarchical matrix approximation.
    """

    def build(self,  mean=0.0, white_noise=0, tol=1e-6):
        """
        Construct the GP
        """


        self.gp = george.GP(self.kernel,
                            solver=george.HODLRSolver,
                            tol=tol,
                            min_size=50,
                            mean=mean, white_noise=white_noise)
        self.yerr = np.ones(len(self.training_data)) * 0
	
        self.gp.compute(self.training_data[:, :self.x_dimensions], self.yerr)  
        
    def _generate_eval_matrix(self, p, times):
        """
        Create the matrix of parameter points at which to evaluate the model.
        """

        nt = len(times)
        points = np.ones((nt, self.x_dimensions))
        points[:,self.c_ind['time']] = times

        for column, value in p.items():
            points[:, self.c_ind[column]] *= value

        return points



class Heron2dHodlrIMR(HodlrGPR, BBHNonSpinSurrogate, HofTSurrogate):
    """
    Produce a BBH waveform generator using the Hodlr method with IMRPhenomPv2 training data.
    """


    def __init__(self):

        
        waveforms = [{"mass ratio": q,
                      "spin 1x": 0, "spin 1y": 0, "spin 1z": 0,
                      "spin 2x": 0, "spin 2y": 0, "spin 2z": 0}
                     for q in np.linspace(0.1, 1.0, 6)]
        
        self.kernel = 1.46 * george.kernels.ExpSquaredKernel(0.0285,
                                                      ndim=self.problem_dims,
                                                      axes=self.c_ind['mass ratio']) \
        * george.kernels.ExpSquaredKernel(0.0157,
                                          ndim=self.problem_dims,
                                          axes=self.c_ind['time'],) 

        self.total_mass = 60
        self.f_min = 95.0
        self.ma = [(2,2), (2,-2)]
        self.t_max = 0.05
        self.t_min = -0.05
        self.f_sample = 4096 #1024

        self.catalogue = elk.catalogue.PPCatalogue("IMRPhenomPv2", total_mass=self.total_mass, fmin = self.f_min, waveforms=waveforms)

        mean = 0.0
        tol = 1e-3
        white_noise = 0.0
        
        self.training_data = self.catalogue.create_training_data(self.total_mass,
                                                               f_min = self.f_min,
                                                               sample_rate=self.f_sample,
                                                               ma=self.ma,
                                                               tmax=self.t_max,
                                                               tmin=self.t_min,
        )

        self.training_data[:,self.c_ind['time']] *= 100
        self.training_data[:,self.c_ind['mass ratio']] = np.log(self.training_data[:,self.c_ind['mass ratio']])
        self.training_data[:,self.c_ind['h+']] *= 1e19
        self.training_data[:,self.c_ind['hx']] *= 1e19

        self.x_dimensions = self.kernel.ndim
        
        self.build(mean, white_noise, tol)
        

    def log_evidence(self, k):
        """
        Evaluate the log-evidence of the model at a hyperparameter location k.
        """
        old_k = self.gp.get_parameter_vector()
        self.gp.set_parameter_vector(k)
        ll, grad_ll =  self.gp.log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True), self.gp.grad_log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True)
        self.gp.set_parameter_vector(old_k)

        return ll, grad_ll
    

class HeronHodlr(HodlrGPR, BBHSurrogate, HofTSurrogate):
    """
    Produce a BBH waveform generator using the Hodlr method.
    """


    def __init__(self):

        self.catalogue = elk.catalogue.NRCatalogue(origin="GeorgiaTech")
        #self.problem_dims = 8
        self.kernel = 1.0 * george.kernels.ExpSquaredKernel(0.005,
                                                      ndim=self.problem_dims,
                                                      axes=self.c_ind['mass ratio']) \
        * george.kernels.ExpSquaredKernel(100,
                                          ndim=self.problem_dims,
                                          axes=self.c_ind['time'],) \
        * george.kernels.ExpSquaredKernel([0.005, 0.005, 0.005, 
                                           0.005, 0.005, 0.005], 
                                          ndim=self.problem_dims, 
                                          axes=[2,3,4,5,6,7])

        self.total_mass = 60
        self.f_min = None
        self.ma = [(2,2), (2,-2)]
        self.t_max = 0.03
        self.t_min = -0.01
        self.f_sample = 512 #1024

        mean = 0.0
        tol = 1e-3
        white_noise = 0
        
        self.training_data = self.catalogue.create_training_data(self.total_mass,
                                                               f_min = self.f_min,
                                                               sample_rate=self.f_sample,
                                                               ma=self.ma,
                                                               tmax=self.t_max,
                                                               tmin=self.t_min)

        self.training_data[:,self.c_ind['time']] *= 100
        self.training_data[:,self.c_ind['mass ratio']] = np.log(self.training_data[:,self.c_ind['mass ratio']])
        self.training_data[:,self.c_ind['h+']] *= 1e19
        self.training_data[:,self.c_ind['hx']] *= 1e19

        self.x_dimensions = self.kernel.ndim
        
        self.build(mean, white_noise, tol)
        self.gp.set_parameter_vector(np.log([1.46, 0.0285, 0.0157, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]))
        

    def log_evidence(self, k):
        """
        Evaluate the log-evidence of the model at a hyperparameter location k.
        """
        old_k = self.gp.get_parameter_vector()
        self.gp.set_parameter_vector(k)
        ll, grad_ll =  self.gp.log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True), self.gp.grad_log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True)
        self.gp.set_parameter_vector(old_k)

        return ll, grad_ll


   
class Heron2dHodlr(HodlrGPR, BBHNonSpinSurrogate, HofTSurrogate):
    """
    Produce a BBH waveform generator using the Hodlr method.
    """


    def __init__(self):

        self.catalogue = elk.catalogue.NRCatalogue(origin="GeorgiaTech")
        self.kernel = 1.0 * george.kernels.ExpSquaredKernel(0.005,
                                                      ndim=self.problem_dims,
                                                      axes=self.c_ind['mass ratio']) \
        * george.kernels.ExpSquaredKernel(100,
                                          ndim=self.problem_dims,
                                          axes=self.c_ind['time'],) \

        self.total_mass = 60
        self.f_min = None
        self.ma = [(2,2), (2,-2)]
        self.t_max = 0.03
        self.t_min = -0.01
        self.f_sample = 512 #1024

        mean = 0.0
        tol = 1e-3
        white_noise = 0
        
        self.training_data = self.catalogue.create_training_data(self.total_mass,
                                                               f_min = self.f_min,
                                                               sample_rate=self.f_sample,
                                                               ma=self.ma,
                                                               tmax=self.t_max,
                                                               tmin=self.t_min)

        self.training_data[:,self.c_ind['time']] *= 100
        self.training_data[:,self.c_ind['mass ratio']] = np.log(self.training_data[:,self.c_ind['mass ratio']])
        self.training_data[:,self.c_ind['h+']] *= 1e19
        self.training_data[:,self.c_ind['hx']] *= 1e19

        self.x_dimensions = self.kernel.ndim
        
        self.build(mean, white_noise, tol)
        self.gp.set_parameter_vector(np.log([1.46, 0.0285, 0.0157]))
        

    def log_evidence(self, k):
        """
        Evaluate the log-evidence of the model at a hyperparameter location k.
        """
        old_k = self.gp.get_parameter_vector()
        self.gp.set_parameter_vector(k)
        ll, grad_ll =  self.gp.log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True), self.gp.grad_log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True)
        self.gp.set_parameter_vector(old_k)

        return ll, grad_ll


    
