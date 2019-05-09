from . import Model
from .gw import BBHSurrogate

import numpy as np
import george
from george import HODLRSolver
import elk
from elk.waveform import Waveform, Timeseries
from elk.catalogue import Catalogue


def train(model):
    """
    Train a george-based Gaussian process model.
    """
    pass

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
                            min_size=100,
                            mean=mean, white_noise=white_noise)
        self.yerr = np.ones(len(self.training_data)) * 0
	
        self.gp.compute(self.training_data[:, :self.x_dimensions], self.yerr)  
        
    def nll(k):
        self.gp.set_parameter_vector(k)
        #print(np.exp(p))
        ll = self.log_evidence(k)[0]
        #print(-ll)
        return -ll if np.isfinite(ll) else 1e25

        # # And the gradient of the objective function.
    def grad_nll(k):
        #print(np.exp(p))
        
        #print(gp.log_likelihood(data[c_ind['h+']]*1e19, quiet=True))
        return - self.log_evidence(k)
        
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
    

class HeronHodlr(HodlrGPR, BBHSurrogate):
    """
    Produce a BBH waveform generator using the Hodlr method.
    """


    def __init__(self):

        self.catalogue = elk.catalogue.NRCatalogue(origin="GeorgiaTech")
        self.problem_dims = 8
        self.kernel = 1.0 * george.kernels.ExpSquaredKernel(0.005,
                                                      ndim=self.problem_dims,
                                                      axes=self.c_ind['mass ratio']) \
        + george.kernels.ExpSquaredKernel(100,
                                          ndim=self.problem_dims,
                                          axes=self.c_ind['time'],) \
        + george.kernels.ExpSquaredKernel([0.005, 0.005, 0.005, 
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
        tol = 1e-6
        white_noise = 0
        
        self.training_data = self.catalogue.create_training_data(self.total_mass,
                                                               f_min = self.f_min,
                                                               sample_rate=self.f_sample,
                                                               ma=self.ma,
                                                               tmax=self.t_max,
                                                               tmin=self.t_min,
        )

        self.training_data[:,self.c_ind['time']] *= 100
        #self.training_data[:,self.c_ind['mass ratio']] = np.log(self.training_data[:,self.c_ind['mass ratio']])
        #self.training_data[:,self.c_ind['time']] -= self.training_data[np.argmax(self.training_data[:,self.c_ind['time']]),self.c_ind['time']]
        self.training_data[:,self.c_ind['h+']] *= 1e19
        self.training_data[:,self.c_ind['hx']] *= 1e19

        self.x_dimensions = self.kernel.ndim
        
        self.build(mean, white_noise, tol)
        self.gp.set_parameter_vector(np.log([1.33e+00,
                                       2.04e-04,
                                       .95,
                                       8.78725810e-04, 7.28872939e-04, 6.98418530e-04,
                                       8.91739716e-04, 7.28836193e-04, 7.05170046e-04]))
        

    def mean(self, p, times):
        """
        Return the mean waveform at a given location in the 
        BBH parameter space.
        """

        points = self._generate_eval_matrix(p, times)
        
        mean, var = self.gp.predict(self.training_data[:,self.c_ind['h+']],
                                    points,
                                    return_var=True,
        )
        return Timeseries(data=mean/1e19, times=points[:,self.c_ind['time']])

    def log_evidence(self, k):
        """
        Evaluate the log-evidence of the model at a hyperparameter location k.
        """
        old_k = self.gp.get_parameter_vector()
        self.gp.set_parameter_vector(k)
        ll, grad_ll =  self.gp.log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True), self.gp.grad_log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True)
        self.gp.set_parameter_vector(old_k)

        return ll, grad_ll
