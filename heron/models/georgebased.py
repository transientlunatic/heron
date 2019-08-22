"""
Models utilising the `george` GPR library in Python and C++.
"""
from . import Model
from .gw import BBHSurrogate, BBHNonSpinSurrogate, HofTSurrogate

import numpy as np
import george
from george import HODLRSolver
import elk
from elk.waveform import Timeseries

from elk.catalogue import Catalogue

import scipy.optimize as op

import pkg_resources

DATA_PATH = pkg_resources.resource_filename('heron', 'models/data/')

def train(model, batch_size=100, algorithm="adam", max_iter=1000):
    """
    Train a george-based Gaussian process model.
    """

    def callback(p):
        print('{}\t{}'.format(np.exp(p),  model.log_evidence(p, n=batch_size)[0]))

    def nll(k):
        ll = model.log_evidence(k, n=batch_size)[0]
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(k):
        return - model.log_evidence(k, n=batch_size)[1]

    def grad_ll(k):
        return model.log_evidence(k, n=batch_size)[1]

    # Get the default value of the hyperparameters as the initial point for the optimisation
    p0 = model.gp.get_parameter_vector()

    model.train()
    
    if not batch_size == None:
        if algorithm == "adam":
            """
            Optimise using the adam algorithm.
            """
            import climin
            opt = climin.Adam(p0, grad_nll)

        for info in opt:
                    if info['n_iter']%10 == 0:
                        k = model.gp.get_parameter_vector()
                        print("{} - {} - {}".format(info['n_iter'],
                                                    model.log_evidence(k, n=batch_size)[0],
                                                    np.exp(k)
                        ))
                        
                    if info['n_iter'] > max_iter: break
        results = model.gp.get_parameter_vector()

    else:
                
        results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B", callback=callback)
        model.gp.set_parameter_vector(results.x)
        

    model.eval()

    return results

class HodlrGPR(Model):
    """
    A GPR model using the hierarchical matrix approximation.
    """

    training = False
    evaluate = True

    time_factor = 100
    strain_input_factor = 1e19
    
    def eval(self):
        """
        Prepare the model to be evaluated.
        """
        self.training = False
        self.gp.white_noise.set_parameter_vector(0.0)
        self.training = False
        self.gp.compute(self.training_data[:,:self.x_dimensions], self.yerr)
        self.evaluate = True

    def train(self):
        """
        Prepare the model to be trained.
        """
        # Put the model into training mode, 
        model.training = True
        model.evaluate = False
        # and set the white noise to a slightly higher level to improve stability
        model.gp.white_noise.set_parameter_vector(0.1)
    
    def _process_inputs(self, times, p):
        """
        Apply regularisations and normalisations to any input point dictionary.

        Parameters
        ----------
        p : dict
           A dictionary of the input locations
        """

        times *= self.time_factor
        p['mass ratio'] = np.log(p['mass ratio'])
        return times, p

    def build(self,  mean=0.0, white_noise=0, tol=1e-6):
        """
        Construct the GP object
        """


        self.gp = george.GP(self.kernel,
                            solver=george.HODLRSolver,
                            tol=tol,
                            min_size=100,
                            mean=mean, white_noise=white_noise)
        self.yerr = np.ones(len(self.training_data)) * 0
	
        self.gp.compute(self.training_data[:, :self.x_dimensions], self.yerr)

        
    def log_evidence(self, k, n):
        """
        Evaluate the log-evidence of the model at a hyperparameter location k.

        Parameters
        ----------
        n : int
           The number of points to be used to calculate the log likelihood.
        """
        
        

        if self.training:
            rix = np.random.randint(len(self.training_data), size=(n))
            y = self.training_data[rix, -1]
            x = self.training_data[rix, :self.x_dimensions]

            self.gp.compute(x)
        else:
            old_k = self.gp.get_parameter_vector()
            y = self.training_data[:, self.c_ind['h+']]

        self.gp.set_parameter_vector(k)
        ll, grad_ll = self.gp.log_likelihood(y, quiet=True), self.gp.grad_log_likelihood(y, quiet=True)
        if not self.training:
            self.gp.set_parameter_vector(old_k)

        return ll, grad_ll

    def mean(self, p, times):
        """
        Return the mean waveform at a given location in the 
        BBH parameter space.

        Parameters
        ----------
        
        """

        points = self._generate_eval_matrix(p, times)
        
        mean, var = self.gp.predict(self.training_data[:,self.c_ind['h+']],
                                    points,
                                    return_var=True,
        )
        mean_x, var_x = self.gp.predict(self.training_data[:,self.c_ind['hx']],
                                    points,
                                    return_var=True,
        )
        return Timeseries(data=mean/self.strain_input_factor, times=points[:,self.c_ind['time']]/self.time_factor, variance=var), \
               Timeseries(data=mean_x/self.strain_input_factor, times=points[:,self.c_ind['time']]/self.time_factor, variance=var_x)

    def distribution(self, p, times, samples=100, polarisation="h+"):
        """
        Return the mean waveform and the variance at a given location in the 
        BBH parameter space.

        Parameters
        ----------
        p : dict
           A dictionary of parameter locations.
        times : array-like
           The timestamps at which the model should be evaluated.
        samples : int
           The number of samples to draw from the GP.
        polarisation : str {"h+", "hx"}
           The polarisation which should be evaluated.
        """

        points = self._generate_eval_matrix(p, times)

        samples = self.gp.sample_conditional(self.training_data[:,self.c_ind[polarisation]],
                                             points,
                                             size=samples
        )
        return_samples = [Timeseries(data=sample/1e19, times=points[:,self.c_ind['time']]/self.time_factor) for sample in samples]
        
        return np.array(return_samples)


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
        

    

class HeronHodlr(HodlrGPR, BBHSurrogate, HofTSurrogate):
    """
    Produce a BBH waveform generator using the Hodlr method.
    """


    def __init__(self):

        #self.catalogue = elk.catalogue.NRCatalogue(origin="GeorgiaTech")
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
        self.t_max = 0.02
        self.t_min = -0.015
        self.f_sample =  1024

        mean = 0.0
        tol = 1e-6
        white_noise = 0

        DB_FILE = pkg_resources.resource_filename('heron', 'models/data/gt-M60-F1024.dat')
        self.training_data = np.genfromtxt(DB_FILE)
        
        self.time_factor = 1e2
        self.strain_input_factor = 1e19
        
        self.training_data[:,self.c_ind['time']] *= self.time_factor
        self.training_data[:,self.c_ind['mass ratio']] = np.log(self.training_data[:,self.c_ind['mass ratio']])
        self.training_data[:,self.c_ind['h+']] *= self.strain_input_factor
        self.training_data[:,self.c_ind['hx']] *= self.strain_input_factor

        self.x_dimensions = self.kernel.ndim
        
        self.build(mean, white_noise, tol)
        # self.gp.set_parameter_vector(np.log([1.46, 0.0285, 0.0157, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]))
        #self.gp.set_parameter_vector(np.log([0.346, 0.0285, 0.0148, 0.006, 0.004, 0.007, 0.007, 0.005, 0.005]))
        self.gp.set_parameter_vector(np.log([0.606, 0.005380, 0.0041315, 0.006, 0.004, 0.007, 0.007, 0.005, 0.005]))


        
        

    # def log_evidence(self, k):
    #     """
    #     Evaluate the log-evidence of the model at a hyperparameter location k.
    #     """
    #     old_k = self.gp.get_parameter_vector()
    #     self.gp.set_parameter_vector(k)
    #     ll, grad_ll =  self.gp.log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True), self.gp.grad_log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True)
    #     self.gp.set_parameter_vector(old_k)

    #     return ll, grad_ll


   
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
        

    # def log_evidence(self, k):
    #     """
    #     Evaluate the log-evidence of the model at a hyperparameter location k.
    #     """
    #     old_k = self.gp.get_parameter_vector()
    #     self.gp.set_parameter_vector(k)
    #     ll, grad_ll =  self.gp.log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True), self.gp.grad_log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True)
    #     self.gp.set_parameter_vector(old_k)

    #     return ll, grad_ll


    
