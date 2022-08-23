"""
Models which use the GPyTorch GPR package as their backbone.
"""

from functools import reduce
import operator

import pkg_resources

import numpy as np

import os

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel
from gpytorch.constraints import GreaterThan, LessThan, Interval
from lal import antenna, cached_detector_by_prefix, TimeDelayFromEarthCenter, LIGOTimeGPS
import tqdm

from matplotlib import pyplot as plt

from elk.waveform import Timeseries, FrequencySeries
from elk.catalogue import PPCatalogue

from . import Model
from ..data import DataWrapper
from .gw import BBHSurrogate, HofTSurrogate, BBHNonSpinSurrogate, FrequencyMixin

DATA_PATH = pkg_resources.resource_filename('heron', 'models/data/')
disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train(model, iterations=1000, lr=0.1):
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.model_plus.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    model.model_plus.train()
    model.model_cross.train()
    model.likelihood.train()
    model.likelihood_cross.train()
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model.model_plus)
    for i in range(iterations):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model.model_plus(model.training_x)
        # Calc loss and backprop gradients
        loss = -mll(output, model.training_y)
        loss.backward()
        optimizer.step()
        model.model_cross.load_state_dict(model.model_plus.state_dict())
    model.model_plus.eval()
    model.model_cross.eval()
    model.likelihood.eval()
    model.likelihood.eval()
        
class CUDAModel(Model):
    """
    The factory class for all CUDA-based models.
    """
    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
    def eval(self):
        """
        Prepare the model to be evaluated.
        """
        if hasattr(self, "model_plus"):
            self.model_plus.eval()
        if hasattr(self, "model_cross"):
            self.model_cross.eval()
        self.likelihood.eval()
        self.likelihood_cross.eval()

    def _process_inputs(self, times, p):
        times *= self.time_factor
        times  += 0.40
        times[times < 0] = times[times < 0] / 4.0
        times -= 0.40
        times -= 0.0017
        p = {k: self.time_factor*v for k, v in p.items()}
        return times, p
    
    def _predict(self, times, p, polarisation="plus"):
        """
        Query the model for the mean and covariance tensors.
        Optionally include the covariance.

        Parameters
        -------------
        times : ndarray
           The times at which the model should be evaluated.
        p : dict
           A dictionary of locations in parameter space where the model
           should be evaluated.

        Returns
        -------
        mean : torch.tensor
           The mean waveform
        var : torch.tensor
            The variance of the waveform
        cov : torch.tensor, optional
            The covariance matrix of the waveform.
            Only returned if covariance was True.
        """

        if polarisation == "plus":
            model = self.model_plus
            likelihood = self.likelihood
        elif polarisation == "cross":
            model = self.model_cross
            likelihood = self.likelihood_cross
            

        if not isinstance(times, torch.Tensor):
            times = torch.tensor(times)
        
        points = self._generate_eval_matrix(p, times)
        points = torch.tensor(points, device=self.device, dtype=torch.float)

        with torch.no_grad(), gpytorch.settings.fast_pred_var(): #, gpytorch.settings.fast_pred_var(num_probe_vectors=10), gpytorch.settings.max_root_decomposition_size(5000):
                f_preds = likelihood(model(points))

        mean = f_preds.mean/self.strain_input_factor
        var = f_preds.variance.double().abs()/self.strain_input_factor
        covariance = f_preds.covariance_matrix.double()
        covariance /= (self.strain_input_factor**2)

        return mean, var, covariance
        
    
    def mean(self, times, p, covariance=False, polarisation=None):
        """
        Provide the mean waveform and its variance.
        Optionally include the covariance.

        Parameters
        ----------
        times : ndarray
           The times at which the model should be evaluated.
        p : dict
           A dictionary of locations in parameter space where the
           model should be evaluated.
        covariance : bool, optional
           A flag to determine if the whole covariance matrix
           should be returned.

        Returns
        -------
        mean : torch.tensor
           The mean waveform
        var : torch.tensor
            The variance of the waveform
        cov : torch.tensor, optional
            The covariance matrix of the waveform.
            Only returned if covariance was True.

        """
        covariance_flag = covariance

        timeseries = {}
        if not isinstance(times, torch.Tensor):
            times = times - 0.0017
            times = torch.tensor(times)
        
        if hasattr(self, "model_cross"):
            polarisations = ["plus", "cross"]
        else:
            polarisations = ["plus"]
        for polarisation in polarisations:

            mean, var, covariance = \
                self._predict(times.clone(), p, polarisation=polarisation)

            timeseries[polarisation] = \
                Timeseries(data=mean,
                           times=times+0.0017,
                           covariance=covariance,
                           variance=var)
        return timeseries

class HeronCUDA(CUDAModel, BBHSurrogate, HofTSurrogate):
    """
    A GPR BBH waveform model which is capable of using CUDA resources.
    """

    time_factor = 100
    other_input_factor = 100
    strain_input_factor = 1e22
    x_dimensions = 8
    decimation = 1

    def __init__(self,
                 datafile: str = None,
                 datalabel: str = None,
                 device: str = None,
                 size: int = None,
                 noise: list = [0.001, 0.1],
                 lengths: dict = {"mass ratio": [0.001, 1],
                                  "time": [0.001, 0.1]}
                 ):
        """
        Construct a CUDA-based waveform model with pyTorch
        """
        super().__init__(device=device)
        #
        self.training_data = DataWrapper(datafile)
        self.datalabel = datalabel
        self.data_size = size
        # Kernel and likelihood settings
        self.noise = noise
        self.lengths = lengths
        #
        self.x_dimensions = len(self.training_data[self.datalabel]['meta']['parameters'])
        self.parameters = list(self.training_data[self.datalabel]['meta']['parameters'])
        #
        self.columns = dict(enumerate(self.training_data[self.datalabel]['meta']['parameters']))
        self.c_ind = {j:i for i,j in self.columns.items()}
        #
        self.time_factor = 100
        self.strain_input_factor = 1e22
        #
        (self.model_plus, self.model_cross), (self.likelihood, self.likelihood_cross) = self.build()
        #
        self.eval()
        
    # def _process_inputs(self, times, p):
    #     times *= self.time_factor
    #     p = {k: self.other_input_factor*v for k, v in p.items()}
    #     return times, p

    def build(self):
        """
        Right now this isn't need by this method
        """

        def prod(iterable):
            return reduce(operator.mul, iterable)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_cross = gpytorch.likelihoods.GaussianLikelihood()
        # This needs to be changed when the model has more than one dimension, but I'm trying to
        # get anything to work right now.
        time_kernel = RBFKernel(active_dims=self.c_ind[b"time"])
        mass_kernel = RBFKernel(active_dims=self.c_ind[b"mass ratio"])
        # TODO: Add mass ratio support to the model
        # TODO: Add spin support to the model

        class ExactGPModel(gpytorch.models.ExactGP):
            """
            Use the GpyTorch Exact GP
            """
            def __init__(self, train_x, train_y, likelihood):
                """Initialise the model"""
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                
                self.mean_module = gpytorch.means.ZeroMean()
                inner_kernels = time_kernel * mass_kernel
                self.covar_module = gpytorch.kernels.ScaleKernel(inner_kernels)

            def forward(self, x):
                """Run the forward method of the model"""
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

        # Prepare the training data by rescaling it
        x, y = self.training_data.get_training_data(label=self.datalabel, polarisation=b"+")
        training_x = self.training_x = torch.tensor(x*self.other_input_factor,
                                                    device=self.device, dtype=torch.float).T[::2]#[1, :398]
        print(self.training_x.shape)
        training_y = self.training_y = torch.tensor(y*self.strain_input_factor,
                                                    device=self.device, dtype=torch.float)[::2]#[:398]

        _, y2 = self.training_data.get_training_data(label=self.datalabel, polarisation=b"x")
        
        y_mean = training_y.mean()
        training_y -= y_mean
        self.training_y = training_y

        self.training_x[:, self.c_ind[b'time']] += 0.40
        self.training_x[self.training_x[:,self.c_ind[b'time']]<0, self.c_ind[b'time']] = \
            self.training_x[self.training_x[:,self.c_ind[b'time']]<0, self.c_ind[b'time']]/4
        self.training_x[:, self.c_ind[b'time']] -= 0.40

        training_y_cross = self.training_y_cross = torch.tensor(y2*self.strain_input_factor,
                                                    device=self.device, dtype=torch.float)[::2]#[:398]

        
        model = ExactGPModel(self.training_x,
                             self.training_y, likelihood)

        model_cross = ExactGPModel(self.training_x,
                                   self.training_y_cross, likelihood_cross)
        if self.device.type == "cuda":
            model = model.cuda()
            model_cross = model_cross.cuda()
            likelihood = likelihood.cuda()
            likelihood_cross = likelihood_cross.cuda()
            # Annoyingly this can't be passed-in as a device keyword

        # Right now we're not returning a different model for the cross-polarisation
        # that's clearly not the correct way to do things
        # TODO: Add the cross polarisation correctly.
        return [model, model_cross], [likelihood, likelihood_cross]

    def distribution(self, times, p, samples=100):
        """
        Return a number of sample waveforms from the GPR distribution.
        """
        times_b = times.clone()
        points = self._generate_eval_matrix(p, times_b)
        points = torch.tensor(points, device=self.device)#.float().cuda()
        return_samples = []
        for polarisation in [self.model_plus, self.model_cross]:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                f_preds = polarisation(points)
                y_preds = self.likelihood(f_preds)

                return_samples.append([Timeseries(data=sample/self.strain_input_factor,
                                         times=times_b)
                                       for sample in y_preds.sample_n(samples)])
        return return_samples

    def frequency_domain_waveform(self, p, window, times, polarisation=None):
        """
        Return the frequency domain waveform.

        Parameters
        ----------
        p : dict
           A dictionary of the parameter values for the waveform.
        window : array
           An array containing the window which should be applied
           to the data before performing the FFT.
        times : array, optional
           The times at which the model should be evaluated.
        polarisation : str
           The polarisation to return. Default is `None` in which case all available polarisations are returned.
        """
        #print("TIMES BEFORE", times)
        timeseries = self.time_domain_waveform(times=times.clone(), p=p)
        #print("TIMES AFTER", timeseries['plus'].times)
        frequencyseries = {pol: ts.to_frequencyseries(window=window)
                           for pol, ts in timeseries.items()}

        polarisations = frequencyseries
        if "ra" in p.keys():
            ra, dec, psi, gpstime = p['ra'], p['dec'], p['psi'], p['gpstime']
            detector = cached_detector_by_prefix[p['detector']]
            response = self._get_antenna_response(detector,
                                                ra,
                                                dec,
                                                psi,
                                                gpstime)
            dt = TimeDelayFromEarthCenter(detector.location, ra, dec, LIGOTimeGPS(gpstime))
            tshiftvec = torch.exp(1j*2*torch.pi*dt*torch.tensor(frequencyseries['plus'].frequencies, device=self.device))
            plus_r = torch.tensor(response.plus, device=self.device)
            cross_r = torch.tensor(response.cross, device=self.device)
            
            waveform_mean = (polarisations['plus'].data * plus_r + polarisations['cross'].data * cross_r)*tshiftvec
            waveform_variance = polarisations['plus'].variance * response.plus**2 + polarisations['cross'].variance * response.cross**2
            waveform = FrequencySeries(data=waveform_mean,
                                       variance=waveform_variance,
                                       frequencies=torch.tensor(frequencyseries['plus'].frequencies, device=self.device)
            )
        else:
            waveform = polarisations
        
        return waveform

    
    def time_domain_waveform(self, p, times):
        """
        Return the timedomain waveform.
        """
        polarisations = self.mean(times, p)

        if "ra" in p.keys():
            ra, dec, psi, gpstime = p['ra'], p['dec'], p['psi'], p['gpstime']
            detector = cached_detector_by_prefix[p['detector']]
            response = self._get_antenna_response(detector,
                                                ra,
                                                dec,
                                                psi,
                                                gpstime)
            dt = TimeDelayFromEarthCenter(detector.location, ra, dec, LIGOTimeGPS(gpstime))
            
            waveform_mean = (polarisations['plus'].data * response.plus + polarisations['cross'].data * response.cross) 
            waveform_variance = polarisations['plus'].variance * response.plus**2 + polarisations['cross'].variance * response.cross**2
            
            waveform = Timeseries(data=waveform_mean,
                                  variance=waveform_variance,
                                  times=torch.tensor(polarisations['plus'].times + dt, device=self.device)
            )
        else:
            waveform = polarisations
        return waveform

    def plot_td(self, parameters, times=np.linspace(-2, 2, 1000), f=None):
        """
        Plot the timedomain waveform.
        """
        if not f:
            f, ax = plt.subplots(1, 1, dpi=300)
        else:
            ax = f.axes[0]

        mean, var = self.time_domain_waveform(parameters, times)
        mean = mean.cpu().numpy()
        var = var.cpu().numpy()

        ax.plot(times, mean)
        ax.fill_between(times, mean+np.abs(var), mean-np.abs(var), alpha=0.1)

        return f
