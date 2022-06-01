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
from gpytorch.kernels import RBFKernel
from gpytorch.constraints import GreaterThan, LessThan, Interval

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
    """
    Train the model.
    """

    training_iterations = iterations
    if hasattr(model, "model_cross"):
        model.model_cross.train()
    model.model_plus.train()
    model.likelihood.train()
    

    optimizer = torch.optim.Adam([
        {'params': model.model_plus.parameters()},
    ], lr=lr)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model.model_plus)

    epochs_iter = tqdm.tqdm(range(training_iterations), desc="Epoch")

    if hasattr(model, "state_vector"):
        state_vector = model.state_vector
    else:
        state_vector = "model_state.pth"
    for i in epochs_iter:
        optimizer.zero_grad()
        # Output from model
        output = model.model_plus(model.training_x)
        # Calc loss and backprop gradients
        loss = torch.distributions.Normal(output.mean.to(device=model.device),
                                          output.variance.sqrt().to(device=model.device)).log_prob(model.training_y).mean()
        
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            torch.save(model.model_plus.state_dict(), state_vector)
    model.eval()
    model.model_plus.eval()


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Use the GpyTorch Exact GP
    """
    def __init__(self, train_x, train_y, likelihood):
        """Initialise the model"""
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        time_kernel = RBFKernel()
        spin_kernels = None
        
        if spin_kernels:
            inner_kernels = time_kernel*mass_kernel*prod(spin_kernels)
        else:
            inner_kernels = time_kernel#*mass_kernel

        self.covar_module = gpytorch.kernels.ScaleKernel(
            inner_kernels
        )

    def forward(self, x):
        """Run the forward method of the model"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,
                                                         covar_x
        )

    
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

    def _process_inputs(self, times, p):
        times *= self.time_factor
        # p['mass ratio'] *= 100 #= np.log(p['mass ratio']) * 100
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
        elif polarisation == "cross":
            model = self.model_cross

        if not isinstance(times, torch.Tensor):
            times = torch.tensor(times)
        times_b = times.clone()
        points = self._generate_eval_matrix(p, times_b)
        points = torch.tensor(points, device=self.device).float()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(num_probe_vectors=10), gpytorch.settings.max_root_decomposition_size(5000):
                f_preds = model(points)

        mean = f_preds.mean/self.strain_input_factor
        var = f_preds.variance.double().abs()/self.strain_input_factor # .detach().double()
        covariance = f_preds.covariance_matrix.double()#.detach().double()
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
            times = torch.tensor(times)
        
        if hasattr(self, "model_cross"):
            polarisations = ["plus", "cross"]
        else:
            polarisations = ["plus"]
        for polarisation in polarisations:

            mean, var, covariance = \
                self._predict(times, p, polarisation=polarisation)

            timeseries[polarisation] = \
                Timeseries(data=mean,
                           times=times,
                           covariance=covariance,
                           variance=var)
        return timeseries

class HeronCUDA(CUDAModel, BBHSurrogate, HofTSurrogate):
    """
    A GPR BBH waveform model which is capable of using CUDA resources.
    """

    time_factor = 100
    other_input_factor = 100
    strain_input_factor = 1e21
    x_dimensions = 8

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
        (self.model_plus, self.model_cross), self.likelihood = self.build()
        #
        self.columns = dict(enumerate(self.training_data[self.datalabel]['meta']['parameters']))
        self.c_ind = {j:i for i,j in self.columns.items()}
        #
        self.time_factor = 100
        self.strain_input_factor = 1e21
        #
        self.eval()
    def _process_inputs(self, times, p):
        times *= self.time_factor
        p = {k: self.other_input_factor*v for k, v in p.items()}
        return times, p

    def build(self):
        """
        Right now this isn't need by this method
        """

        def prod(iterable):
            return reduce(operator.mul, iterable)

        mass_kernel = RBFKernel(active_dims=self.c_ind["mass ratio"],
                                lengthscale_constraint=Interval(self.lengths['mass ratio'][0], self.lengths['mass ratio'][1]))
        time_kernel = RBFKernel(active_dims=self.c_ind["time"],
                                lengthscale_constraint=Interval(self.lengths['time'][0], self.lengths['time'][1]))

        if b"spin 1x" in self.parameters:
            spin_kernels = [RBFKernel(active_dims=dimension,
                                      lengthscale_constraint=GreaterThan(7))
                            for dimension in range(2, 8)]
        else:
            spin_kernels = None


        class ExactGPModel(gpytorch.models.ExactGP):
            """
            Use the GpyTorch Exact GP
            """
            def __init__(self, train_x, train_y, likelihood):
                """Initialise the model"""
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()


                if spin_kernels:
                    inner_kernels = time_kernel*mass_kernel*prod(spin_kernels)
                else:
                    inner_kernels = time_kernel*mass_kernel

                self.covar_module = gpytorch.kernels.ScaleKernel(
                    inner_kernels
                )

            def forward(self, x):
                """Run the forward method of the model"""
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x,
                                                                 covar_x
                )

        

        x, y = self.training_data.get_training_data(label=self.datalabel,
                                                    polarisation=b"+",
                                                    size = self.data_size)

        training_x = self.training_x = torch.tensor(x*self.other_input_factor, device=self.device).float().T
        training_y = self.training_y = torch.tensor(y*self.strain_input_factor, device=self.device).float()
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(self.noise[0], self.noise[1]))
        model = ExactGPModel(training_x, training_y, likelihood)

        if self.device.type == "cuda":
            model = model.cuda() # Annoyingly this can't be passe din as a device keyword
        
        #model2 = ExactGPModel(training_x, training_yx, likelihood)
        state_vector = pkg_resources.resource_filename('heron', 'models/data/gt-gpytorch.pth')

        # FIXME: fix this dirty hack
        model2 = model
        #model.load_state_dict(torch.load(state_vector))
        #model2.load_state_dict(torch.load(state_vector))

        return [model, model2], likelihood

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

    def frequency_domain_waveform(self, p, window, times=np.linspace(-2, 2, 1000), polarisation=None):
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
        timeseries = self.mean(times, p, polarisation)
        frequencyseries = {pol: ts.to_frequencyseries()
                           for pol, ts in timeseries.items()}
        return frequencyseries

    
    def time_domain_waveform(self, p, times=np.linspace(-2, 2, 1000)):
        """
        Return the timedomain waveform.
        """

        return self.mean(times, p)

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



class HeronCUDAIMR(CUDAModel, BBHNonSpinSurrogate, HofTSurrogate, FrequencyMixin):
    """
    A GPR BBH waveform model which is capable of using CUDA resources.
    """

    time_factor = 1000
    strain_input_factor = 1e21

    def __init__(self, device=device):
        """
        Construct a CUDA-based waveform model with pyTorch
        """
        self.device = device
        self.polarisations = ["plus", "cross"]
        # super(HeronCUDA, self).__init__()
        #assert torch.cuda.is_available()  # This is a bit of a kludge
        (self.model_plus, self.model_cross), self.likelihood = self.build()
        self.x_dimensions = 2
        self.time_factor = 1000
        self.strain_input_factor = 1e21
        #
        self.eval()

    def eval(self):
        """
        Prepare the model to be evaluated.
        """
        self.model_plus.eval()
        self.model_cross.eval()
        self.likelihood.eval()

    def _process_inputs(self, times, p):
        times *= self.time_factor
        #p['mass ratio'] *= 10 #= np.log(p['mass ratio']) * 100
        p = {k: v for k, v in p.items()}
        return times, p

    def build(self):
        """
        Right now this isn't need by this method
        """

        def prod(iterable):
            return reduce(operator.mul, iterable)

        mass_kernel = RBFKernel(active_dims=1,
                                lengthscale_constraint=GreaterThan(torch.tensor(.1, device=self.device)))
        time_kernel = RBFKernel(active_dims=0,
                                lengthscale_constraint=GreaterThan(torch.tensor(1, device=self.device)))
        total_kernel = gpytorch.kernels.ScaleKernel((time_kernel * mass_kernel))
        class ExactGPModel(gpytorch.models.ExactGP):
            """
            Use the GpyTorch Exact GP
            """
            def __init__(self, train_x, train_y, likelihood):
                """Initialise the model"""
                super(ExactGPModel, self).__init__(train_x,
                                                   train_y, likelihood)

                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = total_kernel

            def forward(self, x):
                """Run the forward method of the model"""
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        catalogue = PPCatalogue("IMRPhenomPv2", total_mass=10, waveforms=[{"mass ratio": q} for q in np.linspace(1, 4.0, 10)])
        data = catalogue.create_training_data(total_mass=10, ma=[(2,2)])
        data[:,0] *= self.time_factor
        data[:,1] = 1./data[:,1]

        training_x = torch.tensor(np.random.randn(*data[:,[0,1]].shape)*1e-3 + data[:, [0,1]], device=self.device).float()#.cuda()
        training_y = torch.tensor(np.random.rand(*data[:,-1].shape)*1e-6 + data[:, -1]*self.strain_input_factor, device=self.device).float()#.cuda()
        training_yx = torch.tensor(data[:, -2]*self.strain_input_factor, device=self.device).float()#.cuda()


        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(training_x, training_y, likelihood)
        model2 = ExactGPModel(training_x, training_yx, likelihood)
        state_vector = pkg_resources.resource_filename('heron', 'models/data/imr-gpytorch.pth')

        if not (self.device.type == 'cpu') and torch.cuda.is_available():
            model = model.cuda()
            model2 = model2.cuda()
            likelihood = likelihood.cuda()

        model.load_state_dict(torch.load(state_vector))

        model2.load_state_dict(torch.load(state_vector))

        return [model, model2], likelihood

    def _predict(self, times, p, polarisation="plus"):
        """
        Query the model for the mean and covariance tensors.
        Optionally include the covariance.

        Parameters
        ----------
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
        elif polarisation == "cross":
            model = self.model_cross

        times_b = times.clone()

        if "distance" in p:
            distance = p.pop("distance")
        else:
            distance = 1

        points = self._generate_eval_matrix(p, times_b)
        points = torch.tensor(points, device=self.device).float()

        with torch.no_grad():#, gpytorch.settings.fast_pred_var():
            f_preds = model(points)

        mean = f_preds.mean/(distance * self.strain_input_factor)
        var = f_preds.variance.double()/((distance*self.strain_input_factor)) # **2
        covariance = f_preds.covariance_matrix.double()

        
        covariance /= (((self.strain_input_factor)**2))
        covariance /= distance**2

        return mean, var, covariance

    def mean(self, times, p, covariance=False):
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
        for polarisation in ["plus", "cross"]:
            mean, var, covariance = self._predict(times, p, polarisation=polarisation)
            timeseries[polarisation] = \
                Timeseries(data=mean,#.cpu().numpy().astype(np.float64),
                           times=times,
                           covariance=covariance,
                           variance=var),#.cpu().numpy().astype(np.float64))
        return timeseries

    def distribution(self, times, p, samples=100):
        """
        Return a number of sample waveforms from the GPR distribution.
        """
        times_b = times.clone()
        points = self._generate_eval_matrix(p, times_b)
        points = torch.tensor(points, device=self.device).float()
        return_samples = []
        for polarisation in [self.model_plus, self.model_cross]:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                f_preds = polarisation(points)
                y_preds = self.likelihood(f_preds)

                return_samples.append([Timeseries(data=sample.cpu()/self.strain_input_factor,
                                         times=times_b)
                                       for sample in y_preds.sample_n(samples)])
        return return_samples



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




class HeronCUDAMix(CUDAModel, BBHNonSpinSurrogate, HofTSurrogate, FrequencyMixin):
    """
    A GPR BBH waveform model which is capable of using CUDA resources.
    """
    # strain_input_factor = 1e20

    def __init__(self, training_data, specification):
        """
        Construct a CUDA-based waveform model with pyTorch.

        Parameters
        ----------
        training_data: `numpy.ndarray`
               The array of training data.

        specification: dict
           The Heron model specification
        """
        self.specification = specification
        self.data = np.genfromtxt(training_data)
        super().__init__()
        self.polarisations = ["plus"]
        self.x_dimensions = 2
        self.time_factor = 100
        self.strain_input_factor = 1e21
        self.total_mass = specification['total mass']

        self.state_vector = f"{self.specification['name']}.pth"

        self.model_plus, self.likelihood = self.build()
        self.eval()
        
    def build(self):
        """
        """

        def prod(iterable):
            return reduce(operator.mul, iterable)

        # construct the kernels
        kernel_spec = self.specification['kernels']
        kernels = []
        scale = False
        for k in kernel_spec:
            if k['type'] == "scale":
                scale = True
            else:
                if "constraints" in k:
                    if "greater" in k["constraints"]:
                        constraint = GreaterThan(k["constraints"]["greater"])
                else:
                    constraint = None
                kernels += [RBFKernel(active_dims=k['dimension'], has_lengthscale=True, lengthscale_constraint = constraint)]
        kernels = prod(kernels)
        if scale:
            kernels = gpytorch.kernels.ScaleKernel(kernels)
            
        if self.specification["mean"] == "constant":
            mean = gpytorch.means.ConstantMean()
            
        class ExactGPModel(gpytorch.models.ExactGP):
            """
            Use the GpyTorch Exact GP
            """
            def __init__(self, train_x, train_y, likelihood):
                """Initialise the model"""
                super(ExactGPModel, self).__init__(train_x,
                                                   train_y, likelihood)
                self.mean_module = mean
                self.covar_module = kernels

            def forward(self, x):
                """Run the forward method of the model"""
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        data = self.data

        
        
        idx = data[:, 1] == 0
        tx = data[idx][:, [2, 3]]*self.time_factor

        training_x = self.training_x = torch.tensor(tx, device=self.device).float()
        training_y = self.training_y = torch.tensor(data[idx, -1]*self.strain_input_factor, device=self.device).float()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()#noise_constraint=LessThan(1))
        model = ExactGPModel(training_x, training_y, likelihood)
        if not (self.device.type == "cpu") and torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        if os.path.exists(self.state_vector):
            model.load_state_dict(torch.load(self.state_vector))

        return model, likelihood
