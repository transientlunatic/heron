"""
Models which use the GPyTorch GPR package as their backbone.
"""

from functools import reduce
import operator

import pkg_resources

import numpy as np

import torch
import gpytorch
from gpytorch.kernels import RBFKernel
from gpytorch.constraints import GreaterThan, LessThan

import tqdm

from matplotlib import pyplot as plt

from elk.waveform import Timeseries

from . import Model
from ..data import DataWrapper
from .gw import BBHSurrogate, HofTSurrogate, BBHNonSpinSurrogate

DATA_PATH = pkg_resources.resource_filename('heron', 'models/data/')

def train(model, iterations=1000):
    """
    Train the model.
    """

    training_iterations = iterations
    model.model_plus.train()
    model.likelihood.train()
    model.model_cross.train()

    optimizer = torch.optim.Adam([
        {'params': model.model_plus.parameters()},
    ], lr=0.1)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model.model_plus)

    epochs_iter = tqdm.tqdm_notebook(range(training_iterations), desc="Epoch")
    for i in epochs_iter:
        optimizer.zero_grad()
        # Output from model
        output = model.model_plus(model.training_x)
        # Calc loss and backprop gradients
        loss = -mll(output, model.training_y).cuda()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            torch.save(model.model_plus.state_dict(), 'model_state.pth')


class CUDAModel(Model):
    """
    The factory class for all CUDA-based models.
    """
    def __init__(self, device=device):
        self.device = device

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
        points = torch.tensor(points, device=self.device).float()#.cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(num_probe_vectors=10), gpytorch.settings.max_root_decomposition_size(5000):
            f_preds = model(points)

        mean = f_preds.mean/self.strain_input_factor
        var = f_preds.variance.detach().double()/(self.strain_input_factor**2)
        covariance = f_preds.covariance_matrix.detach().double()
        covariance /= (self.strain_input_factor**2)

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

        timeseries = []
        covariances = []
        if not isinstance(times, torch.Tensor):
            times = torch.tensor(times)
        
        if hasattr(self, "model_cross"):
            polarisations = ["plus", "cross"]
        else:
            polarisations = ["plus"]
        for polarisation in polarisations:
            mean, var, covariance = self._predict(times, p, polarisation=polarisation)
            timeseries.append(
                Timeseries(data=mean.cpu().numpy().astype(np.float64),
                           times=times,
                           variance=var.cpu().numpy().astype(np.float64)))
            covariances.append(covariance)
        if covariance_flag:
            return timeseries, covariances
        else:
            return timeseries

            

class HeronCUDA(Model, CUDAModel, BBHSurrogate, HofTSurrogate):
    """
    A GPR BBH waveform model which is capable of using CUDA resources.
    """

    time_factor = 100
    strain_input_factor = 1e21

    def __init__(self,
                 datafile: str = None,
                 datalabel: str = None
                 ):
        """
        Construct a CUDA-based waveform model with pyTorch
        """

        # super(HeronCUDA, self).__init__()
        #
        self.training_data = DataWrapper(datafile)
        self.datalabel = datalabel
        #
        assert torch.cuda.is_available()  # This is a bit of a kludge
        (self.model_plus, self.model_cross), self.likelihood = self.build()
        #
        self.x_dimensions = len(self.training_data[self.datalabel]['meta']['parameters'])
        self.parameters = self.training_data[self.datalabel]['meta']['parameters']
        self.columns = dict(enumerate(self.training_data[self.datalabel]['meta']['parameters']))
        self.c_ind = {j:i for i,j in self.columns.items()}
        #
        self.time_factor = 100
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
        # p['mass ratio'] *= 100 #= np.log(p['mass ratio']) * 100
        p = {k: 100*v for k, v in p.items()}
        return times, p

    def build(self):
        """
        Right now this isn't need by this method
        """

        def prod(iterable):
            return reduce(operator.mul, iterable)

        mass_kernel = RBFKernel(active_dims=1,
                                lengthscale_constraint=GreaterThan(10.))
        time_kernel = RBFKernel(active_dims=0,
                                lengthscale_constraint=GreaterThan(0.1))
        spin_kernels = [RBFKernel(active_dims=dimension,
                                  lengthscale_constraint=GreaterThan(7))
                        for dimension in range(2, 8)]

        class ExactGPModel(gpytorch.models.ExactGP):
            """
            Use the GpyTorch Exact GP
            """
            def __init__(self, train_x, train_y, likelihood):
                """Initialise the model"""
                super(ExactGPModel, self).__init__(train_x,
                                                   train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    time_kernel*mass_kernel*prod(spin_kernels),
                    lengthscale_constraint=gpytorch.constraints.LessThan(0.01)
                )

            def forward(self, x):
                """Run the forward method of the model"""
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        #data = np.genfromtxt(pkg_resources.resource_filename('heron',
        #                                                     'models/data/gt-M60-F1024.dat'))

        # These changes implement the new data interface in the model

        x, y = self.training_data.get_training_data(label=self.datalabel,
                                                    polarisation="+")

        training_x = self.training_x = torch.tensor(x*100).float().cuda()[:, :100].T
        training_y = self.training_y = torch.tensor(y*1e21).float().cuda()[:100]
        #training_yx = torch.tensor(data[:, -1]*1e21).float().cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=LessThan(10))
        model = ExactGPModel(training_x, training_y, likelihood)
        #model2 = ExactGPModel(training_x, training_yx, likelihood)
        state_vector = pkg_resources.resource_filename('heron', 'models/data/gt-gpytorch.pth')

        model = model.cuda()
        #model2 = model2.cuda()
        #FIXME fix this dirty hack
        model2 = model
        likelihood = likelihood.cuda()

        #model.load_state_dict(torch.load(state_vector))
        #model2.load_state_dict(torch.load(state_vector))

        return [model, model2], likelihood

    def _predict(self, times, p, polarisation="plus"):
        """
        Query the model for the mean and covariance tensors.
        Optionally include the covariance.

        Pararameters
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

        times_b = times.copy()
        points = self._generate_eval_matrix(p, times_b)
        points = torch.tensor(points).float().cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_preds = model(points)

        mean = f_preds.mean/self.strain_input_factor
        var = f_preds.variance.detach().double()/(self.strain_input_factor**2)
        covariance = f_preds.covariance_matrix.detach().double()
        covariance /= (self.strain_input_factor**2)

        return mean, var, covariance

    def mean(self, times, p, covariance=False):
        """
        Provide the mean waveform and its variance.
        Optionally include the covariance.

        Pararameters
        -------------
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

        timeseries = []
        covariances = []
        for polarisation in ["plus", "cross"]:
            mean, var, covariance = self._predict(times, p, polarisation=polarisation)
            timeseries.append(
                Timeseries(data=mean.cpu().numpy().astype(np.float64),
                           times=times,
                           variance=var.cpu().numpy().astype(np.float64)))
            covariances.append(covariance)
        if covariance_flag:
            return timeseries, covariances
        else:
            return timeseries

    def distribution(self, times, p, samples=100):
        """
        Return a number of sample waveforms from the GPR distribution.
        """
        times_b = times.copy()
        points = self._generate_eval_matrix(p, times_b)
        points = torch.tensor(points).float().cuda()
        return_samples = []
        for polarisation in [self.model_plus, self.model_cross]:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                f_preds = polarisation(points)
                y_preds = self.likelihood(f_preds)

                return_samples.append([Timeseries(data=sample.cpu()/self.strain_input_factor,
                                         times=times_b)
                                       for sample in y_preds.sample_n(samples)])
        return return_samples

    def frequency_domain_waveform(self, p, times=np.linspace(-2, 2, 1000)):
        """
        Return the frequency domain waveform.
        """
        mean, _, cov = self._predict(times, p)

        strain_f = mean.rfft(1)
        cov_f = cov.rfft(2)

        uncert_f = torch.stack([torch.diag(cov_f[:, :, 0]), torch.diag(cov_f[:, :, 1])]).T

        return strain_f, uncert_f

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
