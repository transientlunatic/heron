"""
pytorch-based waveform models
"""

from . import Model
from .gw import BBHSurrogate, BBHNonSpinSurrogate, HofTSurrogate

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import RBFKernel
from matplotlib import pyplot as plt
import tqdm


def train(model, likelihood, num_epochs = 50, lr = 0.01):
    """
    Use ADAM to train  a Gaussian process regression model.
    """

    model.train()
    likelihood.train()


    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ],
                                 lr=lr)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=training_y.size(0))

    epochs_iter = tqdm.tqdm_notebook(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm_notebook(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    model.eval()
    likelihood.eval()

class GpyTorchGPR(Model):
    """
    The base-class for Torch-based heron models.
    """

    
    training = False
    evaluate = True

    time_factor = 100
    strain_input_factor = 1e19

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

    def mean(self, p, times):
        """
        Return the mean waveform at a given location in the 
        BBH parameter space.

        Parameters
        ----------
        
        """

        times_b = times.copy()
        points = self._generate_eval_matrix(p, times_b)

        points = torch.tensor(points).float()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_preds = self.model(points)
            y_preds = self.likelihood(f_preds)

        return Timeseries(data=f_preds.mean.numpy()[:,0]/self.strain_input_factor,
                          times=points[:,self.c_ind['time']].numpy()/self.time_factor,
                          variance=y_preds.variance.numpy()[:,0]/self.strain_input_factor), \
               Timeseries(data=f_preds.mean.numpy()[:,1]/self.strain_input_factor,
                          times=points[:,self.c_ind['time']].numpy()/self.time_factor,
                          variance=y_preds.variance.numpy()[:,1]/self.strain_input_factor)
    
    def distribution(self, p, times, samples=100):
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
        """

        times_b = times.copy()
        points = self._generate_eval_matrix(p, times_b)

        points = torch.tensor(points).float()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_preds = self.model(points)
            y_preds = self.likelihood(f_preds)

        with torch.no_grad():
            samples =  y_preds.sample_n(samples)
            
        results = []
        for sample in samples:
            sample_list = Timeseries(data=sample.numpy()[:,0]/self.strain_input_factor,
                                     times=points[:,self.c_ind['time']].numpy()/self.time_factor),\
                          Timeseries(data=sample.numpy()[:,1]/self.strain_input_factor,
                                     times=points[:,self.c_ind['time']].numpy()/self.time_factor)
            results.append(sample_list)
        results = np.array(results)

        return results
        
class TorchHeron(GpyTorchGPR, BBHSurrogate, HofTSurrogate):
    class TorchBigHeron(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2), 
                batch_shape=torch.Size([2]))
            
            variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
                VariationalStrategy(self, inducing_points, 
                                    variational_distribution, 
                                    learn_inducing_locations=True),
                num_tasks=2)
            
            super(TorchBigHeron, self).__init__(variational_strategy)
            
            self.mean_module =  gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
            self.covar_module = RBFKernel(batch_shape=torch.Size([2]))#gpytorch.kernels.ScaleKernel(
                #RBFKernel(batch_shape=torch.Size([2])), 
                #prod([RBFKernel(active_dims=torch.tensor([i]), 
                #                batch_shape=torch.Size([2])) for i in range(8)]),
                #batch_shape = torch.Size([2]))
        
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
