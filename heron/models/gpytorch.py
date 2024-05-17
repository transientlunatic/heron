"""
These models utilise Gaussian process regression using GPyTorch.
"""

import torch
import gpytorch

from . import WaveformSurrogate
from ..types import Waveform


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()
    ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                active_dims=[0],
                lengthscale_constraint=gpytorch.constraints.GreaterThan(0.05),
            )
            * gpytorch.kernels.RBFKernel(
                active_dims=[1],
                lengthscale_constraint=gpytorch.constraints.GreaterThan(0.001),
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelKeOps(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()
    ):  # noise_constraint=gpytorch.constraints.LessThan(3))):
        super(ExactGPModelKeOps, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(
                active_dims=[0],
            )
            * gpytorch.kernels.keops.RBFKernel(
                active_dims=[1],
                lengthscale_constraint=gpytorch.constraints.GreaterThan(0.001),
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchSurrogate:
    def train(self, iterations=1000):
        self.model.train()
        self.model.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )

        for i in range(iterations):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.model.likelihood.eval()


class HeronNonSpinningApproximant(WaveformSurrogate, GPyTorchSurrogate):
    """
    A non-spinning model which is trained using a waveform approximant as the feedstock.
    """

    def __init__(self, train_x, train_y):
        self.train_x = train_x.cuda()
        self.train_y = train_y.cuda()
        self.model = ExactGPModelKeOps(self.train_x, self.train_y).cuda()
        self.model.likelihood.cuda()

    def _make_evaluation_manifold(
        self, parameter_min, parameter_max, parameter_n, time_min, time_max, time_n
    ):
        test_mass_ratio = torch.linspace(
            parameter_min, parameter_max, parameter_n, dtype=torch.float32
        ).cuda()
        test_times = torch.linspace(
            time_min, time_max, time_n, dtype=torch.float32
        ).cuda()

        # Warp the time axis
        test_times[test_times < 0] = test_times[test_times < 0] / 5

        test_data = torch.cartesian_prod(test_mass_ratio, test_times)
        return test_data

    def time_domain_manifold(self, parameters):
        a = parameters["mass ratio"]
        t = parameters["time"]
        points = self._make_evaluation_manifold(
            a["lower"], a["upper"], a["number"], t["lower"], t["upper"], t["number"]
        )
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model.likelihood(self.model(points))
            mean = observed_pred.mean
            # lower, upper = observed_pred.confidence_region()

        # Perform the unwarping of the time axis
        points[points[:, 1] < 0, 1] = points[points[:, 1] < 0, 1] * 5
        return mean, observed_pred, points

    def time_domain(self, parameters):
        """
        Return a timedomain waveform.
        """
        a = parameters["mass ratio"]
        t = parameters["time"]

        points = torch.vstack(
            [
                torch.ones(t["number"], dtype=torch.float32).cuda() * a,
                torch.linspace(
                    t["lower"], t["upper"], t["number"], dtype=torch.float32
                ).cuda(),
            ]
        ).T

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model.likelihood(self.model(points))
            mean = observed_pred.mean
            # lower, upper = observed_pred.confidence_region()
        # Perform the unwarping of the time axis
        points[points[:, 1] < 0, 1] = points[points[:, 1] < 0, 1] * 5
        return Waveform(
            data=mean.cpu(),
            times=points[:, 1].cpu(),
            covariance=observed_pred.covariance_matrix,
        )
