"""
These models utilise Gaussian process regression using GPyTorch.
"""

import torch
import gpytorch

from . import WaveformSurrogate
from ..types import Waveform, WaveformDict

disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()
    ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.device = device
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
        self.train_x = train_x
        self.train_y = train_y
        self.device = device
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(
                active_dims=[0],
            )
            * gpytorch.kernels.keops.RBFKernel(
                active_dims=[1],
                lengthscale_constraint=gpytorch.constraints.GreaterThan(
                    0.0015
                ),  # 0.001
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchSurrogate:
    def train(self, iterations=1000):
        for model in self.models.values():
            model.train()
            model.likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

            for i in range(iterations):
                optimizer.zero_grad()
                output = model(model.train_x)
                loss = -mll(output, model.train_y)
                loss.backward()
                optimizer.step()

            model.eval()
            model.likelihood.eval()


class HeronNonSpinningApproximant(WaveformSurrogate, GPyTorchSurrogate):
    """
    A non-spinning model which is trained using a waveform approximant as the feedstock.
    """

    def __init__(
        self,
        train_x_plus,
        train_x_cross,
        train_y_plus,
        train_y_cross,
        total_mass,
        distance,
        warp_scale=2,
        training=400,
    ):
        self.device = device
        self.output_scale = 1e27
        self.warp_scale = warp_scale
        self.mass_factor = total_mass
        self.distance_factor = distance
        self.train_x_plus = train_x_plus.to(self.device)
        self.train_x_cross = train_x_cross.to(self.device)
        times_cross = self.train_x_cross[:, 1]
        times_cross[times_cross < 0] = times_cross[times_cross < 0] / self.warp_scale
        self.train_x_cross[:, 1] = times_cross

        times_plus = self.train_x_plus[:, 1]
        times_plus[times_plus < 0] = times_plus[times_plus < 0] / self.warp_scale
        self.train_x_plus[:, 1] = times_plus

        self.train_y_plus = train_y_plus.cuda() * self.output_scale
        self.train_y_cross = train_y_cross.cuda() * self.output_scale
        self.models = {}
        self.models["plus"] = ExactGPModelKeOps(
            self.train_x_plus, self.train_y_plus
        ).to(self.device)
        self.models["cross"] = ExactGPModelKeOps(
            self.train_x_cross, self.train_y_cross
        ).to(self.device)
        for polarisation in ("plus", "cross"):
            self.models[polarisation].likelihood.cuda()

        self._args = {
            "total_mass": None,
            "mass_ratio": None,
            "luminosity_distance": None,
            "inclination": None,
        }

        self.train(training)

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
        test_times[test_times < 0] = test_times[test_times < 0] / self.warp_scale

        test_data = torch.cartesian_prod(test_mass_ratio, test_times)
        return test_data

    def time_domain_manifold(self, parameters):
        a = parameters["mass_ratio"]
        t = parameters["time"]
        points = self._make_evaluation_manifold(
            a["lower"], a["upper"], a["number"], t["lower"], t["upper"], t["number"]
        )
        M = parameters.get("total mass")
        outputs = {}
        for polarisation, model in self.models.items():

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = model.likelihood(model(points))
                mean = observed_pred.mean

            # Perform the unwarping of the time axis
            points[points[:, 1] < 0, 1] = points[points[:, 1] < 0, 1] * self.warp_scale

            outputs[polarisation] = (mean / self.output_scale, observed_pred, points)

        return outputs

    def time_domain(self, parameters, times=None):
        """
        Return a timedomain waveform.
        """
        a = parameters.get("mass_ratio", parameters.get("mass ratio"))
        epoch = parameters.get("gpstime")

        total_mass = parameters.get("total_mass", self.mass_factor)
        mass_factor = (total_mass / self.mass_factor).value

        if times is None:
            t = parameters.get("time", parameters.get("gpstime"))
            times = (
                torch.linspace(
                    t["lower"],
                    t["upper"],
                    t["number"],
                    dtype=torch.float32,
                )
                / mass_factor
            )
            times_i = times
            # parameters.pop("time")
        else:
            times_i = (
                torch.tensor(times.value, dtype=torch.float32) / mass_factor
            ) - epoch

        distance = parameters.get("luminosity_distance", self.distance_factor)
        distance_factor = distance / self.distance_factor

        points = torch.vstack(
            [
                torch.ones(
                    t["number"],
                    dtype=torch.float32,
                )
                * a,
                torch.linspace(
                    t["lower"],
                    t["upper"],
                    t["number"],
                    dtype=torch.float32,
                )
                / mass_factor,
            ]
        ).T.to(device=self.device)
        # Warp the time axis
        points[points[:, 1] < 0, 1] = points[points[:, 1] < 0, 1] / self.warp_scale
        # Extract the waveform

        parameters.pop("time")
        output = WaveformDict(parameters=parameters)
        for polarisation in ("plus", "cross"):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.models[polarisation].likelihood(
                    self.models[polarisation](points)
                )
                mean = observed_pred.mean
            # Perform the unwarping of the time axis
            points[points[:, 1] < 0, 1] = points[points[:, 1] < 0, 1] * self.warp_scale

            output.waveforms[polarisation] = Waveform(
                data=mean.cpu() / self.output_scale / distance_factor,
                times=times,
                covariance=observed_pred.covariance_matrix.cpu()
                / self.output_scale
                / self.output_scale
                / distance_factor**2,
            )

        return output
