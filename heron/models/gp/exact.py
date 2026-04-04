"""
Exact Gaussian Process surrogate model with Matern kernels.

This is the core Heron surrogate: a GPyTorch ExactGP that interpolates
gravitational waveforms in (parameter, warped-time) space, returning
full covariance matrices alongside mean predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import gpytorch

from ..base import WaveformSurrogate
from ...types import Waveform, WaveformDict
from ..warping import get_warping, SimpleWarping, ChirpTimeWarping

logger = logging.getLogger("heron.models.gp.exact")


class _ExactGPModel(gpytorch.models.ExactGP):
    """Low-level GPyTorch ExactGP with product Matern kernel.

    Parameters
    ----------
    train_x : Tensor, shape (N, D)
    train_y : Tensor, shape (N,)
    mean_module : gpytorch.means.Mean
        Mean function (default: ZeroMean).
    nu : float
        Matern smoothness parameter (1.5 or 2.5).
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        mean_module: gpytorch.means.Mean | None = None,
        nu: float = 2.5,
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.mean_module = mean_module or gpytorch.means.ZeroMean()

        n_dims = train_x.shape[1]
        kernels = []
        for dim in range(n_dims):
            kernels.append(
                gpytorch.kernels.MaternKernel(
                    nu=nu,
                    active_dims=[dim],
                    lengthscale_constraint=gpytorch.constraints.GreaterThan(0.0005),
                )
            )
        product_kernel = kernels[0]
        for k in kernels[1:]:
            product_kernel = product_kernel * k
        self.covar_module = gpytorch.kernels.ScaleKernel(product_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPSurrogate(WaveformSurrogate):
    """Exact GP waveform surrogate with Matern kernels.

    Trains separate GPs for plus and cross polarisations on
    (parameter, warped-time) input space. Returns waveforms with
    full covariance matrices from the GP posterior.

    Parameters
    ----------
    train_x : Tensor, shape (N, D)
        Training input coordinates. Last column is time, preceding
        columns are physical parameters (e.g. mass_ratio).
    train_y_plus, train_y_cross : Tensor, shape (N,)
        Training strain values for each polarisation.
    warping : str or TimeWarping
        Time warping strategy ('chirp', 'simple', or a warping object).
    nu : float
        Matern smoothness (default 2.5).
    output_scale : float
        Rescaling factor for numerical stability (default 1e27).
    device : str
        Torch device ('cpu' or 'cuda').
    mean_module : gpytorch.means.Mean or None
        Custom mean function (e.g. PN-based). None → ZeroMean.
    total_mass : float
        Reference total mass used during training (solar masses).
    distance : float
        Reference luminosity distance used during training (Mpc).
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y_plus: torch.Tensor,
        train_y_cross: torch.Tensor,
        warping: str = "chirp",
        nu: float = 2.5,
        output_scale: float = 1e27,
        device: str = "cpu",
        mean_module: gpytorch.means.Mean | None = None,
        total_mass: float = 60.0,
        distance: float = 100.0,
        training_iterations: int = 400,
    ):
        self._device = torch.device(device)
        self.output_scale = output_scale
        self.nu = nu
        self.mass_factor = total_mass
        self.distance_factor = distance

        # Set up warping
        if isinstance(warping, str):
            self.warping = get_warping(warping)
        else:
            self.warping = warping

        # Store unwarped training data for checkpointing
        self._train_x_raw = train_x.clone()

        # Warp the time column (last column)
        train_x_warped = train_x.clone().to(self._device)
        train_x_warped[:, -1] = self.warping.warp(train_x_warped[:, -1])

        # Scale outputs
        train_y_plus_scaled = train_y_plus.to(self._device) * self.output_scale
        train_y_cross_scaled = train_y_cross.to(self._device) * self.output_scale

        # Build GP models for each polarisation
        self.models: dict[str, _ExactGPModel] = {}
        for name, y in [("plus", train_y_plus_scaled), ("cross", train_y_cross_scaled)]:
            model = _ExactGPModel(
                train_x_warped, y,
                mean_module=mean_module,
                nu=nu,
            ).to(self._device)
            model.likelihood.to(self._device)
            self.models[name] = model

        if training_iterations > 0:
            self._train(training_iterations)

    def _train(self, iterations: int, lr: float = 0.05):
        """Train all GP models via MLL optimisation."""
        for name, model in self.models.items():
            model.train()
            model.likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

            for i in range(iterations):
                optimizer.zero_grad()
                output = model(model.train_x)
                loss = -mll(output, model.train_y)
                loss.backward()
                optimizer.step()

            model.eval()
            model.likelihood.eval()
            logger.info(f"Trained {name} model for {iterations} iterations")

    def predict(self, parameters: dict) -> WaveformDict:
        """Generate waveform with uncertainty.

        Parameters
        ----------
        parameters : dict
            Must contain 'mass_ratio' and 'time' (dict with lower/upper/number)
            or 'times' (array). Optional: 'total_mass', 'luminosity_distance'.
        """
        mass_ratio = parameters.get("mass_ratio")
        total_mass = parameters.get("total_mass", self.mass_factor)
        mass_factor = total_mass / self.mass_factor
        distance = parameters.get("luminosity_distance", self.distance_factor)
        distance_factor = distance / self.distance_factor

        # Build time array
        if "times" in parameters:
            times = torch.tensor(parameters["times"], dtype=torch.float32) / mass_factor
        elif "time" in parameters:
            t = parameters["time"]
            times = torch.linspace(
                t["lower"], t["upper"], t["number"], dtype=torch.float32
            ) / mass_factor
        else:
            raise ValueError("parameters must contain 'times' or 'time'")

        n_times = len(times)

        # Build evaluation points: (mass_ratio, time)
        points = torch.column_stack([
            torch.full((n_times,), mass_ratio, dtype=torch.float32),
            times,
        ]).to(self._device)

        # Warp the time column
        points_warped = points.clone()
        points_warped[:, -1] = self.warping.warp(points_warped[:, -1])

        # Predict
        times_np = times.numpy()
        output = WaveformDict(
            parameters={k: v for k, v in parameters.items() if k != "time" and k != "times"}
        )

        for pol_name in ("plus", "cross"):
            model = self.models[pol_name]
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = model.likelihood(model(points_warped))
                mean = pred.mean.cpu()
                covar = pred.covariance_matrix.cpu()

            output[pol_name] = Waveform(
                data=(mean / self.output_scale / distance_factor).numpy(),
                times=times_np,
                covariance=(covar / self.output_scale**2 / distance_factor**2).numpy(),
            )

        return output

    def save(self, path: str | Path) -> None:
        """Save checkpoint."""
        warping = self.warping
        if isinstance(warping, SimpleWarping):
            warping_config = {"type": "simple", "scale": warping.scale}
        elif isinstance(warping, ChirpTimeWarping):
            warping_config = {
                "type": "chirp",
                "alpha": warping.alpha,
                "t_ref": warping.t_ref,
            }
        else:
            warping_config = {"type": str(type(warping).__name__)}

        checkpoint = {
            "version": 2,
            "model_states": {
                name: model.state_dict()
                for name, model in self.models.items()
            },
            "train_x": self._train_x_raw.cpu(),
            "train_y": {
                name: model.train_y.cpu() for name, model in self.models.items()
            },
            "mass_factor": self.mass_factor,
            "distance_factor": self.distance_factor,
            "output_scale": self.output_scale,
            "nu": self.nu,
            "warping": warping_config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> ExactGPSurrogate:
        """Load a pre-trained model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        warp_cfg = checkpoint["warping"]
        warping_obj = get_warping(
            warp_cfg["type"],
            **{k: v for k, v in warp_cfg.items() if k != "type"},
        )

        # Handle v1 checkpoints (from old HeronNonSpinningApproximantMatern)
        if "version" not in checkpoint:
            return cls._load_v1(checkpoint, warping_obj, device)

        instance = cls(
            train_x=checkpoint["train_x"],
            train_y_plus=checkpoint["train_y"]["plus"] / checkpoint["output_scale"],
            train_y_cross=checkpoint["train_y"]["cross"] / checkpoint["output_scale"],
            warping=warping_obj,
            nu=checkpoint["nu"],
            output_scale=checkpoint["output_scale"],
            device=device,
            total_mass=checkpoint["mass_factor"],
            distance=checkpoint["distance_factor"],
            training_iterations=0,
        )

        for name, state in checkpoint["model_states"].items():
            instance.models[name].load_state_dict(state)
            instance.models[name].eval()
            instance.models[name].likelihood.eval()

        logger.info(f"Loaded checkpoint from {path}")
        return instance

    @classmethod
    def _load_v1(cls, checkpoint, warping_obj, device):
        """Load a v1 checkpoint (old HeronNonSpinningApproximantMatern format)."""
        # V1 saved warped+scaled data; unwarp before passing to __init__
        train_x_plus = checkpoint["train_x_plus"].clone()
        train_x_plus[:, 1] = warping_obj.unwarp(train_x_plus[:, 1])

        instance = cls(
            train_x=train_x_plus,
            train_y_plus=checkpoint["train_y_plus"] / checkpoint["output_scale"],
            train_y_cross=checkpoint["train_y_cross"] / checkpoint["output_scale"],
            warping=warping_obj,
            nu=checkpoint["nu"],
            output_scale=checkpoint["output_scale"],
            device=device,
            total_mass=checkpoint["mass_factor"],
            distance=checkpoint["distance_factor"],
            training_iterations=0,
        )

        instance.models["plus"].load_state_dict(checkpoint["model_plus_state"])
        instance.models["cross"].load_state_dict(checkpoint["model_cross_state"])
        for model in instance.models.values():
            model.eval()
            model.likelihood.eval()

        return instance

    @property
    def parameter_names(self) -> list[str]:
        return ["mass_ratio"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        # Infer from training data (first column = mass_ratio)
        q_vals = self._train_x_raw[:, 0]
        return {
            "mass_ratio": (float(q_vals.min()), float(q_vals.max())),
        }
