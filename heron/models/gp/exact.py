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
        ls_min_per_dim: list[float] | None = None,
        noise_floor_rel: float = 1e-6,
    ):
        import math
        from gpytorch.priors import LogNormalPrior

        y_var = float(train_y.var())
        # Hard floor on noise: prevents ill-conditioning when training points
        # are highly correlated (ls >> spacing). When ls_min_time > training
        # spacing the kernel matrix is nearly rank-deficient; noise_floor_rel
        # controls the minimum noise / y_var so CG converges. Using the latent
        # GP for K means this noise does NOT inflate the surrogate uncertainty.
        noise_floor = max(noise_floor_rel * y_var, 1e-10)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(noise_floor),
        )
        super().__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.mean_module = mean_module or gpytorch.means.ZeroMean()

        n_dims = train_x.shape[1]
        kernels = []
        for dim in range(n_dims):
            ls_min = ls_min_per_dim[dim] if ls_min_per_dim is not None else 0.0005
            data_range = float(train_x[:, dim].max() - train_x[:, dim].min())
            init_ls = max(data_range / 4.0, ls_min) if data_range > 0 else 1.0
            k = gpytorch.kernels.MaternKernel(
                nu=nu,
                active_dims=[dim],
                lengthscale_constraint=gpytorch.constraints.GreaterThan(ls_min),
            )
            k.lengthscale = init_ls
            # LogNormal prior centred on the initialised lengthscale with
            # σ=1 in log-space. Penalises collapse toward zero (overfitting
            # one mass-ratio grid) and runaway growth (underfitting).
            k.register_prior(
                "lengthscale_prior",
                LogNormalPrior(loc=math.log(init_ls), scale=1.0),
                "lengthscale",
            )
            kernels.append(k)

        product_kernel = kernels[0]
        for k in kernels[1:]:
            product_kernel = product_kernel * k
        self.covar_module = gpytorch.kernels.ScaleKernel(product_kernel)

        # Initialise outputscale and noise from data statistics.
        if y_var > 0:
            self.covar_module.outputscale = y_var
            self.likelihood.noise = max(1e-4 * y_var, noise_floor)

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
        optimizer: str = "lbfgs",
        lr: float | None = None,
        ls_min_time: float = 0.0005,
        ls_min_q: float = 0.0005,
        noise_floor_rel: float = 1e-6,
        cholesky_size: int = 2000,
    ):
        self._device = torch.device(device)
        self.output_scale = output_scale
        self.nu = nu
        self.mass_factor = total_mass
        self.distance_factor = distance
        self.ls_min_time = ls_min_time
        self.ls_min_q = ls_min_q
        self.noise_floor_rel = noise_floor_rel
        self.cholesky_size = cholesky_size

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

        # Build GP models for each polarisation.
        # ls_min_per_dim: per-dimension minimum lengthscale (in warped space).
        # The time dimension (last) uses ls_min_time; parameter dims use ls_min_q.
        n_dims = train_x_warped.shape[1]
        ls_min_per_dim = [ls_min_q] * (n_dims - 1) + [ls_min_time]

        self.models: dict[str, _ExactGPModel] = {}
        for name, y in [("plus", train_y_plus_scaled), ("cross", train_y_cross_scaled)]:
            model = _ExactGPModel(
                train_x_warped, y,
                mean_module=mean_module,
                nu=nu,
                ls_min_per_dim=ls_min_per_dim,
                noise_floor_rel=noise_floor_rel,
            ).to(self._device)
            model.likelihood.to(self._device)
            self.models[name] = model

        if training_iterations > 0:
            self._train(training_iterations, optimizer_type=optimizer, lr=lr)

    def _train(
        self,
        iterations: int,
        optimizer_type: str = "lbfgs",
        lr: float | None = None,
    ):
        """Train all GP models via MLL optimisation.

        Parameters
        ----------
        iterations : int
            Number of optimiser steps (L-BFGS: ~100 is usually enough; Adam: ~1000).
        optimizer_type : str
            ``"lbfgs"`` (default) or ``"adam"``.
        lr : float or None
            Learning rate / initial step size. Defaults to 1.0 for L-BFGS, 0.05 for Adam.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        for name, model in self.models.items():
            model.train()
            model.likelihood.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            loss_history = []

            # Force Cholesky up to cholesky_size: exact, no CG NaN risk.
            # Default 2000 matches the CPU-era "safe" ceiling; raise it (GPU
            # can afford a bigger N^3 factorisation) to train past N=2000.
            cholesky_ctx = gpytorch.settings.max_cholesky_size(self.cholesky_size)

            if optimizer_type == "lbfgs":
                _lr = lr if lr is not None else 1.0
                opt = torch.optim.LBFGS(
                    model.parameters(),
                    lr=_lr,
                    line_search_fn="strong_wolfe",
                )
                eval_count = 0
                bar = tqdm(total=iterations, desc=f"  {name}", unit="step") if tqdm else None

                def closure():
                    nonlocal eval_count
                    opt.zero_grad()
                    with cholesky_ctx:
                        output = model(model.train_x)
                        loss = -mll(output, model.train_y)
                    loss.backward()
                    loss_val = float(loss.item())
                    loss_history.append(loss_val)
                    eval_count += 1
                    if bar is not None:
                        bar.set_postfix(loss=f"{loss_val:.4f}", evals=eval_count)
                    logger.debug(f"  [{name}] eval {eval_count}: loss={loss_val:.4f}")
                    return loss

                for _ in range(iterations):
                    opt.step(closure)
                    if bar is not None:
                        bar.update(1)

                if bar is not None:
                    bar.close()

            else:  # adam
                _lr = lr if lr is not None else 0.05
                opt = torch.optim.Adam(model.parameters(), lr=_lr)
                iter_range = (
                    tqdm(range(iterations), desc=f"  {name}", unit="iter")
                    if tqdm else range(iterations)
                )
                for i in iter_range:
                    opt.zero_grad()
                    with cholesky_ctx:
                        output = model(model.train_x)
                        loss = -mll(output, model.train_y)
                    loss.backward()
                    opt.step()
                    loss_val = float(loss.item())
                    loss_history.append(loss_val)
                    if tqdm:
                        iter_range.set_postfix(loss=f"{loss_val:.4f}")
                    logger.debug(f"  [{name}] iter {i + 1:4d}/{iterations}: loss={loss_val:.4f}")

            logger.info(
                f"  [{name}] training complete: "
                f"final loss={loss_history[-1]:.4f} over {len(loss_history)} evals"
            )
            model.eval()
            model.likelihood.eval()
            model.loss_history = loss_history

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
            # gpytorch's default max_cholesky_size is 800; above that it
            # silently falls back to CG, which can fail to converge (seen
            # directly: N=2000 with tight lengthscales left CG short of
            # tolerance, producing an inaccurate, inflated covariance).
            # _train() already raises this for training — predict() needs
            # the same override (self.cholesky_size) for consistent, exact
            # covariances at the same N.
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                    gpytorch.settings.max_cholesky_size(self.cholesky_size):
                # Use the LATENT distribution (no observation noise) for the
                # covariance. The training noise σ² is a regularisation artefact
                # (LALSuite training data is noiseless); the physical surrogate
                # uncertainty is the latent posterior covariance K_latent.
                latent = model(points_warped)
                mean = latent.mean.cpu()
                covar = latent.covariance_matrix.cpu()

            # Cast to float64 before dividing: covar / output_scale² ~ 1e12/1e54 = 1e-42,
            # which underflows float32 (min ~1.2e-38).
            output[pol_name] = Waveform(
                data=(mean.double() / self.output_scale / distance_factor).numpy(),
                times=times_np,
                covariance=(covar.double() / self.output_scale**2 / distance_factor**2).numpy(),
            )

        return output

    def save(self, path: str | Path) -> None:
        """Save checkpoint."""
        import datetime
        from heron import __version__ as heron_version

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
            "format_version": 4,
            "heron_version": heron_version,
            "model_class": type(self).__name__,
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
            "parameter_names": self.parameter_names,
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
            "ls_min_time": self.ls_min_time,
            "ls_min_q": self.ls_min_q,
            "noise_floor_rel": self.noise_floor_rel,
            "cholesky_size": self.cholesky_size,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (heron {heron_version})")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> ExactGPSurrogate:
        """Load a pre-trained model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        fmt = checkpoint.get("format_version", checkpoint.get("version"))

        saved_class = checkpoint.get("model_class")
        if saved_class is not None and saved_class != cls.__name__:
            logger.warning(
                f"Checkpoint was saved by {saved_class} but is being loaded by {cls.__name__}"
            )

        saved_heron = checkpoint.get("heron_version")
        if saved_heron is not None:
            logger.info(f"Checkpoint saved with heron {saved_heron} on {checkpoint.get('saved_at', 'unknown date')}")

        warp_cfg = checkpoint["warping"]
        warping_obj = get_warping(
            warp_cfg["type"],
            **{k: v for k, v in warp_cfg.items() if k != "type"},
        )

        # Handle v1 checkpoints (from old HeronNonSpinningApproximantMatern)
        if fmt is None:
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
            ls_min_time=checkpoint.get("ls_min_time", 0.0005),
            ls_min_q=checkpoint.get("ls_min_q", 0.0005),
            noise_floor_rel=checkpoint.get("noise_floor_rel", 1e-6),
            cholesky_size=checkpoint.get("cholesky_size", 2000),
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
