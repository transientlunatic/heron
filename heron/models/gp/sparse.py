"""
Sparse Variational Gaussian Process surrogate model.

Uses inducing points to reduce complexity from O(N^3) to O(NM^2),
where M is the number of inducing points. With N=4000 training points
and M=200 inducing points, this is ~400x faster than exact GP.

Produces full covariance matrices from the variational posterior,
same interface as ExactGPSurrogate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    NaturalVariationalDistribution,
    VariationalStrategy,
)
from gpytorch.optim import NGD

from ..base import WaveformSurrogate
from ...types import Waveform, WaveformDict
from ..warping import get_warping, SimpleWarping, ChirpTimeWarping

logger = logging.getLogger("heron.models.gp.sparse")


def _kmeans_inducing_points(
    data: torch.Tensor, n_inducing: int, n_iters: int = 50
) -> torch.Tensor:
    """Select inducing points via k-means clustering.

    Parameters
    ----------
    data : Tensor, shape (N, D)
    n_inducing : int
        Number of inducing points.
    n_iters : int
        K-means iterations.

    Returns
    -------
    Tensor, shape (n_inducing, D)
    """
    n = data.shape[0]
    if n_inducing >= n:
        return data.clone()

    # Initialize with random subset
    indices = torch.randperm(n)[:n_inducing]
    centroids = data[indices].clone()

    for _ in range(n_iters):
        # Assign each point to nearest centroid
        dists = torch.cdist(data, centroids)
        assignments = dists.argmin(dim=1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_inducing, device=data.device)
        for i in range(n_inducing):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = data[mask].mean(dim=0)
                counts[i] = mask.sum()
            else:
                new_centroids[i] = centroids[i]

        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids


class _SparseGPModel(ApproximateGP):
    """Low-level GPyTorch Sparse Variational GP with Matern kernel.

    Parameters
    ----------
    inducing_points : Tensor, shape (M, D)
    mean_module : gpytorch.means.Mean or None
    nu : float
        Matern smoothness.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        mean_module: gpytorch.means.Mean | None = None,
        nu: float = 2.5,
        ls_min_per_dim: list[float] | None = None,
        train_y_var: float | None = None,
    ):
        import math
        from gpytorch.priors import LogNormalPrior

        n_inducing = inducing_points.shape[0]
        n_dims = inducing_points.shape[1]

        # Natural-gradient parameterisation: plain Adam on a
        # CholeskyVariationalDistribution converges very slowly for the
        # variational covariance factor (M*(M+1)/2 parameters) at M in the
        # hundreds-to-thousands range — verified directly on this problem
        # (predicted waveform ended up uncorrelated with the true one,
        # r~0.03, regardless of M). NGD is the standard fix.
        variational_distribution = NaturalVariationalDistribution(n_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = mean_module or gpytorch.means.ZeroMean()

        # Lengthscale floor as in ExactGPSurrogate, but init differently:
        # ExactGPSurrogate inits at data_range/4, which is fine when the
        # caller's grid spacing happens to be close to range/4 (true of the
        # 5-mass-ratio baseline). On a dense grid built specifically to
        # resolve fine structure (e.g. 30 mass ratios), data_range/4 can be
        # many multiples of the intended resolution — the LogNormal prior
        # then anchors near that oversized value and 400-ish Adam steps
        # aren't enough gradient pressure to pull it down, so the SVGP
        # converges to an over-smoothed, washed-out fit. When the caller
        # supplies ls_min_per_dim (i.e. states the intended resolution
        # explicitly), start there instead of at the coarse range/4 guess.
        kernels = []
        for dim in range(n_dims):
            ls_min = ls_min_per_dim[dim] if ls_min_per_dim is not None else 0.0005
            data_range = float(inducing_points[:, dim].max() - inducing_points[:, dim].min())
            if ls_min_per_dim is not None:
                init_ls = ls_min
            else:
                init_ls = max(data_range / 4.0, ls_min) if data_range > 0 else 1.0
            k = gpytorch.kernels.MaternKernel(
                nu=nu,
                active_dims=[dim],
                lengthscale_constraint=gpytorch.constraints.GreaterThan(ls_min),
            )
            k.lengthscale = init_ls
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
        if train_y_var:
            self.covar_module.outputscale = train_y_var

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseGPSurrogate(WaveformSurrogate):
    """Sparse Variational GP waveform surrogate.

    Same interface as ExactGPSurrogate but uses inducing points for
    O(NM^2) scaling instead of O(N^3).

    Parameters
    ----------
    train_x : Tensor, shape (N, D)
        Training input coordinates (last column is time).
    train_y_plus, train_y_cross : Tensor, shape (N,)
        Training strain values.
    n_inducing : int
        Number of inducing points (default 200).
    warping : str or TimeWarping
    nu : float
    output_scale : float
    device : str
    mean_module : gpytorch.means.Mean or None
    total_mass, distance : float
    training_iterations : int
    learning_rate : float
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y_plus: torch.Tensor,
        train_y_cross: torch.Tensor,
        n_inducing: int = 200,
        warping: str = "chirp",
        nu: float = 2.5,
        output_scale: float = 1e27,
        device: str = "cpu",
        mean_module: gpytorch.means.Mean | None = None,
        total_mass: float = 60.0,
        distance: float = 100.0,
        training_iterations: int = 400,
        learning_rate: float = 0.01,
        ngd_lr: float = 0.1,
        ls_min_time: float = 0.0005,
        ls_min_q: float = 0.0005,
        noise_floor_rel: float = 1e-6,
    ):
        self._device = torch.device(device)
        self.output_scale = output_scale
        self.nu = nu
        self.n_inducing = n_inducing
        self.mass_factor = total_mass
        self.distance_factor = distance
        self.ls_min_time = ls_min_time
        self.ls_min_q = ls_min_q
        self.noise_floor_rel = noise_floor_rel

        if isinstance(warping, str):
            self.warping = get_warping(warping)
        else:
            self.warping = warping

        self._train_x_raw = train_x.clone()

        # Warp time column
        train_x_warped = train_x.clone().to(self._device)
        train_x_warped[:, -1] = self.warping.warp(train_x_warped[:, -1])

        # Scale outputs
        train_y_plus_scaled = train_y_plus.to(self._device) * self.output_scale
        train_y_cross_scaled = train_y_cross.to(self._device) * self.output_scale

        # Select inducing points via k-means in warped space
        inducing_points = _kmeans_inducing_points(train_x_warped, n_inducing)

        # Per-dimension lengthscale floor: parameter dims use ls_min_q, the
        # last (time) dim uses ls_min_time. Mirrors ExactGPSurrogate.
        n_dims = train_x_warped.shape[1]
        ls_min_per_dim = [ls_min_q] * (n_dims - 1) + [ls_min_time]

        # Build models
        self.models: dict[str, _SparseGPModel] = {}
        self.likelihoods: dict[str, gpytorch.likelihoods.GaussianLikelihood] = {}
        self._train_data: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        for name, y in [("plus", train_y_plus_scaled), ("cross", train_y_cross_scaled)]:
            y_var = float(y.var())
            noise_floor = max(noise_floor_rel * y_var, 1e-10) if y_var > 0 else 1e-10
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(noise_floor),
            ).to(self._device)
            if y_var > 0:
                likelihood.noise = max(1e-4 * y_var, noise_floor)
            model = _SparseGPModel(
                inducing_points=inducing_points.clone(),
                mean_module=mean_module,
                nu=nu,
                ls_min_per_dim=ls_min_per_dim,
                train_y_var=y_var,
            ).to(self._device)
            self.models[name] = model
            self.likelihoods[name] = likelihood
            self._train_data[name] = (train_x_warped, y)

        if training_iterations > 0:
            self._train(training_iterations, lr=learning_rate, ngd_lr=ngd_lr)

    @staticmethod
    def _jitter_for(model: "_SparseGPModel") -> float:
        """Output-scale-aware Cholesky jitter for this model's K_mm.

        gpytorch's cholesky_jitter is an ABSOLUTE value, but our covariance
        entries are ~outputscale (output_scale=1e27 => outputscale ~1e12), so
        a fixed small jitter is numerically meaningless. Read the live
        outputscale (correct whether freshly initialised, mid-training, or
        just loaded from a checkpoint) the same way noise_floor_rel scales
        to y_var for ExactGPSurrogate.
        """
        outputscale = float(model.covar_module.outputscale.detach())
        return max(1e-3 * outputscale, 1e-6)

    def _train(self, iterations: int, lr: float = 0.01, ngd_lr: float = 0.1):
        """Train via variational ELBO.

        Uses two optimisers, as is standard for SVGP with a natural
        variational parameterisation: NGD for the variational distribution
        (natural_vec/natural_mat — the parameters plain Adam converges on
        very slowly for M in the hundreds-to-thousands range) and Adam for
        everything else (kernel/likelihood hyperparameters, inducing-point
        locations).
        """
        # Inducing points near merger are packed close together in warped
        # time (uniform-in-warped-time sampling) and, at the lengthscale
        # floors needed for smooth interpolation, K_mm (inducing-inducing
        # covariance) is nearly rank-deficient — the SVGP analogue of the
        # ill-conditioning that noise_floor_rel fixes for ExactGPSurrogate.
        # There's no observation noise on K_mm to lean on here, so raise the
        # Cholesky jitter directly instead.
        for name, model in self.models.items():
            likelihood = self.likelihoods[name]
            train_x, train_y = self._train_data[name]
            jitter = self._jitter_for(model)

            model.train()
            likelihood.train()

            variational_optimizer = NGD(
                model.variational_parameters(), num_data=train_y.size(0), lr=ngd_lr
            )
            hyperparameter_optimizer = torch.optim.Adam([
                {"params": model.hyperparameters()},
                {"params": likelihood.parameters()},
            ], lr=lr)

            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

            with gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
                for i in range(iterations):
                    variational_optimizer.zero_grad()
                    hyperparameter_optimizer.zero_grad()
                    output = model(train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    variational_optimizer.step()
                    hyperparameter_optimizer.step()
                    if i == 0 or (i + 1) % max(iterations // 10, 1) == 0 or i == iterations - 1:
                        logger.info(f"  [{name}] step {i + 1}/{iterations}  loss={loss.item():.4f}")

                model.eval()
                likelihood.eval()
                logger.info(f"Trained {name} SVGP for {iterations} iterations")

    def predict(self, parameters: dict) -> WaveformDict:
        """Generate waveform with uncertainty."""
        mass_ratio = parameters.get("mass_ratio")
        total_mass = parameters.get("total_mass", self.mass_factor)
        mass_factor = total_mass / self.mass_factor
        distance = parameters.get("luminosity_distance", self.distance_factor)
        distance_factor = distance / self.distance_factor

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
        points = torch.column_stack([
            torch.full((n_times,), mass_ratio, dtype=torch.float32),
            times,
        ]).to(self._device)

        points_warped = points.clone()
        points_warped[:, -1] = self.warping.warp(points_warped[:, -1])

        times_np = times.numpy()
        output = WaveformDict(
            parameters={k: v for k, v in parameters.items() if k not in ("time", "times")}
        )

        for pol_name in ("plus", "cross"):
            model = self.models[pol_name]
            jitter = self._jitter_for(model)
            # Latent GP distribution, not likelihood(model(x)): training data
            # is noiseless, so the learned noise is a regularisation artefact
            # and the predictive (latent + noise) covariance would inflate K
            # unnecessarily. Mirrors ExactGPSurrogate.predict().
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                    gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
                pred = model(points_warped)
                mean = pred.mean.cpu()
                covar = pred.covariance_matrix.cpu()

            # Cast to float64 before dividing: covar / output_scale^2 ~
            # 1e12/1e54 = 1e-42, which underflows float32 (min ~1.2e-38).
            # Mirrors ExactGPSurrogate.predict().
            output[pol_name] = Waveform(
                data=(mean.double() / self.output_scale / distance_factor).numpy(),
                times=times_np,
                covariance=(covar.double() / self.output_scale**2 / distance_factor**2).numpy(),
            )

        return output

    def save(self, path: str | Path) -> None:
        """Save checkpoint."""
        warping = self.warping
        if isinstance(warping, SimpleWarping):
            warping_config = {"type": "simple", "scale": warping.scale}
        elif isinstance(warping, ChirpTimeWarping):
            warping_config = {"type": "chirp", "alpha": warping.alpha, "t_ref": warping.t_ref}
        else:
            warping_config = {"type": str(type(warping).__name__)}

        checkpoint = {
            "version": 2,
            "model_type": "sparse",
            "model_states": {
                name: model.state_dict() for name, model in self.models.items()
            },
            "likelihood_states": {
                name: lik.state_dict() for name, lik in self.likelihoods.items()
            },
            "train_x": self._train_x_raw.cpu(),
            "n_inducing": self.n_inducing,
            "mass_factor": self.mass_factor,
            "distance_factor": self.distance_factor,
            "output_scale": self.output_scale,
            "nu": self.nu,
            "warping": warping_config,
            "ls_min_time": self.ls_min_time,
            "ls_min_q": self.ls_min_q,
            "noise_floor_rel": self.noise_floor_rel,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved SVGP checkpoint to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> SparseGPSurrogate:
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        warp_cfg = checkpoint["warping"]
        warping_obj = get_warping(
            warp_cfg["type"],
            **{k: v for k, v in warp_cfg.items() if k != "type"},
        )

        # We need dummy training data to initialize; the actual state
        # comes from the checkpoint
        train_x = checkpoint["train_x"]
        n = train_x.shape[0]

        instance = cls(
            train_x=train_x,
            train_y_plus=torch.zeros(n),
            train_y_cross=torch.zeros(n),
            n_inducing=checkpoint["n_inducing"],
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
        )

        for name, state in checkpoint["model_states"].items():
            instance.models[name].load_state_dict(state)
            instance.models[name].eval()
        for name, state in checkpoint["likelihood_states"].items():
            instance.likelihoods[name].load_state_dict(state)
            instance.likelihoods[name].eval()

        logger.info(f"Loaded SVGP checkpoint from {path}")
        return instance

    @property
    def parameter_names(self) -> list[str]:
        return ["mass_ratio"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        q_vals = self._train_x_raw[:, 0]
        return {"mass_ratio": (float(q_vals.min()), float(q_vals.max()))}
