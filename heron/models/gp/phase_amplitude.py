"""
Phase-amplitude decomposition GP surrogate.

An alternative to `ExactGPSurrogate` (heron/models/gp/exact.py) for
representing gravitational waveforms. Instead of modelling h(q, t) directly
with a Matern kernel -- a structurally poor fit for an oscillatory target,
see design.md's "Architectural fork" note -- this decomposes the complex
strain into amplitude and phase,

    h_plus(t) - i * h_cross(t) = A(t) * exp(-i * Phi(t))
    => h_plus = A * cos(Phi),  h_cross = A * sin(Phi)

and fits two independent GPs -- one on log-amplitude, one on unwrapped
phase -- both smooth, non-oscillatory functions well suited to a Matern
kernel. Plus/cross strain and its covariance are reconstructed at
predict() time via first-order (delta-method) error propagation.

Not a replacement for ExactGPSurrogate -- both implement the same
WaveformSurrogate interface and are interchangeable wherever a surrogate
is consumed (heron/gw_likelihood.py, heron/evaluate.py, ...).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import gpytorch

from ..base import WaveformSurrogate
from ...types import Waveform, WaveformDict
from ..warping import get_warping, SimpleWarping, ChirpTimeWarping, MassRatioChirpTimeWarping
from .exact import _ExactGPModel

logger = logging.getLogger("heron.models.gp.phase_amplitude")


def strain_to_amplitude_phase(
    x: torch.Tensor,
    y_plus: torch.Tensor,
    y_cross: torch.Tensor,
    phase_reference: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose (h_plus, h_cross) training data into (log-amplitude, phase).

    Convention: h_plus - i*h_cross = A * exp(-i*Phi), so
    h_plus = A*cos(Phi), h_cross = A*sin(Phi).

    Groups rows by mass_ratio (x[:, 0]) and sorts each group by time
    (x[:, -1]) before unwrapping -- phase must be continuous in time within
    each mass ratio for `np.unwrap` to produce a smooth target rather than
    a sawtooth.

    Parameters
    ----------
    x : Tensor, shape (N, D)
        Training coordinates; first column mass_ratio, last column time.
    y_plus, y_cross : Tensor, shape (N,)
        Plus/cross strain training targets.
    phase_reference : ndarray, shape (N,), optional
        An accurate a-priori unwrapped phase evaluated at the same rows as
        `x` (e.g. a full-IMR mean function). When given, the phase target
        is recovered by *mean-referenced unwrapping*: wrap the principal
        (atan2) phase relative to the reference, unwrap the slowly-varying
        difference, and add the reference back. This is essential wherever
        the training grid undersamples the raw phase: plain `np.unwrap`
        needs consecutive samples to advance by < pi, which the
        warped-uniform training grids violate near merger at EVERY mass
        ratio (7-9 samples per q on the dense30 grid, up to 177/199 at
        q=0.10) -- silently deleting cycles and corrupting the target. The
        mean-referenced difference advances by ~0.1 rad/sample instead, so
        its unwrap is valid on the sparse grid. Only pass a reference
        accurate to well under pi per training-sample step (a full IMR
        approximant qualifies; a PN mean at M~60 does not).

    Returns
    -------
    x_reordered, log_amplitude, phase : Tensor, shape (N, D), (N,), (N,)
        Rows grouped by mass_ratio then sorted by time; log_amplitude and
        phase are aligned with x_reordered.
    """
    x_np = x.detach().cpu().numpy()
    yp_np = y_plus.detach().cpu().numpy()
    yc_np = y_cross.detach().cpu().numpy()

    q_col = x_np[:, 0]
    t_col = x_np[:, -1]

    x_chunks, logA_chunks, phase_chunks = [], [], []
    for q in np.unique(q_col):
        idx = np.where(q_col == q)[0]
        idx = idx[np.argsort(t_col[idx])]

        # h_plus = A*cos(Phi), h_cross = A*sin(Phi) => Phi = atan2(h_cross, h_plus).
        amplitude = np.sqrt(yp_np[idx] ** 2 + yc_np[idx] ** 2)
        principal = np.arctan2(yc_np[idx], yp_np[idx])
        if phase_reference is None:
            phase = np.unwrap(principal)
        else:
            ref = phase_reference[idx]
            diff = np.unwrap(np.angle(np.exp(1j * (principal - ref))))
            # Canonicalise the (physically meaningless) overall 2*pi branch
            # per mass ratio: anchoring on the first sample alone lets taper
            # artefacts land isolated q's a full 2*pi from their neighbours,
            # which needlessly breaks the residual's smoothness in q.
            diff -= 2 * np.pi * np.round(np.median(diff) / (2 * np.pi))
            phase = ref + diff

        x_chunks.append(x_np[idx])
        logA_chunks.append(np.log(amplitude + 1e-30))
        phase_chunks.append(phase)

    return (
        torch.tensor(np.concatenate(x_chunks), dtype=x.dtype),
        torch.tensor(np.concatenate(logA_chunks), dtype=y_plus.dtype),
        torch.tensor(np.concatenate(phase_chunks), dtype=y_plus.dtype),
    )


class PhaseAmplitudeGPSurrogate(WaveformSurrogate):
    """Phase-amplitude decomposition GP waveform surrogate.

    Fits two independent GPs -- log-amplitude and unwrapped phase -- on
    (parameter, warped-time) input space, and reconstructs plus/cross
    strain (with covariance, via first-order error propagation) at
    predict() time. See module docstring for the decomposition convention.

    Parameters mirror `ExactGPSurrogate`, with two differences: no
    `output_scale` (log-amplitude and phase are already O(1-100), unlike
    raw strain ~1e-21 -- no numerical-stability rescaling needed), and
    `mean_module_amplitude`/`mean_module_phase` instead of one
    `mean_module`.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y_plus: torch.Tensor,
        train_y_cross: torch.Tensor,
        warping: str = "chirp",
        nu: float = 2.5,
        device: str = "cpu",
        mean_module_amplitude: gpytorch.means.Mean | None = None,
        mean_module_phase: gpytorch.means.Mean | None = None,
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
        self.nu = nu
        self.mass_factor = total_mass
        self.distance_factor = distance
        self.ls_min_time = ls_min_time
        self.ls_min_q = ls_min_q
        self.noise_floor_rel = noise_floor_rel
        self.cholesky_size = cholesky_size

        if isinstance(warping, str):
            self.warping = get_warping(warping)
        else:
            self.warping = warping

        # Store raw (unwarped, un-decomposed) training data for
        # checkpointing -- log-amplitude/phase are re-derived deterministically
        # in this same constructor on load(), same pattern as ExactGPSurrogate.
        self._train_x_raw = train_x.clone()
        self._train_y_plus_raw = train_y_plus.clone()
        self._train_y_cross_raw = train_y_cross.clone()

        # Mean-referenced unwrapping (see strain_to_amplitude_phase): only
        # valid when the phase mean is accurate to << pi per training-time
        # step, i.e. a full-IMR approximant mean -- not a PN mean, whose
        # residual at M~60 is itself many radians per step.
        from .mean import _LALApproximantMeanBase

        phase_reference = None
        if isinstance(mean_module_phase, _LALApproximantMeanBase):
            x_ref = train_x.clone().to(torch.float64)
            x_ref[:, -1] = self.warping.warp(
                x_ref[:, -1], mass_ratio=x_ref[:, 0]
            )
            with torch.no_grad():
                phase_reference = mean_module_phase(x_ref).cpu().numpy()
            logger.info(
                "Using mean-referenced phase unwrapping "
                f"({mean_module_phase.approximant})"
            )

        x_sorted, log_amplitude, phase = strain_to_amplitude_phase(
            train_x, train_y_plus, train_y_cross,
            phase_reference=phase_reference,
        )

        x_warped = x_sorted.clone().to(self._device)
        x_warped[:, -1] = self.warping.warp(
            x_warped[:, -1], mass_ratio=x_warped[:, 0]
        )
        log_amplitude = log_amplitude.to(self._device)
        phase = phase.to(self._device)

        # ls_min_per_dim: per-dimension minimum lengthscale (in warped space).
        # The time dimension (last) uses ls_min_time; parameter dims use ls_min_q.
        n_dims = x_warped.shape[1]
        ls_min_per_dim = [ls_min_q] * (n_dims - 1) + [ls_min_time]

        mean_modules = {"log_amplitude": mean_module_amplitude, "phase": mean_module_phase}
        targets = {"log_amplitude": log_amplitude, "phase": phase}

        self.models: dict[str, _ExactGPModel] = {}
        for name, y in targets.items():
            model = _ExactGPModel(
                x_warped, y,
                mean_module=mean_modules[name],
                nu=nu,
                ls_min_per_dim=ls_min_per_dim,
                noise_floor_rel=noise_floor_rel,
            ).to(self._device)
            model.likelihood.to(self._device)
            self.models[name] = model

        # Lazily-built float64 clones of self.models, used only by predict().
        # Same roundoff-noise rationale as ExactGPSurrogate._get_predict_models.
        self._predict_models: dict[str, _ExactGPModel] | None = None

        if training_iterations > 0:
            self._train(training_iterations, optimizer_type=optimizer, lr=lr)

    def _train(
        self,
        iterations: int,
        optimizer_type: str = "lbfgs",
        lr: float | None = None,
    ):
        """Train the log-amplitude and phase GPs via MLL optimisation.

        Structurally identical to ExactGPSurrogate._train (same optimiser
        choices, Cholesky-forced training) applied to this model's two
        targets instead of plus/cross.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        self._predict_models = None  # invalidate float64 predict cache

        for name, model in self.models.items():
            model.train()
            model.likelihood.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            loss_history = []

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

    def _get_predict_models(self) -> dict[str, _ExactGPModel]:
        """Float64 clones of self.models, built lazily and cached.

        See ExactGPSurrogate._get_predict_models -- same float32 roundoff
        rationale applies here (cond(K) ~1e5 combined with float32's ~7
        digits leaves too few clean digits at fine query resolution).
        """
        if self._predict_models is None:
            import copy

            self._predict_models = {}
            for name, model in self.models.items():
                pm = copy.deepcopy(model).double()
                pm.eval()
                pm.likelihood.eval()
                self._predict_models[name] = pm
        return self._predict_models

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

        if "times" in parameters:
            times = torch.tensor(parameters["times"], dtype=torch.float64) / mass_factor
        elif "time" in parameters:
            t = parameters["time"]
            times = torch.linspace(
                t["lower"], t["upper"], t["number"], dtype=torch.float64
            ) / mass_factor
        else:
            raise ValueError("parameters must contain 'times' or 'time'")

        n_times = len(times)

        points = torch.column_stack([
            torch.full((n_times,), mass_ratio, dtype=torch.float64),
            times,
        ]).to(self._device)

        points_warped = points.clone()
        points_warped[:, -1] = self.warping.warp(
            points_warped[:, -1], mass_ratio=points_warped[:, 0]
        )

        times_np = times.numpy()
        predict_models = self._get_predict_models()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.max_cholesky_size(self.cholesky_size):
            latent_logA = predict_models["log_amplitude"](points_warped)
            mean_logA = latent_logA.mean.cpu()
            cov_logA = latent_logA.covariance_matrix.cpu()

            latent_phase = predict_models["phase"](points_warped)
            mean_phase = latent_phase.mean.cpu()
            cov_phase = latent_phase.covariance_matrix.cpu()

        amplitude = torch.exp(mean_logA)
        cos_phase = torch.cos(mean_phase)
        sin_phase = torch.sin(mean_phase)

        h_plus = amplitude * cos_phase
        h_cross = amplitude * sin_phase

        # First-order (delta-method) error propagation from the independent
        # (log-amplitude, phase) GPs to (h_plus, h_cross). Each output
        # sample depends only on the logA/phase value at that same time, so
        # the Jacobian is diagonal and the propagated covariance is a
        # diagonal congruence: Cov(h) = diag(d) @ Cov(x) @ diag(d) =
        # outer(d, d) * Cov(x) (elementwise). Accurate while sigma_phase is
        # small (roughly <~0.3 rad); breaks down for large phase
        # uncertainty since cos/sin are only linearised locally.
        d_hp_dlogA = amplitude * cos_phase
        d_hp_dphase = -amplitude * sin_phase
        d_hc_dlogA = amplitude * sin_phase
        d_hc_dphase = amplitude * cos_phase

        cov_plus = torch.outer(d_hp_dlogA, d_hp_dlogA) * cov_logA \
            + torch.outer(d_hp_dphase, d_hp_dphase) * cov_phase
        cov_cross = torch.outer(d_hc_dlogA, d_hc_dlogA) * cov_logA \
            + torch.outer(d_hc_dphase, d_hc_dphase) * cov_phase

        output = WaveformDict(
            parameters={k: v for k, v in parameters.items() if k != "time" and k != "times"}
        )
        output["plus"] = Waveform(
            data=(h_plus / distance_factor).numpy(),
            times=times_np,
            covariance=(cov_plus / distance_factor**2).numpy(),
        )
        output["cross"] = Waveform(
            data=(h_cross / distance_factor).numpy(),
            times=times_np,
            covariance=(cov_cross / distance_factor**2).numpy(),
        )
        return output

    def save(self, path: str | Path) -> None:
        """Save checkpoint."""
        import datetime
        from heron import __version__ as heron_version

        warping = self.warping
        if isinstance(warping, SimpleWarping):
            warping_config = {"type": "simple", "scale": warping.scale}
        elif isinstance(warping, MassRatioChirpTimeWarping):
            warping_config = {
                "type": "chirp_adaptive",
                "alpha": warping.alpha,
                "t_ref": warping.t_ref,
                "ref_mass_ratio": warping.ref_mass_ratio,
            }
        elif isinstance(warping, ChirpTimeWarping):
            warping_config = {
                "type": "chirp",
                "alpha": warping.alpha,
                "t_ref": warping.t_ref,
            }
        else:
            warping_config = {"type": str(type(warping).__name__)}

        from .mean import mean_to_config

        checkpoint = {
            "format_version": 2,
            "heron_version": heron_version,
            "model_class": type(self).__name__,
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
            "parameter_names": self.parameter_names,
            "model_states": {
                name: model.state_dict()
                for name, model in self.models.items()
            },
            # Mean functions have no trainable parameters, so state_dicts
            # cannot restore them -- they must be recorded explicitly or
            # load() silently reverts to ZeroMean (format_version 1 bug).
            "mean_functions": {
                name: mean_to_config(model.mean_module)
                for name, model in self.models.items()
            },
            "train_x": self._train_x_raw.cpu(),
            "train_y_plus": self._train_y_plus_raw.cpu(),
            "train_y_cross": self._train_y_cross_raw.cpu(),
            "mass_factor": self.mass_factor,
            "distance_factor": self.distance_factor,
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
    def load(cls, path: str | Path, device: str = "cpu") -> "PhaseAmplitudeGPSurrogate":
        """Load a pre-trained model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

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

        from .mean import mean_from_config

        if "mean_functions" not in checkpoint:
            logger.warning(
                f"Checkpoint {path} predates mean-function serialization "
                "(format_version 1): if this model was trained with non-zero "
                "mean functions they CANNOT be recovered from the checkpoint "
                "and predictions will silently use ZeroMean. Re-save the "
                "checkpoint with its mean_functions field populated."
            )
        mean_cfgs = checkpoint.get("mean_functions", {})

        instance = cls(
            train_x=checkpoint["train_x"],
            train_y_plus=checkpoint["train_y_plus"],
            train_y_cross=checkpoint["train_y_cross"],
            warping=warping_obj,
            nu=checkpoint["nu"],
            device=device,
            mean_module_amplitude=mean_from_config(
                mean_cfgs.get("log_amplitude"), warping=warping_obj
            ),
            mean_module_phase=mean_from_config(
                mean_cfgs.get("phase"), warping=warping_obj
            ),
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

    @property
    def parameter_names(self) -> list[str]:
        return ["mass_ratio"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        q_vals = self._train_x_raw[:, 0]
        return {
            "mass_ratio": (float(q_vals.min()), float(q_vals.max())),
        }
