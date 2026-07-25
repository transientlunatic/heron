"""
Training pipeline for heron waveform surrogate models.

Supports three training modes:
  - fixed:  Generate training data at fixed mass ratios from a reference approximant
  - active: Iterative active learning with uncertainty-guided refinement
  - data:   Load pre-existing training data from HDF5

Usage:
  heron train --settings config.yaml
"""

import logging
import os

import click
import numpy as np
import torch

from heron.models.warping import get_warping
from heron.training.dataset import TrainingSet
from heron.utils import load_yaml

logger = logging.getLogger("heron.train")

# Model registry — maps config string to (module, class) for lazy import
MODEL_REGISTRY = {
    "exact": ("heron.models.gp.exact", "ExactGPSurrogate"),
    "sparse": ("heron.models.gp.sparse", "SparseGPSurrogate"),
    "phase_amplitude": ("heron.models.gp.phase_amplitude", "PhaseAmplitudeGPSurrogate"),
    "delta": ("heron.models.gp.delta", "DeltaGPSurrogate"),
}

# Approximant registry — maps config string to (module, class) for lazy import
APPROXIMANT_REGISTRY = {
    "IMRPhenomPv2": ("heron.models.lalsimulation", "IMRPhenomPv2"),
    "IMRPhenomD": ("heron.models.lalsimulation", "IMRPhenomD"),
    "IMRPhenomXAS": ("heron.models.lalsimulation", "IMRPhenomXAS"),
    "SEOBNRv3": ("heron.models.lalsimulation", "SEOBNRv3"),
}


def _import_class(module_path: str, class_name: str):
    """Lazily import a class from a module path."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{name}'. Available: {list(MODEL_REGISTRY)}")
    return _import_class(*MODEL_REGISTRY[name])


def _get_approximant(name: str):
    if name not in APPROXIMANT_REGISTRY:
        raise ValueError(
            f"Unknown approximant '{name}'. Available: {list(APPROXIMANT_REGISTRY)}"
        )
    cls = _import_class(*APPROXIMANT_REGISTRY[name])
    return cls()


def _build_mean_module(settings: dict, warping=None, target: str = "strain"):
    """Build a mean function from config, or return None for ZeroMean.

    Parameters
    ----------
    target : str
        'strain' reads the `mean_function` block and returns a combined
        h=A*cos(Phi) mean, shared between polarisations (SparseGPSurrogate,
        or ExactGPSurrogate with a PN mean_function -- PN means combine
        amplitude*cos(phase) identically for plus/cross).
        'plus'/'cross' also read `mean_function`, but require
        `type: approximant` and return a polarisation-specific full-IMR
        mean (ExactGPSurrogate only -- a waveform mean is inherently
        polarisation-specific, cross is 90 deg out of phase, so it can't be
        shared the way the PN means are).
        'amplitude'/'phase' read `mean_function_amplitude`/
        `mean_function_phase` and return a standalone log-amplitude/phase
        mean (for PhaseAmplitudeGPSurrogate).
    """
    cfg_key = {
        "strain": "mean_function",
        "plus": "mean_function",
        "cross": "mean_function",
        "amplitude": "mean_function_amplitude",
        "phase": "mean_function_phase",
    }[target]
    mean_cfg = settings.get(cfg_key)
    if mean_cfg is None:
        return None

    mean_type = mean_cfg.get("type", "zero")
    if mean_type == "zero":
        return None

    from heron.models.gp import mean as mean_module

    total_mass = settings.get("total_mass", 60.0)
    distance = settings.get("distance", 100.0)

    if target == "strain":
        classes = {"newtonian": "NewtonianInspiralMean", "taylort2": "TaylorT2Mean"}
        if mean_type == "approximant":
            raise ValueError(
                "mean_function type 'approximant' is not supported as a "
                "single shared mean: a waveform mean is polarisation-"
                "specific (cross is 90 deg out of phase). "
                "ExactGPSurrogate builds this automatically as two "
                "LALApproximantPlusMean/LALApproximantCrossMean instances "
                "(see heron_train's model_kwargs dispatch for model: "
                "exact) -- this 'strain' path is only reached for model: "
                "sparse, which doesn't support per-polarisation means. Use "
                "model: exact, or model: phase_amplitude with "
                "mean_function_amplitude/mean_function_phase, instead."
            )
        if mean_type not in classes:
            raise ValueError(f"Unknown mean function type '{mean_type}'")
        output_scale = settings.get("output_scale", 1e27)
        return getattr(mean_module, classes[mean_type])(
            total_mass=total_mass, distance=distance,
            output_scale=output_scale, warping=warping,
        )
    elif target in ("plus", "cross"):
        if mean_type != "approximant":
            raise ValueError(
                f"mean_function type '{mean_type}' is not supported "
                f"per-polarisation (target={target!r}); only 'approximant' "
                "is -- PN means are polarisation-agnostic here and are "
                "built once via target='strain' instead."
            )
        target_cap = "Plus" if target == "plus" else "Cross"
        output_scale = settings.get("output_scale", 1e27)
        mean_approximant = mean_cfg.get("approximant", "IMRPhenomXAS")
        # The mean and training-target approximants use different
        # phi_ref/f_ref conventions (measured: IMRPhenomD vs IMRPhenomXAS
        # differ by a near-constant ~-2.15 rad, not a time-alignment issue
        # -- see mean.compute_phase_correction docstring). Uncorrected,
        # this makes cos(Phi) vs cos(Phi+2.15) two nearly-uncorrelated
        # oscillating functions and the strain-domain mean makes the GP's
        # job HARDER than ZeroMean, not easier. `training.approximant` is
        # the training-target oracle for `mode: fixed`/`active`; `mode:
        # data` configs (loading a pre-generated HDF5) don't set it, so
        # this defaults to IMRPhenomD -- the only oracle used anywhere in
        # this repo so far.
        target_approximant = settings.get("approximant", "IMRPhenomD")
        phase_correction = mean_module.compute_phase_correction(
            mean_approximant=mean_approximant,
            target_approximant=target_approximant,
            total_mass=total_mass, distance=distance,
            f_low=settings.get("f_low", 20.0),
        )
        return getattr(mean_module, f"LALApproximant{target_cap}Mean")(
            total_mass=total_mass, distance=distance,
            output_scale=output_scale, warping=warping,
            approximant=mean_approximant,
            phase_correction=phase_correction,
        )
    else:
        target_cap = "Amplitude" if target == "amplitude" else "Phase"
        classes = {
            "newtonian": f"NewtonianInspiral{target_cap}Mean",
            "taylort2": f"TaylorT2{target_cap}Mean",
            "approximant": f"LALApproximant{target_cap}Mean",
        }
        if mean_type not in classes:
            raise ValueError(f"Unknown mean function type '{mean_type}'")
        kwargs = dict(total_mass=total_mass, distance=distance, warping=warping)
        if mean_type == "approximant":
            kwargs["approximant"] = mean_cfg.get("approximant", "IMRPhenomXAS")
        return getattr(mean_module, classes[mean_type])(**kwargs)


def generate_training_data_fixed(settings: dict) -> TrainingSet:
    """Generate training data at fixed mass ratios from a reference approximant.

    Parameters
    ----------
    settings : dict
        Must contain: mass_ratios, total_mass, distance.
        Optional: n_samples (default 200), warping, approximant.
    """
    from astropy import units as u

    mass_ratios = settings["mass_ratios"]
    total_mass = settings["total_mass"]
    distance = settings["distance"]
    n_samples = settings.get("n_samples", 200)

    warping_cfg = settings.get("warping", {})
    warping = get_warping(
        warping_cfg.get("type", "chirp"),
        **{k: v for k, v in warping_cfg.items() if k != "type"},
    )

    approximant_name = settings.get("approximant", "IMRPhenomPv2")
    approximant = _get_approximant(approximant_name)

    all_q = []
    all_t = []
    all_plus = []
    all_cross = []

    for q in mass_ratios:
        logger.info(f"Generating waveform for q={q}")

        params = {
            "mass_ratio": q,
            "total_mass": total_mass * u.solMass,
            "luminosity_distance": distance * u.Mpc,
            "f_min": 20.0 * u.Hertz,
            "delta_t": (1.0 / 4096) * u.second,
        }

        waveform = approximant.time_domain(params)

        times = waveform["plus"].times
        plus_strain = waveform["plus"].data
        cross_strain = waveform["cross"].data

        # Sample uniformly in warped space
        times_tensor = torch.tensor(times, dtype=torch.float32)
        warped_times = warping.warp(times_tensor, mass_ratio=q).numpy()

        uniform_warped = np.linspace(warped_times[0], warped_times[-1], n_samples)
        plus_sampled = np.interp(uniform_warped, warped_times, plus_strain)
        cross_sampled = np.interp(uniform_warped, warped_times, cross_strain)

        physical_times = warping.unwarp(
            torch.tensor(uniform_warped, dtype=torch.float32), mass_ratio=q
        ).numpy()

        all_q.extend([q] * n_samples)
        all_t.extend(physical_times)
        all_plus.extend(plus_sampled)
        all_cross.extend(cross_sampled)

        logger.info(f"  Sampled {n_samples} points for q={q}")

    coords = np.column_stack([all_q, all_t]).astype(np.float32)

    return TrainingSet(
        x=torch.tensor(coords, dtype=torch.float32),
        y_plus=torch.tensor(np.array(all_plus), dtype=torch.float32),
        y_cross=torch.tensor(np.array(all_cross), dtype=torch.float32),
        parameter_names=["mass_ratio"],
        metadata={
            "source": "fixed_grid",
            "approximant": settings.get("approximant", "IMRPhenomPv2"),
            "total_mass": total_mass,
            "distance": distance,
            "n_mass_ratios": len(mass_ratios),
        },
    )


def heron_train(settings):
    """Run the full training pipeline.

    Parameters
    ----------
    settings : str or dict
        Path to a YAML settings file, or a parsed settings dict.

    Returns
    -------
    model : WaveformSurrogate
        The trained surrogate model.
    """
    if isinstance(settings, str):
        settings = load_yaml(settings)

    train_settings = settings["training"]

    if "logging" in settings:
        level = settings.get("logging", {}).get("level", "warning")
        LOGGER_LEVELS = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }
        logging.basicConfig(level=LOGGER_LEVELS[level])

    # --- Step 1: Get training data ---
    mode = train_settings.get("mode", "fixed")

    if mode == "fixed":
        logger.info("Generating training data (fixed grid)")
        training_set = generate_training_data_fixed(train_settings)
    elif mode == "active":
        logger.info("Running active learning loop")
        from heron.training.active import active_learning_loop

        approximant_name = train_settings.get("approximant", "IMRPhenomPv2")
        approximant = _get_approximant(approximant_name)

        parameter_bounds = train_settings.get(
            "parameter_space", {"mass_ratio": (0.1, 1.0)}
        )
        # Convert lists to tuples if needed (YAML gives lists)
        parameter_bounds = {k: tuple(v) for k, v in parameter_bounds.items()}

        warping_cfg = train_settings.get("warping", {})
        warping = get_warping(
            warping_cfg.get("type", "chirp"),
            **{k: v for k, v in warping_cfg.items() if k != "type"},
        )

        model_type = train_settings.get("model", "exact")
        model_class = _get_model_class(model_type)

        model_kwargs = {
            "nu": train_settings.get("nu", 2.5),
            "output_scale": train_settings.get("output_scale", 1e27),
            "training_iterations": train_settings.get("iterations", 400),
        }
        if model_type == "sparse":
            model_kwargs["n_inducing"] = train_settings.get("n_inducing", 200)

        mean_module = _build_mean_module(train_settings, warping=warping)
        if mean_module is not None:
            model_kwargs["mean_module"] = mean_module

        result = active_learning_loop(
            approximant=approximant,
            parameter_bounds=parameter_bounds,
            model_class=model_class,
            model_kwargs=model_kwargs,
            n_initial=train_settings.get("initial_samples", 50),
            n_time_samples=train_settings.get("n_samples", 200),
            active_iterations=train_settings.get("active_iterations", 5),
            points_per_iteration=train_settings.get("points_per_iteration", 20),
            warping=warping,
            total_mass=train_settings.get("total_mass", 60.0),
            distance=train_settings.get("distance", 100.0),
            seed=train_settings.get("seed"),
        )
        training_set = result.training_set
        logger.info(
            f"Active learning complete: {result.n_iterations} iterations, "
            f"{len(training_set)} training points"
        )
    elif mode == "data":
        data_path = train_settings["data_path"]
        logger.info(f"Loading training data from {data_path}")
        training_set = TrainingSet.load(data_path)
    else:
        raise ValueError(f"Unknown training mode '{mode}'. Use: fixed, active, data")

    logger.info(f"Training data: {len(training_set)} samples")

    # --- Step 2: Build and train model ---
    model_type = train_settings.get("model", "exact")
    model_class = _get_model_class(model_type)

    warping_cfg = train_settings.get("warping", {})
    warping = get_warping(
        warping_cfg.get("type", "chirp"),
        **{k: v for k, v in warping_cfg.items() if k != "type"},
    )

    total_mass = train_settings.get("total_mass", 60.0)
    distance = train_settings.get("distance", 100.0)

    # Kwargs common to ExactGPSurrogate, SparseGPSurrogate and
    # PhaseAmplitudeGPSurrogate.
    model_kwargs = {
        "train_x": training_set.x,
        "train_y_plus": training_set.y_plus,
        "train_y_cross": training_set.y_cross,
        "warping": warping,
        "nu": train_settings.get("nu", 2.5),
        "device": train_settings.get("device", "cpu"),
        "total_mass": total_mass,
        "distance": distance,
        "training_iterations": train_settings.get("iterations", 100),
        "ls_min_time": train_settings.get("ls_min_time", 0.0005),
        "ls_min_q": train_settings.get("ls_min_q", 0.0005),
        "noise_floor_rel": train_settings.get("noise_floor_rel", 1e-6),
    }

    if model_type == "exact" and train_settings.get("merger_kernel", False):
        model_kwargs["merger_kernel"] = True
        model_kwargs["ls_min_time_merger"] = train_settings.get("ls_min_time_merger")
        model_kwargs["merger_center"] = train_settings.get("merger_center", 0.0)
        model_kwargs["merger_width_init"] = train_settings.get("merger_width_init")

    if model_type in ("exact", "phase_amplitude") and train_settings.get("q_floor_kernel", False):
        model_kwargs["q_floor_kernel"] = True
        model_kwargs["q_floor_lengthscale"] = train_settings.get("q_floor_lengthscale")
        model_kwargs["q_floor_outputscale_min"] = train_settings.get(
            "q_floor_outputscale_min", 0.05
        )
        model_kwargs["q_floor_outputscale_init"] = train_settings.get("q_floor_outputscale_init")

    if model_type in ("exact", "phase_amplitude") and train_settings.get("q_warping"):
        model_kwargs["q_warping"] = train_settings["q_warping"]

    if model_type == "phase_amplitude":
        # log-amplitude/phase targets are already well-scaled — no
        # output_scale needed (unlike raw strain ~1e-21).
        mean_module_amplitude = _build_mean_module(train_settings, warping=warping, target="amplitude")
        mean_module_phase = _build_mean_module(train_settings, warping=warping, target="phase")
        if mean_module_amplitude is not None:
            model_kwargs["mean_module_amplitude"] = mean_module_amplitude
        if mean_module_phase is not None:
            model_kwargs["mean_module_phase"] = mean_module_phase
    elif model_type == "delta":
        # Delta targets (residuals against the base approximant) are
        # already well-scaled — no output_scale — and the base approximant
        # is the mean, so no mean_module either (delta GPs use ZeroMean).
        model_kwargs["base_approximant"] = train_settings.get(
            "base_approximant", "IMRPhenomD"
        )
        # The training approximant doubles as the delta model's oracle
        # evaluator so both sides of the residual are decomposed on dense
        # native grids (sparse-sample phase unwrap aliases -- see
        # heron/models/gp/delta.py). In `data` mode with no approximant
        # setting this stays None and the sparse fallback path is used.
        model_kwargs["oracle_approximant"] = train_settings.get("approximant")
        model_kwargs["f_low"] = train_settings.get("f_low", 20.0)
        model_kwargs["phase_alignment"] = train_settings.get(
            "phase_alignment", "anchor"
        )
        model_kwargs["amp_floor_rel"] = train_settings.get("amp_floor_rel", 1e-4)
    else:
        model_kwargs["output_scale"] = train_settings.get("output_scale", 1e27)
        mean_cfg = train_settings.get("mean_function") or {}
        if model_type == "exact" and mean_cfg.get("type") == "approximant":
            # A waveform mean is polarisation-specific -- build two
            # separate LALApproximantPlusMean/CrossMean instances instead
            # of the single shared mean_module used by PN/zero means.
            mean_module_plus = _build_mean_module(train_settings, warping=warping, target="plus")
            mean_module_cross = _build_mean_module(train_settings, warping=warping, target="cross")
            if mean_module_plus is not None:
                model_kwargs["mean_module_plus"] = mean_module_plus
            if mean_module_cross is not None:
                model_kwargs["mean_module_cross"] = mean_module_cross
        else:
            mean_module = _build_mean_module(train_settings, warping=warping)
            if mean_module is not None:
                model_kwargs["mean_module"] = mean_module

    if model_type == "sparse":
        model_kwargs["n_inducing"] = train_settings.get("n_inducing", 200)
        if "lr" in train_settings:
            model_kwargs["learning_rate"] = train_settings["lr"]
        if "ngd_lr" in train_settings:
            model_kwargs["ngd_lr"] = train_settings["ngd_lr"]
    else:
        model_kwargs["optimizer"] = train_settings.get("optimizer", "lbfgs")
        model_kwargs["lr"] = train_settings.get("lr", None)
        model_kwargs["cholesky_size"] = train_settings.get("cholesky_size", 2000)

    logger.info(
        f"Training {model_type} GP "
        f"(nu={model_kwargs['nu']}, iterations={model_kwargs['training_iterations']})"
    )
    model = model_class(**model_kwargs)

    # --- Step 3: Save checkpoint ---
    checkpoint_path = train_settings.get("checkpoint", "heron_checkpoint.pt")
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    model.save(checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Optionally save training data
    if train_settings.get("save_training_data"):
        data_path = train_settings["save_training_data"]
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        training_set.save(data_path)
        logger.info(f"Training data saved to {data_path}")

    return model


@click.command()
@click.option("--settings", required=True, help="Path to training settings YAML")
def train(settings):
    """Train a heron waveform surrogate model."""
    click.echo("Training heron surrogate model")
    heron_train(settings)
    click.echo("Training complete")
