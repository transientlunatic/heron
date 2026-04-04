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
}

# Approximant registry — maps config string to (module, class) for lazy import
APPROXIMANT_REGISTRY = {
    "IMRPhenomPv2": ("heron.models.lalsimulation", "IMRPhenomPv2"),
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


def _build_mean_module(settings: dict, warping=None):
    """Build a mean function from config, or return None for ZeroMean."""
    mean_cfg = settings.get("mean_function")
    if mean_cfg is None:
        return None

    mean_type = mean_cfg.get("type", "zero")
    if mean_type == "zero":
        return None

    from heron.models.gp.mean import NewtonianInspiralMean, TaylorT2Mean

    total_mass = settings.get("total_mass", 60.0)
    distance = settings.get("distance", 100.0)
    output_scale = settings.get("output_scale", 1e27)

    if mean_type == "newtonian":
        return NewtonianInspiralMean(
            total_mass=total_mass, distance=distance,
            output_scale=output_scale, warping=warping,
        )
    elif mean_type == "taylort2":
        return TaylorT2Mean(
            total_mass=total_mass, distance=distance,
            output_scale=output_scale, warping=warping,
        )
    else:
        raise ValueError(f"Unknown mean function type '{mean_type}'")


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
        warped_times = warping.warp(times_tensor).numpy()

        uniform_warped = np.linspace(warped_times[0], warped_times[-1], n_samples)
        plus_sampled = np.interp(uniform_warped, warped_times, plus_strain)
        cross_sampled = np.interp(uniform_warped, warped_times, cross_strain)

        physical_times = warping.unwarp(
            torch.tensor(uniform_warped, dtype=torch.float32)
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

    mean_module = _build_mean_module(train_settings, warping=warping)

    model_kwargs = {
        "train_x": training_set.x,
        "train_y_plus": training_set.y_plus,
        "train_y_cross": training_set.y_cross,
        "warping": warping,
        "nu": train_settings.get("nu", 2.5),
        "output_scale": train_settings.get("output_scale", 1e27),
        "device": train_settings.get("device", "cpu"),
        "total_mass": total_mass,
        "distance": distance,
        "training_iterations": train_settings.get("iterations", 400),
    }
    if mean_module is not None:
        model_kwargs["mean_module"] = mean_module
    if model_type == "sparse":
        model_kwargs["n_inducing"] = train_settings.get("n_inducing", 200)

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
