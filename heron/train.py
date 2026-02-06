"""
Training pipeline for heron GPR waveform models.

Generates training data from a reference approximant, trains
a HeronNonSpinningApproximantMatern model, and saves a checkpoint.
"""

import logging
import os

import click
import numpy as np
import torch
import astropy.units as u

from heron.models.lalsimulation import IMRPhenomPv2
from heron.models.gpytorch import HeronNonSpinningApproximantMatern
from heron.models.warping import ChirpTimeWarping
from heron.utils import load_yaml

logger = logging.getLogger("heron.train")

KNOWN_APPROXIMANTS = {
    "IMRPhenomPv2": IMRPhenomPv2,
}


def generate_training_data(settings):
    """Generate training data from a reference approximant.

    Parameters
    ----------
    settings : dict
        Training settings with keys: mass_ratios, total_mass, distance,
        warping (alpha, t_ref), n_samples, approximant.

    Returns
    -------
    train_x_plus, train_x_cross : torch.Tensor
        Training input coordinates (mass_ratio, time), shape (N, 2).
    train_y_plus, train_y_cross : torch.Tensor
        Training strain values, shape (N,).
    """
    mass_ratios = settings["mass_ratios"]
    total_mass = settings["total_mass"]
    distance = settings["distance"]
    n_samples = settings.get("n_samples", 200)
    alpha = settings.get("warping", {}).get("alpha", 0.625)
    t_ref = settings.get("warping", {}).get("t_ref", 0.1)

    approximant_name = settings.get("approximant", "IMRPhenomPv2")
    approximant = KNOWN_APPROXIMANTS[approximant_name]()

    warping = ChirpTimeWarping(alpha=alpha, t_ref=t_ref)

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

        times = waveform["plus"].times.value
        plus_strain = waveform["plus"].data
        cross_strain = waveform["cross"].data

        # Sample uniformly in warped space
        times_tensor = torch.tensor(times, dtype=torch.float32)
        warped_times = warping.warp(times_tensor).numpy()

        uniform_warped = np.linspace(warped_times[0], warped_times[-1], n_samples)
        plus_sampled = np.interp(uniform_warped, warped_times, plus_strain)
        cross_sampled = np.interp(uniform_warped, warped_times, cross_strain)

        # Convert back to physical time
        physical_times = warping.unwarp(
            torch.tensor(uniform_warped, dtype=torch.float32)
        ).numpy()

        all_q.extend([q] * n_samples)
        all_t.extend(physical_times)
        all_plus.extend(plus_sampled)
        all_cross.extend(cross_sampled)

        logger.info(f"  Sampled {n_samples} points for q={q}")

    # Build training tensors: (mass_ratio, time) pairs
    coords = np.column_stack([all_q, all_t]).astype(np.float32)
    train_x = torch.tensor(coords, dtype=torch.float32)
    train_y_plus = torch.tensor(np.array(all_plus), dtype=torch.float32)
    train_y_cross = torch.tensor(np.array(all_cross), dtype=torch.float32)

    return train_x, train_x.clone(), train_y_plus, train_y_cross


def heron_train(settings):
    """Run the full training pipeline.

    Parameters
    ----------
    settings : str or dict
        Path to a YAML settings file, or a parsed settings dict.
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

    logger.info("Generating training data")
    train_x_plus, train_x_cross, train_y_plus, train_y_cross = (
        generate_training_data(train_settings)
    )

    logger.info(
        f"Training data: {train_x_plus.shape[0]} samples, "
        f"{len(train_settings['mass_ratios'])} mass ratios"
    )

    # Build model
    warping_cfg = train_settings.get("warping", {})
    warping_type = warping_cfg.get("type", "chirp")
    nu = train_settings.get("nu", 2.5)
    training_iterations = train_settings.get("iterations", 400)

    total_mass = train_settings["total_mass"]
    distance = train_settings["distance"]

    logger.info(
        f"Training HeronNonSpinningApproximantMatern "
        f"(nu={nu}, warping={warping_type}, iterations={training_iterations})"
    )

    model = HeronNonSpinningApproximantMatern(
        train_x_plus=train_x_plus,
        train_x_cross=train_x_cross,
        train_y_plus=train_y_plus,
        train_y_cross=train_y_cross,
        total_mass=total_mass,
        distance=distance,
        warping=warping_type,
        nu=nu,
        training=training_iterations,
    )

    # Save checkpoint
    checkpoint_path = train_settings.get(
        "checkpoint", "heron_matern_checkpoint.pt"
    )
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    model.save_checkpoint(checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    return model


@click.command()
@click.option("--settings", required=True, help="Path to training settings YAML")
def train(settings):
    """Train a heron GPR waveform model."""
    click.echo("Training heron GPR model")
    heron_train(settings)
    click.echo("Training complete")
