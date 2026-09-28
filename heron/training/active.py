"""
Active learning loop for waveform surrogate training.

When training from a callable approximant (e.g., LAL), we can query new
waveforms cheaply. The active learning loop iteratively identifies
high-uncertainty regions and adds training data there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from .dataset import TrainingSet
from .sampling import sobol_sample
from ..models.warping import TimeWarping

logger = logging.getLogger("heron.training.active")


@dataclass
class ActiveLearningResult:
    """Result from an active learning run."""
    training_set: TrainingSet
    uncertainty_history: list[float]
    n_iterations: int


def generate_waveform_at_params(
    approximant,
    param_values: dict,
    n_time_samples: int,
    warping: TimeWarping,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single waveform and sample it uniformly in warped time.

    Returns
    -------
    times : ndarray, shape (n_time_samples,)
    plus_strain : ndarray, shape (n_time_samples,)
    cross_strain : ndarray, shape (n_time_samples,)
    """
    from astropy import units as u

    params = {
        "mass_ratio": param_values["mass_ratio"],
        "total_mass": param_values.get("total_mass", 60.0) * u.solMass,
        "luminosity_distance": param_values.get("distance", 100.0) * u.Mpc,
        "f_min": param_values.get("f_min", 20.0) * u.Hertz,
        "delta_t": (1.0 / 4096) * u.second,
    }

    waveform = approximant.time_domain(params)
    times = waveform["plus"].times
    plus_strain = waveform["plus"].data
    cross_strain = waveform["cross"].data

    # Sample uniformly in warped space
    times_tensor = torch.tensor(times, dtype=torch.float32)
    warped_times = warping.warp(times_tensor).numpy()

    uniform_warped = np.linspace(warped_times[0], warped_times[-1], n_time_samples)
    plus_sampled = np.interp(uniform_warped, warped_times, plus_strain)
    cross_sampled = np.interp(uniform_warped, warped_times, cross_strain)

    physical_times = warping.unwarp(
        torch.tensor(uniform_warped, dtype=torch.float32)
    ).numpy()

    return physical_times, plus_sampled, cross_sampled


def generate_initial_training_set(
    approximant,
    parameter_bounds: dict[str, tuple[float, float]],
    n_initial: int,
    n_time_samples: int,
    warping: TimeWarping,
    total_mass: float = 60.0,
    distance: float = 100.0,
    seed: int | None = None,
) -> TrainingSet:
    """Generate initial training set using Sobol sampling.

    Parameters
    ----------
    approximant : WaveformApproximant
        Callable waveform generator.
    parameter_bounds : dict
        Parameter name → (lower, upper) bounds.
    n_initial : int
        Number of parameter-space points to sample.
    n_time_samples : int
        Number of time samples per waveform.
    warping : TimeWarping
        Time warping for uniform sampling.
    total_mass, distance : float
        Reference total mass (solar masses) and distance (Mpc).
    seed : int or None
        Random seed.
    """
    param_samples = sobol_sample(parameter_bounds, n_initial, seed=seed)
    param_names = list(parameter_bounds.keys())

    all_x = []
    all_plus = []
    all_cross = []

    for i in range(n_initial):
        param_dict = {name: float(param_samples[name][i]) for name in param_names}
        param_dict["total_mass"] = total_mass
        param_dict["distance"] = distance

        logger.info(f"Generating waveform {i+1}/{n_initial}: {param_dict}")

        times, plus, cross = generate_waveform_at_params(
            approximant, param_dict, n_time_samples, warping
        )

        # Build input coordinates: (param1, ..., paramN, time)
        coords = np.column_stack([
            *[np.full(n_time_samples, param_dict[name]) for name in param_names],
            times,
        ])
        all_x.append(coords)
        all_plus.append(plus)
        all_cross.append(cross)

    return TrainingSet(
        x=torch.tensor(np.vstack(all_x), dtype=torch.float32),
        y_plus=torch.tensor(np.concatenate(all_plus), dtype=torch.float32),
        y_cross=torch.tensor(np.concatenate(all_cross), dtype=torch.float32),
        parameter_names=param_names,
        metadata={
            "source": "active_learning_initial",
            "total_mass": total_mass,
            "distance": distance,
        },
    )


def identify_high_uncertainty_points(
    model,
    parameter_bounds: dict[str, tuple[float, float]],
    n_candidates: int = 500,
    n_select: int = 10,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Find parameter-space points where the model is most uncertain.

    Evaluates the model at candidate points and returns the ones with
    highest integrated predictive variance.

    Parameters
    ----------
    model : WaveformSurrogate
        Trained surrogate model.
    parameter_bounds : dict
        Parameter search space.
    n_candidates : int
        Number of candidate points to evaluate.
    n_select : int
        Number of highest-uncertainty points to return.

    Returns
    -------
    dict
        Parameter names → arrays of selected values.
    """
    candidates = sobol_sample(parameter_bounds, n_candidates, seed=seed)
    param_names = list(parameter_bounds.keys())

    # Evaluate uncertainty at each candidate
    uncertainties = []
    for i in range(n_candidates):
        params = {name: float(candidates[name][i]) for name in param_names}
        params["time"] = {"lower": -0.5, "upper": 0.02, "number": 200}

        try:
            wf = model.predict(params)
            variance = wf["plus"].variance
            if variance is not None:
                uncertainties.append(float(np.sum(variance)))
            else:
                uncertainties.append(0.0)
        except Exception:
            uncertainties.append(0.0)

    uncertainties = np.array(uncertainties)
    top_indices = np.argsort(uncertainties)[-n_select:]

    return {name: candidates[name][top_indices] for name in param_names}


def active_learning_loop(
    approximant,
    parameter_bounds: dict[str, tuple[float, float]],
    model_class,
    model_kwargs: dict,
    n_initial: int = 50,
    n_time_samples: int = 200,
    active_iterations: int = 5,
    points_per_iteration: int = 20,
    warping: TimeWarping | None = None,
    total_mass: float = 60.0,
    distance: float = 100.0,
    seed: int | None = None,
) -> ActiveLearningResult:
    """Run the full active learning training loop.

    1. Generate initial training set (Sobol sampling)
    2. Train model
    3. Find high-uncertainty regions
    4. Generate new training data there
    5. Repeat until budget exhausted

    Parameters
    ----------
    approximant : WaveformApproximant
        Callable that generates reference waveforms.
    parameter_bounds : dict
        Parameter name → (lower, upper) bounds.
    model_class : type
        Surrogate model class (e.g., ExactGPSurrogate).
    model_kwargs : dict
        Additional kwargs for model construction.
    n_initial : int
        Initial Sobol sample count.
    n_time_samples : int
        Time samples per waveform.
    active_iterations : int
        Number of refinement rounds.
    points_per_iteration : int
        New waveforms per round.
    warping : TimeWarping or None
        Time warping (default: ChirpTimeWarping).
    total_mass, distance : float
        Reference mass and distance.
    seed : int or None
        Random seed.

    Returns
    -------
    ActiveLearningResult
    """
    from ..models.warping import get_warping

    if warping is None:
        warping = get_warping("chirp")

    logger.info(f"Starting active learning: {n_initial} initial + "
                f"{active_iterations} × {points_per_iteration} refinement")

    # Step 1: Initial training set
    training_set = generate_initial_training_set(
        approximant, parameter_bounds, n_initial, n_time_samples,
        warping, total_mass, distance, seed,
    )
    logger.info(f"Initial training set: {len(training_set)} points")

    uncertainty_history = []

    for iteration in range(active_iterations):
        # Step 2: Train model
        logger.info(f"Active learning iteration {iteration + 1}/{active_iterations}")
        model = model_class(
            train_x=training_set.x,
            train_y_plus=training_set.y_plus,
            train_y_cross=training_set.y_cross,
            warping=warping,
            total_mass=total_mass,
            distance=distance,
            **model_kwargs,
        )

        # Step 3: Find high-uncertainty regions
        new_params = identify_high_uncertainty_points(
            model, parameter_bounds,
            n_candidates=500,
            n_select=points_per_iteration,
            seed=(seed + iteration + 1) if seed is not None else None,
        )

        # Track max uncertainty
        # (crude proxy: sum of selected point uncertainties)
        param_names = list(parameter_bounds.keys())
        max_unc = 0.0
        for i in range(points_per_iteration):
            params = {name: float(new_params[name][i]) for name in param_names}
            params["time"] = {"lower": -0.5, "upper": 0.02, "number": 200}
            try:
                wf = model.predict(params)
                v = wf["plus"].variance
                if v is not None:
                    max_unc = max(max_unc, float(np.max(v)))
            except Exception:
                pass
        uncertainty_history.append(max_unc)
        logger.info(f"  Max variance: {max_unc:.2e}")

        # Step 4: Generate new training data
        new_x = []
        new_plus = []
        new_cross = []
        for i in range(points_per_iteration):
            param_dict = {name: float(new_params[name][i]) for name in param_names}
            param_dict["total_mass"] = total_mass
            param_dict["distance"] = distance

            times, plus, cross = generate_waveform_at_params(
                approximant, param_dict, n_time_samples, warping
            )
            coords = np.column_stack([
                *[np.full(n_time_samples, param_dict[name]) for name in param_names],
                times,
            ])
            new_x.append(coords)
            new_plus.append(plus)
            new_cross.append(cross)

        new_training = TrainingSet(
            x=torch.tensor(np.vstack(new_x), dtype=torch.float32),
            y_plus=torch.tensor(np.concatenate(new_plus), dtype=torch.float32),
            y_cross=torch.tensor(np.concatenate(new_cross), dtype=torch.float32),
            parameter_names=param_names,
        )
        training_set = training_set.append(new_training)
        logger.info(f"  Training set: {len(training_set)} points")

    return ActiveLearningResult(
        training_set=training_set,
        uncertainty_history=uncertainty_history,
        n_iterations=active_iterations,
    )
