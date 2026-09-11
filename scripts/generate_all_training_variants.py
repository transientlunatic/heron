#!/usr/bin/env python3
"""
Generate all training data variants for empirical comparison.

Tests different mass ratio placement strategies:
A: Chebyshev nodes
B: Uniform linear
C: Variation-weighted
D: Boundary + midpoints
E: Log-space uniform

All use:
- Chirp warping (α=0.625)
- N_SAMPLES=100 (from convergence analysis)
- 5 mass ratios for better coverage
"""

import numpy as np
import astropy.units as u
from pathlib import Path
import torch

from heron.models.lalsimulation import IMRPhenomPv2
from heron.models.warping import ChirpTimeWarping
from heron.training.data import DataWrapper001

# Configuration
ALPHA = 0.625
N_SAMPLES = 600  # Testing if Matérn needs more data even with warping
Q_MIN = 0.1
Q_MAX = 1.0
N_MASS_RATIOS = 5

print("=" * 70)
print("Generating All Training Data Variants")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Chirp warping: α = {ALPHA}")
print(f"  Samples per waveform: {N_SAMPLES}")
print(f"  Number of mass ratios: {N_MASS_RATIOS}")
print(f"  Mass ratio range: [{Q_MIN}, {Q_MAX}]")
print()

# Define all sampling strategies
strategies = {}

# Strategy A: Chebyshev nodes
k = np.arange(1, N_MASS_RATIOS + 1)
cheb_norm = np.cos((2*k - 1) * np.pi / (2 * N_MASS_RATIOS))
strategies['A_chebyshev'] = 0.55 + 0.45 * cheb_norm
strategies['A_chebyshev'] = np.sort(strategies['A_chebyshev'])

# Strategy B: Uniform linear
strategies['B_uniform_linear'] = np.linspace(Q_MIN, Q_MAX, N_MASS_RATIOS)

# Strategy C: Variation-weighted (approximate - cluster near boundaries)
# More points at low q where variation is high
strategies['C_variation'] = np.array([0.1, 0.25, 0.5, 0.75, 1.0])

# Strategy D: Boundary + midpoints
strategies['D_boundary_mid'] = np.array([0.1, 0.4, 0.7, 0.9, 1.0])

# Strategy E: Log-space uniform
# More natural for parameters spanning orders of magnitude
strategies['E_log_space'] = np.logspace(np.log10(Q_MIN), np.log10(Q_MAX), N_MASS_RATIOS)

print("Strategies defined:")
for name, q_values in strategies.items():
    print(f"  {name:20s}: q = {[f'{q:.3f}' for q in q_values]}")
print()

# Generate training data for each strategy
approximant = IMRPhenomPv2()
warping = ChirpTimeWarping(alpha=ALPHA)

for strategy_name, mass_ratios in strategies.items():
    output_file = f"training_data_{strategy_name}.h5"

    print("-" * 70)
    print(f"Generating: {strategy_name}")
    print("-" * 70)

    # Remove old file if it exists
    if Path(output_file).exists():
        Path(output_file).unlink()

    # Create new data file using DataWrapper001
    data_wrapper = DataWrapper001.create(output_file)

    for q in mass_ratios:
        print(f"  q = {q:.3f}...", end=" ", flush=True)

        params = {
            "mass_ratio": q,
            "total_mass": 20.0 * u.solMass,
            "luminosity_distance": 100.0 * u.Mpc,
            "f_min": 20.0 * u.Hertz,
            "delta_t": (1.0/4096) * u.second,
        }

        waveform = approximant.time_domain(params)
        times = waveform['plus'].times.value
        plus_strain = waveform['plus'].data
        cross_strain = waveform['cross'].data

        # Apply warping
        times_tensor = torch.tensor(times, dtype=torch.float32)
        warped_times = warping.warp(times_tensor).cpu().numpy()

        # Sample uniformly in warped space
        uniform_warped = np.linspace(warped_times[0], warped_times[-1], N_SAMPLES)

        # Interpolate
        plus_sampled = np.interp(uniform_warped, warped_times, plus_strain)
        cross_sampled = np.interp(uniform_warped, warped_times, cross_strain)

        # Unwarp for storage
        uniform_warped_tensor = torch.tensor(uniform_warped, dtype=torch.float32)
        physical_times_sampled = warping.unwarp(uniform_warped_tensor).cpu().numpy()

        # Add plus polarization waveform
        data_wrapper.add_waveform(
            group='IMR_training_test',
            polarisation='p',
            reference_mass=20.0,
            source='IMRPhenomPv2',
            locations={'mass_ratio': q},
            times=physical_times_sampled,
            data=plus_sampled,
        )

        # Add cross polarization waveform
        data_wrapper.add_waveform(
            group='IMR_training_test',
            polarisation='c',
            reference_mass=20.0,
            source='IMRPhenomPv2',
            locations={'mass_ratio': q},
            times=physical_times_sampled,
            data=cross_sampled,
        )

        print("✓")

    # Close the file
    data_wrapper.h5file.close()

    total_samples = len(mass_ratios) * N_SAMPLES * 2  # Plus and cross
    print(f"  Saved: {output_file} ({total_samples} samples)")

print()
print("=" * 70)
print("Summary")
print("=" * 70)
print(f"Generated {len(strategies)} training datasets")
print(f"Each with {N_MASS_RATIOS} mass ratios × {N_SAMPLES} samples = {N_MASS_RATIOS * N_SAMPLES} total")
print()
print("Files created:")
for strategy_name in strategies.keys():
    print(f"  training_data_{strategy_name}.h5")
print()
print("Next: Test each dataset and compare mismatch performance")
