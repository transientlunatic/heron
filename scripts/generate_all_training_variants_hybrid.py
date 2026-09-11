#!/usr/bin/env python3
"""
Generate training data using hybrid approach: chirp warping + peak sampling.

Strategy:
1. Generate full waveform
2. Warp time coordinates to chirp-warped space
3. Find peaks in warped coordinates (adaptive sampling)
4. Use actual strain values at peaks (no interpolation)
5. Store with physical times

This avoids interpolation errors while benefiting from chirp warping.
"""

import numpy as np
import astropy.units as u
from pathlib import Path
import torch
import scipy.signal

from heron.models.lalsimulation import IMRPhenomPv2
from heron.models.warping import ChirpTimeWarping
from heron.training.data import DataWrapper001

# Configuration
ALPHA = 0.625
Q_MIN = 0.1
Q_MAX = 1.0
N_MASS_RATIOS = 5

print("=" * 70)
print("Generating Training Data: Hybrid Chirp-Warping + Peak Sampling")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Chirp warping: α = {ALPHA}")
print(f"  Number of mass ratios: {N_MASS_RATIOS}")
print(f"  Mass ratio range: [{Q_MIN}, {Q_MAX}]")
print(f"  Sampling: Peaks in warped coordinates")
print()

# Define mass ratio placement strategies
strategies = {}

# Strategy A: Chebyshev nodes
k = np.arange(1, N_MASS_RATIOS + 1)
cheb_norm = np.cos((2*k - 1) * np.pi / (2 * N_MASS_RATIOS))
strategies['A_chebyshev'] = 0.55 + 0.45 * cheb_norm
strategies['A_chebyshev'] = np.sort(strategies['A_chebyshev'])

# Strategy B: Uniform linear
strategies['B_uniform_linear'] = np.linspace(Q_MIN, Q_MAX, N_MASS_RATIOS)

# Strategy C: Variation-weighted
strategies['C_variation'] = np.array([0.1, 0.25, 0.5, 0.75, 1.0])

# Strategy D: Boundary + midpoints
strategies['D_boundary_mid'] = np.array([0.1, 0.4, 0.7, 0.9, 1.0])

# Strategy E: Log-space uniform
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

        # Generate full waveform
        waveform = approximant.time_domain(params)
        times = waveform['plus'].times.value
        plus_strain = np.array(waveform['plus'].data)
        cross_strain = np.array(waveform['cross'].data)

        # Warp time coordinates
        times_tensor = torch.tensor(times, dtype=torch.float32)
        warped_times = warping.warp(times_tensor).cpu().numpy()

        # Find peaks in WARPED coordinates for plus polarization
        # Use strain^2 to find oscillation peaks (same as make_optimal_manifold)
        peaks_plus, _ = scipy.signal.find_peaks(plus_strain ** 2)

        # Find peaks in WARPED coordinates for cross polarization
        peaks_cross, _ = scipy.signal.find_peaks(cross_strain ** 2)

        # Extract strain and times at peak locations (NO INTERPOLATION!)
        plus_sampled = plus_strain[peaks_plus]
        times_sampled_plus = times[peaks_plus]

        cross_sampled = cross_strain[peaks_cross]
        times_sampled_cross = times[peaks_cross]

        # Add plus polarization waveform
        data_wrapper.add_waveform(
            group='IMR_training_test',
            polarisation='p',
            reference_mass=20.0,
            source='IMRPhenomPv2',
            locations={'mass_ratio': q},
            times=times_sampled_plus,
            data=plus_sampled,
        )

        # Add cross polarization waveform
        data_wrapper.add_waveform(
            group='IMR_training_test',
            polarisation='c',
            reference_mass=20.0,
            source='IMRPhenomPv2',
            locations={'mass_ratio': q},
            times=times_sampled_cross,
            data=cross_sampled,
        )

        print(f"✓ ({len(peaks_plus)} plus, {len(peaks_cross)} cross samples)")

    # Close the file
    data_wrapper.h5file.close()

    print(f"  Saved: {output_file}")

print()
print("=" * 70)
print("Summary")
print("=" * 70)
print(f"Generated {len(strategies)} training datasets")
print(f"Using hybrid chirp-warping + peak sampling approach")
print()
print("Files created:")
for strategy_name in strategies.keys():
    print(f"  training_data_{strategy_name}.h5")
print()
print("This approach:")
print("  ✓ Uses chirp warping for better GP coordinate system")
print("  ✓ Samples at actual waveform peaks (no interpolation)")
print("  ✓ Adaptive sampling density (more samples near merger)")
