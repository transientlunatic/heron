#!/usr/bin/env python3
"""
Generate training data with three mass ratios: q={0.1, 0.8, 1.0}

This extends our training set to include equal mass binaries.
"""

import numpy as np
import astropy.units as u
from pathlib import Path

from heron.training.makedata import make_optimal_manifold
from heron.training.data import DataWrapper001
from heron.models.lalsimulation import IMRPhenomPv2

output_file = "training_data_100mpc_q3.h5"

print("=" * 70)
print("Generating Heron Training Data: 3 mass ratios")
print("=" * 70)

# Target mass ratios
mass_ratios = [0.1, 0.8, 1.0]

print(f"\nParameters:")
print(f"  Output file: {output_file}")
print(f"  Total mass: 20 M_sun")
print(f"  Distance: 100 Mpc")
print(f"  Mass ratios: {mass_ratios}")
print(f"  f_min: 20 Hz")
print(f"  delta_t: 1/4096 s")
print(f"  Warp factor: 3")

# Fixed parameters
fixed_params = {
    "total_mass": 20 * u.solMass,
    "luminosity_distance": 100 * u.Mpc,
    "f_min": 20 * u.Hertz,
    "delta_t": (1.0/4096) * u.second,
}

# Generate for each mass ratio separately
all_plus_data = []
all_cross_data = []
all_plus_times = []
all_cross_times = []

for q in mass_ratios:
    print(f"\nGenerating waveform for q={q}...")

    # Use tiny step to get single waveform at this q
    varied_params = {
        "mass_ratio": {
            "lower": q,
            "upper": q + 0.001,  # Small range to get just one point
            "step": 0.01,        # Step larger than range = single point
        }
    }

    # Generate the optimal manifold
    manifold_plus, manifold_cross = make_optimal_manifold(
        approximant=IMRPhenomPv2,
        warp_factor=3,
        varied=varied_params,
        fixed=fixed_params,
    )

    print(f"  Plus: {len(manifold_plus.locations)} waveforms")
    print(f"  Cross: {len(manifold_cross.locations)} waveforms")

    # Collect data
    all_plus_data.extend(manifold_plus.locations)
    all_cross_data.extend(manifold_cross.locations)
    all_plus_times.append(manifold_plus.times)
    all_cross_times.append(manifold_cross.times)

# Verify we got the right number
print(f"\nTotal waveforms generated:")
print(f"  Plus polarization: {len(all_plus_data)}")
print(f"  Cross polarization: {len(all_cross_data)}")

# The times should all be the same (fixed total mass)
print(f"  Time samples: {len(all_plus_times[0])}")

# Save to HDF5
print(f"\nSaving to {output_file}...")

# Remove old file if exists
if Path(output_file).exists():
    print(f"  Removing existing file...")
    Path(output_file).unlink()

# Create DataWrapper and save
# Note: We need to format data correctly for DataWrapper001
import h5py

with h5py.File(output_file, 'w') as f:
    group = f.create_group('IMR_training_test')

    # Stack data: shape should be (2, n_samples) where row 0 is mass ratios, row 1 is times
    # Then we have n_waveforms columns

    # For plus polarization
    n_waveforms = len(all_plus_data)
    n_samples = len(all_plus_data[0])

    # Create arrays
    plus_data = np.array([loc for loc in all_plus_data]).T  # Transpose to (n_samples, n_waveforms)
    cross_data = np.array([loc for loc in all_cross_data]).T

    # Create coordinate arrays (mass_ratio, time) for each sample
    # Each waveform has same time array, but different mass ratio
    mass_ratio_coords = np.repeat(mass_ratios, n_samples)
    time_coords = np.tile(all_plus_times[0], len(mass_ratios))

    coords = np.vstack([mass_ratio_coords, time_coords])

    # Save datasets
    group.create_dataset('plus_polarisation_x', data=coords)
    group.create_dataset('plus_polarisation_y', data=plus_data.flatten())
    group.create_dataset('cross_polarisation_x', data=coords)
    group.create_dataset('cross_polarisation_y', data=cross_data.flatten())

    print(f"  Saved datasets:")
    print(f"    Coordinates shape: {coords.shape}")
    print(f"    Plus data shape: {plus_data.flatten().shape}")
    print(f"    Cross data shape: {cross_data.flatten().shape}")

print(f"\n✓ Training data saved to {output_file}")
print(f"\nTo use this data, update your training scripts to load:")
print(f"  dw = DataWrapper001('{output_file}')")
