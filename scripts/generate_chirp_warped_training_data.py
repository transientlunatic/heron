#!/usr/bin/env python3
"""
Generate training data with chirp warping applied during sampling.

This ensures the GP sees uniformly-sampled data in the warped coordinate space,
not just post-hoc warped uniform samples.

Mass ratios: q = {0.1, 0.8, 1.0}
Warping: ChirpTimeWarping(alpha=0.625) - optimal from grid search
"""

import numpy as np
import astropy.units as u
from pathlib import Path
import h5py
import torch

from heron.models.lalsimulation import IMRPhenomPv2
from heron.models.warping import ChirpTimeWarping

# Configuration
OUTPUT_FILE = "training_data_100mpc_chirp_warped.h5"
MASS_RATIOS = [0.1, 0.8, 1.0]
ALPHA = 0.625  # Optimal chirp warping exponent from grid search

# Fixed waveform parameters
TOTAL_MASS = 20.0  # Solar masses
DISTANCE = 100.0   # Mpc
F_MIN = 20.0       # Hz
DELTA_T = 1.0/4096 # seconds

# Sampling in warped space
N_SAMPLES = 200  # Number of samples per waveform in warped space

print("=" * 70)
print("Generating Chirp-Warped Training Data")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Output: {OUTPUT_FILE}")
print(f"  Mass ratios: {MASS_RATIOS}")
print(f"  Chirp warping: α = {ALPHA}")
print(f"  Total mass: {TOTAL_MASS} M☉")
print(f"  Distance: {DISTANCE} Mpc")
print(f"  Samples per waveform: {N_SAMPLES}")
print()

# Initialize approximant and warping
approximant = IMRPhenomPv2()
warping = ChirpTimeWarping(alpha=ALPHA)

# Collect all training data
all_mass_ratios = []
all_times = []
all_plus_data = []
all_cross_data = []

for q in MASS_RATIOS:
    print(f"Generating waveform for q = {q}...")

    # Generate full waveform
    params = {
        "mass_ratio": q,
        "total_mass": TOTAL_MASS * u.solMass,
        "luminosity_distance": DISTANCE * u.Mpc,
        "f_min": F_MIN * u.Hertz,
        "delta_t": DELTA_T * u.second,
    }

    waveform = approximant.time_domain(params)

    # Extract plus and cross polarization
    times = waveform['plus'].times.value  # Physical time
    plus_strain = waveform['plus'].data
    cross_strain = waveform['cross'].data

    print(f"  Generated {len(times)} samples in [{times[0]:.3f}, {times[-1]:.3f}] s")

    # Apply warping to time coordinates (convert to torch)
    times_tensor = torch.tensor(times, dtype=torch.float32)
    warped_times_tensor = warping.warp(times_tensor)
    warped_times = warped_times_tensor.cpu().numpy()

    print(f"  Warped to [{warped_times[0]:.3f}, {warped_times[-1]:.3f}] (warped coords)")

    # Create uniform sampling in WARPED space
    warped_min = warped_times[0]
    warped_max = warped_times[-1]
    uniform_warped = np.linspace(warped_min, warped_max, N_SAMPLES)

    # Interpolate strain values at uniform warped times
    plus_sampled = np.interp(uniform_warped, warped_times, plus_strain)
    cross_sampled = np.interp(uniform_warped, warped_times, cross_strain)

    # Convert back to physical time for storage
    # (GP will warp again during training, but we want to know physical times)
    uniform_warped_tensor = torch.tensor(uniform_warped, dtype=torch.float32)
    physical_times_sampled_tensor = warping.unwarp(uniform_warped_tensor)
    physical_times_sampled = physical_times_sampled_tensor.cpu().numpy()

    print(f"  Sampled {N_SAMPLES} points uniformly in warped space")
    print(f"  Physical time range: [{physical_times_sampled[0]:.3f}, {physical_times_sampled[-1]:.3f}] s")

    # Store data
    all_mass_ratios.extend([q] * N_SAMPLES)
    all_times.extend(physical_times_sampled)
    all_plus_data.extend(plus_sampled)
    all_cross_data.extend(cross_sampled)
    print()

# Convert to arrays
all_mass_ratios = np.array(all_mass_ratios)
all_times = np.array(all_times)
all_plus_data = np.array(all_plus_data)
all_cross_data = np.array(all_cross_data)

print("=" * 70)
print("Saving to HDF5")
print("=" * 70)
print(f"Total samples: {len(all_mass_ratios)}")
print(f"  {len(MASS_RATIOS)} mass ratios × {N_SAMPLES} samples = {len(MASS_RATIOS) * N_SAMPLES}")

# Remove old file if exists
if Path(OUTPUT_FILE).exists():
    print(f"Removing existing file...")
    Path(OUTPUT_FILE).unlink()

# Save in DataWrapper001-compatible format
with h5py.File(OUTPUT_FILE, 'w') as f:
    # Create group structure
    training_group = f.create_group('training data')
    imr_group = training_group.create_group('IMR_training_test')

    # Create coordinates array: shape (2, n_samples)
    # Row 0: mass ratios, Row 1: times
    coords = np.vstack([all_mass_ratios, all_times])

    # Save datasets
    imr_group.create_dataset('plus_polarisation_x', data=coords)
    imr_group.create_dataset('plus_polarisation_y', data=all_plus_data)
    imr_group.create_dataset('cross_polarisation_x', data=coords)
    imr_group.create_dataset('cross_polarisation_y', data=all_cross_data)

    # Add metadata
    imr_group.attrs['total_mass'] = TOTAL_MASS
    imr_group.attrs['distance'] = DISTANCE
    imr_group.attrs['f_min'] = F_MIN
    imr_group.attrs['delta_t'] = DELTA_T
    imr_group.attrs['warping_alpha'] = ALPHA
    imr_group.attrs['mass_ratios'] = MASS_RATIOS
    imr_group.attrs['n_samples_per_waveform'] = N_SAMPLES

    print(f"\nDataset shapes:")
    print(f"  Coordinates: {coords.shape}")
    print(f"  Plus data: {all_plus_data.shape}")
    print(f"  Cross data: {all_cross_data.shape}")

print(f"\n✓ Training data saved to {OUTPUT_FILE}")
print(f"\nTo use this data:")
print(f"  dw = DataWrapper001('{OUTPUT_FILE}')")
print(f"  train_x, train_y = dw.get_training_data('IMR_training_test', polarisation=b'p')")
print()
print("Next steps:")
print("  1. Retrain model with this data")
print("  2. Validate across mass ratio range")
print("  3. Compare with old training data performance")
