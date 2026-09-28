#!/usr/bin/env python3
"""
Compare different time-warping strategies on the cluster.

Tests simple linear warping vs physical chirp-time warping to measure:
- Mismatch reduction
- Training time
- Memory usage

Usage: python test_warping_comparison.py <warping_type> <iterations>
    warping_type: 'simple' or 'chirp'
    iterations: number of training iterations
"""

import sys
import torch
import numpy as np
import astropy.units as u
from heron.training.data import DataWrapper001
from heron.models.gpytorch import HeronNonSpinningApproximantMatern
from heron.models.lalsimulation import IMRPhenomPv2

if len(sys.argv) < 3:
    print("Usage: python test_warping_comparison.py <warping_type> <iterations>")
    sys.exit(1)

warping_type = sys.argv[1]
n_iter = int(sys.argv[2])

print("=" * 70)
print(f"Time Warping Test: {warping_type} warping, {n_iter} iterations")
print("=" * 70)

# Load data
dw = DataWrapper001('training_data_100mpc.h5')
train_x_plus, train_y_plus = dw.get_training_data('IMR_training_test', polarisation=b'p')
train_x_cross, train_y_cross = dw.get_training_data('IMR_training_test', polarisation=b'c')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

train_x_plus_t = torch.tensor(train_x_plus.T, dtype=torch.float32, device=device)
train_y_plus_t = torch.tensor(train_y_plus, dtype=torch.float32, device=device)
train_x_cross_t = torch.tensor(train_x_cross.T, dtype=torch.float32, device=device)
train_y_cross_t = torch.tensor(train_y_cross, dtype=torch.float32, device=device)

# Test parameters
q = 0.5
params = {
    "mass_ratio": q,
    "total_mass": 20 * u.solMass,
    "distance": 100 * u.Mpc,
    "time": {"lower": -0.1, "upper": 0.05, "number": 500},
}

print(f"\nTraining model with {warping_type} warping, {n_iter} iterations...")
import time
start_time = time.time()

# Create model with specified warping
if warping_type == 'simple':
    model = HeronNonSpinningApproximantMatern(
        train_x_plus=train_x_plus_t,
        train_y_plus=train_y_plus_t,
        train_x_cross=train_x_cross_t,
        train_y_cross=train_y_cross_t,
        total_mass=20 * u.solMass,
        distance=100 * u.Mpc,
        warping='simple',
        warp_scale=2,
        training=n_iter,
    )
elif warping_type == 'chirp':
    model = HeronNonSpinningApproximantMatern(
        train_x_plus=train_x_plus_t,
        train_y_plus=train_y_plus_t,
        train_x_cross=train_x_cross_t,
        train_y_cross=train_y_cross_t,
        total_mass=20 * u.solMass,
        distance=100 * u.Mpc,
        warping='chirp',
        training=n_iter,
    )
else:
    print(f"Unknown warping type: {warping_type}")
    sys.exit(1)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.1f} seconds")

print(f"\nGenerating waveforms...")
ref_model = IMRPhenomPv2()

# Generate waveforms
wf_model = model.time_domain(parameters=params.copy())
wf_ref = ref_model.time_domain(parameters=params.copy())

# Extract data
model_times = np.array(wf_model['plus'].times.value)
model_data = np.array(wf_model['plus'].data)
ref_times = np.array(wf_ref['plus'].times.value)
ref_data = np.array(wf_ref['plus'].data)

# Peak analysis
model_peak_idx = np.argmax(np.abs(model_data))
ref_peak_idx = np.argmax(np.abs(ref_data))
model_peak_time = model_times[model_peak_idx]
ref_peak_time = ref_times[ref_peak_idx]
time_error = abs(model_peak_time - ref_peak_time)

# Amplitude
model_amp = np.max(np.abs(model_data))
ref_amp = np.max(np.abs(ref_data))
amp_ratio = model_amp / ref_amp

# Simple overlap
norm_model = model_data / np.sqrt(np.sum(np.abs(model_data)**2))
norm_ref = ref_data / np.sqrt(np.sum(np.abs(ref_data)**2))
overlap = np.abs(np.sum(norm_model * np.conj(norm_ref)))
mismatch = (1 - overlap) * 100

print("=" * 70)
print("Results")
print("=" * 70)
print(f"\nWarping:       {warping_type}")
print(f"Iterations:    {n_iter}")
print(f"Training time: {training_time:.1f} s")
print(f"Time error:    {time_error:.4f} s")
print(f"Amp ratio:     {amp_ratio:.3f}")
print(f"Mismatch:      {mismatch:.2f}%")

# Save to file
output_file = f'warping_results_{warping_type}.txt'
with open(output_file, 'a') as f:
    f.write(f"{warping_type}\t{n_iter}\t{training_time:.2f}\t{time_error:.6f}\t{amp_ratio:.6f}\t{mismatch:.4f}\n")

print(f"\nResults appended to: {output_file}")

# Clean up
del model
if device == 'cuda':
    torch.cuda.empty_cache()
