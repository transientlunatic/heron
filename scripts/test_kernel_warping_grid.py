#!/usr/bin/env python3
"""
Comprehensive test of kernel types and warping strategies.

Tests different combinations to find optimal configuration:
- Kernel: RBF vs Matérn with different ν values
- Warping: Simple vs Chirp with different exponents

Goal: Break through the 10% mismatch barrier.

Usage: python test_kernel_warping_grid.py <kernel> <nu> <warping> <alpha> <iterations>
    kernel: 'rbf' or 'matern'
    nu: Matérn smoothness parameter (1.5, 2.5, 3.5) - ignored for RBF
    warping: 'simple' or 'chirp'
    alpha: chirp time exponent (0.375, 0.5, 0.625) - ignored for simple
    iterations: number of training iterations
"""

import sys
import torch
import numpy as np
import astropy.units as u
from heron.training.data import DataWrapper001
from heron.models.gpytorch import HeronNonSpinningApproximant, HeronNonSpinningApproximantMatern
from heron.models.lalsimulation import IMRPhenomPv2
from heron.models.warping import ChirpTimeWarping

if len(sys.argv) < 6:
    print("Usage: python test_kernel_warping_grid.py <kernel> <nu> <warping> <alpha> <iterations>")
    sys.exit(1)

kernel_type = sys.argv[1]
nu = float(sys.argv[2]) if sys.argv[2] != 'none' else None
warping_type = sys.argv[3]
alpha = float(sys.argv[4]) if sys.argv[4] != 'none' else None
n_iter = int(sys.argv[5])

print("=" * 70)
print(f"Kernel-Warping Grid Test")
print("=" * 70)
print(f"Kernel: {kernel_type}" + (f" (ν={nu})" if kernel_type == 'matern' else ""))
print(f"Warping: {warping_type}" + (f" (α={alpha})" if warping_type == 'chirp' else ""))
print(f"Iterations: {n_iter}")
print("=" * 70)

# Load data (use absolute path from repository root)
import os
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
training_data_path = os.path.join(repo_root, 'training_data_100mpc.h5')
dw = DataWrapper001(training_data_path)
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

print(f"\nTraining model...")
import time
start_time = time.time()

# Create model based on kernel and warping type
if kernel_type == 'rbf':
    # RBF kernel (original)
    if warping_type == 'simple':
        # Simple warping is built into HeronNonSpinningApproximant
        model = HeronNonSpinningApproximant(
            train_x_plus=train_x_plus_t,
            train_y_plus=train_y_plus_t,
            train_x_cross=train_x_cross_t,
            train_y_cross=train_y_cross_t,
            total_mass=20 * u.solMass,
            distance=100 * u.Mpc,
            warp_scale=2,
            training=n_iter,
        )
    elif warping_type == 'chirp':
        # Need to use custom warping with RBF
        # Note: HeronNonSpinningApproximant doesn't support warping parameter
        # So we need to modify the training data directly
        print("Warning: RBF + chirp warping requires manual warping")
        warping = ChirpTimeWarping(alpha=alpha)

        # Apply warping to training data
        train_x_plus_warped = train_x_plus_t.clone()
        train_x_cross_warped = train_x_cross_t.clone()
        train_x_plus_warped[:, 1] = warping.warp(train_x_plus_warped[:, 1])
        train_x_cross_warped[:, 1] = warping.warp(train_x_cross_warped[:, 1])

        # Create RBF model with warped data
        # This is a workaround since HeronNonSpinningApproximant doesn't support warping parameter
        model = HeronNonSpinningApproximant(
            train_x_plus=train_x_plus_warped,
            train_y_plus=train_y_plus_t,
            train_x_cross=train_x_cross_warped,
            train_y_cross=train_y_cross_t,
            total_mass=20 * u.solMass,
            distance=100 * u.Mpc,
            warp_scale=1,  # Already warped
            training=n_iter,
        )
        # Store warping for evaluation
        model.warping = warping
        model._using_custom_warp = True
    else:
        print(f"Unknown warping type: {warping_type}")
        sys.exit(1)

elif kernel_type == 'matern':
    # Matérn kernel with configurable ν
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
            nu=nu,
        )
    elif warping_type == 'chirp':
        # Create custom chirp warping with specified alpha
        warping = ChirpTimeWarping(alpha=alpha)
        model = HeronNonSpinningApproximantMatern(
            train_x_plus=train_x_plus_t,
            train_y_plus=train_y_plus_t,
            train_x_cross=train_x_cross_t,
            train_y_cross=train_y_cross_t,
            total_mass=20 * u.solMass,
            distance=100 * u.Mpc,
            warping=warping,
            training=n_iter,
            nu=nu,
        )
    else:
        print(f"Unknown warping type: {warping_type}")
        sys.exit(1)
else:
    print(f"Unknown kernel type: {kernel_type}")
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
kernel_str = f"{kernel_type}_nu{nu}" if kernel_type == 'matern' else kernel_type
warp_str = f"{warping_type}_alpha{alpha}" if warping_type == 'chirp' else warping_type
print(f"\nConfiguration: {kernel_str} + {warp_str}")
print(f"Iterations:    {n_iter}")
print(f"Training time: {training_time:.1f} s")
print(f"Time error:    {time_error:.4f} s")
print(f"Amp ratio:     {amp_ratio:.3f}")
print(f"Mismatch:      {mismatch:.2f}%")

# Save to file
output_file = 'kernel_warping_grid_results.txt'
with open(output_file, 'a') as f:
    f.write(f"{kernel_type}\t{nu}\t{warping_type}\t{alpha}\t{n_iter}\t{training_time:.2f}\t{time_error:.6f}\t{amp_ratio:.6f}\t{mismatch:.4f}\n")

print(f"\nResults appended to: {output_file}")

# Clean up
del model
if device == 'cuda':
    torch.cuda.empty_cache()
