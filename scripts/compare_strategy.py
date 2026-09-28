#!/usr/bin/env python3
"""
Test a specific training data strategy at a given mass ratio.

Usage: python compare_strategy.py <strategy_name> <mass_ratio>
"""

import sys
import torch
import numpy as np
import astropy.units as u
import os

from heron.training.data import DataWrapper001
from heron.models.gpytorch import HeronNonSpinningApproximantMatern
from heron.models.lalsimulation import IMRPhenomPv2
from heron.models.warping import ChirpTimeWarping

if len(sys.argv) < 3:
    print("Usage: python compare_strategy.py <strategy_name> <mass_ratio>")
    sys.exit(1)

strategy = sys.argv[1]
q = float(sys.argv[2])

# Configuration
NU = 2.5
ALPHA = 0.625
N_ITER = 5000

print("=" * 70)
print(f"Strategy Comparison: {strategy} at q={q}")
print("=" * 70)

# Load training data for this strategy
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
training_data_path = os.path.join(repo_root, f'training_data_{strategy}.h5')

if not os.path.exists(training_data_path):
    print(f"ERROR: Training data not found: {training_data_path}")
    sys.exit(1)

print(f"Loading: {training_data_path}")
dw = DataWrapper001(training_data_path)
train_x_plus, train_y_plus = dw.get_training_data('IMR_training_test', polarisation=b'p')
train_x_cross, train_y_cross = dw.get_training_data('IMR_training_test', polarisation=b'c')

# Get training mass ratios from data
train_q_values = np.unique(train_x_plus[0, :])
print(f"  Training mass ratios: {[f'{q:.3f}' for q in train_q_values]}")
print(f"  Total samples: {train_x_plus.shape[1]}")

# Determine test type
test_type = "UNKNOWN"
for tq in train_q_values:
    if abs(q - tq) < 0.02:
        test_type = "TRAINING"
        break
if test_type == "UNKNOWN":
    if q > train_q_values.min() and q < train_q_values.max():
        test_type = "INTERPOLATION"
    else:
        test_type = "EXTRAPOLATION"

print(f"  Test type: {test_type}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")
print()

train_x_plus_t = torch.tensor(train_x_plus.T, dtype=torch.float32, device=device)
train_y_plus_t = torch.tensor(train_y_plus, dtype=torch.float32, device=device)
train_x_cross_t = torch.tensor(train_x_cross.T, dtype=torch.float32, device=device)
train_y_cross_t = torch.tensor(train_y_cross, dtype=torch.float32, device=device)

# Test parameters
params = {
    "mass_ratio": q,
    "total_mass": 20 * u.solMass,
    "distance": 100 * u.Mpc,
    "time": {"lower": -0.1, "upper": 0.05, "number": 500},
}

print(f"Training model...")
import time
start_time = time.time()

# Create model
warping = ChirpTimeWarping(alpha=ALPHA)
model = HeronNonSpinningApproximantMatern(
    train_x_plus=train_x_plus_t,
    train_y_plus=train_y_plus_t,
    train_x_cross=train_x_cross_t,
    train_y_cross=train_y_cross_t,
    total_mass=20 * u.solMass,
    distance=100 * u.Mpc,
    warping=warping,
    training=N_ITER,
    nu=NU,
)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.1f} seconds")

print(f"\nGenerating waveforms...")
ref_model = IMRPhenomPv2()

wf_model = model.time_domain(parameters=params.copy())
wf_ref = ref_model.time_domain(parameters=params.copy())

# Analysis
model_times = np.array(wf_model['plus'].times.value)
model_data = np.array(wf_model['plus'].data)
ref_times = np.array(wf_ref['plus'].times.value)
ref_data = np.array(wf_ref['plus'].data)

model_peak_idx = np.argmax(np.abs(model_data))
ref_peak_idx = np.argmax(np.abs(ref_data))
model_peak_time = model_times[model_peak_idx]
ref_peak_time = ref_times[ref_peak_idx]
time_error = abs(model_peak_time - ref_peak_time)

model_amp = np.max(np.abs(model_data))
ref_amp = np.max(np.abs(ref_data))
amp_ratio = model_amp / ref_amp

norm_model = model_data / np.sqrt(np.sum(np.abs(model_data)**2))
norm_ref = ref_data / np.sqrt(np.sum(np.abs(ref_data)**2))
overlap = np.abs(np.sum(norm_model * np.conj(norm_ref)))
mismatch = (1 - overlap) * 100

print("=" * 70)
print("Results")
print("=" * 70)
print(f"\nStrategy:      {strategy}")
print(f"Mass ratio:    {q} ({test_type})")
print(f"Training time: {training_time:.1f} s")
print(f"Time error:    {time_error:.4f} s")
print(f"Amp ratio:     {amp_ratio:.3f}")
print(f"Mismatch:      {mismatch:.2f}%")

# Save to file
output_file = 'strategy_comparison_results.txt'
with open(output_file, 'a') as f:
    f.write(f"{strategy}\t{q}\t{test_type}\t{training_time:.2f}\t{time_error:.6f}\t{amp_ratio:.6f}\t{mismatch:.4f}\n")

print(f"\nResults appended to: {output_file}")

# Clean up
del model
if device == 'cuda':
    torch.cuda.empty_cache()
