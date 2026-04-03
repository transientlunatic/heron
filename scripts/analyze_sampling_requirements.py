#!/usr/bin/env python3
"""
Analyze optimal sampling requirements for training data.

Determines:
1. Minimum number of time samples needed (convergence test)
2. Optimal mass ratio placements (Chebyshev nodes or greedy)
"""

import numpy as np
import astropy.units as u
import torch
from heron.models.lalsimulation import IMRPhenomPv2
from heron.models.warping import ChirpTimeWarping
import matplotlib.pyplot as plt

ALPHA = 0.625

print("=" * 70)
print("Analyzing Sampling Requirements")
print("=" * 70)

# Test waveform parameters
approximant = IMRPhenomPv2()
warping = ChirpTimeWarping(alpha=ALPHA)

### Part 1: Time Sampling Convergence Test ###
print("\n" + "=" * 70)
print("Part 1: Time Sampling Convergence Test")
print("=" * 70)

test_q = 0.5  # Test at mid-range mass ratio

params = {
    "mass_ratio": test_q,
    "total_mass": 20.0 * u.solMass,
    "luminosity_distance": 100.0 * u.Mpc,
    "f_min": 20.0 * u.Hertz,
    "delta_t": (1.0/4096) * u.second,
}

print(f"\nGenerating reference waveform at q={test_q}...")
waveform = approximant.time_domain(params)
times = waveform['plus'].times.value
strain = waveform['plus'].data

print(f"  Full waveform: {len(times)} samples")
print(f"  Time range: [{times[0]:.3f}, {times[-1]:.3f}] s")

# Warp times
times_tensor = torch.tensor(times, dtype=torch.float32)
warped_times = warping.warp(times_tensor).cpu().numpy()

print(f"  Warped range: [{warped_times[0]:.3f}, {warped_times[-1]:.3f}]")

# Test different numbers of samples
n_samples_list = [50, 100, 200, 400, 800]
reconstruction_errors = []

print(f"\nTesting reconstruction error vs. number of samples:")
for n_samples in n_samples_list:
    # Sample uniformly in warped space
    uniform_warped = np.linspace(warped_times[0], warped_times[-1], n_samples)

    # Interpolate strain
    strain_sampled = np.interp(uniform_warped, warped_times, strain)

    # Reconstruct on original grid
    strain_reconstructed = np.interp(warped_times, uniform_warped, strain_sampled)

    # Compute reconstruction error
    error = np.sqrt(np.mean((strain - strain_reconstructed)**2)) / np.std(strain)
    reconstruction_errors.append(error)

    print(f"  N={n_samples:4d}: Normalized RMSE = {error:.6f}")

# Find knee point (where improvement < 10%)
print(f"\nConvergence analysis:")
for i in range(1, len(n_samples_list)):
    improvement = (reconstruction_errors[i-1] - reconstruction_errors[i]) / reconstruction_errors[i-1] * 100
    print(f"  {n_samples_list[i-1]:4d} → {n_samples_list[i]:4d}: {improvement:.1f}% improvement")
    if improvement < 10:
        print(f"  → Converged! Recommend N ≥ {n_samples_list[i]}")
        optimal_n_samples = n_samples_list[i]
        break
else:
    print(f"  → Not converged, need more samples")
    optimal_n_samples = n_samples_list[-1]

### Part 2: Mass Ratio Placement Strategy ###
print("\n" + "=" * 70)
print("Part 2: Mass Ratio Placement Strategy")
print("=" * 70)

# Generate waveforms at many q values to analyze variation
q_dense = np.linspace(0.1, 1.0, 20)
waveform_features = []

print(f"\nAnalyzing waveform variation across q ∈ [0.1, 1.0]...")
for q in q_dense:
    params['mass_ratio'] = q
    wf = approximant.time_domain(params)

    # Extract key features
    times_wf = wf['plus'].times.value
    strain_wf = wf['plus'].data

    # Peak amplitude and time
    peak_amp = np.max(np.abs(strain_wf))
    peak_time = times_wf[np.argmax(np.abs(strain_wf))]

    # Duration (time when amplitude > 10% of peak)
    threshold = 0.1 * peak_amp
    duration_mask = np.abs(strain_wf) > threshold
    if np.any(duration_mask):
        duration = times_wf[duration_mask][-1] - times_wf[duration_mask][0]
    else:
        duration = 0

    waveform_features.append({
        'q': q,
        'peak_amp': peak_amp,
        'peak_time': peak_time,
        'duration': duration,
    })

# Compute variation metrics
peak_amps = np.array([f['peak_amp'] for f in waveform_features])
peak_times = np.array([f['peak_time'] for f in waveform_features])
durations = np.array([f['duration'] for f in waveform_features])

# Normalized variation (derivative)
d_peak_amp = np.abs(np.gradient(peak_amps))
d_peak_time = np.abs(np.gradient(peak_times))
d_duration = np.abs(np.gradient(durations))

# Combined variation score
variation_score = d_peak_amp / np.max(d_peak_amp) + d_peak_time / np.max(d_peak_time) + d_duration / np.max(d_duration)

print(f"\nWaveform variation analysis:")
print(f"  Peak amplitude variation: {np.std(peak_amps) / np.mean(peak_amps) * 100:.1f}% relative std")
print(f"  Peak time variation: {np.std(peak_times):.4f} s std")
print(f"  Duration variation: {np.std(durations):.4f} s std")

# Suggest mass ratios based on variation
print(f"\nRegions of high variation (need more training points):")
high_var_indices = np.where(variation_score > np.percentile(variation_score, 75))[0]
high_var_q = q_dense[high_var_indices]
print(f"  q ∈ {high_var_q[[0, -1]]}: High variation region")

### Part 3: Specific Recommendations ###
print("\n" + "=" * 70)
print("Part 3: Recommendations")
print("=" * 70)

print(f"\n1. Time Sampling:")
print(f"   Recommended N_SAMPLES = {optimal_n_samples}")
print(f"   (Convergence: reconstruction error < {reconstruction_errors[n_samples_list.index(optimal_n_samples)]:.4f})")

print(f"\n2. Mass Ratio Placement:")
print(f"\n   Option A: Chebyshev Nodes (optimal for polynomial interpolation)")
# Chebyshev nodes on [0.1, 1.0]
n_mass_ratios = 5  # Can adjust
k = np.arange(1, n_mass_ratios + 1)
chebyshev_nodes_norm = np.cos((2*k - 1) * np.pi / (2 * n_mass_ratios))
# Map from [-1, 1] to [0.1, 1.0]
chebyshev_q = 0.55 + 0.45 * chebyshev_nodes_norm  # Maps to [0.1, 1.0]
chebyshev_q = np.sort(chebyshev_q)
print(f"   5 points: q = {[f'{q:.2f}' for q in chebyshev_q]}")

print(f"\n   Option B: Uniform Spacing")
uniform_q = np.linspace(0.1, 1.0, 5)
print(f"   5 points: q = {[f'{q:.2f}' for q in uniform_q]}")

print(f"\n   Option C: Variation-Weighted (more points where waveforms change rapidly)")
# Place points where cumulative variation is uniform
cumsum_var = np.cumsum(variation_score)
cumsum_var = cumsum_var / cumsum_var[-1]  # Normalize to [0, 1]
target_fractions = np.linspace(0, 1, 5)
weighted_indices = [np.argmin(np.abs(cumsum_var - frac)) for frac in target_fractions]
weighted_q = q_dense[weighted_indices]
print(f"   5 points: q = {[f'{q:.2f}' for q in weighted_q]}")

print(f"\n   Option D: Boundary + Midpoint (current-like)")
boundary_q = [0.1, 0.4, 0.7, 0.9, 1.0]
print(f"   5 points: q = {boundary_q}")

print(f"\n3. Training Data Size Estimate:")
print(f"   {len(boundary_q)} mass ratios × {optimal_n_samples} time samples = {len(boundary_q) * optimal_n_samples} total samples")
print(f"   Current: 3 mass ratios × 200 samples = 600 samples")

print(f"\n4. Recommended Strategy:")
print(f"   - Start with Chebyshev nodes (Option A) - theoretically optimal")
print(f"   - Use N_SAMPLES = {optimal_n_samples}")
print(f"   - Validate and refine with greedy selection if needed")

print("\n" + "=" * 70)
