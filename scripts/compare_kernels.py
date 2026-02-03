#!/usr/bin/env python3
"""
Compare RBF vs Matérn(ν=1.5) vs Matérn(ν=2.5) kernels.

Goal: Find which kernel best handles the sharp peak + ringdown structure.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from heron.training.data import DataWrapper001
from heron.models.gpytorch import HeronNonSpinningApproximant
from heron.models.lalsimulation import IMRPhenomPv2

# Import GP models directly to create custom approximants
from heron.models.gpytorch import ExactGPModelKeOps, ExactGPModelMatern
from heron.models.gpytorch import WaveformSurrogate, GPyTorchSurrogate
from heron.types import Waveform, WaveformDict
import gpytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create custom approximant for Matérn ν=1.5
class HeronMatern15(WaveformSurrogate, GPyTorchSurrogate):
    """HeronNonSpinningApproximant with Matérn(ν=1.5) kernels."""
    def __init__(self, train_x_plus, train_x_cross, train_y_plus, train_y_cross,
                 total_mass, distance, warp_scale=2, training=400):
        self.device = device
        self.output_scale = 1e27
        self.warp_scale = warp_scale
        self.mass_factor = total_mass
        self.distance_factor = distance

        # Warp training data
        self.train_x_plus = train_x_plus.to(self.device)
        self.train_x_cross = train_x_cross.to(self.device)
        times_plus = self.train_x_plus[:, 1]
        times_plus[times_plus < 0] = times_plus[times_plus < 0] / self.warp_scale
        self.train_x_plus[:, 1] = times_plus
        times_cross = self.train_x_cross[:, 1]
        times_cross[times_cross < 0] = times_cross[times_cross < 0] / self.warp_scale
        self.train_x_cross[:, 1] = times_cross

        self.train_y_plus = train_y_plus.cuda() * self.output_scale
        self.train_y_cross = train_y_cross.cuda() * self.output_scale

        # Use Matérn(ν=1.5) kernels
        self.models = {}
        self.models["plus"] = ExactGPModelMatern(
            self.train_x_plus, self.train_y_plus, nu=1.5
        ).to(self.device)
        self.models["cross"] = ExactGPModelMatern(
            self.train_x_cross, self.train_y_cross, nu=1.5
        ).to(self.device)

        for polarisation in ("plus", "cross"):
            self.models[polarisation].likelihood.cuda()

        self._args = {"total_mass": None, "mass_ratio": None,
                      "luminosity_distance": None, "inclination": None}

        self.train(training)

    def time_domain(self, parameters, times=None):
        """Return a timedomain waveform."""
        a = parameters.get("mass_ratio", parameters.get("mass ratio"))
        t = parameters.get("time", parameters.get("gpstime"))
        total_mass = parameters.get("total_mass", self.mass_factor)
        mass_factor = (total_mass / self.mass_factor).value
        distance = parameters.get("luminosity_distance", self.distance_factor)
        distance_factor = distance / self.distance_factor

        times = torch.linspace(t["lower"], t["upper"], t["number"],
                               dtype=torch.float32) / mass_factor

        points = torch.vstack([
            torch.ones(t["number"], dtype=torch.float32) * a,
            torch.linspace(t["lower"], t["upper"], t["number"],
                          dtype=torch.float32) / mass_factor,
        ]).T.to(device=self.device)

        # Warp the time axis
        points[points[:, 1] < 0, 1] = points[points[:, 1] < 0, 1] / self.warp_scale

        parameters.pop("time")
        output = WaveformDict(parameters=parameters)

        for polarisation in ("plus", "cross"):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.models[polarisation].likelihood(
                    self.models[polarisation](points)
                )
                mean = observed_pred.mean

            # Unwarp
            points[points[:, 1] < 0, 1] = points[points[:, 1] < 0, 1] * self.warp_scale

            output.waveforms[polarisation] = Waveform(
                data=mean.cpu() / self.output_scale / distance_factor,
                times=times,
                covariance=observed_pred.covariance_matrix.cpu()
                / self.output_scale / self.output_scale / distance_factor**2,
            )

        return output

print("=" * 70)
print("Kernel Comparison: RBF vs Matérn(ν=1.5) vs Matérn(ν=2.5)")
print("=" * 70)

# Load data
dw = DataWrapper001('training_data_100mpc.h5')
train_x_plus, train_y_plus = dw.get_training_data('IMR_training_test', polarisation=b'p')
train_x_cross, train_y_cross = dw.get_training_data('IMR_training_test', polarisation=b'c')

device_str = 'cuda'
train_x_plus_t = torch.tensor(train_x_plus.T, dtype=torch.float32, device=device_str)
train_y_plus_t = torch.tensor(train_y_plus, dtype=torch.float32, device=device_str)
train_x_cross_t = torch.tensor(train_x_cross.T, dtype=torch.float32, device=device_str)
train_y_cross_t = torch.tensor(train_y_cross, dtype=torch.float32, device=device_str)

# Train all three models
models = {}
names = ["RBF", "Matérn(ν=1.5)", "Matérn(ν=2.5)"]

print("\n" + "=" * 70)
print("Training Models (500 iterations each)")
print("=" * 70)

# Import the existing Matérn approximant (defaults to ν=2.5)
from heron.models.gpytorch import HeronNonSpinningApproximantMatern

for name in names:
    print(f"\n{name}...")
    if name == "RBF":
        models[name] = HeronNonSpinningApproximant(
            train_x_plus=train_x_plus_t.clone(),
            train_y_plus=train_y_plus_t.clone(),
            train_x_cross=train_x_cross_t.clone(),
            train_y_cross=train_y_cross_t.clone(),
            total_mass=20 * u.solMass,
            distance=100 * u.Mpc,
            training=500,
            warp_scale=2,
        )
    elif name == "Matérn(ν=1.5)":
        models[name] = HeronMatern15(
            train_x_plus=train_x_plus_t.clone(),
            train_x_cross=train_x_cross_t.clone(),
            train_y_plus=train_y_plus_t.clone(),
            train_y_cross=train_y_cross_t.clone(),
            total_mass=20 * u.solMass,
            distance=100 * u.Mpc,
            training=500,
            warp_scale=2,
        )
    else:  # Matérn(ν=2.5)
        models[name] = HeronNonSpinningApproximantMatern(
            train_x_plus=train_x_plus_t.clone(),
            train_y_plus=train_y_plus_t.clone(),
            train_x_cross=train_x_cross_t.clone(),
            train_y_cross=train_y_cross_t.clone(),
            total_mass=20 * u.solMass,
            distance=100 * u.Mpc,
            training=500,
            warp_scale=2,
        )

# Reference model
ref_model = IMRPhenomPv2()

# Test parameters
q = 0.5
params = {
    "mass_ratio": q,
    "total_mass": 20 * u.solMass,
    "distance": 100 * u.Mpc,
    "time": {"lower": -0.1, "upper": 0.05, "number": 500}
}

print("\n" + "=" * 70)
print("Generating Waveforms")
print("=" * 70)

waveforms = {}
for name in names:
    waveforms[name] = models[name].time_domain(parameters=params.copy())

waveforms["Reference"] = ref_model.time_domain(parameters=params.copy())

# Analysis
print("\n" + "=" * 70)
print("Results Summary")
print("=" * 70)

ref_times = np.array(waveforms["Reference"]['plus'].times.value)
ref_data = np.array(waveforms["Reference"]['plus'].data)
ref_peak_idx = np.argmax(np.abs(ref_data))
ref_peak_time = ref_times[ref_peak_idx]
ref_amp = np.max(np.abs(ref_data))

print(f"\n{'Model':<20} {'Peak Time':<12} {'Time Error':<12} {'Amp Ratio':<12} {'Mismatch %':<12}")
print("-" * 75)
print(f"{'Reference':<20} {ref_peak_time:<12.4f} {'—':<12} {'1.000':<12} {'—':<12}")

results = {}
for name in names:
    times = np.array(waveforms[name]['plus'].times.value)
    data = np.array(waveforms[name]['plus'].data)

    peak_idx = np.argmax(np.abs(data))
    peak_time = times[peak_idx]
    time_error = abs(peak_time - ref_peak_time)

    amp = np.max(np.abs(data))
    amp_ratio = amp / ref_amp

    # Simple overlap
    norm_data = data / np.sqrt(np.sum(np.abs(data)**2))
    norm_ref = ref_data / np.sqrt(np.sum(np.abs(ref_data)**2))
    overlap = np.abs(np.sum(norm_data * np.conj(norm_ref)))
    mismatch = (1 - overlap) * 100

    results[name] = {
        'time_error': time_error,
        'amp_ratio': amp_ratio,
        'mismatch': mismatch,
        'peak_time': peak_time,
        'peak_idx': peak_idx
    }

    print(f"{name:<20} {peak_time:<12.4f} {time_error:<12.4f} {amp_ratio:<12.3f} {mismatch:<12.2f}")

# Find best model
best_mismatch = min(results[name]['mismatch'] for name in names)
best_model = [name for name in names if results[name]['mismatch'] == best_mismatch][0]

print("\n" + "=" * 70)
print("Conclusion")
print("=" * 70)

print(f"\nBest model: {best_model}")
print(f"  Mismatch: {results[best_model]['mismatch']:.2f}%")
print(f"  Time error: {results[best_model]['time_error']:.4f}s")
print(f"  Amplitude ratio: {results[best_model]['amp_ratio']:.3f}")

if results[best_model]['mismatch'] < 1.0:
    print(f"\n✓ SUCCESS: {best_model} achieves <1% mismatch!")
elif results[best_model]['mismatch'] < 5.0:
    print(f"\n✓ GOOD: {best_model} achieves <5% mismatch")
    print(f"  May be sufficient for some applications")
elif results[best_model]['mismatch'] < best_mismatch * 2:
    print(f"\n⚠ IMPROVEMENT: {best_model} is better but still needs work")
    print(f"  Consider:")
    print(f"  - Adaptive warping (physical or learned)")
    print(f"  - Spectral mixture kernels")
    print(f"  - More training iterations")
else:
    print(f"\n✗ INSUFFICIENT: All kernels struggle with this problem")
    print(f"  Need more fundamental changes:")
    print(f"  - Adaptive/physical warping")
    print(f"  - Mean function approach (next paper)")

# Detailed comparison
print(f"\n" + "=" * 70)
print("Detailed Comparison")
print("=" * 70)

for name in names:
    improvement = (results["RBF"]['mismatch'] - results[name]['mismatch'])
    if name != "RBF":
        print(f"\n{name} vs RBF:")
        print(f"  Mismatch: {results['RBF']['mismatch']:.2f}% → {results[name]['mismatch']:.2f}%")
        print(f"  Improvement: {improvement:.2f}% ({improvement/results['RBF']['mismatch']*100:.1f}% reduction)")
        print(f"  Time error: {results['RBF']['time_error']:.4f}s → {results[name]['time_error']:.4f}s")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Full waveforms
ax = axes[0, 0]
colors = {'Reference': 'black', 'RBF': 'red', 'Matérn(ν=1.5)': 'blue', 'Matérn(ν=2.5)': 'green'}
linestyles = {'Reference': '-', 'RBF': '--', 'Matérn(ν=1.5)': '-.', 'Matérn(ν=2.5)': ':'}
for name in ['Reference'] + names:
    times = np.array(waveforms[name]['plus'].times.value)
    data = np.array(waveforms[name]['plus'].data)
    ax.plot(times, data, color=colors[name], linestyle=linestyles[name],
            linewidth=2 if name == 'Reference' else 1.5, label=name, alpha=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain')
ax.set_title('Waveform Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Amplitude envelopes (log scale)
ax = axes[0, 1]
for name in ['Reference'] + names:
    times = np.array(waveforms[name]['plus'].times.value)
    data = np.array(waveforms[name]['plus'].data)
    ax.plot(times, np.abs(data), color=colors[name], linestyle=linestyles[name],
            linewidth=2 if name == 'Reference' else 1.5, label=name, alpha=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('|Strain|')
ax.set_title('Amplitude Envelopes')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Mismatch comparison
ax = axes[1, 0]
x_pos = np.arange(len(names))
mismatches = [results[name]['mismatch'] for name in names]
bars = ax.bar(x_pos, mismatches, color=['red', 'blue', 'green'], alpha=0.7)
ax.set_ylabel('Mismatch (%)')
ax.set_title('Mismatch Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.axhline(y=1.0, color='black', linestyle='--', label='1% target')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (bar, mismatch) in enumerate(zip(bars, mismatches)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{mismatch:.1f}%', ha='center', va='bottom')

# Plot 4: Time error comparison
ax = axes[1, 1]
time_errors = [results[name]['time_error'] for name in names]
bars = ax.bar(x_pos, time_errors, color=['red', 'blue', 'green'], alpha=0.7)
ax.set_ylabel('Time Error (s)')
ax.set_title('Peak Timing Error')
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.axhline(y=0.01, color='black', linestyle='--', label='0.01s threshold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (bar, error) in enumerate(zip(bars, time_errors)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{error:.3f}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('kernel_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: kernel_comparison.png")
