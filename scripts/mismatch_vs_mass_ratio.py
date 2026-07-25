"""
Scan surrogate-vs-reference mismatch across mass_ratio to test whether the
large PE bias found in scripts/injection_nested_sampling.py (surrogate mean at
q=0.5 only has ~92% overlap with the true IMRPhenomD waveform) is explained by
GP interpolation error between training-grid nodes.

Uses the same fitting-factor overlap (maximised over phase and time shift,
aLIGO-PSD-weighted) as heron.evaluation.mismatch.MismatchEvaluator, but on a
deterministic fine grid in q rather than Sobol samples, so mismatch can be
plotted against distance to the nearest training node.

Usage::

    python scripts/mismatch_vs_mass_ratio.py \\
        --checkpoint checkpoints/phenomd_nonspinning_dense30.pt \\
        --mass-ratios 0.10 0.13 0.16 ... \\
        --output results/mismatch_vs_q_dense30.png
"""
from __future__ import annotations

import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heron.evaluation.mismatch import compute_mismatch
from heron.evaluation.psd import aligo_design_psd
from heron.train import _get_approximant


def load_surrogate(path: str, device: str = "cpu"):
    """Load a checkpoint into whichever WaveformSurrogate class trained it.

    Checkpoints self-describe their class via the `model_class` field (see
    ExactGPSurrogate.save/PhaseAmplitudeGPSurrogate.save), so this dispatches
    without the caller needing to know which representation is on disk.
    """
    import torch

    model_class = torch.load(path, map_location="cpu", weights_only=False)["model_class"]
    if model_class == "PhaseAmplitudeGPSurrogate":
        from heron.models.gp.phase_amplitude import PhaseAmplitudeGPSurrogate
        return PhaseAmplitudeGPSurrogate.load(path, device=device)
    if model_class == "DeltaGPSurrogate":
        from heron.models.gp.delta import DeltaGPSurrogate
        return DeltaGPSurrogate.load(path, device=device)
    from heron.models.gp.exact import ExactGPSurrogate
    return ExactGPSurrogate.load(path, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mismatch vs mass_ratio scan")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mass-ratios", type=float, nargs="+", required=True,
                         help="Training-grid mass ratios for this checkpoint")
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--total-mass", type=float, default=60.0)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument("--q-lo", type=float, default=0.12)
    parser.add_argument("--q-hi", type=float, default=0.95)
    parser.add_argument("--n-scan", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--t-lo", type=float, default=-0.5,
                         help="Window lower bound (s, relative to merger). "
                              "Narrow this (e.g. -0.05) to isolate "
                              "merger/ringdown mismatch from the inspiral.")
    parser.add_argument("--t-hi", type=float, default=0.02,
                         help="Window upper bound (s, relative to merger).")
    args = parser.parse_args()

    from astropy import units as u

    grid = np.array(sorted(args.mass_ratios))
    print(f"Loading surrogate from {args.checkpoint} on {args.device} ...")
    surrogate = load_surrogate(args.checkpoint, device=args.device)
    approximant = _get_approximant(args.approximant)

    lower, upper, number = args.t_lo, args.t_hi, 512
    times = np.linspace(lower, upper, number)
    dt = times[1] - times[0]
    freqs = np.fft.rfftfreq(number, d=dt)
    psd = aligo_design_psd(freqs)

    q_scan = np.linspace(args.q_lo, args.q_hi, args.n_scan)
    dist_to_node = np.array([np.min(np.abs(q - grid)) for q in q_scan])

    mismatches = np.empty(args.n_scan)
    print(f"Scanning {args.n_scan} mass-ratio points ...")
    for i, q in enumerate(q_scan):
        surr_wf = surrogate.predict({"mass_ratio": float(q), "times": times})
        surr_plus = surr_wf["plus"].data

        ref_params = {
            "mass_ratio": float(q),
            "total_mass": args.total_mass * u.solMass,
            "luminosity_distance": args.distance * u.Mpc,
        }
        ref_wf = approximant.time_domain(ref_params, times=times)
        ref_plus = ref_wf["plus"].data

        mismatches[i] = compute_mismatch(surr_plus, ref_plus, dt, psd)
        if (i + 1) % 25 == 0 or i == args.n_scan - 1:
            print(f"  {i+1}/{args.n_scan}  q={q:.4f}  mismatch={mismatches[i]:.4e}")

    np.savez(args.output.replace(".png", ".npz"),
              q_scan=q_scan, mismatch=mismatches, dist_to_node=dist_to_node,
              grid=grid)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.semilogy(q_scan, mismatches, "-", color="#1f77b4", lw=1.3)
    for g in grid:
        ax.axvline(g, color="k", lw=0.5, alpha=0.3)
    ax.axhline(1e-2, color="r", ls="--", lw=1, label="PE-grade (1e-2)")
    ax.axhline(1e-3, color="g", ls="--", lw=1, label="Detection-grade (1e-3)")
    ax.set_xlabel("mass_ratio")
    ax.set_ylabel("mismatch (1 - fitting factor)")
    ax.set_title(f"Mismatch vs q  ({len(grid)}-node grid)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.loglog(dist_to_node + 1e-6, mismatches, ".", color="#1f77b4", ms=4)
    ax.axhline(1e-2, color="r", ls="--", lw=1)
    ax.axhline(1e-3, color="g", ls="--", lw=1)
    ax.set_xlabel("distance to nearest training node")
    ax.set_ylabel("mismatch")
    ax.set_title("Mismatch vs distance-to-node")

    fig.suptitle(args.checkpoint)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved figure -> {args.output}")
    print(f"\nMedian mismatch: {np.median(mismatches):.4e}  "
          f"Worst: {np.max(mismatches):.4e} at q={q_scan[np.argmax(mismatches)]:.4f}")


if __name__ == "__main__":
    main()
