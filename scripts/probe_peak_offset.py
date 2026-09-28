"""
Diagnose whether the recovered-peak offset seen in
demo_uncertainty_comparison.py (matched-filter/GP-marginalised posteriors
peaking measurably away from the injected truth, at scales far finer than
the training grid spacing) is real waveform-model systematic error or a
numerical artefact in evaluating the GP surrogate at fine parameter
resolution.

Bypasses noise and the likelihood machinery entirely: computes the direct
noise-weighted match between the surrogate's own mean prediction and the
true reference (LALSuite) waveform, scanned over a fine q grid around
q_true. A smooth match curve peaking slightly off q_true would indicate
real model mismatch; a jagged/noisy curve would indicate a numerical
artefact in the GP evaluation itself.

Usage::

    python scripts/probe_peak_offset.py \\
        --checkpoint checkpoints/phenomd_nonspinning_dense30.pt \\
        --q-true 0.8 --half-width 0.002 --n-grid 400
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from astropy import units as u

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heron.models.gp.exact import ExactGPSurrogate
from heron.evaluation.psd import aligo_design_psd
from heron.noise import noise_covariance
from heron.detector import antenna_patterns, project_waveform
from heron.train import _get_approximant


def match(template: np.ndarray, reference: np.ndarray, C: np.ndarray) -> float:
    """Noise-weighted normalised overlap between template and reference."""
    Cinv_ref = np.linalg.solve(C, reference)
    num = template @ Cinv_ref
    denom = np.sqrt((template @ np.linalg.solve(C, template)) * (reference @ Cinv_ref))
    return float(num / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe source of peak-offset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--q-true", type=float, default=0.8)
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--total-mass", type=float, default=60.0)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument("--sample-rate", type=float, default=512.0)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--detector", default="H1")
    parser.add_argument("--half-width", type=float, default=0.002,
                        help="Half-width of the fine q scan around q-true")
    parser.add_argument("--n-grid", type=int, default=400)
    parser.add_argument("--output", default="results/peak_offset_probe.png")
    args = parser.parse_args()

    tc_true = 1187008882.43
    ra_true, dec_true, psi_true = 3.446, -0.408, 0.0

    n = int(args.duration * args.sample_rate)
    t_start = tc_true - args.duration * 0.75
    times = t_start + np.arange(n) / args.sample_rate
    t_rel = times - tc_true

    print(f"Loading surrogate from {args.checkpoint} ...")
    surrogate = ExactGPSurrogate.load(args.checkpoint)

    fp, fc = antenna_patterns(ra_true, dec_true, psi_true, tc_true, args.detector)
    C = noise_covariance(times, aligo_design_psd, f_low=20.0, jitter_rel=1e-8)

    # HP filter, matching gw_likelihood.py's convention.
    dt = 1.0 / args.sample_rate
    freqs = np.fft.rfftfreq(n, d=dt)
    hp_mask = freqs >= 20.0

    def hp_filter(x):
        xf = np.fft.rfft(x)
        xf[~hp_mask] = 0.0
        return np.fft.irfft(xf, n=n)

    print(f"Building reference (true) waveform at q={args.q_true} via {args.approximant} ...")
    approximant = _get_approximant(args.approximant)
    params = {
        "mass_ratio": args.q_true,
        "total_mass": args.total_mass * u.solMass,
        "luminosity_distance": args.distance * u.Mpc,
        "f_min": 20.0 * u.Hertz,
        "delta_t": (1.0 / 4096) * u.second,
    }
    wf_true = approximant.time_domain(params, times=t_rel)
    reference = hp_filter(fp * wf_true["plus"].data + fc * wf_true["cross"].data)

    q_grid = np.linspace(args.q_true - args.half_width, args.q_true + args.half_width, args.n_grid)
    matches = np.empty(args.n_grid)
    print(f"Scanning surrogate mean vs. true waveform over {args.n_grid} q points "
          f"in [{q_grid[0]:.6f}, {q_grid[-1]:.6f}] ...")
    for i, q in enumerate(q_grid):
        wf = surrogate.predict({"mass_ratio": float(q), "times": t_rel})
        template = hp_filter(fp * wf["plus"].data + fc * wf["cross"].data)
        matches[i] = match(template, reference, C)
        if (i + 1) % 50 == 0 or i == args.n_grid - 1:
            print(f"  {i + 1}/{args.n_grid}", end="\r", flush=True)
    print()

    wf_at_true = surrogate.predict({"mass_ratio": args.q_true, "times": t_rel})
    template_at_true = hp_filter(fp * wf_at_true["plus"].data + fc * wf_at_true["cross"].data)
    match_at_true = match(template_at_true, reference, C)

    i_peak = int(np.argmax(matches))
    q_peak = q_grid[i_peak]
    print(f"\nMatch at q_true         : {match_at_true:.6f}")
    print(f"Max match in scan       : {matches[i_peak]:.6f}  at q={q_peak:.6f}  (q-q_true={q_peak - args.q_true:+.6f})")

    # Roughness diagnostic: compare the actual curve to a smoothed version.
    # A physically real (smooth) mismatch should track its own running mean
    # closely; numerical noise/artefacts show up as high-frequency residuals.
    window = max(3, args.n_grid // 40)
    kernel = np.ones(window) / window
    smoothed = np.convolve(matches, kernel, mode="same")
    residual = matches - smoothed
    print(f"Roughness (std of match - smoothed(match)): {residual.std():.3e}")
    print(f"Overall match variation (max - min):         {matches.max() - matches.min():.3e}")
    print(f"Roughness / variation ratio:                 {residual.std() / (matches.max() - matches.min() + 1e-300):.3f}")
    print("  (near 0 => smooth curve, real systematic offset;"
          " near/above ~0.3 => dominated by noise/jaggedness)")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(q_grid - args.q_true, matches, lw=1.2)
    ax.axvline(0.0, color="k", ls=":", lw=1.0, label="Injected truth")
    ax.axvline(q_peak - args.q_true, color="r", ls="--", lw=1.0, label=f"Peak match (Δq={q_peak - args.q_true:+.5f})")
    ax.set_xlabel(r"$q - q_\mathrm{true}$")
    ax.set_ylabel("Noise-weighted match (GP mean vs. true waveform)")
    ax.set_title(f"Peak-offset probe — {args.checkpoint}, q_true={args.q_true}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved figure -> {args.output}")


if __name__ == "__main__":
    main()
