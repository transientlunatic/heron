"""
Scan K/C (GP uncertainty / noise) across a range of mass ratios.

Unlike demo_uncertainty_comparison.py (which checks one injection point in
depth, including full posterior scans), this checks the core log-det-bias
condition from CLAUDE.md -- K/C < 1 everywhere -- across the whole trained
q range cheaply (no posterior scanning, just the K_ii/C_ii diagonal ratio
at each q).

Usage::

    python scripts/kc_profile_scan.py \\
        --checkpoint checkpoints/phenomd_nonspinning_dense30.pt \\
        --q-min 0.12 --q-max 0.95 --n-q 30
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from heron.models.gp.exact import ExactGPSurrogate
from heron.evaluation.psd import aligo_design_psd
from heron.noise import noise_covariance
from heron.detector import antenna_patterns, project_waveform


def main() -> None:
    parser = argparse.ArgumentParser(description="K/C profile scan across mass ratio")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-rate", type=float, default=512.0)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--detector", default="H1")
    parser.add_argument("--q-min", type=float, default=0.12)
    parser.add_argument("--q-max", type=float, default=0.95)
    parser.add_argument("--n-q", type=int, default=30)
    args = parser.parse_args()

    tc_true = 1187008882.43
    ra_true = 3.446
    dec_true = -0.408
    psi_true = 0.0

    n = int(args.duration * args.sample_rate)
    t_start = tc_true - args.duration * 0.75
    times = t_start + np.arange(n) / args.sample_rate

    print(f"Loading surrogate from {args.checkpoint} ...")
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if raw_checkpoint.get("model_type") == "sparse":
        from heron.models.gp.sparse import SparseGPSurrogate
        surrogate = SparseGPSurrogate.load(args.checkpoint)
    else:
        surrogate = ExactGPSurrogate.load(args.checkpoint)

    fp, fc = antenna_patterns(ra_true, dec_true, psi_true, tc_true, args.detector)
    C = noise_covariance(times, aligo_design_psd, f_low=20.0, jitter_rel=1e-8)

    q_grid = np.linspace(args.q_min, args.q_max, args.n_q)
    print(f"\n{'q':>6}  {'KC_min':>8}  {'KC_mean':>8}  {'KC_max':>8}")
    kc_means = []
    for q in q_grid:
        wf = surrogate.predict({"mass_ratio": float(q), "times": times - tc_true})
        _, K = project_waveform(wf, fp, fc)
        ratio = K.diagonal() / C.diagonal()
        kc_means.append(ratio.mean())
        print(f"{q:6.3f}  {ratio.min():8.2f}  {ratio.mean():8.2f}  {ratio.max():8.2f}")

    kc_means = np.array(kc_means)
    print(f"\nOverall across {args.n_q} q values:")
    print(f"  mean(K/C mean) = {kc_means.mean():.2f}")
    print(f"  max(K/C mean)  = {kc_means.max():.2f}  at q={q_grid[kc_means.argmax()]:.3f}")
    print(f"  frac q-points with mean K/C < 1: {(kc_means < 1).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
