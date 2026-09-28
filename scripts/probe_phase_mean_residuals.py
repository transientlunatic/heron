"""Quantify how well candidate phase/log-amplitude mean functions track the
PhaseAmplitudeGPSurrogate's actual training targets, without any GP training.

Compares, against the decomposed training targets:
  - PN means (Newtonian, TaylorT2/1PN) -- measured 2026-07-15: only ~12%
    phase-residual reduction at M=60 (PN truncation error is comparable to
    the total in-band accumulated phase at this mass);
  - full-IMR approximant means (LALApproximant*Mean, e.g. IMRPhenomXAS) --
    the multi-fidelity idea: the GP fits only the (small) difference
    between the training approximant and the mean approximant, which stays
    phase-coherent through merger/ringdown by construction.

For approximant means this also checks the phase-unwrap branch alignment:
the residual must not contain spurious per-q 2*pi*k offsets (a
non-smooth-in-q residual would break GP interpolation between mass-ratio
nodes even if each node's residual is tiny).

Usage:
    python scripts/probe_phase_mean_residuals.py \
        [--data checkpoints/phenomd_nonspinning_dense30_phase_amplitude_data.h5] \
        [--approximant-mean IMRPhenomXAS]
"""

import argparse

import h5py
import numpy as np
import torch

from heron.models.gp.mean import (
    LALApproximantAmplitudeMean,
    LALApproximantPhaseMean,
    NewtonianInspiralAmplitudeMean,
    NewtonianInspiralPhaseMean,
    TaylorT2PhaseMean,
)
from heron.models.gp.phase_amplitude import strain_to_amplitude_phase
from heron.models.warping import get_warping


def per_q_demeaned(resid: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Residual after removing each mass ratio's constant offset.

    The unwrapped-phase target carries an arbitrary per-q additive constant
    (set by atan2 at the earliest sample); a constant per-q offset in the
    residual is expected and easy for the GP as long as it is smooth in q.
    """
    out = resid.copy()
    for qv in np.unique(q):
        m = q == qv
        out[m] -= out[m].mean()
    return out


def report_phase_mean(name, mean_fn, x_warped, phase_np, q, t, inspiral):
    with torch.no_grad():
        mu = mean_fn(x_warped).to(torch.float64).numpy()
    resid = phase_np - mu
    r_all = np.corrcoef(phase_np, mu)[0, 1]
    print(f"\n--- {name} phase mean ---")
    print(f"  corr(target, mean): {r_all:+.6f}")
    print(f"  residual std (all):                   {resid.std():10.3f} rad")
    print(f"  residual std (inspiral):              {resid[inspiral].std():10.3f} rad")
    print(f"  residual std (inspiral, per-q demeaned): "
          f"{per_q_demeaned(resid[inspiral], q[inspiral]).std():10.3f} rad")
    print(f"  residual std (post-merger):           {resid[~inspiral].std():10.3f} rad")
    print(f"  residual std (post-merger, per-q demeaned): "
          f"{per_q_demeaned(resid[~inspiral], q[~inspiral]).std():10.3f} rad")

    # Per-q constant offsets: smooth in q is fine; 2*pi-scale jumps between
    # adjacent q nodes would break GP interpolation between nodes.
    qs = np.unique(q)
    offsets = np.array([resid[q == qv].mean() for qv in qs])
    doff = np.diff(offsets)
    print(f"  per-q offset: range [{offsets.min():+.2f}, {offsets.max():+.2f}] rad, "
          f"max |adjacent-q jump| = {np.abs(doff).max():.2f} rad "
          f"(2*pi = {2*np.pi:.2f})")
    if np.abs(doff).max() > np.pi:
        print("  WARNING: adjacent-q offset jump exceeds pi -- possible "
              "unwrap-branch misalignment; inspect offsets:")
        for qv, off in zip(qs, offsets):
            print(f"    q={qv:.2f}: {off:+8.2f} rad")

    # Merger continuity of the residual, per q.
    jumps = []
    for qv in qs:
        m = q == qv
        tm, rm = t[m], resid[m]
        order = np.argsort(tm)
        tm, rm = tm[order], rm[order]
        pre = rm[tm < -1e-6]
        post = rm[tm >= -1e-6]
        if len(pre) and len(post):
            jumps.append(post[0] - pre[-1])
    jumps = np.array(jumps)
    print(f"  merger residual jump (last inspiral -> first post-merger sample, "
          f"per q): median |jump| = {np.median(np.abs(jumps)):.2f} rad, "
          f"max |jump| = {np.abs(jumps).max():.2f} rad")


def report_amplitude_mean(name, mean_fn, x_warped, logA_np, q, inspiral):
    with torch.no_grad():
        mu = mean_fn(x_warped).to(torch.float64).numpy()
    resid = logA_np - mu
    r = np.corrcoef(logA_np, mu)[0, 1]
    print(f"\n--- {name} log-amplitude mean ---")
    print(f"  corr(target, mean): {r:+.6f}")
    print(f"  residual std (all): {resid.std():.4f}   "
          f"(inspiral): {resid[inspiral].std():.4f}   "
          f"(post-merger): {resid[~inspiral].std():.4f}")
    print(f"  residual std per-q demeaned: "
          f"(inspiral): {per_q_demeaned(resid[inspiral], q[inspiral]).std():.4f}   "
          f"(post-merger): {per_q_demeaned(resid[~inspiral], q[~inspiral]).std():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="checkpoints/phenomd_nonspinning_dense30_phase_amplitude_data.h5",
    )
    parser.add_argument("--alpha", type=float, default=0.625)
    parser.add_argument("--total-mass", type=float, default=60.0)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument(
        "--approximant-mean", default="IMRPhenomXAS",
        help="LAL approximant to test as a mean function ('' to skip)",
    )
    args = parser.parse_args()

    with h5py.File(args.data, "r") as f:
        x = torch.tensor(f["x"][...], dtype=torch.float64)
        y_plus = torch.tensor(f["y_plus"][...], dtype=torch.float64)
        y_cross = torch.tensor(f["y_cross"][...], dtype=torch.float64)

    # Reproduce the surrogate constructor's exact preprocessing.
    x_sorted, log_amplitude, phase = strain_to_amplitude_phase(x, y_plus, y_cross)
    warping = get_warping("chirp", alpha=args.alpha)
    x_warped = x_sorted.clone()
    x_warped[:, -1] = warping.warp(x_warped[:, -1], mass_ratio=x_warped[:, 0])

    q = x_sorted[:, 0].numpy()
    t = x_sorted[:, -1].numpy()  # unwarped time
    phase_np = phase.numpy()
    logA_np = log_amplitude.numpy()
    inspiral = t < -1e-6

    print(f"data: {args.data}")
    print(f"N = {len(q)}, {len(np.unique(q))} mass ratios, "
          f"{inspiral.sum()} inspiral / {(~inspiral).sum()} post-merger samples")

    print("\n=== PHASE target ===")
    print(f"target std (all): {phase_np.std():10.3f} rad   "
          f"(inspiral, per-q demeaned): "
          f"{per_q_demeaned(phase_np[inspiral], q[inspiral]).std():10.3f} rad")

    kwargs = dict(total_mass=args.total_mass, distance=args.distance, warping=warping)
    report_phase_mean("newtonian", NewtonianInspiralPhaseMean(**kwargs),
                      x_warped, phase_np, q, t, inspiral)
    report_phase_mean("taylort2", TaylorT2PhaseMean(**kwargs),
                      x_warped, phase_np, q, t, inspiral)
    if args.approximant_mean:
        report_phase_mean(
            args.approximant_mean,
            LALApproximantPhaseMean(approximant=args.approximant_mean, **kwargs),
            x_warped, phase_np, q, t, inspiral,
        )

    print("\n=== LOG-AMPLITUDE target ===")
    print(f"target std (all): {logA_np.std():.4f}   "
          f"(inspiral, per-q demeaned): "
          f"{per_q_demeaned(logA_np[inspiral], q[inspiral]).std():.4f}   "
          f"(post-merger, per-q demeaned): "
          f"{per_q_demeaned(logA_np[~inspiral], q[~inspiral]).std():.4f}")

    report_amplitude_mean("newtonian(=taylort2)",
                          NewtonianInspiralAmplitudeMean(**kwargs),
                          x_warped, logA_np, q, inspiral)
    if args.approximant_mean:
        report_amplitude_mean(
            args.approximant_mean,
            LALApproximantAmplitudeMean(approximant=args.approximant_mean, **kwargs),
            x_warped, logA_np, q, inspiral,
        )


if __name__ == "__main__":
    main()
