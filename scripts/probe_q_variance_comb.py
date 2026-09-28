#!/usr/bin/env python3
"""Measure the training-grid-periodic posterior-variance "comb" in q.

Background
----------
``ExactGPSurrogate`` is trained on a tensor-product grid: a handful of
distinct mass ratios (e.g. dense30 = 30 q's at spacing 0.03), each repeated
over many time samples. A GP posterior variance is always minimised at/near
training inputs, so with only ~30 distinct q's the q-marginal posterior
variance develops a periodic "comb" — deep dips at the training nodes,
higher between them, with period equal to the q-grid spacing. That comb
feeds the with-K likelihood's log-det term ``-1/2 log|C+K(theta)|`` and pulls
posteriors toward training nodes (the log-det grid-snap bias; see CLAUDE.md).

This script measures the comb directly on any exact checkpoint: it scans q
finely, evaluates the GP's own diagonal posterior variance at each q
(reduced to a scalar over a fixed time window), and quantifies how strongly
that variance oscillates at the training-grid period. The *scattered*
sampling experiment (heron/train.py `mode: scattered`) predicts that
training on many more distinct, quasi-random q's flattens this comb at its
root. Run this on a grid checkpoint and on a scattered checkpoint and
compare the comb metrics.

No nested sampling required — this measures the mechanism, cheaply.

Usage
-----
    python scripts/probe_q_variance_comb.py CHECKPOINT.pt [options]

    # baseline (grid) vs scattered, same axes:
    python scripts/probe_q_variance_comb.py checkpoints/grid.pt --label grid
    python scripts/probe_q_variance_comb.py checkpoints/scatter.pt --label scatter
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from heron.models.gp.exact import ExactGPSurrogate


def scalar_variance_vs_q(model, q_grid, t_lo, t_hi, n_t, polarisation="plus"):
    """Diagonal posterior variance reduced to one scalar per q.

    For each q, evaluates the GP diagonal predictive variance over a fixed
    window of ``n_t`` physical times in ``[t_lo, t_hi]`` and takes the mean.
    Uses ``_covariance_diag`` (mean module never evaluated), so it is cheap
    and independent of the possibly-expensive LAL mean function.
    """
    times = np.linspace(t_lo, t_hi, n_t)
    out = np.empty(len(q_grid))
    for i, q in enumerate(q_grid):
        params = {"mass_ratio": float(q), "times": times}
        diag = model._covariance_diag(params)[polarisation]
        out[i] = float(np.mean(diag))
    return out


def comb_metrics(q_grid, var, node_qs):
    """Quantify the periodic comb in ``var(q)``.

    Returns a dict with:
      - ``node_antinode_ratio``: median(antinode var) / median(node var),
        using the known training-node locations. >1 means variance dips at
        nodes (the comb). ~1 means flat. Only meaningful when nodes are
        (near-)evenly spaced.
      - ``rel_oscillation``: std of the grid-period-band-pass-filtered
        variance / mean variance — a node-location-free comb strength that
        works for scattered checkpoints too.
      - ``spectral_comb_fraction``: fraction of the detrended-variance power
        spectrum concentrated near the training-grid frequency (and its
        first harmonic). High ⇒ a clean periodic comb; low/broadband ⇒ the
        comb is gone.
    """
    metrics = {}

    # --- node/antinode ratio (needs node locations) ---
    node_qs = np.sort(np.asarray(node_qs, dtype=float))
    if len(node_qs) >= 3:
        var_at_nodes = np.interp(node_qs, q_grid, var)
        antinodes = 0.5 * (node_qs[:-1] + node_qs[1:])
        var_at_antinodes = np.interp(antinodes, q_grid, var)
        node_med = np.median(var_at_nodes)
        metrics["node_antinode_ratio"] = (
            float(np.median(var_at_antinodes) / node_med)
            if node_med > 0 else float("nan")
        )
        metrics["mean_q_spacing"] = float(np.mean(np.diff(node_qs)))
    else:
        metrics["node_antinode_ratio"] = float("nan")
        metrics["mean_q_spacing"] = float("nan")

    # --- detrend: remove smooth large-scale variation, keep the ripple ---
    # Smooth with a window ~2x the grid spacing so the comb survives but the
    # slow trend (variance genuinely growing toward the q-range edges) is
    # removed.
    dq = q_grid[1] - q_grid[0]
    spacing = metrics["mean_q_spacing"]
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = (q_grid[-1] - q_grid[0]) / 30.0
    win = max(3, int(round(2.0 * spacing / dq)) | 1)  # odd
    kernel = np.ones(win) / win
    trend = np.convolve(var, kernel, mode="same")
    # convolution edge effects: blend back to raw near the ends
    ripple = var - trend
    metrics["rel_oscillation"] = float(np.std(ripple) / np.mean(var))

    # --- spectral concentration near the grid frequency ---
    ripple_win = ripple * np.hanning(len(ripple))
    spec = np.abs(np.fft.rfft(ripple_win)) ** 2
    freqs = np.fft.rfftfreq(len(ripple), d=dq)  # cycles per unit q
    grid_freq = 1.0 / spacing if spacing > 0 else np.nan
    total = spec.sum()
    if np.isfinite(grid_freq) and total > 0:
        band = np.zeros_like(freqs, dtype=bool)
        for harm in (1, 2):
            f0 = harm * grid_freq
            band |= np.abs(freqs - f0) <= (0.35 * grid_freq)
        metrics["spectral_comb_fraction"] = float(spec[band].sum() / total)
        metrics["grid_frequency"] = float(grid_freq)
    else:
        metrics["spectral_comb_fraction"] = float("nan")
        metrics["grid_frequency"] = float("nan")

    metrics["_ripple"] = ripple
    metrics["_trend"] = trend
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="Path to an ExactGPSurrogate .pt checkpoint")
    ap.add_argument("--label", default=None, help="Label for outputs (default: basename)")
    ap.add_argument("--n-q", type=int, default=2000,
                    help="Number of q scan points (default 2000)")
    ap.add_argument("--q-lo", type=float, default=None,
                    help="Lower q (default: training-node min)")
    ap.add_argument("--q-hi", type=float, default=None,
                    help="Upper q (default: training-node max)")
    ap.add_argument("--t-lo", type=float, default=-0.3, help="Window start (s)")
    ap.add_argument("--t-hi", type=float, default=0.01, help="Window end (s)")
    ap.add_argument("--n-t", type=int, default=64, help="Times in the window")
    ap.add_argument("--polarisation", default="plus", choices=["plus", "cross"])
    ap.add_argument("--outdir", default="results", help="Output directory")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    label = args.label or os.path.splitext(os.path.basename(args.checkpoint))[0]
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.checkpoint} ...")
    model = ExactGPSurrogate.load(args.checkpoint)
    node_qs = np.unique(model._train_x_raw[:, 0].cpu().numpy())
    print(f"  {len(node_qs)} distinct training q's, "
          f"range [{node_qs.min():.4f}, {node_qs.max():.4f}]")

    q_lo = args.q_lo if args.q_lo is not None else float(node_qs.min())
    q_hi = args.q_hi if args.q_hi is not None else float(node_qs.max())
    q_grid = np.linspace(q_lo, q_hi, args.n_q)

    print(f"Scanning variance over {args.n_q} q in [{q_lo:.4f}, {q_hi:.4f}], "
          f"window t=[{args.t_lo}, {args.t_hi}] ({args.n_t} times), "
          f"pol={args.polarisation} ...")
    var = scalar_variance_vs_q(
        model, q_grid, args.t_lo, args.t_hi, args.n_t, args.polarisation
    )

    m = comb_metrics(q_grid, var, node_qs)
    print("\n=== Comb metrics ===")
    print(f"  distinct training q's   : {len(node_qs)}")
    print(f"  mean q spacing          : {m['mean_q_spacing']:.4f}")
    print(f"  node/antinode var ratio : {m['node_antinode_ratio']:.3f}  "
          "(1.0 = flat; >1 = comb dips at nodes)")
    print(f"  relative oscillation    : {m['rel_oscillation']:.4f}  "
          "(std of ripple / mean var)")
    print(f"  spectral comb fraction  : {m['spectral_comb_fraction']:.3f}  "
          "(power near grid freq / total ripple power)")

    npz_path = os.path.join(args.outdir, f"q_variance_comb_{label}.npz")
    np.savez(
        npz_path,
        q_grid=q_grid, var=var, node_qs=node_qs,
        ripple=m["_ripple"], trend=m["_trend"],
        node_antinode_ratio=m["node_antinode_ratio"],
        rel_oscillation=m["rel_oscillation"],
        spectral_comb_fraction=m["spectral_comb_fraction"],
        mean_q_spacing=m["mean_q_spacing"],
        polarisation=args.polarisation,
    )
    print(f"\nSaved {npz_path}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            ax1.plot(q_grid, var, lw=1.0, color="C0")
            ax1.plot(q_grid, m["_trend"], lw=1.0, color="C3", alpha=0.7,
                     label="smooth trend")
            for nq in node_qs:
                ax1.axvline(nq, color="k", lw=0.4, alpha=0.25)
            ax1.set_ylabel(f"mean diag var ({args.polarisation})")
            ax1.set_title(
                f"{label}: {len(node_qs)} distinct q's | "
                f"node/antinode={m['node_antinode_ratio']:.2f} | "
                f"comb frac={m['spectral_comb_fraction']:.2f}"
            )
            ax1.legend(loc="upper left", fontsize=8)

            ax2.plot(q_grid, m["_ripple"], lw=1.0, color="C0")
            for nq in node_qs:
                ax2.axvline(nq, color="k", lw=0.4, alpha=0.25)
            ax2.axhline(0, color="C3", lw=0.6)
            ax2.set_ylabel("ripple (var - trend)")
            ax2.set_xlabel("mass ratio q  (vertical lines = training nodes)")
            fig.tight_layout()
            png_path = os.path.join(args.outdir, f"q_variance_comb_{label}.png")
            fig.savefig(png_path, dpi=120)
            print(f"Saved {png_path}")
        except Exception as e:  # pragma: no cover - plotting is best-effort
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
