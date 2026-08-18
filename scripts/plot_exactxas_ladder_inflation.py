"""Compare the exact+XAS SNR ladder's bare vs synthetically-inflated with-K arms.

Three columns (SNR~20/75/250) x three rows (no-K, with-K bare, with-K with
K's variance inflated x100 via NetworkLikelihood's covariance_inflation) of
mass_ratio posterior histograms, from the two condor campaigns:
condor/net_injection_ladder.sub's exactxasladder_snr*_{noK,withK} and
exactxasladder_snr*_withK_infl100 tags.

Point: the bare with-K arm shows K too small to bite even at SNR~250
(K/C~0.08 at the peak-amplitude sample -- see the campaign memory). x100
inflation was a deliberately large, likely-unrealistic synthetic multiplier
chosen to force the effect into view quickly, not a claim about real
miscalibration magnitude -- see the caveat in the memory writeup. The
resulting failure mode (bimodality at SNR~75, a severe near-training-node
snap at SNR~250) reproduces the log-det grid-snap pathology documented
throughout CLAUDE.md's Known Issues, on demand.

Usage:
    python scripts/plot_exactxas_ladder_inflation.py
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNGS = ("snr20", "snr75", "snr250")
ARMS = (
    ("noK", "no-K (matched filter)", "C0"),
    ("withK", "with-K (bare)", "C1"),
    ("withK_infl100", "with-K (K var x100)", "C3"),
)
TRUTH = 0.8


def main():
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, rung in enumerate(RUNGS):
        for j, (arm, label, colour) in enumerate(ARMS):
            npz = np.load(f"results/net_exactxasladder_{rung}_{arm}.npz", allow_pickle=True)
            summary = json.load(open(f"results/net_exactxasladder_{rung}_{arm}_summary.json"))
            q = npz["post_mass_ratio"]
            p = summary["params"]["mass_ratio"]

            ax = axes[j, i]
            ax.hist(q, bins=60, color=colour, alpha=0.8)
            ax.axvline(TRUTH, color="k", ls="--", lw=1.5, label="truth")
            ax.set_title(
                f"{rung} {label}\nSNR={summary['network_snr']:.1f}  "
                f"median={p['median']:.4f} ({p['sigma']:+.2f}sigma)",
                fontsize=9,
            )
            ax.set_xlabel("mass_ratio")
            ax.legend(fontsize=8)

    fig.suptitle(
        "exact_xas_lsminq006 SNR ladder: bare vs synthetically K-inflated (x100) with-K",
        fontsize=12,
    )
    fig.tight_layout()
    out = "results/exactxasladder_covariance_inflation_check.png"
    fig.savefig(out, dpi=120)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
