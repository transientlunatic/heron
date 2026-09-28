"""
Wide fixed-tc log-likelihood profile scan over a full mass-ratio prior, to
detect FAR spurious peaks ("universal attractor" failure mode).

Motivation: at SNR~100 (distance=300 Mpc) the phase-amplitude+XAS checkpoint
collapsed onto a spurious far peak at q~0.41-0.42 regardless of the injected q
(see CLAUDE.md / memory snr100_universal_attractor). A narrow scan around truth
cannot see this; this scans the whole prior. Fixed tc=tc_true, injecting an
IMRPhenomD signal at q_true, for no_K (matched filter) and with_K_raw.

Companion to scripts/profile_scan_k_smoothing.py (which does the narrow,
near-truth, +k_smoothing scan). Shares its injection/loading conventions.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from astropy import units as u

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heron.gw_likelihood import GWLikelihood
from heron.evaluation.psd import aligo_design_psd
from heron.noise import noise_covariance
from heron.detector import antenna_patterns
from heron.train import _get_approximant
from scripts.profile_scan_k_smoothing import make_injection, load_surrogate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--q-true", type=float, required=True)
    p.add_argument("--distance", type=float, default=300.0)
    p.add_argument("--total-mass", type=float, default=60.0)
    p.add_argument("--approximant", default="IMRPhenomD")
    p.add_argument("--sample-rate", type=float, default=512.0)
    p.add_argument("--duration", type=float, default=0.5)
    p.add_argument("--detector", default="H1")
    p.add_argument("--q-lo", type=float, default=0.25)
    p.add_argument("--q-hi", type=float, default=0.97)
    p.add_argument("--n-grid", type=int, default=37)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    tc_true = 1187008882.43
    ra, dec, psi = 3.446, -0.408, 0.0
    n = int(args.duration * args.sample_rate)
    times = (tc_true - args.duration * 0.75) + np.arange(n) / args.sample_rate

    print(f"Loading {args.checkpoint} ...", flush=True)
    surrogate = load_surrogate(args.checkpoint, args.device)
    fp, fc = antenna_patterns(ra, dec, psi, tc_true, args.detector)
    C = noise_covariance(times, aligo_design_psd, f_low=20.0, jitter_rel=1e-8)
    approximant = _get_approximant(args.approximant)
    data, signal = make_injection(approximant, times, tc_true, args.q_true,
                                  args.total_mass, args.distance, fp, fc, rng, C)

    freqs = np.fft.rfftfreq(n, d=1.0 / args.sample_rate)
    sig_f = np.fft.rfft(signal); sig_f[freqs < 20.0] = 0.0
    sig_hp = np.fft.irfft(sig_f, n=n)
    snr = float(np.sqrt(sig_hp @ np.linalg.solve(C, sig_hp)))
    print(f"q_true={args.q_true}  distance={args.distance} Mpc  SNR={snr:.1f}", flush=True)

    # Evaluate the template at the injection distance. Without this, the
    # surrogate predicts at its trained default (100 Mpc) while data injected
    # at args.distance is (100/distance)x the amplitude -> a pure amplitude
    # mismatch that slides the likelihood toward the smallest-amplitude (low-q)
    # template (the spurious "universal attractor").
    extr = {"ra": ra, "dec": dec, "psi": psi, "tc": tc_true,
            "luminosity_distance": args.distance}
    q_grid = np.linspace(args.q_lo, args.q_hi, args.n_grid)
    results = {}
    for key, uwu in [("no_K", False), ("with_K_raw", True)]:
        gw = GWLikelihood(data=data, times=times, psd_fn=aligo_design_psd,
                          surrogate=surrogate, detector=args.detector,
                          device=args.device, use_waveform_uncertainty=uwu,
                          k_smoothing_offsets=None)
        ll = np.array([gw({**extr, "mass_ratio": float(q)}) for q in q_grid])
        results[key] = ll
        i = int(np.argmax(ll))
        # is the global peak near truth, or far?
        near = abs(q_grid[i] - args.q_true) < 0.05
        print(f"[{key}] GLOBAL peak q={q_grid[i]:.4f} logL={ll[i]:.2f} "
              f"({'NEAR truth' if near else 'FAR from truth -- ATTRACTOR'})", flush=True)

    np.savez(args.output.replace(".png", ".npz"), q_grid=q_grid, q_true=args.q_true,
             snr=snr, **{f"ll_{k}": v for k, v in results.items()})
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, c in [("no_K", "#2ca02c"), ("with_K_raw", "#d62728")]:
        ll = results[k]
        ax.plot(q_grid, ll - ll.max(), label=k, color=c, marker="o", ms=3)
    ax.axvline(args.q_true, color="k", ls=":", label="Injected")
    ax.set_xlabel("mass_ratio"); ax.set_ylabel("log L - max")
    ax.set_ylim(-2000, 50); ax.legend()
    ax.set_title(f"WIDE q_true={args.q_true} SNR={snr:.0f} {args.checkpoint.split('/')[-1]}")
    fig.tight_layout(); fig.savefig(args.output, dpi=150)
    print(f"Saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
