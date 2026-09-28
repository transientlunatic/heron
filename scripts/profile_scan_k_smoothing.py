"""
Direct log-likelihood profile scan comparing raw vs K-smoothed GWLikelihood,
at fixed tc=tc_true, around a true mass ratio.

Motivation: the with-K nested-sampling posteriors on
checkpoints/phenomd_nonspinning_dense30_phase_amplitude_xas_lsminq006.pt show a
~6-16sigma bias relative to the no-K (matched-filter) posterior at every one of
q_true=0.50/0.60/0.80 (see CLAUDE.md's log-det bias entries). Direct scanning of
the GP's own reported variance across q confirmed a ~2x periodic dip exactly at
training-grid nodes (period = grid spacing) even after the ls_min_q=0.06 fix --
a mathematically expected feature of any GP posterior (variance is minimised
at/near training inputs), not a residual tuning bug. GWLikelihood's
k_smoothing_offsets envelopes (max) the variance over nearby offsets in
mass_ratio to remove this artefact before it reaches the log-det term.

This script scans logL(mass_ratio) at fixed tc=tc_true, with and without
k_smoothing, to check -- cheaply, without a full nested-sampling run -- whether
smoothing moves the with-K peak back toward the true (matched-filter) peak.
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


def make_injection(approximant, times, tc_true, q_true, total_mass, distance, fp, fc, rng, C):
    t_rel = times - tc_true
    params = {
        "mass_ratio": q_true,
        "total_mass": total_mass * u.solMass,
        "luminosity_distance": distance * u.Mpc,
        "f_min": 20.0 * u.Hertz,
        "delta_t": (1.0 / 4096) * u.second,
    }
    wf = approximant.time_domain(params, times=t_rel)
    signal = fp * wf["plus"].data + fc * wf["cross"].data
    L_C = np.linalg.cholesky(C)
    noise = L_C @ rng.standard_normal(len(times))
    return signal + noise, signal


def load_surrogate(checkpoint, device):
    model_class = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_class"]
    if model_class == "PhaseAmplitudeGPSurrogate":
        from heron.models.gp.phase_amplitude import PhaseAmplitudeGPSurrogate
        return PhaseAmplitudeGPSurrogate.load(checkpoint, device=device)
    if model_class == "DeltaGPSurrogate":
        from heron.models.gp.delta import DeltaGPSurrogate
        return DeltaGPSurrogate.load(checkpoint, device=device)
    from heron.models.gp.exact import ExactGPSurrogate
    return ExactGPSurrogate.load(checkpoint, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--q-true", type=float, required=True)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument("--total-mass", type=float, default=60.0)
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--sample-rate", type=float, default=512.0)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--detector", default="H1")
    parser.add_argument("--half-width", type=float, default=0.045,
                         help="Half-width of the q scan window around q_true.")
    parser.add_argument("--n-grid", type=int, default=41)
    parser.add_argument("--grid-spacing", type=float, default=0.03,
                         help="Training mass-ratio grid spacing, used to derive "
                              "the smoothing offsets.")
    parser.add_argument("--n-smoothing-offsets", type=int, default=4,
                         help="Number of extra offsets spanning +/- grid_spacing/2 "
                              "(evenly spaced, excluding the center point).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tc_true = 1187008882.43
    ra_true, dec_true, psi_true = 3.446, -0.408, 0.0

    n = int(args.duration * args.sample_rate)
    t_start = tc_true - args.duration * 0.75
    times = t_start + np.arange(n) / args.sample_rate

    print(f"Loading surrogate from {args.checkpoint} on {args.device} ...")
    surrogate = load_surrogate(args.checkpoint, args.device)

    fp, fc = antenna_patterns(ra_true, dec_true, psi_true, tc_true, args.detector)
    C = noise_covariance(times, aligo_design_psd, f_low=20.0, jitter_rel=1e-8)

    approximant = _get_approximant(args.approximant)
    data, signal = make_injection(approximant, times, tc_true, args.q_true,
                                   args.total_mass, args.distance, fp, fc, rng, C)

    dt = 1.0 / args.sample_rate
    freqs = np.fft.rfftfreq(n, d=dt)
    hp_mask = freqs >= 20.0
    sig_f = np.fft.rfft(signal); sig_f[~hp_mask] = 0.0
    sig_hp = np.fft.irfft(sig_f, n=n)
    snr_hp = float(np.sqrt(sig_hp @ np.linalg.solve(C, sig_hp)))
    print(f"Injected q={args.q_true}, distance={args.distance} Mpc, SNR={snr_hp:.1f}")

    # Offsets spanning a full training-grid period, e.g. n=4 -> [-0.015, -0.005, 0.005, 0.015]
    # for grid_spacing=0.03 (deliberately excludes 0.0, already the center point).
    half = args.grid_spacing / 2.0
    offsets = np.linspace(-half, half, args.n_smoothing_offsets + 1)
    offsets = [float(o) for o in offsets if abs(o) > 1e-9]
    print(f"Smoothing offsets: {offsets}")

    # Evaluate the template at the injection distance -- see profile_scan_q_wide
    # / injection_nested_sampling: omitting it evaluates the template at the
    # default 100 Mpc against data injected at args.distance, an amplitude
    # mismatch that fakes a low-q attractor. No-op at the default 100 Mpc.
    extrinsic = {"ra": ra_true, "dec": dec_true, "psi": psi_true, "tc": tc_true,
                 "luminosity_distance": args.distance}
    q_grid = np.linspace(args.q_true - args.half_width, args.q_true + args.half_width, args.n_grid)

    variants = {
        "no_K": dict(use_waveform_uncertainty=False, k_smoothing_offsets=None),
        "with_K_raw": dict(use_waveform_uncertainty=True, k_smoothing_offsets=None),
        "with_K_smoothed": dict(use_waveform_uncertainty=True, k_smoothing_offsets=offsets),
    }

    results = {}
    for key, kwargs in variants.items():
        gw_ll = GWLikelihood(data=data, times=times, psd_fn=aligo_design_psd,
                              surrogate=surrogate, detector=args.detector,
                              device=args.device, **kwargs)
        ll = np.empty(args.n_grid)
        for i, qv in enumerate(q_grid):
            ll[i] = gw_ll({**extrinsic, "mass_ratio": float(qv)})
            if (i + 1) % 10 == 0:
                print(f"  [{key}] {i+1}/{args.n_grid}")
        results[key] = ll
        best_idx = np.argmax(ll)
        print(f"[{key}] peak at q={q_grid[best_idx]:.5f}  logL={ll[best_idx]:.3f}  "
              f"(truth q={args.q_true})")

    np.savez(args.output.replace(".png", ".npz"), q_grid=q_grid, q_true=args.q_true,
             snr=snr_hp, **{f"ll_{k}": v for k, v in results.items()})

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, color in [("no_K", "#2ca02c"), ("with_K_raw", "#d62728"), ("with_K_smoothed", "#1f77b4")]:
        ll = results[key]
        ax.plot(q_grid, ll - ll.max(), label=key, color=color, marker="o", ms=3)
    ax.axvline(args.q_true, color="k", ls=":", label="Injected")
    ax.set_xlabel("mass_ratio")
    ax.set_ylabel("log L - max(log L)")
    ax.legend()
    ax.set_title(f"q_true={args.q_true}  SNR={snr_hp:.0f}  {args.checkpoint.split('/')[-1]}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
