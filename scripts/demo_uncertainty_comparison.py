"""
Compare GW parameter estimation with and without GP waveform-model uncertainty.

Injects a noiseless signal at known (mass_ratio, tc) into coloured Gaussian
noise, then scans the 1-D log-likelihood profiles over (mass_ratio, tc):

  1. With model uncertainty  (K included) — GP-marginalised likelihood
  2. Without model uncertainty (K = 0)    — standard matched-filter likelihood

Sky position and polarisation angle are fixed at the injected values throughout.
The profiles are converted to normalised posteriors for comparison.

Usage::

    python scripts/demo_uncertainty_comparison.py \\
        --checkpoint checkpoints/phenomd_nonspinning.pt \\
        --output results/uncertainty_comparison.png
"""
from __future__ import annotations

import argparse
import os
import time
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heron.models.gp.exact import ExactGPSurrogate
from heron.gw_likelihood import GWLikelihood
from heron.evaluation.psd import aligo_design_psd
from heron.noise import noise_covariance
from heron.detector import antenna_patterns, project_waveform


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def make_injection(
    surrogate: ExactGPSurrogate,
    times: np.ndarray,
    tc_true: float,
    q_true: float,
    fp: float,
    fc: float,
    rng: np.random.Generator,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (data, noiseless_signal)."""
    t_rel = times - tc_true
    wf = surrogate.predict({"mass_ratio": q_true, "times": t_rel})
    signal, _ = project_waveform(wf, fp, fc)
    L_C = np.linalg.cholesky(C)
    noise = L_C @ rng.standard_normal(len(times))
    return signal + noise, signal


# ---------------------------------------------------------------------------
# 1-D likelihood profiles
# ---------------------------------------------------------------------------

def scan_1d(
    ll,
    param_name: str,
    values: np.ndarray,
    fixed: dict,
    label: str = "",
) -> np.ndarray:
    """Evaluate the log-likelihood along a 1-D slice.

    Parameters
    ----------
    ll : callable
        Accepts a dict and returns a float log-likelihood.
    param_name : str
        The parameter being scanned.
    values : ndarray, shape (N,)
        Grid of values to evaluate at.
    fixed : dict
        All other parameter values (held constant).
    label : str
        Progress label.

    Returns
    -------
    log_like : ndarray, shape (N,)
    """
    n = len(values)
    log_like = np.empty(n)
    t0 = time.perf_counter()
    for i, v in enumerate(values):
        params = {**fixed, param_name: float(v)}
        log_like[i] = ll(params)
        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            print(f"    {label} {i+1}/{n}  ({elapsed:.1f}s)", end="\r", flush=True)
    print()
    return log_like


def to_posterior(log_like: np.ndarray) -> np.ndarray:
    """Convert a log-likelihood array to a normalised (dx=1) posterior density."""
    log_like = log_like - log_like.max()  # stabilise exp
    post = np.exp(log_like)
    return post / post.sum()  # normalise (unit step assumed)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_profiles(
    profiles: dict,
    tc_true: float,
    q_true: float,
    q_grids: dict,
    tc_grids: dict,
    output: str,
    sigma_q: dict,
    sigma_tc: dict,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    styles = {
        "with_K":    {"color": "#1f77b4", "ls": "-",
                      "label": r"With $K$  (GP-marginalised)"},
        "without_K": {"color": "#d62728", "ls": "--",
                      "label": r"Without $K$  (matched filter)"},
    }

    # --- left panel: mass-ratio profile ---
    ax = axes[0]
    for key, sty in styles.items():
        q_vals = q_grids[key] - q_true
        post = profiles[key]["q"]
        sig = sigma_q[key]
        ax.plot(q_vals, post / post.max(), lw=1.8,
                color=sty["color"], ls=sty["ls"],
                label=f"{sty['label']}  ($\\sigma_q = {sig:.4f}$)")
    ax.axvline(0.0, color="k", lw=1.0, ls=":", label="Injected")
    ax.set_xlabel(r"$q - q_\mathrm{inj}$", fontsize=11)
    ax.set_ylabel("Normalised posterior (peak = 1)", fontsize=9)
    ax.set_title("Mass ratio", fontsize=10)
    ax.legend(fontsize=7.5)

    # --- right panel: tc profile ---
    ax = axes[1]
    for key, sty in styles.items():
        tc_vals = (tc_grids[key] - tc_true) * 1e3  # convert to ms
        post = profiles[key]["tc"]
        sig = sigma_tc[key]
        ax.plot(tc_vals, post / post.max(), lw=1.8,
                color=sty["color"], ls=sty["ls"],
                label=f"{sty['label']}  ($\\sigma_{{t_c}} = {sig*1e3:.3f}$ ms)")
    ax.axvline(0.0, color="k", lw=1.0, ls=":", label="Injected")
    ax.set_xlabel(r"$t_c - t_c^\mathrm{inj}$ [ms]", fontsize=11)
    ax.set_ylabel("Normalised posterior (peak = 1)", fontsize=9)
    ax.set_title("Coalescence time", fontsize=10)
    ax.legend(fontsize=7.5)

    fig.suptitle(
        rf"GP uncertainty comparison  —  $q_{{\rm inj}}={q_true}$",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"\nSaved figure → {output}")


def gaussian_sigma(x: np.ndarray, post: np.ndarray) -> float:
    """Estimate the posterior width from a normalised 1-D profile.

    Fits a Gaussian by computing the weighted standard deviation.
    """
    w = post / post.sum()
    mean = np.sum(x * w)
    return float(np.sqrt(np.sum(w * (x - mean) ** 2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GP uncertainty comparison demo")
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--q-true",       type=float, default=0.8)
    parser.add_argument("--sample-rate",  type=float, default=512.0)
    parser.add_argument("--duration",     type=float, default=0.5,
                        help="Segment duration (s); 0.5 s covers 95%% of SNR at n=256")
    parser.add_argument("--detector",     default="H1")
    parser.add_argument("--output",       default="uncertainty_comparison.png")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n-grid",       type=int,   default=200,
                        help="Number of grid points per 1-D profile")
    parser.add_argument("--q-half-width", type=float, default=0.08,
                        help="Half-width of q scan around the truth")
    parser.add_argument("--tc-half-ms",   type=float, default=3.0,
                        help="Half-width of tc scan in milliseconds")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Fixed extrinsic parameters (not sampled)
    tc_true  = 1187008882.43
    ra_true  = 3.446
    dec_true = -0.408
    psi_true = 0.0

    n = int(args.duration * args.sample_rate)
    t_start = tc_true - args.duration * 0.75
    times = t_start + np.arange(n) / args.sample_rate

    print(f"Segment: {n} samples @ {args.sample_rate} Hz  ({args.duration} s)")
    print(f"Loading surrogate from {args.checkpoint} ...")
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if raw_checkpoint.get("model_type") == "sparse":
        from heron.models.gp.sparse import SparseGPSurrogate
        surrogate = SparseGPSurrogate.load(args.checkpoint)
    else:
        surrogate = ExactGPSurrogate.load(args.checkpoint)
    m = surrogate.models["plus"]
    ls_q = float(m.covar_module.base_kernel.kernels[0].lengthscale[0, 0])
    ls_t = float(m.covar_module.base_kernel.kernels[1].lengthscale[0, 0])
    print(f"  Learned hyperparams (plus): ls_q={ls_q:.4f}  ls_t(warp)={ls_t:.4f}  "
          f"ls_min_q={surrogate.ls_min_q}  ls_min_time={surrogate.ls_min_time}")

    fp, fc = antenna_patterns(ra_true, dec_true, psi_true, tc_true, args.detector)
    print(f"Antenna patterns ({args.detector}): F+ = {fp:.3f}  F× = {fc:.3f}")

    print("Building noise covariance matrix ...")
    C = noise_covariance(times, aligo_design_psd, f_low=20.0, jitter_rel=1e-8)

    print(f"Injecting signal at q={args.q_true}, tc={tc_true}")
    data, signal = make_injection(surrogate, times, tc_true, args.q_true, fp, fc, rng, C)

    # HP-filtered SNR (what the likelihood actually uses)
    dt = 1.0 / args.sample_rate
    freqs = np.fft.rfftfreq(n, d=dt)
    hp_mask = freqs >= 20.0
    sig_f = np.fft.rfft(signal); sig_f[~hp_mask] = 0.0
    sig_hp = np.fft.irfft(sig_f, n=n)
    snr_hp = float(np.sqrt(sig_hp @ np.linalg.solve(C, sig_hp)))
    print(f"Optimal SNR (HP-filtered, ≥20 Hz) ≈ {snr_hp:.1f}")

    # GP uncertainty level
    wf_check = surrogate.predict({"mass_ratio": args.q_true, "times": times - tc_true})
    _, K_check = project_waveform(wf_check, fp, fc)
    ratio = K_check.diagonal() / C.diagonal()
    print(f"GP uncertainty / noise (K_ii/C_ii): "
          f"min={ratio.min():.0f}  max={ratio.max():.0f}  mean={ratio.mean():.0f}")

    # Fixed extrinsic params for the likelihood wrapper
    extrinsic = {"ra": ra_true, "dec": dec_true, "psi": psi_true}

    # Scan grids
    q_grid_base = np.linspace(
        max(0.05, args.q_true - args.q_half_width),
        min(1.05, args.q_true + args.q_half_width),
        args.n_grid,
    )
    tc_grid_base = np.linspace(
        tc_true - args.tc_half_ms * 1e-3,
        tc_true + args.tc_half_ms * 1e-3,
        args.n_grid,
    )

    fixed_at_truth = {"mass_ratio": args.q_true, "tc": tc_true, **extrinsic}

    profiles = {}
    q_grids  = {}
    tc_grids = {}

    for use_unc, key in [(True, "with_K"), (False, "without_K")]:
        tag = "with K" if use_unc else "no K"
        print(f"\n[{tag}] building likelihood ...")
        gw_ll = GWLikelihood(
            data=data, times=times, psd_fn=aligo_design_psd,
            surrogate=surrogate, detector=args.detector,
            use_waveform_uncertainty=use_unc,
        )
        ll = lambda p: gw_ll({**p, **extrinsic})

        print(f"[{tag}] scanning q ...")
        lq = scan_1d(ll, "mass_ratio", q_grid_base,
                     fixed={**fixed_at_truth}, label=f"[{key}:q]")
        print(f"[{tag}] scanning tc ...")
        ltc = scan_1d(ll, "tc", tc_grid_base,
                      fixed={**fixed_at_truth}, label=f"[{key}:tc]")

        profiles[key] = {"q": to_posterior(lq), "tc": to_posterior(ltc)}
        q_grids[key]  = q_grid_base.copy()
        tc_grids[key] = tc_grid_base.copy()

    # Posterior widths (Gaussian fit)
    sigma_q  = {k: gaussian_sigma(q_grids[k] - args.q_true, profiles[k]["q"])
                for k in profiles}
    sigma_tc = {k: gaussian_sigma(tc_grids[k] - tc_true,     profiles[k]["tc"])
                for k in profiles}

    print("\n--- Posterior widths ---")
    for key in ["with_K", "without_K"]:
        print(f"  [{key:10s}]  σ_q = {sigma_q[key]:.5f}  "
              f"σ_tc = {sigma_tc[key]*1e3:.4f} ms")

    # Width ratio (the key result)
    ratio_q  = sigma_q["with_K"]  / sigma_q["without_K"]
    ratio_tc = sigma_tc["with_K"] / sigma_tc["without_K"]
    print(f"\n  Width ratio with_K / without_K:  σ_q ×{ratio_q:.1f}  σ_tc ×{ratio_tc:.1f}")

    plot_profiles(profiles, tc_true, args.q_true, q_grids, tc_grids,
                  args.output, sigma_q, sigma_tc)


if __name__ == "__main__":
    main()
