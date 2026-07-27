"""Probability-probability (PP) coverage test for a 2-D (mass_ratio, tc) GW
posterior, via exact grid posteriors instead of nested sampling.

For a well-calibrated pipeline the credible level of the injected truth --
the fraction of posterior mass below it -- is uniform on [0,1] across many
noise realisations. This injects N signals with (q, tc) drawn from the prior,
each with fresh coloured noise, computes the exact 2-D grid posterior for each,
records the credible level of the true q and true tc, and plots the empirical
CDF of those levels against the diagonal (with binomial confidence bands).

Why a grid, not nested sampling: the problem is only 2-D, so a grid posterior
is *exact* and ~100x cheaper than an NS run -- the difference between a
feasible ~100-injection campaign and days of compute. The key speed-up: the
surrogate template mu(q,tc) and covariance K(q,tc) are DATA-INDEPENDENT, so
the (expensive, LAL-backed) grid of predicts is computed ONCE and every
injection's likelihood is then just batched linear algebra against it.

Both the with-K (GP-marginalised) and no-K (matched-filter) posteriors are
produced, so the plot shows directly whether marginalising over the surrogate
covariance K keeps coverage calibrated -- the coverage analogue of the
single-realisation NS with-K/no-K comparison.

Usage (one batch):
    python scripts/pp_plot_demod.py \
        --checkpoint checkpoints/phenomd_nonspinning_dense30_demod.pt \
        --device cuda --n-injections 100 --distance 1500 \
        --output results/pp_demod_snr20.npz --plot

Aggregate several batches (run in parallel with different --seed) and plot:
    python scripts/pp_plot_demod.py --aggregate \
        --inputs 'results/pp_demod_snr20_batch*.npz' \
        --plot-out results/pp_demod_snr20.png
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import torch
import astropy.units as u

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heron.evaluation.psd import aligo_design_psd
from heron.noise import noise_covariance
from heron.detector import antenna_patterns, project_waveform
from heron.train import _get_approximant


# Fixed sky/extrinsic + reference epoch, matching injection_nested_sampling.py.
TC_REF = 1187008882.43
RA, DEC, PSI = 3.446, -0.408, 0.0


def _load_surrogate(checkpoint, device):
    model_class = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_class"]
    if model_class == "DemodGPSurrogate":
        from heron.models.gp.demod import DemodGPSurrogate
        return DemodGPSurrogate.load(checkpoint, device=device)
    if model_class == "PhaseAmplitudeGPSurrogate":
        from heron.models.gp.phase_amplitude import PhaseAmplitudeGPSurrogate
        return PhaseAmplitudeGPSurrogate.load(checkpoint, device=device)
    if model_class == "DeltaGPSurrogate":
        from heron.models.gp.delta import DeltaGPSurrogate
        return DeltaGPSurrogate.load(checkpoint, device=device)
    from heron.models.gp.exact import ExactGPSurrogate
    return ExactGPSurrogate.load(checkpoint, device=device)


def _hp_mask(n, dt, f_low):
    return np.fft.rfftfreq(n, d=dt) >= f_low


def _hp_filter(x, mask, n):
    xf = np.fft.rfft(x)
    xf[~mask] = 0.0
    return np.fft.irfft(xf, n=n)


def _surrogate_projected(surrogate, q, t_rel, fp, fc, distance):
    wf = surrogate.predict({"mass_ratio": float(q), "times": t_rel,
                            "luminosity_distance": distance})
    mu, K = project_waveform(wf, fp, fc)
    return np.asarray(mu, dtype=float), np.asarray(K.diagonal(), dtype=float)


def _credible_level(logL_grid, q_grid, tc_grid, q_true, tc_true):
    """Credible level (posterior CDF at truth) for the q and tc marginals of a
    2-D log-likelihood grid under a uniform prior."""
    from scipy.special import logsumexp
    from scipy.integrate import trapezoid

    def cl_1d(logmarg, grid, truth):
        p = np.exp(logmarg - logmarg.max())
        # Normalised density; CDF via cumulative trapezoid; interp at truth.
        area = trapezoid(p, grid)
        p = p / area
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))])
        return float(np.interp(truth, grid, cdf))

    logmarg_q = logsumexp(logL_grid, axis=1)   # marginalise tc
    logmarg_tc = logsumexp(logL_grid, axis=0)  # marginalise q
    return cl_1d(logmarg_q, q_grid, q_true), cl_1d(logmarg_tc, tc_grid, tc_true)


def run_batch(args):
    device = torch.device(args.device)
    surrogate = _load_surrogate(args.checkpoint, args.device)

    n = int(round(args.duration * args.sample_rate))
    dt = 1.0 / args.sample_rate
    t_start = TC_REF - args.duration * 0.75
    times = t_start + np.arange(n) * dt
    mask = _hp_mask(n, dt, 20.0)

    fp, fc = antenna_patterns(RA, DEC, PSI, TC_REF, args.detector)
    C = noise_covariance(times, aligo_design_psd, f_low=20.0, jitter_rel=1e-8)
    L_C_np = np.linalg.cholesky(C)

    q_lo, q_hi = args.q_bounds
    tc_half = args.tc_half_ms * 1e-3
    q_grid = np.linspace(q_lo, q_hi, args.n_q)
    tc_grid = np.linspace(TC_REF - tc_half, TC_REF + tc_half, args.n_tc)
    n_grid = args.n_q * args.n_tc

    # --- Precompute the DATA-INDEPENDENT template + covariance grid once. ---
    print(f"Precomputing {args.n_q}x{args.n_tc}={n_grid} templates "
          f"(mu, Kdiag) on {args.device} ...", flush=True)
    t0 = time.time()
    mu_grid = np.empty((n_grid, n))
    kdiag_grid = np.empty((n_grid, n))
    g = 0
    for qi in q_grid:
        for tcj in tc_grid:
            mu, kdiag = _surrogate_projected(surrogate, qi, times - tcj, fp, fc, args.distance)
            mu_grid[g] = _hp_filter(mu, mask, n)
            kdiag_grid[g] = kdiag
            g += 1
    print(f"  templates precomputed in {time.time()-t0:.0f}s", flush=True)

    # --- Injections: draw truths, build data, evaluate the fixed grid. ---
    rng = np.random.default_rng(args.seed)
    inj = _get_approximant(args.approximant) if not args.self_inject else None
    data_all = np.empty((args.n_injections, n))
    q_true = np.empty(args.n_injections)
    tc_true = np.empty(args.n_injections)
    snrs = np.empty(args.n_injections)

    for i in range(args.n_injections):
        qt = rng.uniform(q_lo, q_hi)
        tct = rng.uniform(TC_REF - tc_half, TC_REF + tc_half)
        t_rel = times - tct
        if args.self_inject:
            sig, _ = _surrogate_projected(surrogate, qt, t_rel, fp, fc, args.distance)
        else:
            wf = inj.time_domain(
                {"mass_ratio": float(qt), "total_mass": args.total_mass * u.solMass,
                 "luminosity_distance": args.distance * u.Mpc,
                 "f_min": 20.0 * u.Hertz, "delta_t": (1.0 / 4096) * u.second},
                times=t_rel,
            )
            sig = fp * wf["plus"].data + fc * wf["cross"].data
        noise = L_C_np @ rng.standard_normal(n)
        data_all[i] = _hp_filter(sig + noise, mask, n)
        q_true[i] = qt
        tc_true[i] = tct
        sig_hp = _hp_filter(sig, mask, n)
        snrs[i] = float(np.sqrt(sig_hp @ np.linalg.solve(C, sig_hp)))

    # --- Batched likelihood: no-K (fixed C) fully vectorised; with-K looped
    #     over grid points (C+K is injection-independent -> factor once each). ---
    C_t = torch.as_tensor(C, dtype=torch.float64, device=device)
    L_C = torch.as_tensor(L_C_np, dtype=torch.float64, device=device)
    mu_t = torch.as_tensor(mu_grid, dtype=torch.float64, device=device)      # (G, N)
    kdiag_t = torch.as_tensor(kdiag_grid, dtype=torch.float64, device=device)
    data_t = torch.as_tensor(data_all, dtype=torch.float64, device=device)   # (M, N)

    # no-K: logL[m,g] = -0.5 || L_C^{-1}(d_m - mu_g) ||^2  (+ const, drops out)
    zC_mu = torch.linalg.solve_triangular(L_C, mu_t.T, upper=False)   # (N, G)
    zC_d = torch.linalg.solve_triangular(L_C, data_t.T, upper=False)  # (N, M)
    dd = (zC_d**2).sum(0)                     # (M,)
    mm = (zC_mu**2).sum(0)                    # (G,)
    cross = zC_d.T @ zC_mu                     # (M, G)
    logL_nok = -0.5 * (dd[:, None] + mm[None, :] - 2.0 * cross)  # (M, G)

    # with-K: per grid point, M_g = C + diag(K_g); factor once, solve all inj.
    logL_withk = torch.empty_like(logL_nok)
    eye = torch.eye(n, dtype=torch.float64, device=device)
    for gi in range(n_grid):
        Mg = C_t + kdiag_t[gi][:, None] * eye
        Lg = torch.linalg.cholesky(Mg)
        logdet = 2.0 * torch.log(torch.diagonal(Lg)).sum()
        resid = data_t - mu_t[gi]                     # (M, N)
        z = torch.linalg.solve_triangular(Lg, resid.T, upper=False)  # (N, M)
        logL_withk[:, gi] = -0.5 * (z**2).sum(0) - 0.5 * logdet

    logL_nok = logL_nok.cpu().numpy().reshape(args.n_injections, args.n_q, args.n_tc)
    logL_withk = logL_withk.cpu().numpy().reshape(args.n_injections, args.n_q, args.n_tc)

    cl = {k: np.empty(args.n_injections) for k in ("q_nok", "tc_nok", "q_withk", "tc_withk")}
    for i in range(args.n_injections):
        cq, ctc = _credible_level(logL_nok[i], q_grid, tc_grid, q_true[i], tc_true[i])
        cl["q_nok"][i], cl["tc_nok"][i] = cq, ctc
        cq, ctc = _credible_level(logL_withk[i], q_grid, tc_grid, q_true[i], tc_true[i])
        cl["q_withk"][i], cl["tc_withk"][i] = cq, ctc

    print(f"  median SNR = {np.median(snrs):.1f}  (range {snrs.min():.1f}-{snrs.max():.1f})")
    np.savez(args.output, q_true=q_true, tc_true=tc_true, snr=snrs, **cl)
    print(f"Saved credible levels -> {args.output}")
    if args.plot:
        make_pp_plot([args.output], args.output.replace(".npz", ".png"))


def _ks_p(cl):
    from scipy.stats import kstest
    return float(kstest(cl, "uniform").pvalue)


def make_pp_plot(input_globs, plot_out):
    files = []
    for gspec in input_globs:
        files.extend(sorted(glob.glob(gspec)))
    if not files:
        raise SystemExit(f"No npz matched {input_globs}")
    keys = ("q_nok", "tc_nok", "q_withk", "tc_withk")
    agg = {k: np.concatenate([np.load(f)[k] for f in files]) for k in keys}
    n = len(agg["q_nok"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    x = np.linspace(0, 1, 200)
    for ax, arm in zip(axes, ("nok", "withk")):
        # Binomial pointwise confidence bands about the diagonal.
        for z, a in ((1, 0.3), (2, 0.18), (3, 0.1)):
            band = z * np.sqrt(np.clip(x * (1 - x), 0, None) / n)
            ax.fill_between(x, np.clip(x - band, 0, 1), np.clip(x + band, 0, 1),
                            color="gray", alpha=a, lw=0)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
        for param, colour in (("q", "C0"), ("tc", "C1")):
            cl = np.sort(agg[f"{param}_{arm}"])
            yy = np.arange(1, n + 1) / n
            ax.plot(cl, yy, color=colour, lw=1.8,
                    label=f"{param}  (KS p={_ks_p(agg[f'{param}_{arm}']):.2f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("credible level of truth")
        ax.set_ylabel("fraction of injections")
        ax.set_title(f"{'with-K (GP-marginalised)' if arm == 'withk' else 'no-K (matched filter)'}")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_aspect("equal")
    fig.suptitle(f"Demod PP-plot: {n} injections (1/2/3σ binomial bands)")
    fig.tight_layout()
    fig.savefig(plot_out, dpi=130)
    print(f"Saved PP-plot -> {plot_out}")
    for arm in ("nok", "withk"):
        for param in ("q", "tc"):
            print(f"  {arm:6s} {param:3s}: KS p = {_ks_p(agg[f'{param}_{arm}']):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aggregate", action="store_true", help="Aggregate npz + plot only.")
    ap.add_argument("--inputs", nargs="+", default=[], help="npz glob(s) for --aggregate.")
    ap.add_argument("--plot-out", default="results/pp_demod.png")

    ap.add_argument("--checkpoint", default="checkpoints/phenomd_nonspinning_dense30_demod.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-injections", type=int, default=100)
    ap.add_argument("--q-bounds", type=float, nargs=2, default=[0.40, 0.95])
    ap.add_argument("--tc-half-ms", type=float, default=5.0)
    ap.add_argument("--approximant", default="IMRPhenomD")
    ap.add_argument("--self-inject", action="store_true")
    ap.add_argument("--total-mass", type=float, default=60.0)
    ap.add_argument("--distance", type=float, default=1500.0, help="Mpc (1500 -> SNR~20).")
    ap.add_argument("--sample-rate", type=float, default=512.0)
    ap.add_argument("--duration", type=float, default=0.5)
    ap.add_argument("--detector", default="H1")
    ap.add_argument("--n-q", type=int, default=180)
    ap.add_argument("--n-tc", type=int, default=41)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/pp_demod.npz")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        make_pp_plot(args.inputs, args.plot_out)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
