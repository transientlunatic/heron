"""Network parameter-estimation injection test, built on ``heron.inference``.

Extends the single-detector 2-D ``(q, tc)`` runs in
``injection_nested_sampling.py`` to the full PE layer:

- a **coherent multi-detector network** (``--detectors H1,L1,V1``),
- **analytic extrinsic sampling** — any subset of
  ``mass_ratio, tc, luminosity_distance, ra, dec, psi, inclination,
  coalescence_phase`` (``--sample ...``); un-sampled parameters are fixed at
  their injected truth,
- **nessai** (default) or **dynesty** as the sampler,

exercising :class:`heron.inference.NetworkLikelihood`,
:class:`heron.inference.Injection`, the analytic
:func:`heron.inference.project_polarisations`, the
:class:`heron.inference.PriorDict`, and :class:`heron.inference.NessaiSampler`.

Rough sampler settings (low ``--nlive``) are appropriate for a machinery
validation: the goal is "does the network PE recover the injected truth across
these parameters", not tight production posteriors.

Example::

    python scripts/network_injection_ns.py \\
        --checkpoint checkpoints/phenomd_nonspinning_dense30_demod.pt \\
        --detectors H1,L1 --sample mass_ratio,tc,luminosity_distance \\
        --q-true 0.8 --distance 1500 --sampler nessai --nlive 250 \\
        --output results/net_h1l1_3d_q080
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heron.inference import (
    Detector, NetworkLikelihood, Injection,
    Uniform, Sine, Cosine, PowerLaw, PriorDict,
    DynestySampler, NessaiSampler,
    credible_level_1d,
)


# Injected truth for every parameter the layer understands.  --distance and
# --q-true override the two most-varied ones; the rest are a fixed, generic
# (loud-ish, off-axis) source so the extrinsic recovery is a real test.
def build_truth(args) -> dict:
    return {
        "mass_ratio": args.q_true,
        "total_mass": args.total_mass,
        "tc": args.tc_true,
        "ra": 1.95,
        "dec": -1.27,
        "psi": 0.82,
        "luminosity_distance": args.distance,
        "inclination": 0.5,
        "coalescence_phase": 1.1,
    }


def build_prior(sample: list[str], truth: dict, args) -> PriorDict:
    """A PriorDict over the requested sampled parameters (standard GW priors)."""
    tc = truth["tc"]
    catalogue = {
        "mass_ratio": Uniform(args.q_bounds[0], args.q_bounds[1],
                              latex_label="q"),
        "tc": Uniform(tc - args.tc_half_ms * 1e-3, tc + args.tc_half_ms * 1e-3,
                      latex_label="t_c"),
        "luminosity_distance": PowerLaw(2.0, args.dist_bounds[0], args.dist_bounds[1],
                                        latex_label="d_L"),
        "ra": Uniform(0.0, 2 * np.pi, periodic=True, latex_label="\\alpha"),
        "dec": Cosine(latex_label="\\delta"),
        "psi": Uniform(0.0, np.pi, periodic=True, latex_label="\\psi"),
        "inclination": Sine(latex_label="\\iota"),
        "coalescence_phase": Uniform(0.0, 2 * np.pi, periodic=True,
                                     latex_label="\\phi_c"),
    }
    unknown = [p for p in sample if p not in catalogue]
    if unknown:
        raise ValueError(f"unknown sampled parameter(s): {unknown}")
    return PriorDict({p: catalogue[p] for p in sample})


def load_surrogate(checkpoint: str, device: str):
    import torch
    model_class = torch.load(checkpoint, map_location="cpu",
                             weights_only=False)["model_class"]
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Network injection PE test")
    ap.add_argument("--checkpoint",
                    default="checkpoints/phenomd_nonspinning_dense30_demod.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--detectors", default="H1,L1",
                    help="Comma-separated detector prefixes, e.g. H1,L1,V1")
    ap.add_argument("--sample", default="mass_ratio,tc",
                    help="Comma-separated parameters to sample; the rest are "
                         "fixed at their injected truth.")
    ap.add_argument("--q-true", type=float, default=0.8)
    ap.add_argument("--q-bounds", type=float, nargs=2, default=[0.4, 0.95])
    ap.add_argument("--tc-true", type=float, default=1187008882.43)
    ap.add_argument("--tc-half-ms", type=float, default=5.0)
    ap.add_argument("--dist-bounds", type=float, nargs=2, default=[100.0, 5000.0])
    ap.add_argument("--total-mass", type=float, default=60.0)
    ap.add_argument("--distance", type=float, default=1500.0,
                    help="Injected luminosity distance (Mpc); sets the SNR.")
    ap.add_argument("--approximant", default="IMRPhenomD")
    ap.add_argument("--self-inject", action="store_true",
                    help="Inject the surrogate's own prediction (mismatch=0).")
    ap.add_argument("--sample-rate", type=float, default=512.0)
    ap.add_argument("--duration", type=float, default=0.5)
    ap.add_argument("--no-uncertainty", action="store_true",
                    help="K=0 matched filter instead of GP-marginalised.")
    ap.add_argument("--k-smoothing-grid-spacing", type=float, default=None)
    ap.add_argument("--sampler", choices=["nessai", "dynesty"], default="nessai")
    ap.add_argument("--nlive", type=int, default=250)
    ap.add_argument("--n-pool", type=int, default=None,
                    help="nessai likelihood-pool workers (serial if unset).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/net_injection",
                    help="Output stem (no extension).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sample = [s for s in args.sample.split(",") if s]
    det_names = [d for d in args.detectors.split(",") if d]
    truth = build_truth(args)

    # --- time grid + network -----------------------------------------------
    n = int(args.duration * args.sample_rate)
    t0 = args.tc_true - 0.75 * args.duration
    times = t0 + np.arange(n) / args.sample_rate
    network = [Detector.from_name(d) for d in det_names]
    print(f"Network: {det_names}   {n} samples @ {args.sample_rate} Hz "
          f"({args.duration}s)")

    surrogate = load_surrogate(args.checkpoint, args.device)

    # --- injection ----------------------------------------------------------
    if args.self_inject:
        source, src_desc = surrogate, "surrogate self-injection (mismatch=0)"
    else:
        from heron.train import _get_approximant
        source, src_desc = _get_approximant(args.approximant), args.approximant
    inj = Injection(times=times, detectors=network, parameters=truth, f_low=20.0)
    injres = inj.generate(source, rng=rng)
    print(f"Injected {src_desc} at q={args.q_true}, d={args.distance} Mpc")
    for p, s in injres.snrs.items():
        print(f"  {p} SNR = {s:.2f}")
    print(f"  network SNR = {injres.network_snr:.2f}")

    # --- likelihood + prior -------------------------------------------------
    use_unc = not args.no_uncertainty
    k_offsets = None
    if args.k_smoothing_grid_spacing is not None:
        half = args.k_smoothing_grid_spacing / 2.0
        k_offsets = [o for o in np.linspace(-half, half, 5) if abs(o) > 1e-9]
    like = NetworkLikelihood(
        data=injres.data, times=times, detectors=network, surrogate=surrogate,
        f_low=20.0, use_waveform_uncertainty=use_unc, device=args.device,
        k_smoothing_offsets=k_offsets,
    )
    fixed = {k: v for k, v in truth.items() if k not in sample}
    log_likelihood = lambda p: like({**fixed, **p})

    prior = build_prior(sample, truth, args)
    print(f"Sampling {sample}  ({'with' if use_unc else 'without'} K)  "
          f"via {args.sampler}, nlive={args.nlive}")

    # --- run ----------------------------------------------------------------
    t_start = time.time()
    if args.sampler == "nessai":
        sampler = NessaiSampler(log_likelihood, prior)
        run_kwargs = dict(output=f"{args.output}_nessai", nlive=args.nlive,
                          seed=args.seed)
        if args.n_pool:
            run_kwargs["n_pool"] = args.n_pool
        result = sampler.run(**run_kwargs)
    else:
        sampler = DynestySampler(log_likelihood, prior, nlive=args.nlive)
        result = sampler.run(dlogz=0.5, print_progress=True)
    elapsed = time.time() - t_start
    print(f"Done in {elapsed/60:.1f} min. logZ = {result.log_evidence:.2f} "
          f"+/- {result.log_evidence_err:.2f}")

    # --- summarise ----------------------------------------------------------
    post = {n_: result.samples[:, i] for i, n_ in enumerate(sample)}
    summary = {"detectors": det_names, "sample": sample,
               "network_snr": injres.network_snr, "with_K": use_unc,
               "elapsed_min": elapsed / 60.0,
               "log_evidence": result.log_evidence,
               "n_samples": int(result.samples.shape[0]), "params": {}}
    print("\n--- Posterior (median [16,84%], truth, offset) ---")
    for p in sample:
        s = post[p]
        med = float(np.median(s))
        lo, hi = np.percentile(s, [16, 84])
        std = float(np.std(s))
        tv = truth[p]
        sig = (med - tv) / std if std > 0 else np.nan
        # credible level of the truth from the 1-D marginal (KDE-free histogram).
        cl = float(np.mean(s < tv))
        summary["params"][p] = dict(median=med, lo=float(lo), hi=float(hi),
                                    std=std, truth=float(tv), sigma=float(sig),
                                    credible_level=cl)
        print(f"  {p:20s} {med:+.5g} [{lo:+.5g}, {hi:+.5g}]  "
              f"truth={tv:+.5g}  ({sig:+.2f}sigma, CL={cl:.2f})")

    np.savez(f"{args.output}.npz",
             samples=result.samples, parameter_names=np.array(sample),
             log_evidence=result.log_evidence,
             log_evidence_err=result.log_evidence_err,
             truths=np.array([truth[p] for p in sample]),
             network_snr=injres.network_snr,
             **{f"post_{p}": post[p] for p in sample})
    with open(f"{args.output}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # --- corner-ish plot ----------------------------------------------------
    m = len(sample)
    fig, axes = plt.subplots(1, m, figsize=(3.2 * m, 3.0), squeeze=False)
    for i, p in enumerate(sample):
        ax = axes[0, i]
        ax.hist(post[p], bins=40, color="#1f77b4", alpha=0.8)
        ax.axvline(truth[p], color="k", ls=":", lw=1.5)
        ax.set_xlabel(p, fontsize=9)
        ax.set_yticks([])
    fig.suptitle(f"{'+'.join(det_names)}  netSNR={injres.network_snr:.0f}  "
                 f"{'with' if use_unc else 'no'} K  logZ={result.log_evidence:.0f}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{args.output}.png", dpi=140)
    print(f"\nSaved -> {args.output}.{{npz,png,_summary.json}}")


if __name__ == "__main__":
    main()
