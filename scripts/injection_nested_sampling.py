"""
First end-to-end nested-sampling run on an injection, using the GP-marginalised
GW likelihood (heron.gw_likelihood.GWLikelihood) and the dynesty backend wired
up in heron.sampling.

Scope: a 2-D parameter space (mass_ratio, tc). Sky position (ra, dec) and
polarisation angle (psi) are fixed at their injected values -- a full 5-D run
with sky location free is a natural follow-up, not attempted here. As documented
in CLAUDE.md's log-det bias section, the q posterior is expected to show visible
bias at current training densities (dense30/dense45); this run is partly meant
to see that bias directly in a real posterior rather than via the 1-D width-ratio
proxy used in demo_uncertainty_comparison.py.

Usage::

    python scripts/injection_nested_sampling.py \\
        --checkpoint checkpoints/phenomd_nonspinning_dense30.pt \\
        --device cuda \\
        --output results/injection_ns_dense30_q05.png
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from astropy import units as u

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heron.gw_likelihood import GWLikelihood
from heron.evaluation.psd import aligo_design_psd
from heron.noise import noise_covariance
from heron.detector import antenna_patterns, project_waveform
from heron.train import _get_approximant
from heron.sampling import Parameter, UniformPrior, DynestySampler


def make_injection(approximant, times, tc_true, q_true, total_mass, distance,
                    fp, fc, rng, C):
    """Inject the reference LALSuite waveform (not the surrogate's own mean)
    plus coloured Gaussian noise drawn from C. See demo_uncertainty_comparison.py
    for why the reference waveform (not the surrogate mean) is used here.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested sampling on a GW injection")
    parser.add_argument("--checkpoint", default="checkpoints/phenomd_nonspinning_dense30.pt")
    parser.add_argument("--device", default="cpu", help="Device for surrogate predict (cpu/cuda)")
    parser.add_argument("--q-true", type=float, default=0.5)
    parser.add_argument("--q-bounds", type=float, nargs=2, default=[0.4, 0.95],
                         help="Prior bounds on mass_ratio. Default excludes q<0.4: "
                              "scripts/probe_phase_drift.py + mismatch_vs_mass_ratio.py "
                              "found the surrogate mean develops a secular phase-rate "
                              "error there (see CLAUDE.md Known Issues) that training-"
                              "grid density and warping-anchor tuning don't fix. Upper "
                              "bound stays inside the trained grid (max q~0.97-0.98).")
    parser.add_argument("--tc-half-ms", type=float, default=5.0,
                         help="Half-width of the tc prior in milliseconds")
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--total-mass", type=float, default=60.0)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument("--sample-rate", type=float, default=512.0)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--detector", default="H1")
    parser.add_argument("--no-uncertainty", action="store_true",
                         help="Disable GP waveform uncertainty (K=0, matched filter)")
    parser.add_argument("--k-smoothing-grid-spacing", type=float, default=None,
                         help="If set, envelope the GP variance over offsets spanning "
                              "+/- half this spacing in mass_ratio (5 points total) "
                              "before using it in the likelihood -- removes the "
                              "training-grid-periodic dip in K(theta) that otherwise "
                              "biases the log-det term toward training nodes. Set to "
                              "the surrogate's training mass-ratio grid spacing. "
                              "See heron.gw_likelihood.GWLikelihood's k_smoothing_offsets.")
    parser.add_argument("--nlive", type=int, default=250)
    parser.add_argument("--dlogz", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/injection_ns.png")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    tc_true = 1187008882.43
    ra_true, dec_true, psi_true = 3.446, -0.408, 0.0

    n = int(args.duration * args.sample_rate)
    t_start = tc_true - args.duration * 0.75
    times = t_start + np.arange(n) / args.sample_rate

    print(f"Segment: {n} samples @ {args.sample_rate} Hz ({args.duration} s)")
    print(f"Loading surrogate from {args.checkpoint} on {args.device} ...")
    # Checkpoints self-describe their class via the `model_class` field --
    # dispatch like scripts/mismatch_vs_mass_ratio.py so phase-amplitude /
    # delta checkpoints work too.
    import torch

    model_class = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )["model_class"]
    if model_class == "PhaseAmplitudeGPSurrogate":
        from heron.models.gp.phase_amplitude import PhaseAmplitudeGPSurrogate

        surrogate = PhaseAmplitudeGPSurrogate.load(args.checkpoint, device=args.device)
    elif model_class == "DeltaGPSurrogate":
        from heron.models.gp.delta import DeltaGPSurrogate

        surrogate = DeltaGPSurrogate.load(args.checkpoint, device=args.device)
    else:
        from heron.models.gp.exact import ExactGPSurrogate

        surrogate = ExactGPSurrogate.load(args.checkpoint, device=args.device)

    fp, fc = antenna_patterns(ra_true, dec_true, psi_true, tc_true, args.detector)
    print(f"Antenna patterns ({args.detector}): F+ = {fp:.3f}  F× = {fc:.3f}")

    print("Building noise covariance matrix ...")
    C = noise_covariance(times, aligo_design_psd, f_low=20.0, jitter_rel=1e-8)

    print(f"Injecting signal at q={args.q_true}, tc={tc_true} "
          f"(reference approximant: {args.approximant}) ...")
    approximant = _get_approximant(args.approximant)
    data, signal = make_injection(
        approximant, times, tc_true, args.q_true,
        args.total_mass, args.distance, fp, fc, rng, C,
    )

    dt = 1.0 / args.sample_rate
    freqs = np.fft.rfftfreq(n, d=dt)
    hp_mask = freqs >= 20.0
    sig_f = np.fft.rfft(signal); sig_f[~hp_mask] = 0.0
    sig_hp = np.fft.irfft(sig_f, n=n)
    snr_hp = float(np.sqrt(sig_hp @ np.linalg.solve(C, sig_hp)))
    print(f"Optimal SNR (HP-filtered, >=20 Hz) = {snr_hp:.1f}")

    use_unc = not args.no_uncertainty
    k_smoothing_offsets = None
    if args.k_smoothing_grid_spacing is not None:
        half = args.k_smoothing_grid_spacing / 2.0
        k_smoothing_offsets = [o for o in np.linspace(-half, half, 5) if abs(o) > 1e-9]
        print(f"K-smoothing enabled: offsets = {k_smoothing_offsets}")
    gw_ll = GWLikelihood(
        data=data, times=times, psd_fn=aligo_design_psd,
        surrogate=surrogate, detector=args.detector,
        use_waveform_uncertainty=use_unc,
        device=args.device,
        k_smoothing_offsets=k_smoothing_offsets,
    )
    # Evaluate the surrogate template at the injection distance. Without this,
    # the surrogate predicts at its trained default (checkpoint distance_factor,
    # 100 Mpc) while data injected at args.distance is (100/distance)x the
    # amplitude -- a pure template/data amplitude mismatch. That mismatch slides
    # the likelihood toward the smallest-amplitude (low-q) template and produces
    # a spurious far peak (the "universal attractor" previously misattributed to
    # a surrogate/kernel defect at SNR~100). No-op at the default 100 Mpc, so
    # every distance=100 result on record is unaffected.
    extrinsic = {"ra": ra_true, "dec": dec_true, "psi": psi_true,
                 "luminosity_distance": args.distance}
    log_likelihood = lambda p: gw_ll({**p, **extrinsic})

    prior = UniformPrior([
        Parameter("mass_ratio", args.q_bounds[0], args.q_bounds[1]),
        Parameter("tc", tc_true - args.tc_half_ms * 1e-3, tc_true + args.tc_half_ms * 1e-3),
    ])
    print(f"Prior: mass_ratio in {args.q_bounds}, "
          f"tc in [{tc_true - args.tc_half_ms*1e-3:.6f}, {tc_true + args.tc_half_ms*1e-3:.6f}]")

    sampler = DynestySampler(log_likelihood, prior, nlive=args.nlive)
    print(f"Running dynesty (nlive={args.nlive}, dlogz={args.dlogz}, "
          f"{'with' if use_unc else 'without'} waveform uncertainty) ...")
    t0 = time.time()
    result = sampler.run(dlogz=args.dlogz, print_progress=True)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. log Z = {result.log_evidence:.2f} "
          f"+/- {result.log_evidence_err:.2f}")

    post = result.posterior_dict()
    q_samples, tc_samples = post["mass_ratio"], post["tc"]
    med = result.posterior_median()
    q_lo, q_hi = np.percentile(q_samples, [16, 84])
    tc_lo, tc_hi = np.percentile(tc_samples, [16, 84])

    print("\n--- Posterior (median, 68% interval) ---")
    print(f"  mass_ratio = {med['mass_ratio']:.4f}  "
          f"[{q_lo:.4f}, {q_hi:.4f}]  (truth = {args.q_true})")
    print(f"  tc         = {med['tc']:.6f}  "
          f"[{tc_lo:.6f}, {tc_hi:.6f}]  (truth = {tc_true})")
    print(f"  n effective samples = {len(q_samples)}")

    np.savez(args.output.replace(".png", ".npz"),
              q=q_samples, tc=tc_samples,
              log_evidence=result.log_evidence,
              log_evidence_err=result.log_evidence_err,
              q_true=args.q_true, tc_true=tc_true, snr=snr_hp)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(q_samples, bins=40, color="#1f77b4", alpha=0.8)
    axes[0].axvline(args.q_true, color="k", ls=":", label="Injected")
    axes[0].set_xlabel("mass_ratio")
    axes[0].legend(fontsize=8)
    axes[1].hist((tc_samples - tc_true) * 1e3, bins=40, color="#1f77b4", alpha=0.8)
    axes[1].axvline(0.0, color="k", ls=":", label="Injected")
    axes[1].set_xlabel("tc - tc_true [ms]")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Nested sampling posterior  (SNR={snr_hp:.0f}, "
                 f"logZ={result.log_evidence:.1f}, "
                 f"{'with' if use_unc else 'without'} K)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved figure -> {args.output}")


if __name__ == "__main__":
    main()
