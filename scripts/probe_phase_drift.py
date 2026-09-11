"""
Characterise *where* the surrogate mean diverges from the reference waveform
across mass ratio: a secular (monotonically growing) phase drift over the
inspiral means the surrogate's phase-evolution *rate* is wrong for that q
(a warping/kernel-bandwidth mismatch); a flat/bounded residual means
whatever error is present is localised rather than a rate error.

Motivated by scripts/mismatch_vs_mass_ratio.py finding a non-monotonic
mismatch-vs-q bump (peaking ~q=0.27-0.30) that isn't explained by GP
interpolation error (density-independent, per CLAUDE.md) or the fixed
evaluation window (still present with the native, untruncated window --
see the "native_window_mismatch" check in probe_window_effect.py, folded in
here). This script isolates the *inspiral* (pre-merger) phase residual to
distinguish a rate/warping defect from a local one.

predict() builds an NxN covariance matrix, so evaluating at full native
time resolution (~20k points for the longest, lowest-q waveforms) OOMs --
this restricts to a bounded, sub-merger inspiral window at a fixed number
of points instead.

Usage::

    python scripts/probe_phase_drift.py \\
        --checkpoint checkpoints/phenomd_nonspinning_dense30.pt \\
        --mass-ratios 0.30 0.90
"""
from __future__ import annotations

import argparse

import numpy as np
from astropy import units as u

from heron.train import _get_approximant


def load_surrogate(path: str, device: str = "cpu"):
    """Load a checkpoint into whichever WaveformSurrogate class trained it.

    Checkpoints self-describe their class via the `model_class` field (see
    ExactGPSurrogate.save/PhaseAmplitudeGPSurrogate.save), so this dispatches
    without the caller needing to know which representation is on disk.
    """
    import torch

    model_class = torch.load(path, map_location="cpu", weights_only=False)["model_class"]
    if model_class == "PhaseAmplitudeGPSurrogate":
        from heron.models.gp.phase_amplitude import PhaseAmplitudeGPSurrogate
        return PhaseAmplitudeGPSurrogate.load(path, device=device)
    if model_class == "DeltaGPSurrogate":
        from heron.models.gp.delta import DeltaGPSurrogate
        return DeltaGPSurrogate.load(path, device=device)
    from heron.models.gp.exact import ExactGPSurrogate
    return ExactGPSurrogate.load(path, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspiral phase-drift probe")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mass-ratios", type=float, nargs="+", default=[0.30, 0.90])
    parser.add_argument("--approximant", default="IMRPhenomD")
    parser.add_argument("--total-mass", type=float, default=60.0)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument("--merger-margin", type=float, default=0.1,
                         help="Exclude the last N seconds before t=0 (merger) "
                              "to keep the diagnostic in the inspiral band.")
    parser.add_argument("--n-points", type=int, default=2500,
                         help="Points in the inspiral window (bounds the NxN "
                              "covariance predict() builds).")
    parser.add_argument("--amp-threshold", type=float, default=0.02,
                         help="Only report the phase residual where the "
                              "reference amplitude is above this fraction "
                              "of its peak.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    surrogate = load_surrogate(args.checkpoint, device=args.device)
    approximant = _get_approximant(args.approximant)

    for q in args.mass_ratios:
        native_params = {
            "mass_ratio": q,
            "total_mass": args.total_mass * u.solMass,
            "luminosity_distance": args.distance * u.Mpc,
            "f_min": 20.0 * u.Hertz,
            "delta_t": (1.0 / 4096) * u.second,
        }
        native_wf = approximant.time_domain(native_params)
        t_full = native_wf["plus"].times
        duration = t_full[-1] - t_full[0]

        t_lo, t_hi = t_full[0], -args.merger_margin
        times = np.linspace(t_lo, t_hi, args.n_points)

        ref_wf = approximant.time_domain(
            {"mass_ratio": q, "total_mass": args.total_mass * u.solMass,
             "luminosity_distance": args.distance * u.Mpc}, times=times)
        ref_plus, ref_cross = ref_wf["plus"].data, ref_wf["cross"].data

        surr_wf = surrogate.predict({"mass_ratio": float(q), "times": times})
        surr_plus, surr_cross = surr_wf["plus"].data, surr_wf["cross"].data

        h_ref = ref_plus - 1j * ref_cross
        h_surr = surr_plus - 1j * surr_cross
        amp_ref = np.abs(h_ref)
        amp_norm = amp_ref / amp_ref.max()

        phase_ref = np.unwrap(np.angle(h_ref))
        phase_surr = np.unwrap(np.angle(h_surr))

        mask = amp_norm >= args.amp_threshold
        dphi = phase_surr - phase_ref
        dphi -= np.mean(dphi[mask])
        amp_rel_err = (np.abs(h_surr) - np.abs(h_ref)) / np.abs(h_ref).max()

        print(f"--- q={q}  duration={duration:.2f}s  inspiral window "
              f"[{t_lo:.2f},{t_hi:.2f}]s ---")
        idxs = np.where(mask)[0]
        n = len(idxs)
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            i = idxs[min(int(frac * (n - 1)), n - 1)]
            print(f"  t={times[i]:+7.3f}s  amp_norm={amp_norm[i]:.3f}  "
                  f"dphase={dphi[i]:+8.3f} rad  amp_rel_err={amp_rel_err[i]:+.4f}")
        span = dphi[mask].max() - dphi[mask].min()
        print(f"  dphase span over window: {span:.2f} rad = "
              f"{span / (2 * np.pi):.3f} cycles\n")


if __name__ == "__main__":
    main()
