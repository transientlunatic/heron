"""Train a DemodGPSurrogate on the dense30 IMRPhenomD targets.

The demodulated-residual surrogate (heron.models.gp.demod.DemodGPSurrogate)
heterodynes the IMRPhenomD-vs-IMRPhenomXAS strain residual by the XAS phase,
fits smooth Re/Im GPs, and reconstructs strain with exact linear covariance.
See the module docstring and the `exact_xas_mismatch_residual` memory.

The IMRPhenomD training targets and the exact phase_correction are taken from
the exact-XAS checkpoint (its ``train_y`` IS the scaled IMRPhenomD strain, and
its plus mean carries phase_correction), so this trains on the identical data
the exact-XAS / phase-amplitude checkpoints used -- a like-for-like comparison.

Usage (wiay GPU)::

    python scripts/train_demod.py \
        --source checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt \
        --output checkpoints/phenomd_nonspinning_dense30_demod.pt \
        --device cuda --iterations 200
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from heron.models.gp.demod import DemodGPSurrogate
from heron.models.warping import get_warping


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt",
                    help="Exact-XAS checkpoint supplying IMRPhenomD train_y + phase_correction.")
    ap.add_argument("--output", default="checkpoints/phenomd_nonspinning_dense30_demod.pt")
    ap.add_argument("--base-approximant", default="IMRPhenomXAS", help="Reference (heterodyne) approximant.")
    ap.add_argument("--oracle-approximant", default="IMRPhenomD", help="Training-data approximant.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--ls-min-q", type=float, default=None, help="Default: source checkpoint's ls_min_q.")
    ap.add_argument("--ls-min-time", type=float, default=None, help="Default: source checkpoint's ls_min_time.")
    ap.add_argument("--noise-floor-rel", type=float, default=None, help="Default: source checkpoint's noise_floor_rel.")
    ap.add_argument("--cholesky-size", type=int, default=None, help="Default: source checkpoint's cholesky_size.")
    ap.add_argument("--mismatch-scan", action="store_true",
                    help="After training, scan mismatch-vs-D over q=0.45..0.90 and report median.")
    args = ap.parse_args()

    ck = torch.load(args.source, map_location="cpu", weights_only=False)
    output_scale = ck["output_scale"]
    train_x = ck["train_x"].to(torch.float64)
    # train_y in the exact-XAS checkpoint is the IMRPhenomD strain * output_scale.
    y_plus = (ck["train_y"]["plus"].to(torch.float64) / output_scale)
    y_cross = (ck["train_y"]["cross"].to(torch.float64) / output_scale)

    warp_cfg = dict(ck["warping"])
    warping = get_warping(warp_cfg.pop("type"), **warp_cfg)

    # Reproduce the exact-XAS mean's phase_correction exactly (avoids recompute
    # and guarantees the demodulation frame matches the prototype).
    phase_correction = ck["mean_functions"]["plus"].get("phase_correction")

    ls_min_q = args.ls_min_q if args.ls_min_q is not None else ck["ls_min_q"]
    ls_min_time = args.ls_min_time if args.ls_min_time is not None else ck["ls_min_time"]
    noise_floor_rel = args.noise_floor_rel if args.noise_floor_rel is not None else ck["noise_floor_rel"]
    cholesky_size = args.cholesky_size if args.cholesky_size is not None else ck["cholesky_size"]

    print(f"Training DemodGPSurrogate: base={args.base_approximant} oracle={args.oracle_approximant}")
    print(f"  N={train_x.shape[0]}  phase_correction={phase_correction:.6f} rad")
    print(f"  ls_min_q={ls_min_q} ls_min_time={ls_min_time} noise_floor_rel={noise_floor_rel} "
          f"cholesky_size={cholesky_size} iterations={args.iterations} device={args.device}")

    t0 = time.time()
    surrogate = DemodGPSurrogate(
        train_x=train_x,
        train_y_plus=y_plus,
        train_y_cross=y_cross,
        base_approximant=args.base_approximant,
        oracle_approximant=args.oracle_approximant,
        phase_correction=phase_correction,
        warping=warping,
        nu=ck["nu"],
        output_scale=output_scale,
        device=args.device,
        total_mass=ck["mass_factor"],
        distance=ck["distance_factor"],
        training_iterations=args.iterations,
        ls_min_time=ls_min_time,
        ls_min_q=ls_min_q,
        noise_floor_rel=noise_floor_rel,
        cholesky_size=cholesky_size,
    )
    print(f"Trained in {time.time() - t0:.0f}s")
    surrogate.save(args.output)
    print(f"Saved -> {args.output}")

    # Report trained q-lengthscales (a demod win is ls_q OFF its floor).
    for name, model in surrogate._gp.models.items():
        try:
            ls = model.covar_module.base_kernel.kernels[0].lengthscale.detach().cpu().numpy().ravel()
            print(f"  {name} (Re/Im) ls_q = {ls[0]:.4f}  (floor {ls_min_q})")
        except Exception:
            pass

    if args.mismatch_scan:
        import astropy.units as u
        from heron.evaluation.mismatch import compute_mismatch
        from heron.evaluation.psd import aligo_design_psd
        from heron.train import _get_approximant

        times = np.linspace(-0.5, 0.02, 512)
        dt = times[1] - times[0]
        psd = aligo_design_psd(np.fft.rfftfreq(512, d=dt))
        D = _get_approximant(args.oracle_approximant)
        qs = np.linspace(0.45, 0.90, 60)
        mm = np.empty(len(qs))
        for i, q in enumerate(qs):
            hp = surrogate.predict({"mass_ratio": float(q), "times": times})["plus"].data
            ref = D.time_domain({"mass_ratio": float(q), "total_mass": 60 * u.solMass,
                                 "luminosity_distance": 100 * u.Mpc}, times=times)["plus"].data
            mm[i] = compute_mismatch(hp, ref, dt, psd)
        print(f"\nDEMOD mismatch vs {args.oracle_approximant}: "
              f"median={np.median(mm):.3e}  worst={mm.max():.3e}@q={qs[mm.argmax()]:.3f}")
        print("  (compare: exact-XAS 3.19e-3, phase-amp 7.60e-5, prototype demod 1.19e-5)")


if __name__ == "__main__":
    main()
