"""Held-out-error scalar recalibration of DemodGPSurrogate's covariance K.

The demod GPs interpolate their smooth Re/Im targets so confidently that the
bare posterior variance under-reports the true surrogate error by orders of
magnitude (var/err^2 ~ 1e-17 on dense30). That is harmless -- indeed desirable
-- for point-estimate PE (K << C, so the log-det term is flat and there is no
grid-snap bias; see the exact_xas_mismatch_residual memory / the SNR~20 NS
validation). But it makes K useless as a *trustworthy uncertainty* for
coverage / PP-plot work.

This measures the true squared strain error against the oracle at held-out
ANTINODE mass ratios (midpoints between training nodes -- worst case: largest
error, smallest variance) over an in-band time window, and reports the scalar
`covariance_inflation` that makes var * inflation match err^2 in the median.
With --write it stores that scalar in the checkpoint (predict() then returns the
inflated, calibrated covariance; the mean is never touched).

A single scalar cannot capture the q- and time-dependence of the true error, so
this is a first-order coverage fix, not an exact one -- inspect the reported
percentile spread of err^2/var to see how far a scalar can get you.

Usage:
    python scripts/calibrate_demod_k.py \
        --checkpoint checkpoints/phenomd_nonspinning_dense30_demod.pt
    # ... then, to bake it in:
    python scripts/calibrate_demod_k.py --checkpoint ... --write
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import astropy.units as u

from heron.models.gp.demod import DemodGPSurrogate
from heron.train import _get_approximant


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/phenomd_nonspinning_dense30_demod.pt")
    ap.add_argument("--oracle", default=None,
                    help="Oracle approximant (default: the checkpoint's recorded oracle).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--q-lo", type=float, default=0.40, help="Lowest held-out q to include.")
    ap.add_argument("--q-hi", type=float, default=0.90, help="Highest held-out q to include.")
    ap.add_argument("--t-lo", type=float, default=-0.3, help="In-band window start (s).")
    ap.add_argument("--t-hi", type=float, default=0.0, help="In-band window end (s).")
    ap.add_argument("--n-times", type=int, default=512)
    ap.add_argument("--write", action="store_true",
                    help="Write the recommended covariance_inflation into the checkpoint.")
    ap.add_argument("--output", default=None,
                    help="Checkpoint to write with --write (default: overwrite --checkpoint).")
    args = ap.parse_args()

    s = DemodGPSurrogate.load(args.checkpoint, device=args.device)
    s.covariance_inflation = 1.0  # measure the BARE posterior variance
    oracle_name = args.oracle or s._oracle_name or "IMRPhenomD"
    D = _get_approximant(oracle_name)

    # Antinode held-out q's: midpoints between adjacent training nodes.
    nodes = np.unique(s._train_x_raw[:, 0].cpu().numpy())
    mids = 0.5 * (nodes[:-1] + nodes[1:])
    qs = mids[(mids >= args.q_lo) & (mids <= args.q_hi)]
    print(f"Calibrating K on {len(qs)} held-out antinode q's in [{args.q_lo}, {args.q_hi}] "
          f"vs {oracle_name}, window [{args.t_lo}, {args.t_hi}] s")

    times = np.linspace(-0.5, 0.02, args.n_times)
    band = (times >= args.t_lo) & (times <= args.t_hi)

    ratios, all_err2, all_var = [], [], []
    for q in qs:
        wf = s.predict({"mass_ratio": float(q), "times": times})
        ref = D.time_domain(
            {"mass_ratio": float(q), "total_mass": s.mass_factor * u.solMass,
             "luminosity_distance": s.distance_factor * u.Mpc},
            times=times,
        )
        for pol in ("plus", "cross"):
            h = wf[pol].data
            v = np.diag(wf[pol].covariance)
            r = ref[pol].data
            err2 = (h - r) ** 2
            m = band & (v > 0)
            ratios.append(err2[m] / v[m])
            all_err2.append(err2[m])
            all_var.append(v[m])

    ratios = np.concatenate(ratios)
    err2_all = np.concatenate(all_err2)
    var_all = np.concatenate(all_var)

    infl_median = float(np.median(ratios))          # median(var*infl / err2) = 1
    infl_rms = float(err2_all.mean() / var_all.mean())  # matches total power
    p16, p84, p97 = np.percentile(ratios, [16, 84, 97.5])

    print(f"\n  bare var/err^2 median : {1.0/infl_median:.3e}  (under-report factor)")
    print(f"  err^2/var percentiles : 16%={p16:.3e}  50%={infl_median:.3e}  "
          f"84%={p84:.3e}  97.5%={p97:.3e}")
    print(f"  RECOMMENDED covariance_inflation (median-calibrated): {infl_median:.4e}")
    print(f"    -> std inflated by {np.sqrt(infl_median):.3e}x")
    print(f"  alternative (total-power / RMS-calibrated)          : {infl_rms:.4e}")
    print("  NOTE: a scalar cannot flatten the q/time spread above; for full "
          "coverage inspect the percentiles (97.5% would need a larger factor).")

    if args.write:
        out = args.output or args.checkpoint
        s.covariance_inflation = infl_median
        s.save(out)
        print(f"\nWrote covariance_inflation={infl_median:.4e} -> {out}")
    else:
        print("\n(dry run; pass --write to store this in the checkpoint)")


if __name__ == "__main__":
    main()
