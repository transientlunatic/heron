"""Effect of the peak-alignment convention on the IMRPhenomD - IMRPhenomXAS residual.

LAL archaeology (manager task #399), follow-up to scripts/lal_peak_jumps.py.
For each q, generates the oracle (D) and reference (XAS) with
``peak_alignment="sample"`` (LAL's convention) and ``"interpolated"``, then on
a fixed grid around merger reports:
  * the relative residual norm ||D - e^{i phi} XAS|| / ||D|| maximised over a
    constant phase only (like the XAS mean's phase_correction), not over time;
  * the continuous best-fit D-XAS time shift.
With "sample", both jump between q~0.2837 (XAS jump) and q~0.2913 (D jump);
with "interpolated" they are smooth. Requires lalsuite.

Usage: python scripts/lal_peak_alignment_residual.py [--out results/lal_peak_alignment_residual.npz]
"""

import argparse

import numpy as np
from astropy import units as u
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from heron.models.lalsimulation import IMRPhenomD, IMRPhenomXAS

T = np.arange(-0.25, 0.03, 1.0 / 16384)  # fixed comparison grid (s)


def complex_on_grid(approx, q):
    wf = approx.time_domain({
        "mass_ratio": q,
        "total_mass": 60 * u.solMass,
        "luminosity_distance": 100 * u.Mpc,
        "f_min": 20 * u.Hertz,
    })
    t = np.asarray(wf["plus"].times)
    h = np.asarray(wf["plus"].data) - 1j * np.asarray(wf["cross"].data)
    return CubicSpline(t, h.real), CubicSpline(t, h.imag)


def compare(d, x):
    D = d[0](T) + 1j * d[1](T)
    X = x[0](T) + 1j * x[1](T)
    z = np.vdot(X, D)
    resid = np.linalg.norm(D - np.exp(1j * np.angle(z)) * X) / np.linalg.norm(D)

    def cost(s):
        Xs = x[0](T + s) + 1j * x[1](T + s)
        return -abs(np.vdot(Xs, D)) / np.linalg.norm(Xs)

    shift = minimize_scalar(cost, bounds=(-1e-3, 1e-3), method="bounded",
                            options={"xatol": 1e-10}).x
    return resid, shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--qmin", type=float, default=0.25)
    ap.add_argument("--qmax", type=float, default=0.33)
    ap.add_argument("--dq", type=float, default=0.0025)
    args = ap.parse_args()
    qs = np.round(np.arange(args.qmin, args.qmax + 1e-9, args.dq), 5)
    out = {"q": qs}
    for mode in ("sample", "interpolated"):
        D, X = IMRPhenomD(peak_alignment=mode), IMRPhenomXAS(peak_alignment=mode)
        res = np.array([compare(complex_on_grid(D, q), complex_on_grid(X, q)) for q in qs])
        out[f"resid_{mode}"], out[f"shift_{mode}"] = res[:, 0], res[:, 1]
    print("   q    resid(sample)  resid(interp)  shift(sample,us)  shift(interp,us)")
    for i, q in enumerate(qs):
        print(f"{q:.4f}   {out['resid_sample'][i]:.4e}     {out['resid_interpolated'][i]:.4e}"
              f"     {1e6 * out['shift_sample'][i]:+8.2f}         {1e6 * out['shift_interpolated'][i]:+8.2f}")
    if args.out:
        np.savez(args.out, **out)
        print("saved", args.out)


if __name__ == "__main__":
    main()
