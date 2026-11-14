"""Is the time warping the right one? Training-free diagnostics (manager task #398).

A stationary Matern kernel in the warped time u suits the strain-domain GP
best when the target oscillates at a constant rate in u, i.e. the GW phase
advances uniformly: dPhi/du = const. Newtonian: Phi ~ tau^(5/8) and
f ~ tau^(-3/8) (tau = time to merger), so u ~ tau^(5/8) makes the phase
uniform; 3/8 is the frequency exponent.

For IMRPhenomD (M=60, f_min=20 Hz, 4096 Hz, interpolated-peak alignment)
this script measures, per warping:
  * dPhi/du over the in-band inspiral (spread = max/min, and over the last
    100 ms before the peak) and after the peak (ringdown), relative to the
    inspiral median: 1 everywhere would be ideal;
  * with the training scheme (200 samples uniform in u over each q's
    support): the physical sampling density, minimum samples per GW cycle
    (Nyquist needs > 2), and the fraction of samples below 20 Hz;
  * the error of the linear (np.interp) target interpolation train.py uses.

Warpings: the code's ChirpTimeWarping(alpha, t_ref=0.1) for several alpha,
and the paper's form, -|t|^(3/8) before the peak and linear after (the slope
of the linear segment is not fixed by the description; it does not affect
the pre-peak numbers).

Usage: python scripts/warping_uniformity.py [--out results/warping_uniformity.npz]
"""

import argparse

import numpy as np
from astropy import units as u

from heron.models.lalsimulation import IMRPhenomD

T_REF = 0.1
N_SAMPLES = 200


def power_warp(alpha):
    def w(t):
        return np.sign(t) * np.abs(t / T_REF) ** alpha * T_REF
    return w


def paper_warp(t):
    # -|t|^(3/8) before the peak, linear after (slope 1/s with s = T_REF);
    # pre-peak normalised like the code's (t_ref) for comparable units.
    return np.where(t < 0, -np.abs(t / T_REF) ** 0.375 * T_REF, t / T_REF * T_REF)


WARPINGS = {
    "code a=5/8 (production)": power_warp(0.625),
    "a=1/2": power_warp(0.5),
    "a=3/4": power_warp(0.75),
    "a=3/8 symmetric (code default)": power_warp(0.375),
    "paper: a=3/8 pre, linear post": paper_warp,
}


def waveform(q):
    wf = IMRPhenomD(peak_alignment="interpolated").time_domain({
        "mass_ratio": q, "total_mass": 60 * u.solMass,
        "luminosity_distance": 100 * u.Mpc, "f_min": 20 * u.Hertz,
    })
    t = np.asarray(wf["plus"].times)
    hp, hx = np.asarray(wf["plus"].data), np.asarray(wf["cross"].data)
    phase = np.unwrap(np.arctan2(hx, hp))
    phase = phase * np.sign(phase[-1] - phase[0])  # increasing with time
    freq = np.gradient(phase, t) / (2 * np.pi)
    return t, hp, hx, phase, freq


def analyse(q, warp):
    t, hp, hx, phase, freq = waveform(q)
    uu = warp(t)
    dphi_du = np.gradient(phase, uu)

    amp = np.hypot(hp, hx)
    # frequency is only meaningful where the signal is: ignore the
    # numerically empty post-ringdown tail when sampling is assessed.
    significant = amp > 1e-2 * amp.max()
    inband = (freq > 20) & (t < -0.005)  # in-band inspiral, clear of the peak
    late = (t > -0.1) & (t < -0.005)
    post = (t > 0.002) & (t < 0.02) & (amp > 1e-3 * amp.max())
    ref = np.median(dphi_du[inband])
    lo, hi = np.percentile(dphi_du[inband], [5, 95])
    spread = hi / lo  # robust to the noisy tapered start
    late_spread = dphi_du[late].max() / dphi_du[late].min()
    post_ratio = np.median(dphi_du[post]) / ref if post.any() else np.nan

    # training scheme: N_SAMPLES uniform in u over the full support
    us = np.linspace(uu[0], uu[-1], N_SAMPLES)
    order = np.argsort(uu)
    ts = np.interp(us, uu[order], t[order])
    fs = np.interp(ts, t, freq)
    sig_s = np.interp(ts, t, significant.astype(float)) > 0.5
    dts = np.diff(ts)
    cycles_per_gap = fs[:-1] * dts
    masked = np.where(sig_s[:-1], cycles_per_gap, 0.0)
    worst = int(np.argmax(masked))
    min_samples_per_cycle = 1.0 / masked[worst]
    t_worst = ts[worst]
    below_20 = np.mean(fs[sig_s] < 20.0)
    density_early = 1.0 / dts[0]
    density_peak = 1.0 / dts[np.argmin(np.abs(ts[:-1]))]

    return dict(spread=spread, late_spread=late_spread, post_ratio=post_ratio,
                min_spc=min_samples_per_cycle, below20=below_20,
                dens_early=density_early, dens_peak=density_peak,
                t_worst=t_worst)


def interp_error(q):
    """Relative error of linear interpolation at the training points (code
    warping) vs a cubic-accurate reference, for the plus polarisation."""
    from scipy.interpolate import CubicSpline
    t, hp, _, _, _ = waveform(q)
    warp = power_warp(0.625)
    uu = warp(t)
    us = np.linspace(uu[0], uu[-1], N_SAMPLES)
    ts = np.interp(us, uu, t)
    lin = np.interp(ts, t, hp)
    ref = CubicSpline(t, hp)(ts)
    return np.max(np.abs(lin - ref)) / np.max(np.abs(hp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    qs = (0.3, 0.6, 0.9)
    rows = {}
    print("dPhi/du uniformity (1 = ideal) and training-grid sampling, IMRPhenomD M=60")
    print(f"{'warping':34s} {'q':>4s} {'insp p95/p5':>16s} {'last100ms max/min':>17s}"
          f" {'post/inspiral':>13s} {'min samp/cycle':>14s} {'<20Hz':>6s} {'dens early/peak (1/s)':>22s}")
    for name, warp in WARPINGS.items():
        for q in qs:
            r = analyse(q, warp)
            rows[(name, q)] = r
            print(f"{name:34s} {q:4.1f} {r['spread']:16.1f} {r['late_spread']:17.2f}"
                  f" {r['post_ratio']:13.2f} {r['min_spc']:14.2f} {100 * r['below20']:5.0f}%"
                  f" {r['dens_early']:10.0f} / {r['dens_peak']:9.0f}   worst@{1e3 * r['t_worst']:+.1f}ms")
    print("\nlinear-interpolation target error (production warping, max |lin-cubic|/max|h|):")
    for q in qs:
        print(f"  q={q}: {interp_error(q):.2e}")
    if args.out:
        names = list(WARPINGS)
        keys = ["spread", "late_spread", "post_ratio", "min_spc", "below20",
                "dens_early", "dens_peak", "t_worst"]
        np.savez(args.out, warpings=np.array(names), q=np.array(qs),
                 **{k: np.array([[rows[(n, q)][k] for q in qs] for n in names])
                    for k in keys})
        print("saved", args.out)


if __name__ == "__main__":
    main()
