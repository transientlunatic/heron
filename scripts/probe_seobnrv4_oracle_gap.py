"""Cheap local sanity check (no wiay, no training): how much bigger is the
SEOBNRv4-vs-IMRPhenomXAS mismatch than the current IMRPhenomD-vs-IMRPhenomXAS
mismatch, at dense30's parameters? Decides whether SEOBNRv4 is worth an
actual demod retrain before spending GPU time on it.
"""
import time
import numpy as np
from astropy import units as u
from scipy.interpolate import CubicSpline

from heron.train import _get_approximant
from heron.evaluation.mismatch import compute_mismatch
from heron.evaluation.psd import aligo_design_psd

TOTAL_MASS = 60.0
DISTANCE = 100.0
F_LOW = 20.0
DT = 1.0 / 4096

ref = _get_approximant("IMRPhenomXAS")
oracle_d = _get_approximant("IMRPhenomD")
oracle_seob = _get_approximant("SEOBNRv4")


def gen(approx, q):
    params = dict(
        mass_ratio=q,
        total_mass=TOTAL_MASS * u.solMass,
        luminosity_distance=DISTANCE * u.Mpc,
        f_min=F_LOW * u.Hertz,
        delta_t=DT * u.second,
    )
    wf = approx.time_domain(params)
    return np.asarray(wf["plus"].times), np.asarray(wf["plus"].data)


def mismatch_on_common_grid(t1, h1, t2, h2):
    t0 = min(t1[0], t2[0])
    t1_end = max(t1[-1], t2[-1])
    n = int(round((t1_end - t0) / DT)) + 1
    tgrid = t0 + np.arange(n) * DT
    g1 = np.nan_to_num(CubicSpline(t1, h1, extrapolate=False)(tgrid))
    g2 = np.nan_to_num(CubicSpline(t2, h2, extrapolate=False)(tgrid))
    freqs = np.fft.rfftfreq(n, d=DT)
    psd = aligo_design_psd(freqs)
    return compute_mismatch(g1, g2, DT, psd=psd)


print(f"{'q':>6} {'D-vs-XAS':>12} {'SEOBv4-vs-XAS':>14} {'ratio':>8} {'SEOBv4 gen(ms)':>15}")
for q in [0.25, 0.40, 0.50, 0.70, 0.90]:
    t_ref, h_ref = gen(ref, q)
    t_d, h_d = gen(oracle_d, q)
    t0 = time.perf_counter()
    t_s, h_s = gen(oracle_seob, q)
    dt_s = (time.perf_counter() - t0) * 1e3

    mm_d = mismatch_on_common_grid(t_ref, h_ref, t_d, h_d)
    mm_s = mismatch_on_common_grid(t_ref, h_ref, t_s, h_s)
    ratio = mm_s / mm_d if mm_d > 0 else float("nan")
    print(f"{q:6.2f} {mm_d:12.3e} {mm_s:14.3e} {ratio:8.1f} {dt_s:15.1f}")
