"""Follow-up to probe_seobnrv4_alignment.py: does correcting the ~1-2ms
peak-amplitude time offset between SEOBNRv4 and IMRPhenomXAS (measured
there) stabilize the phase offset into a near-constant band across q, the
way it is for IMRPhenomD-vs-IMRPhenomXAS? Or is the phase relationship
itself genuinely unstable, independent of the time misalignment?

Time offset is measured via envelope cross-correlation (sub-sample,
parabolic-refined), not the coarser argmax used in the first probe.
"""
import numpy as np
from astropy import units as u
from scipy.interpolate import CubicSpline

from heron.train import _get_approximant
from heron.evaluation.psd import aligo_design_psd

TOTAL_MASS = 60.0
DISTANCE = 100.0
F_LOW = 20.0
DT = 1.0 / 4096

ref = _get_approximant("IMRPhenomXAS")
oracle = _get_approximant("SEOBNRv4")


def generate(approx, q):
    params = dict(
        mass_ratio=q,
        total_mass=TOTAL_MASS * u.solMass,
        luminosity_distance=DISTANCE * u.Mpc,
        f_min=F_LOW * u.Hertz,
        delta_t=DT * u.second,
    )
    wf = approx.time_domain(dict(params))
    return np.asarray(wf["plus"].times), np.asarray(wf["plus"].data)


def common_grid(t1, h1, t2, h2):
    t0 = min(t1[0], t2[0])
    t1_end = max(t1[-1], t2[-1])
    n = int(round((t1_end - t0) / DT)) + 1
    tgrid = t0 + np.arange(n) * DT
    g1 = np.nan_to_num(CubicSpline(t1, h1, extrapolate=False)(tgrid))
    g2 = np.nan_to_num(CubicSpline(t2, h2, extrapolate=False)(tgrid))
    return tgrid, g1, g2


def envelope_time_offset(t_ref, h_ref, t_oracle, h_oracle):
    """Sub-sample lag (oracle relative to ref) via amplitude-envelope
    cross-correlation, parabolic-refined near the peak."""
    tgrid, e_ref, e_oracle = common_grid(t_ref, np.abs(h_ref), t_oracle, np.abs(h_oracle))
    n = len(tgrid)
    corr = np.correlate(e_oracle, e_ref, mode="full")
    lags = np.arange(-(n - 1), n) * DT
    k = int(np.argmax(corr))
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return lags[k] + delta * DT


def matched_filter_phase(t_ref, h_ref, t_oracle, h_oracle):
    """Same PSD-weighted zero-lag phase measurement as
    heron.models.gp.mean.compute_phase_correction's internals."""
    tgrid, g_ref, g_oracle = common_grid(t_ref, h_ref, t_oracle, h_oracle)
    n = len(tgrid)
    freqs = np.fft.rfftfreq(n, d=DT)
    f_ref = np.fft.rfft(g_ref)
    f_oracle = np.fft.rfft(g_oracle)
    psd = aligo_design_psd(freqs)
    band = freqs >= F_LOW
    z = np.sum(np.conj(f_ref[band]) * f_oracle[band] / psd[band])
    return float(np.angle(z))


print(f"{'q':>6} {'dt_env(ms)':>11} {'phase_raw':>10} {'phase_corrected':>16}")
for q in [0.15, 0.25, 0.40, 0.50, 0.70, 0.90]:
    t_ref, h_ref = generate(ref, q)
    t_oracle, h_oracle = generate(oracle, q)

    dt_env = envelope_time_offset(t_ref, h_ref, t_oracle, h_oracle)
    phase_raw = matched_filter_phase(t_ref, h_ref, t_oracle, h_oracle)
    phase_corrected = matched_filter_phase(t_ref, h_ref, t_oracle - dt_env, h_oracle)

    print(f"{q:6.2f} {dt_env*1e3:11.4f} {phase_raw:10.4f} {phase_corrected:16.4f}")
