"""Joint (time-lag, phase) matched-filter alignment between SEOBNRv4 and
IMRPhenomXAS, across a finer q grid than the first two probes.

Sequentially aligning on the amplitude-envelope peak time and then
measuring phase separately (probe_seobnrv4_alignment_corrected.py) is not
the right way to do this for a chirping signal -- time and phase are
coupled in the matched-filter sense. This instead finds the joint
argmax_tau |z(tau)| (same complex-ifft approach as
heron.evaluation.mismatch.compute_overlap's maximize_time branch) and
reads off both the optimal lag and the phase there, to see whether a
SINGLE (tau*, phi*) pair is even well-defined and how it varies with q.
"""
import numpy as np
from astropy import units as u

from heron.train import _get_approximant
from heron.evaluation.psd import aligo_design_psd

TOTAL_MASS = 60.0
DISTANCE = 100.0
F_LOW = 20.0
DT = 1.0 / 4096
MAX_SHIFT = 0.05  # seconds

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


def joint_lag_phase(t_ref, h_ref, t_oracle, h_oracle):
    from scipy.interpolate import CubicSpline

    t0 = min(t_ref[0], t_oracle[0])
    t1 = max(t_ref[-1], t_oracle[-1])
    n = int(round((t1 - t0) / DT)) + 1
    tgrid = t0 + np.arange(n) * DT
    g_ref = np.nan_to_num(CubicSpline(t_ref, h_ref, extrapolate=False)(tgrid))
    g_oracle = np.nan_to_num(CubicSpline(t_oracle, h_oracle, extrapolate=False)(tgrid))

    freqs = np.fft.rfftfreq(n, d=DT)
    df = freqs[1] - freqs[0]
    psd = aligo_design_psd(freqs)
    inv_psd = np.where(np.isfinite(psd) & (psd > 0), 1.0 / psd, 0.0)

    f_ref = np.fft.rfft(g_ref)
    f_oracle = np.fft.rfft(g_oracle)
    integrand = f_ref * np.conj(f_oracle) * inv_psd
    weighted = integrand.copy()
    weighted[0] *= 0.5
    if n % 2 == 0:
        weighted[-1] *= 0.5
    spectrum = np.zeros(n, dtype=complex)
    spectrum[: len(weighted)] = weighted
    z_t = np.fft.ifft(spectrum) * n * 4.0 * df

    n_shift = max(1, int(round(MAX_SHIFT / DT)))
    idx = np.concatenate([np.arange(0, n_shift + 1), np.arange(n - n_shift, n)])
    window = z_t[idx]
    k = int(np.argmax(np.abs(window)))
    best_idx = idx[k]
    lag = best_idx * DT if best_idx <= n_shift else (best_idx - n) * DT
    return lag, float(np.angle(window[k])), float(np.abs(window[k]))


print(f"{'q':>6} {'lag*(ms)':>10} {'phase*':>9} {'|z*|(rel)':>10}")
peaks = []
for q in np.arange(0.10, 0.96, 0.05):
    q = round(float(q), 3)
    t_ref, h_ref = generate(ref, q)
    t_oracle, h_oracle = generate(oracle, q)
    lag, phase, amp = joint_lag_phase(t_ref, h_ref, t_oracle, h_oracle)
    peaks.append(amp)
    print(f"{q:6.2f} {lag*1e3:10.4f} {phase:9.4f} {amp/peaks[0]:10.4f}")
