"""Check whether SEOBNRv4-vs-IMRPhenomXAS has the same near-constant single
global phase/time alignment that IMRPhenomD-vs-IMRPhenomXAS was validated to
have (compute_phase_correction's docstring: -2.24 to -2.13 rad over
q=0.15-0.9). If it drifts with q instead, a single global phase_correction
(what DemodGPSurrogate applies) would leave a spurious secular residual.
"""
import numpy as np
from astropy import units as u

from heron.train import _get_approximant
from heron.models.gp.mean import compute_phase_correction

TOTAL_MASS = 60.0
DISTANCE = 100.0
F_LOW = 20.0
DT = 1.0 / 4096

ref = _get_approximant("IMRPhenomXAS")
oracle = _get_approximant("SEOBNRv4")

print("Phase offset (compute_phase_correction, single-q PSD-weighted matched filter):")
print(f"{'q':>6} {'phase (rad)':>12}")
for q in [0.15, 0.25, 0.40, 0.50, 0.70, 0.90]:
    c = compute_phase_correction(
        mean_approximant="IMRPhenomXAS",
        target_approximant="SEOBNRv4",
        total_mass=TOTAL_MASS,
        distance=DISTANCE,
        f_low=F_LOW,
        reference_mass_ratio=q,
    )
    print(f"{q:6.2f} {c:12.4f}")

print()
print("Peak-amplitude epoch (argmax sqrt(hp^2+hx^2)), time offset ref vs oracle:")
print(f"{'q':>6} {'t_peak_ref':>12} {'t_peak_seob':>13} {'dt (ms)':>10}")
for q in [0.15, 0.25, 0.40, 0.50, 0.70, 0.90]:
    params = dict(
        mass_ratio=q,
        total_mass=TOTAL_MASS * u.solMass,
        luminosity_distance=DISTANCE * u.Mpc,
        f_min=F_LOW * u.Hertz,
        delta_t=DT * u.second,
    )
    wf_ref = ref.time_domain(dict(params))
    wf_seob = oracle.time_domain(dict(params))
    t_ref = np.asarray(wf_ref["plus"].times)
    A_ref = np.sqrt(np.asarray(wf_ref["plus"].data)**2 + np.asarray(wf_ref["cross"].data)**2)
    t_seob = np.asarray(wf_seob["plus"].times)
    A_seob = np.sqrt(np.asarray(wf_seob["plus"].data)**2 + np.asarray(wf_seob["cross"].data)**2)
    tp_ref = t_ref[np.argmax(A_ref)]
    tp_seob = t_seob[np.argmax(A_seob)]
    print(f"{q:6.2f} {tp_ref:12.6f} {tp_seob:13.6f} {(tp_seob-tp_ref)*1e3:10.3f}")
