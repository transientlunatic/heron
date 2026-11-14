"""Standardised validation metrics for the progressive test framework.

Everything here is numpy/scipy only, so the metric layer can be tested without
torch/gpytorch/lalsuite.  Waveforms are passed as (plus, cross) pairs.

Targets (see ``tests/test_framework/README.md``):

- mismatch        1 - overlap            < 1e-3
- amplitude ratio surrogate/reference    ~ 1
- correlation     Pearson                > 0.99
- phase error     max |dPhi|, degrees    < 5
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    """Pass/fail limits for one validation level."""

    mismatch: float = 1e-3
    correlation: float = 0.99
    amplitude_ratio_tol: float = 0.05   # |median(ratio) - 1|
    phase_error_deg: float = 5.0
    z_score: float = 5.0                # |mean - ref| / sigma at a single point

    def relaxed(self, factor: float) -> "Thresholds":
        """Return a copy with every tolerance loosened by ``factor``."""
        return Thresholds(
            mismatch=self.mismatch * factor,
            correlation=1.0 - (1.0 - self.correlation) * factor,
            amplitude_ratio_tol=self.amplitude_ratio_tol * factor,
            phase_error_deg=self.phase_error_deg * factor,
            z_score=self.z_score,
        )


def envelope(plus: np.ndarray, cross: np.ndarray) -> np.ndarray:
    """Amplitude envelope sqrt(h+^2 + hx^2); smooth, unlike |h+| on a carrier."""
    return np.hypot(plus, cross)


def mismatch(ref_plus, surr_plus, dt, psd=None, maximize_time=False) -> float:
    """Fitting-factor mismatch via the project's own ``compute_mismatch``.

    Time maximisation is off by default: the harness wants timing errors to
    *count*, since ``compute_overlap`` only searches integer-sample lags.
    """
    from heron.evaluation.mismatch import compute_mismatch

    return float(compute_mismatch(
        np.asarray(ref_plus), np.asarray(surr_plus), dt, psd,
        maximize_phase=True, maximize_time=maximize_time,
    ))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def amplitude_ratio(ref, surr, floor: float = 0.01) -> float:
    """Median surr/ref envelope ratio over samples above ``floor`` x peak.

    ``ref``/``surr`` are (plus, cross) tuples.  The floor keeps the ratio away
    from the near-zero ringdown tail, where it is noise.
    """
    e_ref, e_surr = envelope(*ref), envelope(*surr)
    mask = e_ref > floor * e_ref.max()
    return float(np.median(e_surr[mask] / e_ref[mask]))


def max_phase_error_deg(ref, surr, floor: float = 0.05) -> float:
    """Max |phase difference| in degrees after removing a constant offset.

    Phase is ``atan2(hx, h+)`` of each pair; the *difference* is wrapped to
    (-pi, pi] so no unwrap is needed (sampling-rate-safe, unlike unwrapping
    each phase separately).  A constant offset is a convention choice, not an
    error, and is removed with the circular mean.
    """
    p_ref = np.arctan2(ref[1], ref[0])
    p_surr = np.arctan2(surr[1], surr[0])
    d = np.angle(np.exp(1j * (p_surr - p_ref)))
    e = envelope(*ref)
    mask = e > floor * e.max()
    d = d[mask]
    offset = np.angle(np.mean(np.exp(1j * d)))
    d = np.angle(np.exp(1j * (d - offset)))
    return float(np.degrees(np.max(np.abs(d))))


@dataclass
class Check:
    name: str
    value: float
    limit: float
    passed: bool
    direction: str  # "<" value must be below limit, ">" above

    def __str__(self) -> str:
        flag = "ok " if self.passed else "FAIL"
        return f"[{flag}] {self.name}: {self.value:.3e} (need {self.direction} {self.limit:.3e})"


@dataclass
class MetricReport:
    """All metrics for one (waveform, surrogate) comparison."""

    label: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def __str__(self) -> str:
        return "\n".join([self.label] + ["  " + str(c) for c in self.checks])

    def as_dict(self) -> dict:
        return {"label": self.label, "passed": self.passed,
                "checks": {c.name: c.value for c in self.checks}}


def compare_waveforms(ref, surr, dt, thresholds: Thresholds, label="", psd=None) -> MetricReport:
    """Run every time-series metric on a (plus, cross) pair against ``thresholds``."""
    t = thresholds
    rep = MetricReport(label)
    mm = mismatch(ref[0], surr[0], dt, psd)
    rep.checks.append(Check("mismatch", mm, t.mismatch, bool(mm < t.mismatch), "<"))
    r = min(correlation(ref[0], surr[0]), correlation(ref[1], surr[1]))
    rep.checks.append(Check("correlation", r, t.correlation, bool(r > t.correlation), ">"))
    ar = abs(amplitude_ratio(ref, surr) - 1.0)
    rep.checks.append(Check("|amp_ratio-1|", ar, t.amplitude_ratio_tol, bool(ar < t.amplitude_ratio_tol), "<"))
    ph = max_phase_error_deg(ref, surr)
    rep.checks.append(Check("phase_error_deg", ph, t.phase_error_deg, bool(ph < t.phase_error_deg), "<"))
    return rep
