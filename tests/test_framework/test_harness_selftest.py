"""Test the tests: injected defects MUST be caught by the metrics/levels."""
import numpy as np
import pytest

from .fixtures import TIMES, chirp_oracle, OraclePlusError
from .metrics import (Thresholds, amplitude_ratio, compare_waveforms,
                      max_phase_error_deg, mismatch)

DT = float(TIMES[1] - TIMES[0])
REF = chirp_oracle(0.6, TIMES)


def _cmp(**defects):
    p = OraclePlusError(**defects).predict(0.6, TIMES)
    return compare_waveforms(REF, (p.plus, p.cross), DT, Thresholds())


def test_perfect_surrogate_passes():
    assert _cmp().passed


def test_phase_offset_is_not_an_error():
    hp, hx = REF
    c, s = np.cos(2.0), np.sin(2.0)
    assert max_phase_error_deg(REF, (hp * c - hx * s, hp * s + hx * c)) < 1e-6


@pytest.mark.parametrize("defect,check", [
    ({"phase_drift": 0.6}, "phase_error_deg"),        # ~34 deg secular drift
    ({"amp_scale": 1.3}, "|amp_ratio-1|"),
    ({"noise": 0.2}, "correlation"),
    ({"time_shift": 3 * DT}, "mismatch"),
])
def test_defect_is_caught(defect, check):
    rep = _cmp(**defect)
    assert not rep.passed
    assert not next(c for c in rep.checks if c.name == check).passed, str(rep)


def test_amplitude_ratio_is_exact_for_scaling():
    hp, hx = REF
    assert amplitude_ratio(REF, (2 * hp, 2 * hx)) == pytest.approx(2.0)


def test_time_shift_counts_in_mismatch():
    p = OraclePlusError(noise=0.0, time_shift=2 * DT).predict(0.6, TIMES)
    assert mismatch(REF[0], p.plus, DT) > 1e-3
