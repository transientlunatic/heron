"""Level 0: one waveform, one point in time.

Cheapest possible check that a model is alive: at a training node the mean is
finite and close to truth *relative to its own stated uncertainty*, and the
uncertainty itself is finite and positive.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.level(0)


def _mid_index(h):
    return int(np.argmax(np.hypot(*h.reference(h.train_qs[2]))))  # peak sample


def test_prediction_is_finite(harness):
    q = harness.train_qs[2]
    p = harness.predict(q)
    assert np.all(np.isfinite(p.plus)) and np.all(np.isfinite(p.cross))
    assert p.plus.shape == harness.times.shape


def test_variance_positive_and_finite(harness):
    p = harness.predict(harness.train_qs[2])
    if p.var_plus is None:
        pytest.skip("surrogate reports no variance")
    for v in (p.var_plus, p.var_cross):
        assert np.all(np.isfinite(v)) and np.all(v > 0)


def test_single_point_within_stated_uncertainty(harness):
    q = harness.train_qs[2]
    i = _mid_index(harness)
    ref, p = harness.reference(q), harness.predict(q)
    if p.var_plus is None:
        pytest.skip("surrogate reports no variance")
    z = abs(p.plus[i] - ref[0][i]) / np.sqrt(p.var_plus[i])
    assert z < harness.limits(0).z_score, f"|err|/sigma = {z:.1f} at sample {i}"
