"""Level 2: every training node, plus interpolation midway between nodes."""
import numpy as np
import pytest

from ..metrics import compare_waveforms

pytestmark = pytest.mark.level(2)


def test_all_training_nodes(harness):
    failed = []
    for q in harness.train_qs:
        p = harness.predict(q)
        rep = compare_waveforms(harness.reference(q), (p.plus, p.cross), harness.dt,
                                harness.limits(2), label=f"q={q}")
        if not rep.passed:
            failed.append(str(rep))
    assert not failed, "\n" + "\n".join(failed)


def test_error_not_hidden_in_one_polarisation(harness):
    """Plus and cross must be equally good (a phase bug breaks one first)."""
    from ..metrics import mismatch
    q = harness.train_qs[1]
    ref, p = harness.reference(q), harness.predict(q)
    mp = mismatch(ref[0], p.plus, harness.dt)
    mc = mismatch(ref[1], p.cross, harness.dt)
    lim = harness.limits(2).mismatch
    assert mp < lim and mc < lim, f"plus {mp:.2e}, cross {mc:.2e}, limit {lim:.1e}"


def test_midpoint_interpolation_within_holdout_limit(harness):
    """Held-out q between nodes, against the harness's level-3 (holdout) limit."""
    from ..metrics import mismatch
    lim = harness.limits(3).mismatch
    worst = max(
        mismatch(harness.reference(q)[0], harness.predict(q).plus, harness.dt)
        for q in harness.holdout_qs
    )
    assert np.isfinite(worst) and worst < lim, f"worst midpoint mismatch {worst:.2e} >= {lim:.1e}"
