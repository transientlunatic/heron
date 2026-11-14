"""Level 1: one waveform (at a training node), full time series."""
import pytest

from ..metrics import compare_waveforms

pytestmark = pytest.mark.level(1)


@pytest.mark.parametrize("node", [0, 2, 4])
def test_time_series_metrics_at_node(harness, node):
    q = harness.train_qs[node]
    p = harness.predict(q)
    rep = compare_waveforms(harness.reference(q), (p.plus, p.cross), harness.dt,
                            harness.limits(1), label=f"{harness.name} q={q}")
    assert rep.passed, "\n" + str(rep)
