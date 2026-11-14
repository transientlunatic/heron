"""Progressive gating: a level only runs if every lower level passed.

Tests carry ``@pytest.mark.level(n)``.  Failures are recorded per harness, so
a broken model is stopped at the first failing level (with a clear skip
reason) instead of producing a wall of downstream failures.
"""
from __future__ import annotations

import pytest

from .fixtures import make_analytic_harness, make_gp_harness

_failed_levels: dict[str, set[int]] = {}


def pytest_configure(config):
    config.addinivalue_line("markers", "level(n): progressive validation level n (0-3)")


def _level(item):
    m = item.get_closest_marker("level")
    return None if m is None else int(m.args[0])


@pytest.fixture(scope="session", params=["analytic", "gp"])
def harness(request):
    if request.param == "analytic":
        return make_analytic_harness()
    pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    return make_gp_harness()


def _harness_name(item):
    callspec = getattr(item, "callspec", None)
    return callspec.params.get("harness") if callspec else None


def pytest_runtest_setup(item):
    lvl, name = _level(item), _harness_name(item)
    if lvl is None or name is None:
        return
    bad = sorted(n for n in _failed_levels.get(name, ()) if n < lvl)
    if bad:
        pytest.skip(f"[{name}] level {bad[0]} failed; fix it before level {lvl}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    lvl, name = _level(item), _harness_name(item)
    if rep.when == "call" and rep.failed and lvl is not None and name is not None:
        _failed_levels.setdefault(name, set()).add(lvl)
