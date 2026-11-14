"""Base abstractions for the progressive test framework.

A ``Harness`` couples a surrogate with its ground truth behind one tiny
interface, so every level runs unchanged against any model (ExactGP, demod,
a stub, ...) -- adding a model to the framework means writing one adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from .metrics import Thresholds


class Prediction:
    """Surrogate output on a time grid: mean (plus, cross) and diagonal variance."""

    def __init__(self, plus, cross, var_plus=None, var_cross=None):
        self.plus = np.asarray(plus, float)
        self.cross = np.asarray(cross, float)
        self.var_plus = None if var_plus is None else np.asarray(var_plus, float)
        self.var_cross = None if var_cross is None else np.asarray(var_cross, float)


class Surrogate(Protocol):
    def predict(self, q: float, times: np.ndarray) -> Prediction: ...


Oracle = Callable[[float, np.ndarray], tuple]  # (q, times) -> (plus, cross)


@dataclass
class Harness:
    """Surrogate + ground truth + the grids/limits a level needs.

    Attributes
    ----------
    train_qs : mass ratios the surrogate was trained on (level 1/2 'nodes').
    holdout_qs : mass ratios *between* nodes (level 3 interpolation).
    thresholds : per-level limits; models with weaker accuracy relax these
        explicitly rather than editing the test.
    """

    name: str
    surrogate: Surrogate
    oracle: Oracle
    train_qs: Sequence[float]
    holdout_qs: Sequence[float]
    times: np.ndarray
    thresholds: dict = field(default_factory=dict)  # level -> Thresholds

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])

    def limits(self, level: int) -> Thresholds:
        return self.thresholds.get(level, Thresholds())

    def reference(self, q: float):
        return self.oracle(q, self.times)

    def predict(self, q: float, times=None) -> Prediction:
        return self.surrogate.predict(q, self.times if times is None else times)


class WaveformSurrogateAdapter:
    """Adapt a ``heron`` ``WaveformSurrogate`` (``predict(dict) -> WaveformDict``)."""

    def __init__(self, model, **fixed_params):
        self.model = model
        self.fixed = fixed_params

    def predict(self, q, times):
        wf = self.model.predict(
            {"mass_ratio": float(q), "times": np.asarray(times), **self.fixed},
            covariance="diagonal",
        )
        p, c = wf["plus"], wf["cross"]
        return Prediction(p.data, c.data, p.variance, c.variance)
