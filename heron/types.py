"""
Lightweight data types for waveforms with uncertainty.

No external dependencies beyond numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass
class Waveform:
    """A time-domain waveform with optional uncertainty.

    Parameters
    ----------
    data : np.ndarray
        Strain values, shape (N,).
    times : np.ndarray
        Time values, shape (N,).
    covariance : np.ndarray or None
        Full covariance matrix, shape (N, N).
    dt : float or None
        Sample spacing. Inferred from times if not given.
    t0 : float
        Epoch / reference time.
    """

    data: np.ndarray
    times: np.ndarray
    covariance: np.ndarray | None = None
    dt: float | None = None
    t0: float = 0.0

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=np.float64)
        self.times = np.asarray(self.times, dtype=np.float64)
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=np.float64)
        if self.dt is None and len(self.times) > 1:
            self.dt = float(self.times[1] - self.times[0])

    @property
    def variance(self) -> np.ndarray | None:
        if self.covariance is not None:
            return np.diag(self.covariance)
        return None

    @property
    def std(self) -> np.ndarray | None:
        v = self.variance
        if v is not None:
            return np.sqrt(np.maximum(v, 0.0))
        return None

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    def __len__(self) -> int:
        return len(self.data)


class WaveformDict:
    """Container for polarisation components (plus, cross) of a waveform.

    Parameters
    ----------
    parameters : dict or None
        Physical parameters that generated this waveform.
    **kwargs : Waveform
        Polarisation components keyed by name (e.g., plus=..., cross=...).
    """

    def __init__(self, parameters: dict | None = None, **kwargs: Waveform):
        self.waveforms: dict[str, Waveform] = kwargs
        self._parameters = parameters or {}

    def __getitem__(self, item: str) -> Waveform:
        return self.waveforms[item]

    def __setitem__(self, item: str, value: Waveform):
        self.waveforms[item] = value

    def __contains__(self, item: str) -> bool:
        return item in self.waveforms

    def __iter__(self) -> Iterator[str]:
        return iter(self.waveforms)

    def __repr__(self) -> str:
        return f"<WaveformDict components={list(self.waveforms.keys())}>"

    @property
    def times(self) -> np.ndarray:
        if "plus" in self.waveforms:
            return self.waveforms["plus"].times
        first_key = next(iter(self.waveforms))
        return self.waveforms[first_key].times

    @property
    def parameters(self) -> dict:
        return self._parameters

    @parameters.setter
    def parameters(self, value: dict):
        self._parameters = value

    @property
    def hrss(self) -> np.ndarray:
        """Root-sum-square strain amplitude."""
        if "plus" in self.waveforms and "cross" in self.waveforms:
            return np.sqrt(
                self.waveforms["plus"].data ** 2
                + self.waveforms["cross"].data ** 2
            )
        raise ValueError("Need both plus and cross polarisations for hrss")
