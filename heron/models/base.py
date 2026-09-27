"""
Abstract base classes for waveform surrogate models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..types import WaveformDict


class WaveformSurrogate(ABC):
    """Abstract base for all waveform surrogate models with uncertainty.

    Any surrogate model (GP, neural network, ensemble, etc.) must implement
    this interface. The key contract: predict() returns waveforms with
    covariance matrices, not just point estimates.
    """

    @abstractmethod
    def predict(self, parameters: dict) -> WaveformDict:
        """Generate a waveform with uncertainty at given parameters.

        Parameters
        ----------
        parameters : dict
            Physical parameters. Must include at least ``mass_ratio``.
            May include ``total_mass``, ``luminosity_distance``, etc.

        Returns
        -------
        WaveformDict
            Contains Waveform objects for 'plus' and 'cross' polarisations,
            each with ``.data`` (mean), ``.covariance``, and ``.variance``.
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model state to a checkpoint file."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> WaveformSurrogate:
        """Load a pre-trained model from a checkpoint file."""
        ...

    def training_time_bounds(self, parameters: dict) -> tuple[float, float] | None:
        """Physical time window ``(t0, t1)`` (seconds, merger-relative) covered
        by the training data at these parameters, or ``None`` if unknown.

        Outside this window a GP surrogate's variance reverts to its prior,
        which says nothing physical about the waveform there; the likelihood
        uses these bounds to taper that variance away (see
        ``NetworkLikelihood(variance_taper=...)``).
        """
        return None

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Names of the physical parameters this model accepts."""
        ...

    @property
    @abstractmethod
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Valid ranges for each parameter, as (lower, upper) tuples."""
        ...
