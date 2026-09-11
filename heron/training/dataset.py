"""
Training set abstraction.

TrainingSet is the universal format that all data sources produce
and all models consume. Models see tensors, not data sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass
class TrainingSet:
    """A training dataset for waveform surrogate models.

    Parameters
    ----------
    x : Tensor, shape (N, D)
        Input coordinates. Columns are physical parameters followed by
        time as the last column. E.g., for non-spinning: (mass_ratio, time).
    y_plus : Tensor, shape (N,)
        Plus polarisation strain values.
    y_cross : Tensor, shape (N,)
        Cross polarisation strain values.
    parameter_names : list[str]
        Names of the physical parameter columns (excluding time).
    metadata : dict
        Arbitrary metadata (source approximant, mass ranges, etc.).
    """

    x: torch.Tensor
    y_plus: torch.Tensor
    y_cross: torch.Tensor
    parameter_names: list[str] = field(default_factory=lambda: ["mass_ratio"])
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def n_parameters(self) -> int:
        """Number of physical parameters (excluding time)."""
        return self.x.shape[1] - 1

    @property
    def parameter_values(self) -> torch.Tensor:
        """Physical parameter columns (excluding time)."""
        return self.x[:, :-1]

    @property
    def times(self) -> torch.Tensor:
        """Time column."""
        return self.x[:, -1]

    def append(self, other: TrainingSet) -> TrainingSet:
        """Concatenate another training set onto this one.

        Used during active learning to grow the dataset incrementally.
        """
        return TrainingSet(
            x=torch.cat([self.x, other.x], dim=0),
            y_plus=torch.cat([self.y_plus, other.y_plus], dim=0),
            y_cross=torch.cat([self.y_cross, other.y_cross], dim=0),
            parameter_names=self.parameter_names,
            metadata={**self.metadata, "appended": True},
        )

    def save(self, path: str | Path) -> None:
        """Save to HDF5."""
        path = Path(path)
        with h5py.File(path, "w") as f:
            f.create_dataset("x", data=self.x.numpy())
            f.create_dataset("y_plus", data=self.y_plus.numpy())
            f.create_dataset("y_cross", data=self.y_cross.numpy())
            f.attrs["parameter_names"] = self.parameter_names
            for k, v in self.metadata.items():
                if isinstance(v, (int, float, str)):
                    f.attrs[k] = v

    @classmethod
    def load(cls, path: str | Path) -> TrainingSet:
        """Load from HDF5."""
        path = Path(path)
        with h5py.File(path, "r") as f:
            x = torch.tensor(f["x"][:], dtype=torch.float32)
            y_plus = torch.tensor(f["y_plus"][:], dtype=torch.float32)
            y_cross = torch.tensor(f["y_cross"][:], dtype=torch.float32)
            parameter_names = list(f.attrs.get("parameter_names", ["mass_ratio"]))
            metadata = {k: v for k, v in f.attrs.items() if k != "parameter_names"}
        return cls(
            x=x,
            y_plus=y_plus,
            y_cross=y_cross,
            parameter_names=parameter_names,
            metadata=metadata,
        )
