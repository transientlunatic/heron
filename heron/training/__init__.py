"""Training infrastructure for waveform surrogate models."""

from .dataset import TrainingSet
from .sampling import sobol_sample, latin_hypercube_sample

__all__ = ["TrainingSet", "sobol_sample", "latin_hypercube_sample"]
