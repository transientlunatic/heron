"""Evaluation framework for waveform surrogate models."""

from .mismatch import MismatchEvaluator, MismatchResult, compute_overlap, compute_mismatch
from .calibration import CalibrationEvaluator, CalibrationResult
from .report import EvaluationReport

__all__ = [
    "MismatchEvaluator",
    "MismatchResult",
    "CalibrationEvaluator",
    "CalibrationResult",
    "EvaluationReport",
    "compute_overlap",
    "compute_mismatch",
]
