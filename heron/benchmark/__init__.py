"""Shared benchmark-harness metric code (manager task B0, #641).

Everything here is numpy/scipy (plus torch for the likelihood check) and needs
no lalsuite: the metrics take arrays, so any surrogate/oracle pair can be
scored.  The Asimov A1 commands wrap these functions.
"""
from .calibration import (  # noqa: F401
    CalibrationSummary, calibration_by_group, calibration_summary,
    chi2_calibration, whitened_chi2,
)
from .inference_summary import (  # noqa: F401
    bound_position, compare_with_without_k, multi_seed_shift, node_clustering,
    ns_summary, pp_summary, scaling_benchmark, training_cost_summary,
)
from .likelihood_check import likelihood_agreement  # noqa: F401
from .overlap import subsample_mismatch, subsample_overlap  # noqa: F401
from .testset import TestSet, make_test_set  # noqa: F401
