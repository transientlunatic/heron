"""K-calibration metrics: is the predicted variance the right size?

Two complementary views (B0 measurement 2):

* per-sample / integrated ``var / err^2`` (what ``calibrate_demod_k.py`` does);
* whitened ``chi2 = r^T K^-1 r`` against its expected chi-squared law, which
  is what matters when K has off-diagonal terms.

Gotcha, built into the summary: for a *correct* Gaussian the per-sample
``var/err^2`` is not centred on 1 -- ``err^2/var ~ chi2_1`` has median 0.4549,
so the median ratio is ~2.198.  The unbiased calibration statistic is the
*integrated* ratio ``sum(var)/sum(err^2)``, which tends to 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

CHI2_1_MEDIAN = float(stats.chi2.median(1))     # 0.4549...
EXPECTED_MEDIAN_RATIO = 1.0 / CHI2_1_MEDIAN     # ~2.198 for well-calibrated K


@dataclass
class CalibrationSummary:
    n: int
    integrated_ratio: float      # sum(var)/sum(err^2); 1 = calibrated
    median_ratio: float          # median(var/err^2); ~EXPECTED_MEDIAN_RATIO if calibrated
    expected_median_ratio: float = EXPECTED_MEDIAN_RATIO
    frac_within_2sigma: float = float("nan")   # should be ~0.954


def calibration_summary(var, err, floor: float = 1e-300) -> CalibrationSummary:
    var = np.asarray(var, float).ravel()
    e2 = np.asarray(err, float).ravel() ** 2
    return CalibrationSummary(
        n=len(var),
        integrated_ratio=float(var.sum() / max(e2.sum(), floor)),
        median_ratio=float(np.median(var / np.maximum(e2, floor))),
        frac_within_2sigma=float(np.mean(e2 <= 4.0 * var)),
    )


def calibration_by_group(var, err, groups) -> dict:
    """Summaries split by a label array (e.g. ``'node'`` vs ``'midpoint'``)."""
    var, err, groups = (np.asarray(x) for x in (var, err, groups))
    return {g: calibration_summary(var[groups == g], err[groups == g])
            for g in np.unique(groups)}


def whitened_chi2(r, K):
    """``(chi2, dof, two_sided_p)`` for residual ``r`` under covariance ``K``."""
    r = np.asarray(r, float)
    L = np.linalg.cholesky(np.asarray(K, float))
    u = np.linalg.solve(L, r)
    chi2, dof = float(u @ u), len(r)
    cdf = stats.chi2.cdf(chi2, dof)
    return chi2, dof, float(2 * min(cdf, 1 - cdf))


def chi2_calibration(residuals, K) -> dict:
    """Whitened chi2 over many residual vectors (rows) vs chi2(dof).

    Returns the mean chi2/dof (should be ~1) and the KS p-value of the
    probability-integral-transformed chi2 values against uniform.
    """
    residuals = np.atleast_2d(residuals)
    L = np.linalg.cholesky(np.asarray(K, float))
    u = np.linalg.solve(L, residuals.T)
    chi2 = np.sum(u * u, axis=0)
    dof = residuals.shape[1]
    pit = stats.chi2.cdf(chi2, dof)
    return {"n": len(chi2), "dof": dof,
            "mean_chi2_per_dof": float(chi2.mean() / dof),
            "ks_pvalue": float(stats.kstest(pit, "uniform").pvalue)}
