"""
Uncertainty calibration evaluation.

Tests whether the model's predicted uncertainty is well-calibrated:
does the true waveform fall within the X% credible interval X% of
the time?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..training.sampling import sobol_sample
from .mismatch import _build_reference_params

logger = logging.getLogger("heron.evaluation.calibration")


@dataclass
class CalibrationResult:
    """Results from an uncertainty calibration evaluation."""

    # Z-scores at each (point, time-sample)
    z_scores: np.ndarray

    # Coverage: fraction of truth within credible interval
    coverage: dict[str, float] = field(default_factory=dict)

    # Kolmogorov-Smirnov test against standard normal
    ks_statistic: float = 0.0
    ks_pvalue: float = 0.0

    # Log predictive density (higher = better)
    log_predictive_densities: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_log_pred_density: float = 0.0

    # Per-parameter uncertainty vs mismatch correlation
    uncertainty_mismatch_correlation: float = 0.0

    def __post_init__(self):
        if len(self.z_scores) > 0:
            flat_z = self.z_scores.ravel()
            flat_z = flat_z[np.isfinite(flat_z)]

            if len(flat_z) > 0:
                # Coverage fractions
                for level in [0.5, 0.68, 0.9, 0.95, 0.99]:
                    threshold = stats.norm.ppf((1 + level) / 2)
                    frac = float(np.mean(np.abs(flat_z) < threshold))
                    self.coverage[f"{level:.0%}"] = frac

                # KS test
                ks = stats.kstest(flat_z, "norm")
                self.ks_statistic = float(ks.statistic)
                self.ks_pvalue = float(ks.pvalue)

        if len(self.log_predictive_densities) > 0:
            finite = self.log_predictive_densities[np.isfinite(self.log_predictive_densities)]
            if len(finite) > 0:
                self.mean_log_pred_density = float(np.mean(finite))

    def summary(self) -> str:
        lines = [
            f"Uncertainty calibration ({len(self.z_scores)} waveforms):",
            f"  KS test vs N(0,1):  stat={self.ks_statistic:.4f}, p={self.ks_pvalue:.4f}",
        ]
        for level, frac in sorted(self.coverage.items()):
            lines.append(f"  {level} coverage:    {frac:.1%} (expected {level})")
        if self.mean_log_pred_density != 0:
            lines.append(f"  Mean log pred dens: {self.mean_log_pred_density:.2f}")
        return "\n".join(lines)

    @property
    def qq_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (theoretical quantiles, observed quantiles) for a Q-Q plot."""
        flat_z = np.sort(self.z_scores.ravel())
        flat_z = flat_z[np.isfinite(flat_z)]
        n = len(flat_z)
        theoretical = stats.norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))
        return theoretical, flat_z


class CalibrationEvaluator:
    """Evaluate uncertainty calibration of a surrogate model.

    Parameters
    ----------
    surrogate : WaveformSurrogate
        Model to evaluate.
    reference : WaveformApproximant
        Ground truth waveform generator.
    """

    def __init__(self, surrogate, reference):
        self.surrogate = surrogate
        self.reference = reference

    def evaluate(
        self,
        n_points: int = 100,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        time_config: dict | None = None,
        seed: int | None = None,
    ) -> CalibrationResult:
        """Evaluate uncertainty calibration at held-out points.

        At each parameter point:
        1. Predict waveform + covariance from surrogate
        2. Generate "truth" from reference
        3. Compute z-scores: (truth - mean) / std
        4. Optionally compute log predictive density

        Parameters
        ----------
        n_points : int
            Number of evaluation points.
        parameter_bounds : dict or None
            If None, uses surrogate's own parameter_bounds.
        time_config : dict or None
            Time grid config.
        seed : int or None
            Random seed.

        Returns
        -------
        CalibrationResult
        """
        if parameter_bounds is None:
            parameter_bounds = self.surrogate.parameter_bounds

        if time_config is None:
            time_config = {"lower": -0.5, "upper": 0.02, "number": 256}

        samples = sobol_sample(parameter_bounds, n_points, seed=seed)
        param_names = list(parameter_bounds.keys())

        all_z_scores = []
        log_pred_densities = []

        for i in range(n_points):
            params = {name: float(samples[name][i]) for name in param_names}
            params["time"] = time_config

            try:
                surr_wf = self.surrogate.predict(params)
                ref_params = _build_reference_params(self.surrogate, params)
                ref_wf = self.reference.time_domain(ref_params, times=surr_wf["plus"].times)

                surr_mean = surr_wf["plus"].data
                surr_std = surr_wf["plus"].std
                surr_cov = surr_wf["plus"].covariance
                ref_data = ref_wf["plus"].data

                n = min(len(surr_mean), len(ref_data))
                residual = ref_data[:n] - surr_mean[:n]

                # Z-scores (pointwise)
                if surr_std is not None:
                    std = surr_std[:n]
                    std_safe = np.where(std > 0, std, 1.0)
                    z = residual / std_safe
                    all_z_scores.append(z)
                else:
                    all_z_scores.append(np.full(n, np.nan))

                # Log predictive density (full multivariate normal)
                if surr_cov is not None:
                    cov = surr_cov[:n, :n]
                    # Add small jitter for numerical stability
                    cov_reg = cov + 1e-10 * np.eye(n)
                    try:
                        lpd = stats.multivariate_normal.logpdf(
                            ref_data[:n], mean=surr_mean[:n], cov=cov_reg
                        )
                        log_pred_densities.append(float(lpd))
                    except (np.linalg.LinAlgError, ValueError):
                        log_pred_densities.append(np.nan)
                else:
                    log_pred_densities.append(np.nan)

            except Exception as e:
                logger.warning(f"Calibration eval failed at point {i}: {e}")
                all_z_scores.append(np.array([np.nan]))
                log_pred_densities.append(np.nan)

        # Pad z-score arrays to same length for stacking
        max_len = max(len(z) for z in all_z_scores)
        padded = np.full((len(all_z_scores), max_len), np.nan)
        for i, z in enumerate(all_z_scores):
            padded[i, :len(z)] = z

        return CalibrationResult(
            z_scores=padded,
            log_predictive_densities=np.array(log_pred_densities),
        )
