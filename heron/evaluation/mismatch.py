"""
Waveform mismatch computation and evaluation.

Mismatch = 1 - overlap is the standard GW metric for surrogate
faithfulness. A surrogate is detection-grade at mismatch < 1e-3
and PE-grade at mismatch < 1e-2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import rfft, rfftfreq

from ..training.sampling import sobol_sample

logger = logging.getLogger("heron.evaluation.mismatch")


def _build_reference_params(surrogate, sampled_params: dict) -> dict:
    """Build a parameter dict suitable for the reference approximant.

    The surrogate stores the total_mass and distance it was trained at.
    The reference approximant needs these (with astropy units) to convert
    mass_ratio → m1, m2 and to set the luminosity distance.
    """
    from astropy import units as u

    ref_params = dict(sampled_params)

    # Add total_mass and distance from the surrogate's training config
    if hasattr(surrogate, "mass_factor") and "total_mass" not in ref_params:
        ref_params["total_mass"] = surrogate.mass_factor * u.solMass
    if hasattr(surrogate, "distance_factor") and "luminosity_distance" not in ref_params:
        ref_params["luminosity_distance"] = surrogate.distance_factor * u.Mpc

    return ref_params


def compute_overlap(
    h1: np.ndarray,
    h2: np.ndarray,
    dt: float,
    psd: np.ndarray | None = None,
) -> float:
    """Compute the noise-weighted overlap between two waveforms.

    overlap = <h1|h2> / sqrt(<h1|h1> <h2|h2>)

    where <a|b> = 4 Re ∫ a~(f) b~*(f) / Sn(f) df

    Parameters
    ----------
    h1, h2 : ndarray, shape (N,)
        Time-domain waveforms (must be same length).
    dt : float
        Sample spacing in seconds.
    psd : ndarray or None
        One-sided power spectral density at the FFT frequencies.
        If None, uses flat (white noise) weighting.

    Returns
    -------
    float
        Overlap in [0, 1] (or slightly outside due to numerics).
    """
    n = len(h1)
    assert len(h2) == n, "Waveforms must have the same length"

    h1_f = rfft(h1)
    h2_f = rfft(h2)
    freqs = rfftfreq(n, d=dt)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    if psd is not None:
        assert len(psd) == len(freqs), "PSD must match FFT frequency grid"
        # Avoid division by zero at DC and very low frequencies
        inv_psd = np.zeros_like(psd)
        mask = psd > 0
        inv_psd[mask] = 1.0 / psd[mask]
    else:
        inv_psd = np.ones(len(freqs))

    inner = lambda a, b: 4.0 * df * np.sum((a * np.conj(b) * inv_psd).real)

    norm = np.sqrt(inner(h1_f, h1_f) * inner(h2_f, h2_f))
    if norm == 0:
        return 0.0

    return inner(h1_f, h2_f) / norm


def compute_mismatch(
    h1: np.ndarray,
    h2: np.ndarray,
    dt: float,
    psd: np.ndarray | None = None,
) -> float:
    """Compute mismatch = 1 - overlap."""
    return 1.0 - compute_overlap(h1, h2, dt, psd)


@dataclass
class MismatchResult:
    """Results from a mismatch evaluation."""
    mismatches: np.ndarray
    parameters: dict[str, np.ndarray]
    worst_mismatch: float = 0.0
    worst_parameters: dict[str, float] = field(default_factory=dict)
    median_mismatch: float = 0.0
    fraction_below_1e3: float = 0.0
    fraction_below_1e2: float = 0.0

    def __post_init__(self):
        if len(self.mismatches) > 0:
            self.worst_mismatch = float(np.max(self.mismatches))
            self.median_mismatch = float(np.median(self.mismatches))
            self.fraction_below_1e3 = float(np.mean(self.mismatches < 1e-3))
            self.fraction_below_1e2 = float(np.mean(self.mismatches < 1e-2))

            worst_idx = int(np.argmax(self.mismatches))
            self.worst_parameters = {
                name: float(vals[worst_idx])
                for name, vals in self.parameters.items()
            }

    def summary(self) -> str:
        lines = [
            f"Mismatch evaluation ({len(self.mismatches)} points):",
            f"  Median mismatch:  {self.median_mismatch:.2e}",
            f"  Worst mismatch:   {self.worst_mismatch:.2e}",
            f"  Worst at:         {self.worst_parameters}",
            f"  < 1e-3 (detect):  {self.fraction_below_1e3:.1%}",
            f"  < 1e-2 (PE):      {self.fraction_below_1e2:.1%}",
        ]
        return "\n".join(lines)


class MismatchEvaluator:
    """Evaluate a surrogate model against a reference via mismatch distributions.

    Parameters
    ----------
    surrogate : WaveformSurrogate
        The model to evaluate.
    reference : WaveformApproximant
        The reference waveform generator (ground truth).
    psd : ndarray or None
        Power spectral density for noise-weighted overlap.
        None → flat (white noise) weighting.
    """

    def __init__(self, surrogate, reference, psd: np.ndarray | None = None):
        self.surrogate = surrogate
        self.reference = reference
        self.psd = psd

    def evaluate(
        self,
        n_points: int = 100,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        time_config: dict | None = None,
        seed: int | None = None,
    ) -> MismatchResult:
        """Compute mismatch distribution at held-out parameter points.

        Parameters
        ----------
        n_points : int
            Number of evaluation points.
        parameter_bounds : dict or None
            Bounds for parameter sampling. If None, uses surrogate's
            own parameter_bounds.
        time_config : dict or None
            Time grid config (lower, upper, number). If None, uses defaults.
        seed : int or None
            Random seed.

        Returns
        -------
        MismatchResult
        """
        if parameter_bounds is None:
            parameter_bounds = self.surrogate.parameter_bounds

        if time_config is None:
            time_config = {"lower": -0.5, "upper": 0.02, "number": 512}

        samples = sobol_sample(parameter_bounds, n_points, seed=seed)
        param_names = list(parameter_bounds.keys())

        mismatches = []

        for i in range(n_points):
            params = {name: float(samples[name][i]) for name in param_names}
            params["time"] = time_config

            try:
                # Get surrogate prediction
                surr_wf = self.surrogate.predict(params)
                surr_plus = surr_wf["plus"].data
                dt = surr_wf["plus"].dt

                # Get reference waveform at the same times
                # Build params with total_mass/distance from surrogate for LAL
                ref_params = _build_reference_params(self.surrogate, params)
                ref_wf = self.reference.time_domain(ref_params, times=surr_wf["plus"].times)
                ref_plus = ref_wf["plus"].data

                # Ensure same length
                n = min(len(surr_plus), len(ref_plus))
                mm = compute_mismatch(surr_plus[:n], ref_plus[:n], dt, self.psd)
                mismatches.append(mm)

            except Exception as e:
                logger.warning(f"Mismatch computation failed at {params}: {e}")
                mismatches.append(np.nan)

        return MismatchResult(
            mismatches=np.array(mismatches),
            parameters=samples,
        )
