"""
Waveform mismatch computation and evaluation.

Mismatch = 1 - overlap is the standard GW metric for surrogate
faithfulness. A surrogate is detection-grade at mismatch < 1e-3
and PE-grade at mismatch < 1e-2 (both defined relative to the
aLIGO design-sensitivity PSD).

``compute_overlap`` computes the full fitting factor: maximised over
constant phase shift (cheap) and time shift within a ±50 ms window
(FFT-based). This matches the standard used in GW data analysis.
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
    maximize_phase: bool = True,
    maximize_time: bool = True,
    max_time_shift: float = 0.05,
) -> float:
    """Compute the noise-weighted overlap (fitting factor) between two waveforms.

    overlap = max_{Δt, φ} <h1 | h2(Δt, φ)> / sqrt(<h1|h1> <h2|h2>)

    where <a|b> = 4 Re ∫ ã(f) b̃*(f) / Sn(f) df.

    Parameters
    ----------
    h1, h2 : ndarray, shape (N,)
        Time-domain waveforms (must be same length).
    dt : float
        Sample spacing in seconds.
    psd : ndarray or None
        One-sided PSD at the rfft frequencies (length N//2 + 1).
        ``None`` → flat (white noise) weighting.
    maximize_phase : bool
        If True, maximise over constant phase rotation (takes ``abs()``
        of the complex inner product). Default True.
    maximize_time : bool
        If True, maximise over time shifts within ±*max_time_shift* seconds
        using an FFT convolution. Default True.
    max_time_shift : float
        Maximum allowed time shift in seconds. Default 0.05 (50 ms).

    Returns
    -------
    float
        Overlap in [0, 1].
    """
    n = len(h1)
    if len(h2) != n:
        raise ValueError("h1 and h2 must have the same length")

    h1_f = rfft(h1)
    h2_f = rfft(h2)
    freqs = rfftfreq(n, d=dt)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    if psd is not None:
        if len(psd) != len(freqs):
            raise ValueError("PSD length must match rfft output")
        inv_psd = np.where(np.isfinite(psd) & (psd > 0), 1.0 / psd, 0.0)
    else:
        inv_psd = np.ones(len(freqs))

    def _inner(a, b):
        # Trapezoidal rule: DC (k=0) and Nyquist (k=N/2) each appear only once
        # in the two-sided spectrum, so they get half weight. Interior bins are
        # doubled (positive + negative frequency). This is exactly consistent
        # with the irfft-based time-shift computation below.
        z = (a * np.conj(b) * inv_psd).real
        return 4.0 * df * (z[0] / 2.0 + np.sum(z[1:-1]) + z[-1] / 2.0)

    norm = np.sqrt(_inner(h1_f, h1_f) * _inner(h2_f, h2_f))
    if norm == 0.0:
        return 0.0

    integrand = h1_f * np.conj(h2_f) * inv_psd

    if maximize_time:
        # Complex correlation over the one-sided (positive-frequency)
        # spectrum: z(tau) = 4 df sum_f integrand_f e^{2 pi i f tau}. Its
        # modulus at each lag is the inner product maximised analytically
        # over a constant phase rotation. This must NOT be computed with
        # irfft: irfft imposes conjugate symmetry (a real output), i.e. it
        # returns only Re z(tau), and |Re z| equals |z| only when the
        # optimal phase is 0 or pi. For waveform pairs with a genuinely
        # nonzero best-fit phase offset (e.g. IMRPhenomD vs IMRPhenomXAS,
        # ~2 rad), the irfft version undervalued the overlap so badly it
        # reported 23% mismatch where the true fitting-factor mismatch is
        # ~0.1% (bug found+fixed 2026-07-15).
        weighted = integrand.copy()
        # Same trapezoidal DC/Nyquist half-weighting as _inner, so the
        # zero-lag value equals the maximize_time=False branch exactly.
        weighted[0] *= 0.5
        if n % 2 == 0:
            weighted[-1] *= 0.5
        spectrum = np.zeros(n, dtype=complex)
        spectrum[: len(weighted)] = weighted
        z_t = np.fft.ifft(spectrum) * n * 4.0 * df
        n_shift = max(1, int(round(max_time_shift / dt)))
        # Search within ±n_shift samples (circular, so check both ends)
        window = np.concatenate([z_t[:n_shift + 1], z_t[-(n_shift):]])
        peak = (
            float(np.max(np.abs(window)))
            if maximize_phase
            else float(np.max(window.real))
        )
    else:
        # Use same trapezoidal weighting as _inner: half weight at DC and Nyquist
        inner_val = 4.0 * df * (integrand[0] / 2.0 + np.sum(integrand[1:-1]) + integrand[-1] / 2.0)
        peak = float(abs(inner_val)) if maximize_phase else float(inner_val.real)

    return peak / norm


def compute_mismatch(
    h1: np.ndarray,
    h2: np.ndarray,
    dt: float,
    psd: np.ndarray | None = None,
    maximize_phase: bool = True,
    maximize_time: bool = True,
) -> float:
    """Compute mismatch = 1 - overlap (fitting factor)."""
    return 1.0 - compute_overlap(
        h1, h2, dt, psd,
        maximize_phase=maximize_phase,
        maximize_time=maximize_time,
    )


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
        valid = self.mismatches[np.isfinite(self.mismatches)]
        if len(valid) > 0:
            self.worst_mismatch = float(np.max(valid))
            self.median_mismatch = float(np.median(valid))
            self.fraction_below_1e3 = float(np.mean(valid < 1e-3))
            self.fraction_below_1e2 = float(np.mean(valid < 1e-2))

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
    psd : ndarray, "aligo", or None
        Power spectral density for noise-weighted overlap.
        ``"aligo"`` (default) uses the aLIGO Zero-Det High-Power curve, which
        is required for the mismatch thresholds (1e-3, 1e-2) to be meaningful.
        ``None`` → flat weighting (unphysical but useful for unit tests).
    """

    def __init__(self, surrogate, reference, psd: np.ndarray | str | None = "aligo"):
        self.surrogate = surrogate
        self.reference = reference
        self._psd_spec = psd   # resolved lazily once we have the frequency grid

    def _resolve_psd(self, freqs: np.ndarray) -> np.ndarray | None:
        if self._psd_spec is None:
            return None
        if isinstance(self._psd_spec, str) and self._psd_spec == "aligo":
            from .psd import aligo_design_psd
            return aligo_design_psd(freqs)
        return np.asarray(self._psd_spec)

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
        psd = None  # resolved after first waveform

        for i in range(n_points):
            params = {name: float(samples[name][i]) for name in param_names}
            params["time"] = time_config

            try:
                surr_wf = self.surrogate.predict(params)
                surr_plus = surr_wf["plus"].data
                dt = surr_wf["plus"].dt

                ref_params = _build_reference_params(self.surrogate, params)
                ref_wf = self.reference.time_domain(ref_params, times=surr_wf["plus"].times)
                ref_plus = ref_wf["plus"].data

                # Resolve PSD once we have the frequency grid
                if psd is None:
                    n = min(len(surr_plus), len(ref_plus))
                    freqs = rfftfreq(n, d=dt)
                    psd = self._resolve_psd(freqs)

                n = min(len(surr_plus), len(ref_plus))
                mm = compute_mismatch(surr_plus[:n], ref_plus[:n], dt, psd)
                mismatches.append(mm)

            except Exception as e:
                logger.warning(f"Mismatch computation failed at {params}: {e}")
                mismatches.append(np.nan)

        return MismatchResult(
            mismatches=np.array(mismatches),
            parameters=samples,
        )
