"""
Demodulated-residual GP surrogate.

Like `DeltaGPSurrogate`, the GP does not model the waveform itself: a
cheap, exactly-evaluable *reference approximant* (normally IMRPhenomXAS) is
generated at predict time, and two GPs fit only the residual between the
training oracle (e.g. IMRPhenomD) and that reference. The difference from
`DeltaGPSurrogate` is the *representation* of that residual.

`DeltaGPSurrogate` decomposes the residual into (delta-log-amplitude,
delta-phase) and reconstructs strain through exp/cos/sin, so its covariance
is only a first-order (delta-method) approximation. `DemodGPSurrogate`
instead **heterodynes** (demodulates) the complex strain residual by the
reference phase:

    z(t) = (h_D(t) - h_XAS(t)) * exp(+i Phi_ref(t))
         ~= A_XAS(t) * (delta(t) - i * deltaPhi(t))

Multiplying by e^{+i Phi_ref} rotates out the carrier oscillation, leaving
Re(z), Im(z) *smooth and non-oscillatory* -- and, unlike the phase-amplitude
decomposition, Cartesian and bounded (no log, no phase unwrapping). Two GPs
fit Re(z), Im(z); the strain is reconstructed as

    h_plus_res  = Re(z) * cos Phi_ref + Im(z) * sin Phi_ref
    h_cross_res = Re(z) * sin Phi_ref - Im(z) * cos Phi_ref
    h_plus  = h_XAS_plus  + h_plus_res
    h_cross = h_XAS_cross + h_cross_res

which is **linear** in the GP outputs, so the plus/cross covariance is an
*exact* congruence of the (independent) Re/Im GP covariances -- not the
delta-method approximation `DeltaGPSurrogate`/`PhaseAmplitudeGPSurrogate`
use. Phi_ref is the reference approximant's phase minus a constant
`phase_correction` (the phi_ref/f_ref convention offset between the
reference and oracle approximant families -- the same constant
`LALApproximantPlusMean`/`CrossMean` apply, see mean.py). This is the
standard relative-binning / heterodyned-likelihood transform, applied here
to the GP target rather than the likelihood.

Empirically (dense30, IMRPhenomD oracle / IMRPhenomXAS reference) this gives
the lowest mismatch of the three residual representations (~1e-5, vs
exact-XAS ~3e-3 and phase-amplitude ~8e-5), because de-oscillated targets let
the mass-ratio kernel train off its lengthscale floor. Caveat: because the
targets are so smooth, the GPs interpolate the training nodes very
confidently and the returned covariance K under-reports the true
between-node bias -- fine for point-estimate PE at realistic SNR (surrogate
error << statistical width) but not a calibrated uncertainty at high SNR.
See CLAUDE.md / the `exact_xas_mismatch_residual` memory.

Implementation: a `DemodGPSurrogate` composes one inner `ExactGPSurrogate`
whose "plus"/"cross" channels ARE the Re(z)/Im(z) GPs (ZeroMean), reusing
all of its training / float64-predict / Cholesky machinery, plus a reference
approximant evaluator (delta.py's `_ApproximantEvaluator`, which accepts a
registry name or any `time_domain` object and decomposes phase on the dense
native grid, immune to sparse-sample unwrap aliasing). Only mass_ratio and
time are modelled; total_mass/distance are handled by the same rescaling the
other surrogates use.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..base import WaveformSurrogate
from ...types import Waveform, WaveformDict
from ..warping import (
    get_warping,
    SimpleWarping,
    ChirpTimeWarping,
    MassRatioChirpTimeWarping,
)
from .exact import ExactGPSurrogate
from .delta import _ApproximantEvaluator

logger = logging.getLogger("heron.models.gp.demod")


class DemodGPSurrogate(WaveformSurrogate):
    """Demodulated-residual GP waveform surrogate.

    Fits two GPs to the real/imaginary parts of the strain residual between
    an oracle approximant (the training data) and a reference approximant,
    heterodyned by the reference phase so the targets are smooth. The
    reference is evaluated exactly at predict time and the residual added
    back with exact (linear) covariance propagation. See the module
    docstring for the mathematics and motivation.

    Parameters mirror `ExactGPSurrogate` (the inner Re/Im engine), plus:

    base_approximant : str or approximant instance
        The reference model, generated exactly and heterodyned against.
        A string is resolved from `heron.models.lalsimulation`; any object
        with the ``time_domain`` interface is accepted directly. Normally
        ``"IMRPhenomXAS"``.
    oracle_approximant : str or None
        Name of the approximant that produced the training data. Used only
        to auto-compute ``phase_correction`` when it is not given (needs
        both this and ``base_approximant`` to be registry names). Metadata
        otherwise.
    phase_correction : float or None
        Constant phase offset (radians) subtracted from the reference phase
        before heterodyning, absorbing the phi_ref/f_ref convention
        difference between the reference and oracle approximant families
        (e.g. IMRPhenomXAS vs IMRPhenomD ~ -2.15 rad). If None, computed via
        `heron.models.gp.mean.compute_phase_correction` when both
        approximants are registry names, else 0.0.
    f_low : float
        Low-frequency cutoff for reference-waveform generation; match the
        training data's ``f_min``.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y_plus: torch.Tensor,
        train_y_cross: torch.Tensor,
        base_approximant="IMRPhenomXAS",
        oracle_approximant: str | None = "IMRPhenomD",
        phase_correction: float | None = None,
        warping: str = "chirp",
        nu: float = 2.5,
        output_scale: float = 1e27,
        device: str = "cpu",
        total_mass: float = 60.0,
        distance: float = 100.0,
        f_low: float = 20.0,
        training_iterations: int = 200,
        optimizer: str = "lbfgs",
        lr: float | None = None,
        ls_min_time: float = 0.0005,
        ls_min_q: float = 0.0005,
        noise_floor_rel: float = 1e-6,
        cholesky_size: int = 2000,
    ):
        self._device = torch.device(device)
        self.output_scale = output_scale
        self.nu = nu
        self.mass_factor = total_mass
        self.distance_factor = distance
        self.f_low = f_low
        self.ls_min_time = ls_min_time
        self.ls_min_q = ls_min_q
        self.noise_floor_rel = noise_floor_rel
        self.cholesky_size = cholesky_size

        if isinstance(warping, str):
            self.warping = get_warping(warping)
        else:
            self.warping = warping

        # Reference approximant evaluator (native-grid logA/phase; accepts a
        # registry name or an instance stub).
        self._base_name = (
            base_approximant if isinstance(base_approximant, str)
            else type(base_approximant).__name__
        )
        self._oracle_name = oracle_approximant
        self._ref = _ApproximantEvaluator(
            base_approximant, total_mass=total_mass, distance=distance, f_low=f_low
        )

        self.phase_correction = self._resolve_phase_correction(
            phase_correction, base_approximant, oracle_approximant,
            total_mass, distance, f_low,
        )

        # Raw oracle strain retained for checkpointing; the demod targets are
        # re-derived deterministically on load (same pattern as the other GP
        # surrogates).
        self._train_x_raw = train_x.clone()
        self._train_y_plus_raw = train_y_plus.clone()
        self._train_y_cross_raw = train_y_cross.clone()

        # Demodulated targets at the (physical) training coordinates.
        q_np = train_x[:, 0].detach().cpu().numpy().astype(np.float64)
        t_np = train_x[:, -1].detach().cpu().numpy().astype(np.float64)
        Dp = train_y_plus.detach().cpu().numpy().astype(np.float64)
        Dc = train_y_cross.detach().cpu().numpy().astype(np.float64)
        hXp, hXc, cosP, sinP = self._reference(q_np, t_np)
        rp = Dp - hXp
        rc = Dc - hXc
        re_z = rp * cosP + rc * sinP
        im_z = rp * sinP - rc * cosP
        logger.info(
            "Demod targets: |D-XAS_plus| median %.3g, Re(z) std %.3g, Im(z) std %.3g "
            "(reference=%s, phase_correction=%.4f rad)",
            float(np.median(np.abs(rp))), float(re_z.std()), float(im_z.std()),
            self._base_name, self.phase_correction,
        )

        # Inner engine: the "plus"/"cross" channels are Re(z)/Im(z). Fitting
        # small, smooth residuals -- ZeroMean, output_scale rescales them the
        # same way it rescales raw strain for ExactGPSurrogate.
        self._gp = ExactGPSurrogate(
            train_x=train_x,
            train_y_plus=torch.tensor(re_z, dtype=torch.float32),
            train_y_cross=torch.tensor(im_z, dtype=torch.float32),
            warping=self.warping,
            nu=nu,
            output_scale=output_scale,
            device=device,
            mean_module=None,
            total_mass=total_mass,
            distance=distance,
            training_iterations=training_iterations,
            optimizer=optimizer,
            lr=lr,
            ls_min_time=ls_min_time,
            ls_min_q=ls_min_q,
            noise_floor_rel=noise_floor_rel,
            cholesky_size=cholesky_size,
        )

    # -- reference / phase-correction helpers ------------------------------

    def _resolve_phase_correction(
        self, phase_correction, base_approximant, oracle_approximant,
        total_mass, distance, f_low,
    ) -> float:
        if phase_correction is not None:
            return float(phase_correction)
        if isinstance(base_approximant, str) and isinstance(oracle_approximant, str):
            from .mean import compute_phase_correction

            c = compute_phase_correction(
                mean_approximant=base_approximant,
                target_approximant=oracle_approximant,
                total_mass=total_mass, distance=distance, f_low=f_low,
            )
            logger.info(
                "Computed phase_correction %.4f rad (reference=%s vs oracle=%s)",
                c, base_approximant, oracle_approximant,
            )
            return float(c)
        logger.warning(
            "phase_correction not given and base/oracle are not both registry "
            "names; defaulting to 0.0 rad."
        )
        return 0.0

    def _reference(self, q_col: np.ndarray, t_col: np.ndarray):
        """Reference plus/cross strain and heterodyne cos/sin at physical
        (mass_ratio, time) points, at the reference distance.

        cos/sin come straight from the reference phase (minus
        ``phase_correction``); strain is zeroed outside each mass ratio's
        native support (raw strain decays there, unlike the phase, which is
        only used to multiply the near-zero out-of-support residual).
        Returns ``(h_plus, h_cross, cos, sin)`` as float64 arrays.
        """
        q_col = np.asarray(q_col, dtype=np.float64)
        t_col = np.asarray(t_col, dtype=np.float64)
        hXp = np.zeros_like(t_col)
        hXc = np.zeros_like(t_col)
        cosP = np.ones_like(t_col)
        sinP = np.zeros_like(t_col)
        for qv in np.unique(q_col):
            m = q_col == qv
            log_amp, phase = self._ref.log_amplitude_phase(qv, t_col[m])
            t0, t1 = self._ref.support(qv)
            in_sup = (t_col[m] >= t0) & (t_col[m] <= t1)
            corrected = phase - self.phase_correction
            c = np.cos(corrected)
            s = np.sin(corrected)
            amp = np.exp(log_amp)
            cosP[m] = c
            sinP[m] = s
            hXp[m] = np.where(in_sup, amp * c, 0.0)
            hXc[m] = np.where(in_sup, amp * s, 0.0)
        return hXp, hXc, cosP, sinP

    # -- prediction --------------------------------------------------------

    def predict(self, parameters: dict) -> WaveformDict:
        """Generate waveform with (exact-linear) systematics uncertainty.

        Same interface as the other surrogates: ``parameters`` must contain
        'mass_ratio' and 'time' (dict with lower/upper/number) or 'times'
        (array). Optional: 'total_mass', 'luminosity_distance'.
        """
        mass_ratio = float(parameters["mass_ratio"])
        distance = parameters.get("luminosity_distance", self.distance_factor)
        distance_factor = distance / self.distance_factor

        # Predict Re(z)/Im(z) at the reference distance (strip distance so the
        # inner surrogate does not also apply it -- distance is handled once,
        # below, uniformly across reference + residual).
        inner_params = {
            k: v for k, v in parameters.items() if k != "luminosity_distance"
        }
        wf = self._gp.predict(inner_params)
        re_z = wf["plus"].data
        im_z = wf["cross"].data
        cov_re = wf["plus"].covariance
        cov_im = wf["cross"].covariance
        times_np = wf["plus"].times

        # Reference at the same physical times.
        q_arr = np.full(len(times_np), mass_ratio, dtype=np.float64)
        hXp, hXc, cosP, sinP = self._reference(q_arr, times_np)

        h_plus = (hXp + re_z * cosP + im_z * sinP) / distance_factor
        h_cross = (hXc + re_z * sinP - im_z * cosP) / distance_factor

        # Exact linear covariance propagation. Re(z), Im(z) are independent
        # GPs (zero cross-covariance), so only the auto-covariances appear:
        #   Cov(h_plus)  = C C^T . cov_re + S S^T . cov_im
        #   Cov(h_cross) = S S^T . cov_re + C C^T . cov_im
        # with C=cos, S=sin (outer products). The reference is exact and adds
        # no covariance.
        cc = np.outer(cosP, cosP)
        ss = np.outer(sinP, sinP)
        cov_plus = (cc * cov_re + ss * cov_im) / distance_factor**2
        cov_cross = (ss * cov_re + cc * cov_im) / distance_factor**2

        output = WaveformDict(
            parameters={
                k: v for k, v in parameters.items() if k not in ("time", "times")
            }
        )
        output["plus"] = Waveform(data=h_plus, times=times_np, covariance=cov_plus)
        output["cross"] = Waveform(data=h_cross, times=times_np, covariance=cov_cross)
        return output

    # -- persistence -------------------------------------------------------

    def _warping_config(self) -> dict:
        warping = self.warping
        if isinstance(warping, SimpleWarping):
            return {"type": "simple", "scale": warping.scale}
        if isinstance(warping, MassRatioChirpTimeWarping):
            return {
                "type": "chirp_adaptive",
                "alpha": warping.alpha,
                "t_ref": warping.t_ref,
                "ref_mass_ratio": warping.ref_mass_ratio,
            }
        if isinstance(warping, ChirpTimeWarping):
            return {"type": "chirp", "alpha": warping.alpha, "t_ref": warping.t_ref}
        return {"type": str(type(warping).__name__)}

    def save(self, path: str | Path) -> None:
        """Save checkpoint.

        The reference approximant is recorded by name; the demod targets are
        re-derived on load from the stored raw oracle strain and the rebuilt
        reference (with the stored ``phase_correction``, so no recompute).
        """
        import datetime
        from heron import __version__ as heron_version

        checkpoint = {
            "format_version": 1,
            "heron_version": heron_version,
            "model_class": type(self).__name__,
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
            "parameter_names": self.parameter_names,
            "model_states": {
                name: model.state_dict()
                for name, model in self._gp.models.items()
            },
            "base_approximant": self._base_name,
            "oracle_approximant": self._oracle_name,
            "phase_correction": self.phase_correction,
            "f_low": self.f_low,
            "train_x": self._train_x_raw.cpu(),
            "train_y_plus": self._train_y_plus_raw.cpu(),
            "train_y_cross": self._train_y_cross_raw.cpu(),
            "mass_factor": self.mass_factor,
            "distance_factor": self.distance_factor,
            "output_scale": self.output_scale,
            "nu": self.nu,
            "warping": self._warping_config(),
            "ls_min_time": self.ls_min_time,
            "ls_min_q": self.ls_min_q,
            "noise_floor_rel": self.noise_floor_rel,
            "cholesky_size": self.cholesky_size,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (heron {heron_version})")

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        base_approximant=None,
    ) -> "DemodGPSurrogate":
        """Load a pre-trained model from checkpoint.

        Parameters
        ----------
        base_approximant : optional
            Override for the checkpoint's recorded reference approximant
            name -- pass an instance when it is not resolvable from
            `heron.models.lalsimulation` (e.g. a test stub).
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        saved_class = checkpoint.get("model_class")
        if saved_class is not None and saved_class != cls.__name__:
            logger.warning(
                f"Checkpoint was saved by {saved_class} but is being loaded by {cls.__name__}"
            )
        saved_heron = checkpoint.get("heron_version")
        if saved_heron is not None:
            logger.info(
                f"Checkpoint saved with heron {saved_heron} "
                f"on {checkpoint.get('saved_at', 'unknown date')}"
            )

        warp_cfg = checkpoint["warping"]
        warping_obj = get_warping(
            warp_cfg["type"],
            **{k: v for k, v in warp_cfg.items() if k != "type"},
        )

        instance = cls(
            train_x=checkpoint["train_x"],
            train_y_plus=checkpoint["train_y_plus"],
            train_y_cross=checkpoint["train_y_cross"],
            base_approximant=(
                base_approximant if base_approximant is not None
                else checkpoint["base_approximant"]
            ),
            oracle_approximant=checkpoint.get("oracle_approximant"),
            phase_correction=checkpoint.get("phase_correction", 0.0),
            warping=warping_obj,
            nu=checkpoint["nu"],
            output_scale=checkpoint["output_scale"],
            device=device,
            total_mass=checkpoint["mass_factor"],
            distance=checkpoint["distance_factor"],
            f_low=checkpoint.get("f_low", 20.0),
            training_iterations=0,
            ls_min_time=checkpoint.get("ls_min_time", 0.0005),
            ls_min_q=checkpoint.get("ls_min_q", 0.0005),
            noise_floor_rel=checkpoint.get("noise_floor_rel", 1e-6),
            cholesky_size=checkpoint.get("cholesky_size", 2000),
        )

        for name, state in checkpoint["model_states"].items():
            instance._gp.models[name].load_state_dict(state)
            instance._gp.models[name].eval()
            instance._gp.models[name].likelihood.eval()
        instance._gp._predict_models = None  # invalidate float64 predict cache

        logger.info(f"Loaded checkpoint from {path}")
        return instance

    # -- interface ---------------------------------------------------------

    @property
    def parameter_names(self) -> list[str]:
        return ["mass_ratio"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        q_vals = self._train_x_raw[:, 0]
        return {"mass_ratio": (float(q_vals.min()), float(q_vals.max()))}
