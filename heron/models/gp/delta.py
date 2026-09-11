"""
Delta (residual) GP surrogate.

Instead of fitting the waveform itself, fit the *difference between two
waveform models*. A cheap, exactly-evaluable base approximant (e.g.
IMRPhenomD) provides the waveform; the GP models only the residual between
a higher-fidelity oracle (the training approximant -- e.g. IMRPhenomXAS,
an SEOB model, or NR data) and that base, decomposed as delta-log-amplitude
and delta-phase:

    dlogA(q, t) = log A_oracle(q, t) - log A_base(q, t)
    dPhi(q, t)  = Phi_oracle(q, t) - Phi_base(q, t)   (per-q phase-aligned)

Wherever both models agree on the bulk phasing (both NR-calibrated), the
two targets are small and slowly varying, so a Matern kernel is
structurally well suited to them -- unlike raw strain h(q, t), whose
carrier-frequency oscillation underlies the failure modes documented in
CLAUDE.md (floor-pinned time lengthscales, secular low-q phase drift).

predict() evaluates the base approximant exactly at the requested
parameters (zero interpolation error in the mean's base component), adds
the GP-interpolated deltas, and reconstructs plus/cross strain with
first-order (delta-method) covariance. The returned covariance is the
delta GPs' posterior covariance propagated through the reconstruction --
i.e. uncertainty about the *oracle-base model difference* (waveform
systematics), not surrogate self-interpolation error.

Mathematically this is equivalent to `PhaseAmplitudeGPSurrogate` with
`LALApproximantAmplitudeMean`/`LALApproximantPhaseMean` mean functions.
The differences are engineering, and they matter for diagnosis and for
non-LAL oracles:

- The residual targets are explicit arrays computed once at data-prep
  time (`compute_delta_targets`), so they can be inspected and plotted
  directly, and training never calls the base model inside the loss loop.
- When the oracle is itself a callable approximant (``oracle_approximant``,
  the normal case), both sides of the delta are decomposed on their dense
  native time grids and only *evaluated* at the training coordinates.
  Never unwrap phase from sparse training samples: uniform-in-warped-time
  grids routinely have early-inspiral phase steps exceeding pi, and
  np.unwrap then aliases, silently undercounting cycles -- observed
  directly on a real IMRPhenomXAS/IMRPhenomD run, where a 100-sample grid
  produced a spurious ~500 rad secular "delta" that vanished with dense
  decomposition.
- Per-mass-ratio phase alignment is explicit and configurable
  (``phase_alignment``), rather than relying on the two models' phase
  conventions matching by construction.
- Training rows outside the two models' common time support, or where
  either amplitude falls below ``amp_floor_rel`` of its peak (start-up
  taper, post-ringdown numerical noise -- regions where log-amplitude and
  phase are garbage), are dropped.
- The base approximant can be any object with the ``time_domain``
  interface (including test stubs), not only names registered in
  `heron.models.lalsimulation`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import gpytorch

from ...types import Waveform, WaveformDict
from ..warping import get_warping, SimpleWarping, ChirpTimeWarping, MassRatioChirpTimeWarping
from .exact import _ExactGPModel
from .phase_amplitude import PhaseAmplitudeGPSurrogate, strain_to_amplitude_phase

logger = logging.getLogger("heron.models.gp.delta")


class _ApproximantEvaluator:
    """Evaluate an approximant's log-amplitude and phase at arbitrary
    (mass_ratio, time) points, with per-mass-ratio spline caching.

    Follows the same generate-once-per-q-and-spline pattern (and the same
    ``h_plus - i*h_cross = A*exp(-i*Phi)`` decomposition and unwrap
    anchoring) as `_LALApproximantMeanBase` in mean.py, but accepts either
    a `heron.models.lalsimulation` class name or a ready approximant
    instance. Crucially, the phase is unwrapped on the approximant's dense
    native grid, so it is immune to the sparse-sample unwrap aliasing
    described in the module docstring.
    """

    _MAX_Q_CACHE = 256

    def __init__(self, approximant, total_mass: float, distance: float, f_low: float):
        if isinstance(approximant, str):
            self.approximant_name = approximant
            self._generator = None
        else:
            self.approximant_name = type(approximant).__name__
            self._generator = approximant
        self.total_mass = total_mass
        self.distance = distance
        self.f_low = f_low
        self._q_cache: dict[float, dict] = {}

    def _get_generator(self):
        if self._generator is None:
            from heron.models import lalsimulation as lalsim_models

            cls = getattr(lalsim_models, self.approximant_name, None)
            if cls is None:
                raise ValueError(
                    f"Unknown approximant '{self.approximant_name}' "
                    "(no such class in heron.models.lalsimulation)"
                )
            self._generator = cls()
        return self._generator

    def _get_waveform_data(self, q: float) -> dict:
        key = round(float(q), 12)
        if key not in self._q_cache:
            from astropy import units as u
            from scipy.interpolate import CubicSpline

            params = {
                "mass_ratio": key,
                "total_mass": self.total_mass * u.solMass,
                "luminosity_distance": self.distance * u.Mpc,
                "f_min": self.f_low * u.Hertz,
                "delta_t": (1.0 / 4096) * u.second,
            }
            wf = self._get_generator().time_domain(params)
            times = np.asarray(wf["plus"].times, dtype=np.float64)
            hp = np.asarray(wf["plus"].data, dtype=np.float64)
            hx = np.asarray(wf["cross"].data, dtype=np.float64)

            amplitude = np.sqrt(hp**2 + hx**2)
            phase = np.unwrap(np.arctan2(hx, hp))
            log_amplitude = np.log(amplitude + 1e-30)

            if len(self._q_cache) >= self._MAX_Q_CACHE:
                self._q_cache.pop(next(iter(self._q_cache)))
            self._q_cache[key] = {
                "t0": times[0],
                "t1": times[-1],
                "logA_peak": float(log_amplitude.max()),
                "log_amplitude": CubicSpline(times, log_amplitude),
                "phase": CubicSpline(times, phase),
            }
        return self._q_cache[key]

    def support(self, q: float) -> tuple[float, float]:
        """Native time support (t0, t1) of the waveform at this mass ratio."""
        data = self._get_waveform_data(q)
        return data["t0"], data["t1"]

    def peak_log_amplitude(self, q: float) -> float:
        return self._get_waveform_data(q)["logA_peak"]

    def log_amplitude_phase(self, q: float, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Log-amplitude and unwrapped phase at physical times (float64).

        Times outside the native support are clamped to its edges
        (channels freeze at their boundary values), matching
        `_LALApproximantMeanBase._eval_channel`; callers that need to
        exclude out-of-support points should mask on `support()`.
        """
        data = self._get_waveform_data(q)
        t = np.clip(np.asarray(times, dtype=np.float64), data["t0"], data["t1"])
        return data["log_amplitude"](t), data["phase"](t)


def compute_delta_targets(
    train_x: torch.Tensor,
    train_y_plus: torch.Tensor,
    train_y_cross: torch.Tensor,
    base_evaluator: _ApproximantEvaluator,
    phase_alignment: str = "anchor",
    oracle_evaluator: _ApproximantEvaluator | None = None,
    amp_floor_rel: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn oracle strain training data into delta targets against a base model.

    Evaluates the base model (and, when ``oracle_evaluator`` is given, the
    oracle too) at the training (mass_ratio, time) coordinates and forms
    the log-amplitude and phase residuals.

    Parameters
    ----------
    oracle_evaluator : _ApproximantEvaluator, optional
        When given (the normal case -- the oracle is a callable
        approximant), the oracle's log-amplitude/phase come from its dense
        native-grid decomposition, and ``train_y_plus``/``train_y_cross``
        are not used at all. When None (e.g. ``data``-mode training on NR
        waveforms that cannot be regenerated), the oracle side is
        decomposed from the sparse training samples via
        `strain_to_amplitude_phase` -- beware: if the per-sample phase
        steps exceed pi anywhere (common on uniform-in-warped-time grids),
        the unwrap aliases and the phase residual is silently wrong; the
        >50 rad span warning below is the symptom.
    phase_alignment : str
        Per-mass-ratio treatment of the raw phase difference:

        - ``"anchor"`` (default): subtract each mass ratio's earliest-time
          value, so dPhi starts at exactly 0. Discards the (physically
          meaningless) relative reference phase between the two models and
          protects against non-smooth-in-q unwrap anchoring.
        - ``"branch"``: subtract only the nearest multiple of 2*pi at the
          earliest time -- removes spurious unwrap-branch offsets while
          keeping any genuine sub-2*pi reference-phase difference.
        - ``"none"``: use the raw difference.
    amp_floor_rel : float
        Drop training rows where either model's amplitude is below this
        fraction of its per-mass-ratio peak (start-up taper and
        post-ringdown numerical noise, where log-amplitude and phase are
        garbage). Rows outside the two models' common time support are
        always dropped. Set to 0 to disable the amplitude cut.

    Returns
    -------
    x_kept, delta_log_amplitude, delta_phase : Tensor
        Surviving rows grouped by mass ratio and sorted by time; targets
        aligned with x_kept, in the training dtype.
    """
    if phase_alignment not in ("anchor", "branch", "none"):
        raise ValueError(
            f"Unknown phase_alignment '{phase_alignment}' (use anchor, branch, or none)"
        )

    if oracle_evaluator is None:
        x_sorted, logA_all_t, phase_all_t = strain_to_amplitude_phase(
            train_x, train_y_plus, train_y_cross
        )
        x_np = x_sorted.detach().cpu().numpy().astype(np.float64)
        logA_all = logA_all_t.detach().cpu().numpy().astype(np.float64)
        phase_all = phase_all_t.detach().cpu().numpy().astype(np.float64)
    else:
        # Same grouped-by-q, sorted-by-time row ordering that
        # strain_to_amplitude_phase produces.
        x_np = train_x.detach().cpu().numpy().astype(np.float64)
        order = np.lexsort((x_np[:, -1], x_np[:, 0]))
        x_np = x_np[order]
        logA_all = phase_all = None

    q_col = x_np[:, 0]
    t_col = x_np[:, -1]

    x_chunks, dlogA_chunks, dphi_chunks = [], [], []
    n_dropped = 0
    log_floor = np.log(amp_floor_rel) if amp_floor_rel and amp_floor_rel > 0 else None

    for q in np.unique(q_col):
        m = q_col == q
        t = t_col[m]

        if oracle_evaluator is None:
            logA_o = logA_all[m]
            phase_o = phase_all[m]
            o_t0, o_t1 = -np.inf, np.inf  # sample values are exact at these rows
            logA_o_peak = float(logA_o.max())
        else:
            logA_o, phase_o = oracle_evaluator.log_amplitude_phase(q, t)
            o_t0, o_t1 = oracle_evaluator.support(q)
            logA_o_peak = oracle_evaluator.peak_log_amplitude(q)

        logA_b, phase_b = base_evaluator.log_amplitude_phase(q, t)
        b_t0, b_t1 = base_evaluator.support(q)

        keep = (t >= max(o_t0, b_t0)) & (t <= min(o_t1, b_t1))
        if log_floor is not None:
            keep &= logA_o >= logA_o_peak + log_floor
            keep &= logA_b >= base_evaluator.peak_log_amplitude(q) + log_floor

        n_dropped += int((~keep).sum())
        if not keep.any():
            logger.warning(
                "All %d training rows at q=%.4g were dropped (outside the "
                "models' common support or below the amplitude floor)",
                len(t), q,
            )
            continue

        dlogA = logA_o[keep] - logA_b[keep]
        dp = phase_o[keep] - phase_b[keep]
        # Rows within each q group are time-sorted, so dp[0] is the
        # earliest-time (lowest-frequency) sample -- the natural anchor.
        if phase_alignment == "anchor":
            dp = dp - dp[0]
        elif phase_alignment == "branch":
            dp = dp - 2.0 * np.pi * np.round(dp[0] / (2.0 * np.pi))

        x_chunks.append(x_np[m][keep])
        dlogA_chunks.append(dlogA)
        dphi_chunks.append(dp)

    if not x_chunks:
        raise ValueError(
            "No training rows survived the support/amplitude-floor cuts -- "
            "the oracle and base waveforms may not overlap in time at all"
        )

    x_kept = np.concatenate(x_chunks)
    dlogA_kept = np.concatenate(dlogA_chunks)
    dphi_kept = np.concatenate(dphi_chunks)

    if n_dropped:
        logger.info(
            "Dropped %d/%d training rows (common-support / amplitude-floor cuts)",
            n_dropped, len(x_np),
        )
    span_phi = float(np.ptp(dphi_kept))
    logger.info(
        "Delta targets: |dlogA| max %.3g, dPhi span %.3g rad "
        "(alignment=%s, base=%s)",
        float(np.abs(dlogA_kept).max()), span_phi, phase_alignment,
        base_evaluator.approximant_name,
    )
    if span_phi > 50.0:
        hint = (
            "" if oracle_evaluator is not None else
            " The oracle side was decomposed from sparse training samples "
            "(no oracle_evaluator supplied) -- if the training grid's phase "
            "steps exceed pi anywhere, the unwrap aliases and produces "
            "exactly this signature; supply oracle_approximant so both "
            "sides are decomposed on dense native grids."
        )
        logger.warning(
            "Delta-phase span is %.1f rad -- much larger than typical "
            "approximant disagreement. Check that the oracle and base share "
            "time/phase conventions (or that the base isn't simply a poor "
            "model for this region of parameter space); a large oscillatory "
            "residual defeats the purpose of the delta decomposition.%s",
            span_phi, hint,
        )

    return (
        torch.tensor(x_kept, dtype=train_x.dtype),
        torch.tensor(dlogA_kept, dtype=train_y_plus.dtype),
        torch.tensor(dphi_kept, dtype=train_y_plus.dtype),
    )


class DeltaGPSurrogate(PhaseAmplitudeGPSurrogate):
    """Residual GP waveform surrogate: exact base approximant + GP deltas.

    Fits two independent GPs to the (delta-log-amplitude, delta-phase)
    residuals between the training (oracle) strain and a base approximant,
    and reconstructs plus/cross strain at predict() time as

        A = A_base * exp(dlogA_GP),  Phi = Phi_base + dPhi_GP
        h_plus = A cos Phi,  h_cross = A sin Phi

    with delta-method covariance from the delta GPs (the base is exact and
    contributes none). See the module docstring for the motivation and the
    relationship to `PhaseAmplitudeGPSurrogate` (whose `_train` /
    `_get_predict_models` are reused via subclassing -- everything else is
    overridden).

    Parameters mirror `PhaseAmplitudeGPSurrogate`, minus the mean modules
    (the base approximant *is* the mean; the delta GPs use ZeroMean), plus:

    base_approximant : str or approximant instance
        The exactly-evaluated base model. A string is resolved from
        `heron.models.lalsimulation`; any object with the ``time_domain``
        interface is accepted directly.
    oracle_approximant : str or approximant instance, optional
        The model that generated the training data. Strongly recommended
        whenever it is callable: the oracle's amplitude/phase are then
        decomposed on its dense native grid instead of the sparse training
        samples (which alias the phase unwrap -- see module docstring).
        Leave None only when the oracle cannot be regenerated (NR data).
    f_low : float
        Low-frequency cutoff used when generating base/oracle waveforms;
        should match the training data's ``f_min``.
    phase_alignment : str
        Per-mass-ratio alignment of the phase residual -- see
        `compute_delta_targets`.
    amp_floor_rel : float
        Relative amplitude floor below which training rows are dropped --
        see `compute_delta_targets`.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y_plus: torch.Tensor,
        train_y_cross: torch.Tensor,
        base_approximant="IMRPhenomD",
        oracle_approximant=None,
        warping: str = "chirp",
        nu: float = 2.5,
        device: str = "cpu",
        total_mass: float = 60.0,
        distance: float = 100.0,
        f_low: float = 20.0,
        training_iterations: int = 400,
        optimizer: str = "lbfgs",
        lr: float | None = None,
        ls_min_time: float = 0.0005,
        ls_min_q: float = 0.0005,
        noise_floor_rel: float = 1e-6,
        cholesky_size: int = 2000,
        phase_alignment: str = "anchor",
        amp_floor_rel: float = 1e-4,
    ):
        self._device = torch.device(device)
        self.nu = nu
        self.mass_factor = total_mass
        self.distance_factor = distance
        self.f_low = f_low
        self.ls_min_time = ls_min_time
        self.ls_min_q = ls_min_q
        self.noise_floor_rel = noise_floor_rel
        self.cholesky_size = cholesky_size
        self.phase_alignment = phase_alignment
        self.amp_floor_rel = amp_floor_rel

        if isinstance(warping, str):
            self.warping = get_warping(warping)
        else:
            self.warping = warping

        self.base = _ApproximantEvaluator(
            base_approximant, total_mass=total_mass, distance=distance, f_low=f_low
        )
        self.oracle = (
            _ApproximantEvaluator(
                oracle_approximant, total_mass=total_mass, distance=distance, f_low=f_low
            )
            if oracle_approximant is not None else None
        )

        # Raw oracle strain is stored for checkpointing; delta targets are
        # re-derived deterministically here on load(), same pattern as the
        # other GP surrogates.
        self._train_x_raw = train_x.clone()
        self._train_y_plus_raw = train_y_plus.clone()
        self._train_y_cross_raw = train_y_cross.clone()

        x_kept, delta_logA, delta_phase = compute_delta_targets(
            train_x, train_y_plus, train_y_cross, self.base,
            phase_alignment=phase_alignment,
            oracle_evaluator=self.oracle,
            amp_floor_rel=amp_floor_rel,
        )

        x_warped = x_kept.clone().to(self._device)
        x_warped[:, -1] = self.warping.warp(
            x_warped[:, -1], mass_ratio=x_warped[:, 0]
        )
        delta_logA = delta_logA.to(self._device)
        delta_phase = delta_phase.to(self._device)

        n_dims = x_warped.shape[1]
        ls_min_per_dim = [ls_min_q] * (n_dims - 1) + [ls_min_time]

        targets = {"delta_log_amplitude": delta_logA, "delta_phase": delta_phase}
        self.models: dict[str, _ExactGPModel] = {}
        for name, y in targets.items():
            # ZeroMean (mean_module=None): the residual between two decent
            # waveform models hovers around zero, and reverting to zero
            # delta far from training data degrades gracefully to the base
            # approximant.
            model = _ExactGPModel(
                x_warped, y,
                mean_module=None,
                nu=nu,
                ls_min_per_dim=ls_min_per_dim,
                noise_floor_rel=noise_floor_rel,
            ).to(self._device)
            model.likelihood.to(self._device)
            self.models[name] = model

        self._predict_models: dict[str, _ExactGPModel] | None = None

        if training_iterations > 0:
            self._train(training_iterations, optimizer_type=optimizer, lr=lr)

    def predict(self, parameters: dict) -> WaveformDict:
        """Generate waveform with systematics uncertainty.

        Same interface as the other surrogates: parameters must contain
        'mass_ratio' and 'time' (dict with lower/upper/number) or 'times'
        (array). Optional: 'total_mass', 'luminosity_distance'.
        """
        mass_ratio = parameters.get("mass_ratio")
        total_mass = parameters.get("total_mass", self.mass_factor)
        mass_factor = total_mass / self.mass_factor
        distance = parameters.get("luminosity_distance", self.distance_factor)
        distance_factor = distance / self.distance_factor

        if "times" in parameters:
            times = torch.tensor(parameters["times"], dtype=torch.float64) / mass_factor
        elif "time" in parameters:
            t = parameters["time"]
            times = torch.linspace(
                t["lower"], t["upper"], t["number"], dtype=torch.float64
            ) / mass_factor
        else:
            raise ValueError("parameters must contain 'times' or 'time'")

        n_times = len(times)
        times_np = times.numpy()

        points = torch.column_stack([
            torch.full((n_times,), mass_ratio, dtype=torch.float64),
            times,
        ]).to(self._device)
        points_warped = points.clone()
        points_warped[:, -1] = self.warping.warp(
            points_warped[:, -1], mass_ratio=points_warped[:, 0]
        )

        predict_models = self._get_predict_models()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.max_cholesky_size(self.cholesky_size):
            latent_dlogA = predict_models["delta_log_amplitude"](points_warped)
            mean_dlogA = latent_dlogA.mean.cpu()
            cov_dlogA = latent_dlogA.covariance_matrix.cpu()

            latent_dphase = predict_models["delta_phase"](points_warped)
            mean_dphase = latent_dphase.mean.cpu()
            cov_dphase = latent_dphase.covariance_matrix.cpu()

        # Exact base evaluation at the (physical, training-frame) times.
        logA_base, phase_base = self.base.log_amplitude_phase(mass_ratio, times_np)

        amplitude = torch.exp(torch.from_numpy(logA_base) + mean_dlogA)
        phase = torch.from_numpy(phase_base) + mean_dphase
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        h_plus = amplitude * cos_phase
        h_cross = amplitude * sin_phase

        # First-order (delta-method) propagation from the independent
        # (dlogA, dPhi) GPs to (h_plus, h_cross) -- identical in form to
        # PhaseAmplitudeGPSurrogate.predict, since d h/d dlogA = d h/d logA
        # and d h/d dPhi = d h/d Phi. The base contributes no uncertainty.
        d_hp_dlogA = amplitude * cos_phase
        d_hp_dphase = -amplitude * sin_phase
        d_hc_dlogA = amplitude * sin_phase
        d_hc_dphase = amplitude * cos_phase

        cov_plus = torch.outer(d_hp_dlogA, d_hp_dlogA) * cov_dlogA \
            + torch.outer(d_hp_dphase, d_hp_dphase) * cov_dphase
        cov_cross = torch.outer(d_hc_dlogA, d_hc_dlogA) * cov_dlogA \
            + torch.outer(d_hc_dphase, d_hc_dphase) * cov_dphase

        output = WaveformDict(
            parameters={k: v for k, v in parameters.items() if k != "time" and k != "times"}
        )
        output["plus"] = Waveform(
            data=(h_plus / distance_factor).numpy(),
            times=times_np,
            covariance=(cov_plus / distance_factor**2).numpy(),
        )
        output["cross"] = Waveform(
            data=(h_cross / distance_factor).numpy(),
            times=times_np,
            covariance=(cov_cross / distance_factor**2).numpy(),
        )
        return output

    def save(self, path: str | Path) -> None:
        """Save checkpoint.

        The base/oracle approximants are recorded by name only; load()
        re-resolves them from `heron.models.lalsimulation` (or accepts
        instance overrides for non-registry approximants).
        """
        import datetime
        from heron import __version__ as heron_version

        warping = self.warping
        if isinstance(warping, SimpleWarping):
            warping_config = {"type": "simple", "scale": warping.scale}
        elif isinstance(warping, MassRatioChirpTimeWarping):
            warping_config = {
                "type": "chirp_adaptive",
                "alpha": warping.alpha,
                "t_ref": warping.t_ref,
                "ref_mass_ratio": warping.ref_mass_ratio,
            }
        elif isinstance(warping, ChirpTimeWarping):
            warping_config = {
                "type": "chirp",
                "alpha": warping.alpha,
                "t_ref": warping.t_ref,
            }
        else:
            warping_config = {"type": str(type(warping).__name__)}

        checkpoint = {
            "format_version": 2,
            "heron_version": heron_version,
            "model_class": type(self).__name__,
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
            "parameter_names": self.parameter_names,
            "model_states": {
                name: model.state_dict()
                for name, model in self.models.items()
            },
            "base_approximant": self.base.approximant_name,
            "oracle_approximant": (
                self.oracle.approximant_name if self.oracle is not None else None
            ),
            "f_low": self.f_low,
            "phase_alignment": self.phase_alignment,
            "amp_floor_rel": self.amp_floor_rel,
            "train_x": self._train_x_raw.cpu(),
            "train_y_plus": self._train_y_plus_raw.cpu(),
            "train_y_cross": self._train_y_cross_raw.cpu(),
            "mass_factor": self.mass_factor,
            "distance_factor": self.distance_factor,
            "nu": self.nu,
            "warping": warping_config,
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
        oracle_approximant=None,
    ) -> "DeltaGPSurrogate":
        """Load a pre-trained model from checkpoint.

        Parameters
        ----------
        base_approximant, oracle_approximant : optional
            Overrides for the checkpoint's recorded approximant names --
            pass instances when they are not resolvable from
            `heron.models.lalsimulation` (e.g. test stubs).
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
            oracle_approximant=(
                oracle_approximant if oracle_approximant is not None
                else checkpoint.get("oracle_approximant")
            ),
            warping=warping_obj,
            nu=checkpoint["nu"],
            device=device,
            total_mass=checkpoint["mass_factor"],
            distance=checkpoint["distance_factor"],
            f_low=checkpoint.get("f_low", 20.0),
            training_iterations=0,
            ls_min_time=checkpoint.get("ls_min_time", 0.0005),
            ls_min_q=checkpoint.get("ls_min_q", 0.0005),
            noise_floor_rel=checkpoint.get("noise_floor_rel", 1e-6),
            cholesky_size=checkpoint.get("cholesky_size", 2000),
            phase_alignment=checkpoint.get("phase_alignment", "anchor"),
            amp_floor_rel=checkpoint.get("amp_floor_rel", 1e-4),
        )

        for name, state in checkpoint["model_states"].items():
            instance.models[name].load_state_dict(state)
            instance.models[name].eval()
            instance.models[name].likelihood.eval()

        logger.info(f"Loaded checkpoint from {path}")
        return instance
