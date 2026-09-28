"""
Mean functions for GP waveform surrogates.

The GP learns h(q, t) = mean(q, t) + f_GP(q, t), where mean() is
a physics-informed prior and f_GP captures the residual. Using a PN
inspiral as the mean function means the GP only needs to learn
merger/ringdown corrections.
"""

from __future__ import annotations

import torch
import gpytorch
import numpy as np


class ZeroMean(gpytorch.means.Mean):
    """Zero mean function (GP learns the full waveform)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


# Physical constants (SI), shared by every PN mean function below.
_G = 6.67430e-11        # m^3 kg^-1 s^-2
_C = 2.99792458e8       # m/s
_MSUN = 1.98892e30      # kg
_MPC = 3.0857e22        # m


def _symmetric_mass_ratio(q: torch.Tensor) -> torch.Tensor:
    """Convert mass ratio q = m2/m1 (<=1) to symmetric mass ratio eta."""
    return q / (1 + q) ** 2


def _pn_amplitude_and_phase(
    x: torch.Tensor,
    t_scale: float,
    h_scale: float,
    warping,
    pn_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared Newtonian amplitude / Newtonian-or-1PN phase inspiral model.

    Used both by the combined strain means (``amplitude * cos(phase)``) and
    by the standalone amplitude/phase means for the phase-amplitude GP
    surrogate, so the two representations stay derived from the exact same
    PN expressions instead of drifting apart.

    Parameters
    ----------
    x : Tensor, shape (N, D)
        First column is mass_ratio, last column is (warped) time.
    t_scale, h_scale : float
        Precomputed GM/c^3 and GM/(c^2 d) reference scales.
    warping : callable or None
        Unwarps the time column if the GP operates in warped time.
    pn_order : int
        0 for Newtonian-only phase, 1 to add the 1PN phase correction.

    Returns
    -------
    amplitude, phase : Tensor, shape (N,)
        Zero outside the inspiral (t >= 0, i.e. tau <= 0).
    """
    q = x[:, 0]
    t_input = x[:, -1]

    t = warping.unwarp(t_input) if warping is not None else t_input

    eta = _symmetric_mass_ratio(q)
    eta_safe = eta + 1e-30

    # Time to coalescence (t < 0 during inspiral, t = 0 at merger).
    tau = -t
    inspiral_mask = tau > 1e-6

    # Dimensionless PN expansion parameter Theta = eta*tau / (5*t_scale).
    tau_dimless = tau / (5.0 * t_scale)
    safe_tau = torch.where(inspiral_mask, tau_dimless, torch.ones_like(tau_dimless))

    # Amplitude: h ~ eta * Theta^(-1/4) * h_scale (Newtonian order).
    amplitude = torch.where(
        inspiral_mask,
        4.0 * h_scale * eta * safe_tau ** (-0.25),
        torch.zeros_like(t),
    )

    # Phase: Newtonian order, optionally + 1PN correction.
    phase_0 = -2.0 * safe_tau ** 0.625 / eta_safe ** 0.375
    if pn_order >= 1:
        phase_1pn = (3715.0 / 8064.0 + 55.0 * eta / 96.0) * safe_tau ** 0.375 / eta_safe ** 0.625
        phase_full = phase_0 + phase_1pn
    else:
        phase_full = phase_0

    phase = torch.where(inspiral_mask, phase_full, torch.zeros_like(t))

    return amplitude, phase


def _pn_log_amplitude(x: torch.Tensor, t_scale: float, h_scale: float, warping) -> torch.Tensor:
    """Log-amplitude PN mean, without `_pn_amplitude_and_phase`'s hard
    zero-after-merger cutoff.

    The zero-after-merger behaviour is harmless for the combined strain
    mean (`amplitude * cos(phase)`): the residual there is just "actual
    strain minus zero," the same order of magnitude as the strain itself.
    It is catastrophic for a *log*-amplitude mean: `log(0)` diverges right
    at the highest-density, peak-amplitude training region (merger),
    creating a huge, sharp residual discontinuity that destabilises GP
    training (observed directly: `NewtonianInspiralAmplitudeMean` with the
    zero-cutoff caused Cholesky failures during training on real PhenomD
    data). Instead, tau is clamped to a small positive floor for t >= 0, so
    the amplitude saturates at its near-merger value rather than
    discontinuously dropping to (near) zero -- a standard "freeze at the
    edge of validity" extrapolation.
    """
    q = x[:, 0]
    t_input = x[:, -1]
    t = warping.unwarp(t_input) if warping is not None else t_input

    eta = _symmetric_mass_ratio(q)
    tau = -t
    tau_dimless = tau / (5.0 * t_scale)
    floor = 1e-6 / (5.0 * t_scale)
    safe_tau = torch.clamp(tau_dimless, min=floor)

    amplitude = 4.0 * h_scale * eta * safe_tau ** (-0.25)
    return torch.log(amplitude)


class NewtonianInspiralMean(gpytorch.means.Mean):
    """Newtonian-order inspiral mean function.

    Evaluates a restricted (leading-order) inspiral waveform at the
    input coordinates (mass_ratio, time). The GP then only needs to
    learn the residual from this approximation (merger + ringdown).

    This is a closed-form evaluation — no LAL dependency, pure PyTorch,
    GPU-compatible and differentiable.

    The Newtonian inspiral strain goes as:

        h(t) ∝ η^(2/5) * |τ|^(-1/4) * cos(Φ(τ))

    where τ = t_c - t is the time to coalescence, η is the symmetric
    mass ratio, and Φ(τ) is the orbital phase.

    Parameters
    ----------
    total_mass : float
        Reference total mass in solar masses.
    distance : float
        Reference luminosity distance in Mpc.
    output_scale : float
        The output scaling factor used by the GP (e.g. 1e27).
    warping : callable or None
        If provided, the time column is in warped coordinates and
        must be unwarped before evaluating the PN waveform.
    f_low : float
        Low frequency cutoff in Hz (waveform is zero before this).
    """

    # Physical constants (SI)
    G = 6.67430e-11        # m^3 kg^-1 s^-2
    c = 2.99792458e8       # m/s
    MSUN = 1.98892e30      # kg
    MPC = 3.0857e22        # m

    def __init__(
        self,
        total_mass: float = 60.0,
        distance: float = 100.0,
        output_scale: float = 1e27,
        warping=None,
        f_low: float = 20.0,
    ):
        super().__init__()
        self.total_mass = total_mass
        self.distance = distance
        self.output_scale = output_scale
        self.warping = warping
        self.f_low = f_low

        # Precompute reference scales
        M_kg = total_mass * self.MSUN
        d_m = distance * self.MPC
        self._t_scale = self.G * M_kg / self.c**3  # GM/c^3 in seconds
        self._h_scale = (self.G * M_kg / self.c**2) / d_m  # GM/(c^2 d)

    # PN order used for the phase: 0 = Newtonian, 1 = 1PN (overridden by
    # TaylorT2Mean below).
    _pn_order = 0

    def _symmetric_mass_ratio(self, q: torch.Tensor) -> torch.Tensor:
        """Convert mass ratio q = m2/m1 (≤1) to symmetric mass ratio η."""
        return q / (1 + q) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate PN inspiral at input points.

        Parameters
        ----------
        x : Tensor, shape (N, D)
            Input coordinates. First column is mass_ratio,
            last column is (warped) time.

        Returns
        -------
        Tensor, shape (N,)
            Inspiral strain × output_scale.
        """
        amplitude, phase = _pn_amplitude_and_phase(
            x, self._t_scale, self._h_scale, self.warping, self._pn_order
        )
        h = amplitude * torch.cos(phase)
        # Apply output scaling to match GP training targets
        return h * self.output_scale


class TaylorT2Mean(NewtonianInspiralMean):
    """1PN Taylor T2 inspiral mean function.

    Extends the Newtonian inspiral with the first post-Newtonian
    correction to the phase. The amplitude remains Newtonian-order.

    The 1PN phase correction adds:
        δΦ = (3715/8064 + 55η/96) × Θ^(-1/4) / η

    where Θ = η τ / (5 GM/c³).
    """

    _pn_order = 1


class _PNAmplitudeOrPhaseMean(gpytorch.means.Mean):
    """Shared base for the standalone amplitude/phase mean functions.

    Unlike `NewtonianInspiralMean`/`TaylorT2Mean` (which combine amplitude
    and phase into a strain value scaled by `output_scale`), these means
    feed a `PhaseAmplitudeGPSurrogate`, whose GP targets (log-amplitude,
    phase) are already O(1-100) in magnitude — no `output_scale` needed or
    accepted here (unlike raw strain ~1e-21, which does need one).

    Parameters
    ----------
    total_mass : float
        Reference total mass in solar masses.
    distance : float
        Reference luminosity distance in Mpc.
    warping : callable or None
        If provided, the time column is in warped coordinates and must be
        unwarped before evaluating the PN waveform.
    f_low : float
        Low frequency cutoff in Hz (unused by the Newtonian model but kept
        for interface parity with the strain means).
    """

    _pn_order = 0

    def __init__(
        self,
        total_mass: float = 60.0,
        distance: float = 100.0,
        warping=None,
        f_low: float = 20.0,
    ):
        super().__init__()
        self.total_mass = total_mass
        self.distance = distance
        self.warping = warping
        self.f_low = f_low

        M_kg = total_mass * _MSUN
        d_m = distance * _MPC
        self._t_scale = _G * M_kg / _C**3
        self._h_scale = (_G * M_kg / _C**2) / d_m

    def _amplitude_and_phase(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _pn_amplitude_and_phase(
            x, self._t_scale, self._h_scale, self.warping, self._pn_order
        )


class NewtonianInspiralAmplitudeMean(_PNAmplitudeOrPhaseMean):
    """Newtonian-order log-amplitude mean for the phase-amplitude GP surrogate.

    Uses `_pn_log_amplitude` (freezes at merger instead of dropping to
    log(0)) rather than `_amplitude_and_phase`'s zero-after-merger amplitude
    -- see `_pn_log_amplitude` docstring for why.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _pn_log_amplitude(x, self._t_scale, self._h_scale, self.warping)


class NewtonianInspiralPhaseMean(_PNAmplitudeOrPhaseMean):
    """Newtonian-order phase mean for the phase-amplitude GP surrogate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, phase = self._amplitude_and_phase(x)
        return phase


class TaylorT2AmplitudeMean(NewtonianInspiralAmplitudeMean):
    """Log-amplitude mean matching TaylorT2Mean (amplitude is Newtonian-order
    even at 1PN, so this is identical to `NewtonianInspiralAmplitudeMean`;
    kept as a separate class for interface parity with `TaylorT2PhaseMean`)."""


class TaylorT2PhaseMean(_PNAmplitudeOrPhaseMean):
    """1PN Taylor T2 phase mean for the phase-amplitude GP surrogate."""

    _pn_order = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, phase = self._amplitude_and_phase(x)
        return phase


class _LALApproximantMeanBase(gpytorch.means.Mean):
    """Base for mean functions backed by a full LAL IMR approximant.

    Motivation (multi-fidelity mean): at M~60 Msun the in-band chirp is only
    tens of cycles, so closed-form PN means accumulate phase error comparable
    to the total in-band phase and remove almost none of the GP's residual
    (measured: TaylorT2 removes ~12% of the phase target's structure on the
    dense30 grid). A complete IMR approximant (e.g. IMRPhenomXAS) as the
    mean stays phase-coherent through merger and ringdown by construction,
    so the GP only fits the (small) difference between the training
    approximant and the mean approximant.

    Implementation notes:
    - LAL evaluation is CPU/numpy and non-differentiable. That is fine for a
      GP mean: it has no trainable parameters, and MLL gradients w.r.t.
      hyperparameters never differentiate through the mean's *inputs*.
      forward() detaches, computes on CPU in float64, and returns a constant
      tensor on the caller's device/dtype.
    - One waveform is generated per unique mass ratio (via the same
      `heron.models.lalsimulation` code path used to generate training data,
      so time/phase conventions match by construction) and interpolated with
      cubic splines. Splines and full forward() results are cached with a
      bounded size so repeated training iterations cost nothing and long
      sampling runs cannot grow memory without bound.
    - The unwrapped phase is anchored exactly like the training targets in
      `strain_to_amplitude_phase`: principal value (atan2) at the earliest
      native sample, unwrapped forward in time. Both the training
      approximant and the mean approximant pin their phase to phi_ref at
      f_ref, so at the (f~f_low) start of the data the two phases agree to
      much less than pi and the unwrap branches match -- the residual phase
      carries no spurious 2*pi*k(q) offsets. (Checked empirically in
      scripts/probe_phase_mean_residuals.py.)
    """

    _MAX_Q_CACHE = 256
    _MAX_FORWARD_CACHE = 64

    def __init__(
        self,
        approximant: str = "IMRPhenomXAS",
        total_mass: float = 60.0,
        distance: float = 100.0,
        warping=None,
        f_low: float = 20.0,
    ):
        super().__init__()
        self.approximant = approximant
        self.total_mass = total_mass
        self.distance = distance
        self.warping = warping
        self.f_low = f_low
        self._generator = None
        self._q_cache: dict[float, dict] = {}
        self._forward_cache: dict[bytes, torch.Tensor] = {}

    def __deepcopy__(self, memo):
        """Rebuild fresh instead of copying: the lazily-built LAL generator
        holds SWIG-wrapped objects (lal.CreateDict) that cannot be
        deep-copied. Needed because the surrogates' predict() paths
        deep-copy trained models into float64 clones. Caches are dropped
        (they repopulate on first use)."""
        import copy

        new = type(self)(
            approximant=self.approximant,
            total_mass=self.total_mass,
            distance=self.distance,
            warping=copy.deepcopy(self.warping, memo),
            f_low=self.f_low,
        )
        memo[id(self)] = new
        return new

    def _get_generator(self):
        if self._generator is None:
            from heron.models import lalsimulation as lalsim_models

            cls = getattr(lalsim_models, self.approximant, None)
            if cls is None:
                raise ValueError(
                    f"Unknown LAL approximant '{self.approximant}' "
                    "(no such class in heron.models.lalsimulation)"
                )
            self._generator = cls()
        return self._generator

    def _get_waveform_data(self, q: float) -> dict:
        """Generate (or fetch cached) splines for one mass ratio."""
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

            # Same decomposition + anchoring convention as
            # strain_to_amplitude_phase: h+ - i hx = A e^{-i Phi}.
            amplitude = np.sqrt(hp**2 + hx**2)
            phase = np.unwrap(np.arctan2(hx, hp))
            log_amplitude = np.log(amplitude + 1e-30)

            if len(self._q_cache) >= self._MAX_Q_CACHE:
                self._q_cache.pop(next(iter(self._q_cache)))
            self._q_cache[key] = {
                "t0": times[0],
                "t1": times[-1],
                "plus": CubicSpline(times, hp),
                "cross": CubicSpline(times, hx),
                "log_amplitude": CubicSpline(times, log_amplitude),
                "phase": CubicSpline(times, phase),
            }
        return self._q_cache[key]

    def _eval_channel(
        self, channel: str, x64: torch.Tensor, zero_outside_support: bool = False
    ) -> np.ndarray:
        """Evaluate one spline channel at (q, warped-time) inputs.

        Times outside the waveform's native support are clamped to its
        edges (channels are frozen at their boundary values) unless
        `zero_outside_support=True`, in which case they are zeroed instead
        -- used by the strain (plus/cross) means, since raw strain
        physically decays outside the merger/ringdown envelope, unlike
        log-amplitude/phase (meaningful, roughly-constant extrapolations).
        """
        q_col = x64[:, 0]
        t_col = x64[:, -1]
        if self.warping is not None:
            t_col = self.warping.unwarp(t_col, mass_ratio=q_col)
        q_np = q_col.numpy()
        t_np = t_col.numpy()

        out = np.empty_like(t_np)
        for qv in np.unique(q_np):
            m = q_np == qv
            data = self._get_waveform_data(qv)
            t_clamped = np.clip(t_np[m], data["t0"], data["t1"])
            out[m] = data[channel](t_clamped)
        if zero_outside_support:
            out = np.where(self._in_support_mask(x64), out, 0.0)
        return out

    def _in_support_mask(self, x64: torch.Tensor) -> np.ndarray:
        """True where (q, warped-time) falls inside that q's native
        generated-waveform window -- shared by every zero-outside-support
        caller (`_eval_channel` and the plus/cross reconstruction below)
        so the boundary rule lives in one place."""
        q_col = x64[:, 0]
        t_col = x64[:, -1]
        if self.warping is not None:
            t_col = self.warping.unwarp(t_col, mass_ratio=q_col)
        q_np = q_col.numpy()
        t_np = t_col.numpy()

        mask = np.zeros_like(t_np, dtype=bool)
        for qv in np.unique(q_np):
            m = q_np == qv
            data = self._get_waveform_data(qv)
            mask[m] = (t_np[m] >= data["t0"]) & (t_np[m] <= data["t1"])
        return mask

    def _cached_forward(self, x: torch.Tensor, compute) -> torch.Tensor:
        x64 = x.detach().to("cpu", torch.float64).contiguous()
        key = x64.numpy().tobytes()
        if key not in self._forward_cache:
            if len(self._forward_cache) >= self._MAX_FORWARD_CACHE:
                self._forward_cache.pop(next(iter(self._forward_cache)))
            self._forward_cache[key] = torch.from_numpy(compute(x64))
        return self._forward_cache[key].to(device=x.device, dtype=x.dtype)


class LALApproximantAmplitudeMean(_LALApproximantMeanBase):
    """Log-amplitude mean from a full LAL IMR approximant, for the
    phase-amplitude GP surrogate. Covers merger and ringdown natively (no
    PN freeze-at-merger clamp needed)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._cached_forward(x, lambda x64: self._eval_channel("log_amplitude", x64))


class LALApproximantPhaseMean(_LALApproximantMeanBase):
    """Unwrapped-phase mean from a full LAL IMR approximant, for the
    phase-amplitude GP surrogate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._cached_forward(x, lambda x64: self._eval_channel("phase", x64))


class _LALApproximantWaveformMeanBase(_LALApproximantMeanBase):
    """Shared base for the plus/cross strain means used by ExactGPSurrogate.

    Unlike the amplitude/phase means above (which extrapolate as a frozen
    boundary value outside the waveform's native support -- log-amplitude/
    phase are still meaningful there), raw strain is zeroed outside the
    native window: h_plus/h_cross physically decay to (near) zero before/
    after the merger/ringdown envelope, so freezing at a nonzero boundary
    value would inject a spurious constant offset into the GP's residual
    for any query time beyond it.

    `output_scale` matches ExactGPSurrogate's raw-strain scaling (~1e-21 ->
    O(1)); the amplitude/phase means above don't accept it since their
    targets (log-amplitude, phase) are already well-scaled.

    `phase_correction` (radians, default 0.0) is the constant phase
    rotation applied before reconstructing h_plus = A*cos(Phi - correction)
    / h_cross = A*sin(Phi - correction). Different LAL approximant
    families use slightly different `phi_ref`/`f_ref` conventions -- e.g.
    IMRPhenomD vs IMRPhenomXAS differ by a near-constant ~-2.15 rad across
    the whole q range (measured via `compute_phase_correction`, envelope
    peaks already agree to <1ns so this is NOT a time-alignment issue).
    That constant is invisible to `heron.evaluation.mismatch.compute_overlap`
    (which maximises over exactly this phase and reports only the residual
    mismatch), but it is NOT invisible to a raw GP residual: `cos(Phi)` vs
    `cos(Phi + 2.15)` are two almost-uncorrelated oscillating functions of
    comparable magnitude to the raw signal (measured: corr ~= cos(2.15) ~=
    -0.42 on real training data), so an uncorrected mean makes the GP's job
    *harder* than ZeroMean, not easier -- this reconstruction (from
    amplitude/phase, not the raw hp/hx splines) is what fixes that; a
    naive one-time output-level scalar could not, since the correction has
    to happen inside the cos/sin, not after it.
    """

    def __init__(self, *args, output_scale: float = 1e27, phase_correction: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_scale = output_scale
        self.phase_correction = phase_correction

    def __deepcopy__(self, memo):
        import copy

        new = type(self)(
            approximant=self.approximant,
            total_mass=self.total_mass,
            distance=self.distance,
            warping=copy.deepcopy(self.warping, memo),
            f_low=self.f_low,
            output_scale=self.output_scale,
            phase_correction=self.phase_correction,
        )
        memo[id(self)] = new
        return new

    def _reconstruct_channel(self, channel: str, x64: torch.Tensor) -> np.ndarray:
        log_amplitude = self._eval_channel("log_amplitude", x64, zero_outside_support=False)
        phase = self._eval_channel("phase", x64, zero_outside_support=False)
        amplitude = np.exp(log_amplitude)
        corrected_phase = phase - self.phase_correction
        vals = (
            amplitude * np.cos(corrected_phase) if channel == "plus"
            else amplitude * np.sin(corrected_phase)
        )
        return np.where(self._in_support_mask(x64), vals, 0.0)

    def _channel_forward(self, channel: str, x: torch.Tensor) -> torch.Tensor:
        vals = self._cached_forward(x, lambda x64: self._reconstruct_channel(channel, x64))
        return vals * self.output_scale


class LALApproximantPlusMean(_LALApproximantWaveformMeanBase):
    """h_plus mean from a full LAL IMR approximant, for ExactGPSurrogate.

    Covers merger and ringdown natively, same motivation as
    `LALApproximantAmplitudeMean`/`LALApproximantPhaseMean` -- see
    `_LALApproximantMeanBase` docstring. Reconstructed as
    `A*cos(Phi - phase_correction)` rather than looked up directly from a
    raw h_plus spline -- see `_LALApproximantWaveformMeanBase` docstring
    for why the phase correction is essential here.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._channel_forward("plus", x)


class LALApproximantCrossMean(_LALApproximantWaveformMeanBase):
    """h_cross mean from a full LAL IMR approximant, for ExactGPSurrogate.

    Reconstructed as `A*sin(Phi - phase_correction)` -- see
    `_LALApproximantWaveformMeanBase` docstring.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._channel_forward("cross", x)


def compute_phase_correction(
    mean_approximant: str,
    target_approximant: str,
    total_mass: float = 60.0,
    distance: float = 100.0,
    f_low: float = 20.0,
    reference_mass_ratio: float = 0.5,
) -> float:
    """Constant phase offset (radians) between two LAL approximants'
    plus-polarisation phase conventions, for `LALApproximantPlusMean`/
    `CrossMean`'s `phase_correction`.

    Measured via a PSD-weighted matched-filter phase alignment (the
    complex overlap's argument) at one representative mass ratio --
    empirically near-constant across the whole q range for IMRPhenomD vs
    IMRPhenomXAS (-2.24 to -2.13 rad over q=0.15-0.9, envelope peaks
    agreeing to <1ns), so a single reference-q measurement is sufficient;
    this is a `phi_ref`/`f_ref` convention difference between approximant
    families, not a per-q physical effect. See
    `_LALApproximantWaveformMeanBase` for why this matters for a raw
    strain-domain mean specifically.
    """
    from scipy.interpolate import CubicSpline
    from astropy import units as u

    from heron.train import _get_approximant
    from heron.evaluation.psd import aligo_design_psd

    params = dict(
        mass_ratio=reference_mass_ratio,
        total_mass=total_mass * u.solMass,
        luminosity_distance=distance * u.Mpc,
        f_min=f_low * u.Hertz,
        delta_t=(1.0 / 4096) * u.second,
    )
    wf_target = _get_approximant(target_approximant).time_domain(dict(params))
    wf_mean = _get_approximant(mean_approximant).time_domain(dict(params))
    t_target = np.asarray(wf_target["plus"].times)
    h_target = np.asarray(wf_target["plus"].data)
    t_mean = np.asarray(wf_mean["plus"].times)
    h_mean = np.asarray(wf_mean["plus"].data)

    dt = 1.0 / 4096
    t0 = min(t_target[0], t_mean[0])
    t1 = max(t_target[-1], t_mean[-1])
    n = int(round((t1 - t0) / dt)) + 1
    tgrid = t0 + np.arange(n) * dt
    target_grid = np.nan_to_num(CubicSpline(t_target, h_target, extrapolate=False)(tgrid))
    mean_grid = np.nan_to_num(CubicSpline(t_mean, h_mean, extrapolate=False)(tgrid))

    freqs = np.fft.rfftfreq(n, d=dt)
    target_f = np.fft.rfft(target_grid)
    mean_f = np.fft.rfft(mean_grid)
    psd = aligo_design_psd(freqs)
    band = freqs >= f_low
    z = np.sum(np.conj(target_f[band]) * mean_f[band] / psd[band])
    return float(np.angle(z))


# --- Mean-function (de)serialization -------------------------------------
#
# Checkpoints must record which mean function a model was trained with:
# mean functions have no trainable parameters, so load_state_dict() cannot
# detect or restore them -- without an explicit config record, a checkpoint
# trained with a non-zero mean silently loads with ZeroMean and every
# prediction is wrong by the un-interpolated mean. (This actually happened:
# the dense30 phase-amplitude checkpoint was trained with Newtonian means
# that were silently dropped on load.)

_MEAN_CLASS_TO_CONFIG = {
    NewtonianInspiralMean: ("newtonian", "strain"),
    TaylorT2Mean: ("taylort2", "strain"),
    NewtonianInspiralAmplitudeMean: ("newtonian", "amplitude"),
    NewtonianInspiralPhaseMean: ("newtonian", "phase"),
    TaylorT2AmplitudeMean: ("taylort2", "amplitude"),
    TaylorT2PhaseMean: ("taylort2", "phase"),
    LALApproximantAmplitudeMean: ("approximant", "amplitude"),
    LALApproximantPhaseMean: ("approximant", "phase"),
    LALApproximantPlusMean: ("approximant", "plus"),
    LALApproximantCrossMean: ("approximant", "cross"),
}

_CONFIG_TO_MEAN_CLASS = {v: k for k, v in _MEAN_CLASS_TO_CONFIG.items()}


def mean_to_config(mean) -> dict | None:
    """Serialize a mean module to a checkpoint-storable config dict.

    None / ZeroMean / ConstantMean (GPyTorch's trained-constant default)
    serialize to None -- their state, if any, lives in the model
    state_dict.
    """
    if mean is None or isinstance(
        mean, (ZeroMean, gpytorch.means.ZeroMean, gpytorch.means.ConstantMean)
    ):
        return None
    key = _MEAN_CLASS_TO_CONFIG.get(type(mean))
    if key is None:
        raise ValueError(
            f"Cannot serialize mean function of type {type(mean).__name__}; "
            "add it to _MEAN_CLASS_TO_CONFIG in heron/models/gp/mean.py"
        )
    mean_type, target = key
    config = {
        "type": mean_type,
        "target": target,
        "total_mass": mean.total_mass,
        "distance": mean.distance,
        "f_low": mean.f_low,
    }
    if target in ("strain", "plus", "cross"):
        config["output_scale"] = mean.output_scale
    if target in ("plus", "cross"):
        config["phase_correction"] = mean.phase_correction
    if mean_type == "approximant":
        config["approximant"] = mean.approximant
    return config


def mean_from_config(config: dict | None, warping=None):
    """Rebuild a mean module from `mean_to_config` output.

    The warping is supplied by the owning surrogate (it is serialized
    separately in the checkpoint) rather than stored in the mean config.
    """
    if config is None:
        return None
    key = (config["type"], config["target"])
    cls = _CONFIG_TO_MEAN_CLASS.get(key)
    if cls is None:
        raise ValueError(f"Unknown mean-function config {config!r}")
    kwargs = {
        "total_mass": config.get("total_mass", 60.0),
        "distance": config.get("distance", 100.0),
        "f_low": config.get("f_low", 20.0),
        "warping": warping,
    }
    if config["target"] in ("strain", "plus", "cross"):
        kwargs["output_scale"] = config.get("output_scale", 1e27)
    if config["target"] in ("plus", "cross"):
        kwargs["phase_correction"] = config.get("phase_correction", 0.0)
    if config["type"] == "approximant":
        kwargs["approximant"] = config.get("approximant", "IMRPhenomXAS")
    return cls(**kwargs)
