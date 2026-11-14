"""Demodulated-residual GP surrogate in geometric units, on a Kronecker grid.

Same physics as :class:`heron.models.gp.demod.DemodGPSurrogate` -- two GPs
fit the real/imaginary parts of the oracle-minus-reference strain residual,
heterodyned by the reference phase so the targets are smooth --

    z(tau) = (h_oracle - h_ref) * exp(+i Phi_ref),
    h      = h_ref + Re(z) cos Phi_ref + Im(z) sin Phi_ref     (linear => exact K)

but with three structural changes:

1. **Geometric units.** Everything is modelled as a function of
   ``(q, tau = t/M)`` with dimensionless strain ``h_geo = h D / M``
   (:mod:`heron.units`). A non-spinning waveform is exactly self-similar in
   total mass, so ``total_mass`` and ``luminosity_distance`` are analytic
   rescalings at predict time: no total-mass GP dimension, no
   ``output_scale``, and the reference approximant is generated *once per
   mass ratio* (at ``reference_mass``) and rescaled for every M, rather than
   regenerated per (q, M). Total mass can therefore be sampled at no extra
   waveform-generation cost.

2. **Kronecker product grid.** Every mass ratio is sampled at the *same*
   warped-tau nodes, so the training inputs are a Cartesian product and the
   separable kernel's covariance is ``K_q kron K_tau``. Exact inference costs
   ``O(n_q^3 + n_tau^3)`` (:class:`~heron.models.gp.kronecker.KroneckerGP`)
   instead of ``O((n_q n_tau)^3)``: no Cholesky-size limit, no GPU, no
   sparse/variational fallback, and seconds rather than GPU-minutes to train.
   Lengthscale floors default to values derived from the grid.

3. **Low-rank covariance.** :meth:`predict_factor` returns plus/cross
   covariance *factors* (``K = U U^T``) that share latent columns, so the
   temporal correlations of the waveform uncertainty and the correlation
   between h+ and hx (both built from the same Re/Im GPs) are kept exactly.
   :class:`heron.inference.network.NetworkLikelihood` with
   ``waveform_covariance='lowrank'`` consumes these through a Woodbury solve
   (:class:`heron.likelihood.LowRankMarginalLogLikelihood`): ``O(N^2 r)`` per
   call instead of ``O(N^3)``.

**Support.** The residual is only modelled on the training window
``[tau_grid[0], tau_grid[-1]]``. Outside it the residual mean and covariance
are set to zero (the surrogate returns the bare reference) and a warning is
logged once if the reference still has non-negligible amplitude there. Since
the window is in units of M, a fixed physical analysis segment covers a
*longer* tau range at lower total mass: choose ``tau_lo`` for the lightest
system you intend to analyse.

**Phase convention under mass rescaling.** Because one waveform per q is
generated at ``reference_mass`` and rescaled, the LAL reference frequency
(``f_ref``, pinned at 20 Hz in :mod:`heron.models.lalsimulation`) effectively
scales as ``20 Hz * reference_mass / M``. That is just a different (and
mass-consistent) definition of the coalescence phase, which is sampled or
marginalised anyway; oracle and reference share it.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
import torch

from ..base import WaveformSurrogate
from ..warping import ChirpTimeWarping
from ...types import Waveform, WaveformDict
from ...units import MTSUN_SI, strain_scale
from .delta import _ApproximantEvaluator
from .kronecker import KroneckerGP

logger = logging.getLogger("heron.models.gp.demod_kron")

#: 0.1 s at 60 M_sun -- the t_ref the physical-unit chirp warping uses,
#: expressed in units of M.
DEFAULT_TAU_REF = 0.1 / (60.0 * MTSUN_SI)


def symmetric_mass_ratio(q):
    q = np.asarray(q, dtype=np.float64)
    return q / (1.0 + q) ** 2


class _GeometricReference:
    """Reference approximant evaluated in geometric units.

    Wraps :class:`~heron.models.gp.delta._ApproximantEvaluator` (one LAL call
    per mass ratio at ``reference_mass``/``reference_distance``, cached as
    native-grid log-amplitude/phase splines) and maps (q, tau) onto it.
    """

    def __init__(self, approximant, reference_mass, reference_distance, f_low,
                 phase_correction=0.0):
        self.reference_mass = float(reference_mass)
        self.reference_distance = float(reference_distance)
        self.phase_correction = float(phase_correction)
        self._ev = _ApproximantEvaluator(
            approximant, total_mass=reference_mass,
            distance=reference_distance, f_low=f_low,
        )
        self._log_scale = np.log(strain_scale(reference_mass, reference_distance))
        self._t_unit = reference_mass * MTSUN_SI

    @property
    def generator(self):
        return self._ev._generator

    def support(self, q):
        t0, t1 = self._ev.support(q)
        return t0 / self._t_unit, t1 / self._t_unit

    def evaluate(self, q, tau):
        """Geometric reference ``(h_plus, h_cross, cos Phi, sin Phi)`` at (q, tau).

        ``Phi`` is the reference phase minus ``phase_correction``; strain is
        zero outside the native support (the phase is not, since it only
        multiplies the near-zero out-of-support residual).
        """
        tau = np.asarray(tau, dtype=np.float64)
        log_amp, phase = self._ev.log_amplitude_phase(q, tau * self._t_unit)
        t0, t1 = self.support(q)
        in_sup = (tau >= t0) & (tau <= t1)
        phi = phase - self.phase_correction
        c, s = np.cos(phi), np.sin(phi)
        amp = np.where(in_sup, np.exp(log_amp - self._log_scale), 0.0)
        return amp * c, amp * s, c, s


def warped_tau_grid(tau_lo: float, tau_hi: float, n_tau: int,
                    alpha: float = 0.625, tau_ref: float = DEFAULT_TAU_REF) -> np.ndarray:
    """``n_tau`` nodes uniform in chirp-warped tau between ``tau_lo`` and ``tau_hi``."""
    warp = ChirpTimeWarping(alpha=alpha, t_ref=tau_ref)
    ends = warp.warp(torch.tensor([tau_lo, tau_hi], dtype=torch.float64))
    wgrid = torch.linspace(float(ends[0]), float(ends[1]), n_tau, dtype=torch.float64)
    return warp.unwarp(wgrid).numpy()


def generate_kronecker_grid(
    oracle,
    mass_ratios,
    tau,
    reference_mass: float = 60.0,
    reference_distance: float = 100.0,
    f_low: float = 20.0,
):
    """Sample an oracle approximant on the product grid ``mass_ratios x tau``.

    Returns geometric-unit ``(y_plus, y_cross)``, each ``(n_q, n_tau)``. The
    oracle is generated once per mass ratio at ``reference_mass`` and
    evaluated through native-grid log-amplitude/phase splines (immune to
    sparse-sample aliasing). Raises if ``tau[0]`` precedes any mass ratio's
    native start -- lower ``f_low`` to generate longer waveforms.
    """
    ref = _GeometricReference(oracle, reference_mass, reference_distance, f_low)
    tau = np.asarray(tau, dtype=np.float64)
    mass_ratios = np.asarray(mass_ratios, dtype=np.float64)
    yp = np.zeros((len(mass_ratios), len(tau)))
    yc = np.zeros_like(yp)
    for i, q in enumerate(mass_ratios):
        t0, _ = ref.support(q)
        if tau[0] < t0:
            raise ValueError(
                f"tau_lo={tau[0]:.1f} M precedes the oracle's start at q={q:.3f} "
                f"({t0:.1f} M at f_low={f_low} Hz, M={reference_mass}); lower "
                "f_low or raise tau_lo so every mass ratio covers the window."
            )
        hp, hc, _, _ = ref.evaluate(q, tau)
        yp[i], yc[i] = hp, hc
    return yp, yc


class KroneckerDemodGPSurrogate(WaveformSurrogate):
    """Heterodyned-residual GP surrogate in geometric units on a product grid.

    Parameters
    ----------
    mass_ratios : array (n_q,)
        Training mass ratios (any order; sorted internally).
    tau : array (n_tau,)
        Training times in units of M, shared by every mass ratio
        (e.g. from :func:`warped_tau_grid`).
    y_plus, y_cross : array (n_q, n_tau)
        Oracle strain on the grid, in geometric units (``h D / M``).
    base_approximant : str or approximant instance
        Reference model, generated exactly at predict time.
    oracle_approximant : str or None
        Name of the training approximant; used to auto-compute
        ``phase_correction`` when not given.
    phase_correction : float or None
        Constant phase offset between reference and oracle conventions
        (see :class:`~heron.models.gp.demod.DemodGPSurrogate`).
    reference_mass, reference_distance : float
        Mass [M_sun] / distance [Mpc] at which reference waveforms are
        generated (and the default ``total_mass``/``luminosity_distance``
        for :meth:`predict`). ``reference_distance`` is also exposed as
        ``distance_factor`` for the likelihood's analytic distance scaling.
    f_low : float
        Low-frequency cutoff for reference generation at ``reference_mass``.
    alpha, tau_ref : float
        Chirp time warping of tau (``alpha=0.625`` maps time onto Newtonian
        phase: ``Phi ~ tau^(5/8)``).
    q_warping : {'none', 'eta'}
        Kernel coordinate for mass ratio.
    nu : float
        Matern smoothness.
    ls_min_q, ls_min_time : float or None
        Lengthscale floors in kernel coordinates; ``None`` derives them from
        the grid (max spacing in q; 2x max spacing in warped tau).
    noise_floor_rel : float
        Noise variance floor relative to target variance.
    fit : bool
        Fit hyperparameters at construction (``False`` for checkpoint loads).
    maxiter, n_restarts : int
        L-BFGS-B iterations / extra random restarts per GP.
    covariance_inflation : float
        Scalar applied to every returned covariance (variance units).
    factor_rank : int or None
        Default number of modes per Re/Im GP in :meth:`predict_factor`.
    """

    def __init__(
        self,
        mass_ratios,
        tau,
        y_plus,
        y_cross,
        base_approximant="IMRPhenomXAS",
        oracle_approximant: str | None = "IMRPhenomD",
        phase_correction: float | None = None,
        reference_mass: float = 60.0,
        reference_distance: float = 100.0,
        f_low: float = 20.0,
        alpha: float = 0.625,
        tau_ref: float = DEFAULT_TAU_REF,
        q_warping: str = "none",
        nu: float = 2.5,
        ls_min_q: float | None = None,
        ls_min_time: float | None = None,
        noise_floor_rel: float = 1e-6,
        fit: bool = True,
        maxiter: int = 200,
        n_restarts: int = 0,
        covariance_inflation: float = 1.0,
        factor_rank: int | None = None,
    ):
        q = np.asarray(mass_ratios, dtype=np.float64)
        order = np.argsort(q)
        self.mass_ratios = q[order]
        self.tau = np.asarray(tau, dtype=np.float64)
        if np.any(np.diff(self.tau) <= 0):
            raise ValueError("tau must be strictly increasing")
        self._y_plus = np.asarray(y_plus, dtype=np.float64)[order]
        self._y_cross = np.asarray(y_cross, dtype=np.float64)[order]

        self.reference_mass = float(reference_mass)
        self.distance_factor = float(reference_distance)
        self.f_low = float(f_low)
        self.alpha = float(alpha)
        self.tau_ref = float(tau_ref)
        if q_warping not in ("none", "eta"):
            raise ValueError(f"q_warping must be 'none' or 'eta'; got {q_warping!r}")
        self.q_warping = q_warping
        self.nu = float(nu)
        self.covariance_inflation = float(covariance_inflation)
        self.factor_rank = factor_rank
        self._warping = ChirpTimeWarping(alpha=alpha, t_ref=tau_ref)
        self._warned_support = False

        self._base_name = (base_approximant if isinstance(base_approximant, str)
                           else type(base_approximant).__name__)
        self._oracle_name = oracle_approximant
        self.phase_correction = self._resolve_phase_correction(
            phase_correction, base_approximant, oracle_approximant,
        )
        self._ref = _GeometricReference(
            base_approximant, reference_mass, reference_distance, f_low,
            phase_correction=self.phase_correction,
        )

        re_z, im_z = self._demod_targets()
        qk = self._q_kernel(self.mass_ratios)
        wt = self._warp(self.tau)
        self._gps = {
            name: KroneckerGP(qk, wt, target, nu=nu, ls_min=(ls_min_q, ls_min_time),
                              noise_floor_rel=noise_floor_rel)
            for name, target in (("re", re_z), ("im", im_z))
        }
        logger.info(
            "KroneckerDemod targets: Re(z) std %.3g, Im(z) std %.3g on a %d x %d grid "
            "(reference=%s, phase_correction=%.4f rad)",
            re_z.std(), im_z.std(), len(qk), len(wt), self._base_name,
            self.phase_correction,
        )
        if fit:
            for name, gp in self._gps.items():
                gp.fit(maxiter=maxiter, n_restarts=n_restarts)
                logger.info("%s GP hyperparameters: %s", name, gp.hyperparameters)

    # -- construction helpers ----------------------------------------------

    def _resolve_phase_correction(self, phase_correction, base, oracle) -> float:
        if phase_correction is not None:
            return float(phase_correction)
        if isinstance(base, str) and isinstance(oracle, str):
            from .mean import compute_phase_correction

            return float(compute_phase_correction(
                mean_approximant=base, target_approximant=oracle,
                total_mass=self.reference_mass, distance=self.distance_factor,
                f_low=self.f_low,
            ))
        logger.warning("phase_correction not given and base/oracle are not both "
                       "registry names; defaulting to 0.0 rad.")
        return 0.0

    def _q_kernel(self, q):
        q = np.asarray(q, dtype=np.float64)
        return symmetric_mass_ratio(q) if self.q_warping == "eta" else q

    def _warp(self, tau):
        return self._warping.warp(torch.as_tensor(np.asarray(tau, dtype=np.float64))).numpy()

    def _demod_targets(self):
        re_z = np.zeros_like(self._y_plus)
        im_z = np.zeros_like(self._y_plus)
        for i, q in enumerate(self.mass_ratios):
            hp, hc, c, s = self._ref.evaluate(q, self.tau)
            rp = self._y_plus[i] - hp
            rc = self._y_cross[i] - hc
            re_z[i] = rp * c + rc * s
            im_z[i] = rp * s - rc * c
        return re_z, im_z

    # -- prediction --------------------------------------------------------

    @staticmethod
    def _times(parameters) -> np.ndarray:
        if "times" in parameters:
            return np.asarray(parameters["times"], dtype=np.float64)
        t = parameters["time"]
        return np.linspace(t["lower"], t["upper"], t["number"])

    def _evaluate(self, parameters, mode, rank=None):
        """Shared core: reference, residual posteriors and unit conversions.

        Returns ``(times, scale, ref_plus, ref_cross, cos, sin, inside, res)``
        where ``res`` maps 're'/'im' to ``(mean, second)`` from
        :meth:`KroneckerGP.posterior` in geometric units, already zeroed
        outside the training window.
        """
        q = float(parameters["mass_ratio"])
        mass = float(parameters.get("total_mass", self.reference_mass))
        distance = float(parameters.get("luminosity_distance", self.distance_factor))
        times = self._times(parameters)
        tau = times / (mass * MTSUN_SI)
        scale = strain_scale(mass, distance)

        hp, hc, c, s = self._ref.evaluate(q, tau)
        inside = (tau >= self.tau[0]) & (tau <= self.tau[-1])
        if not self._warned_support and not inside.all():
            amp = np.hypot(hp, hc)
            if amp.max() > 0 and amp[~inside].max() > 1e-3 * amp.max():
                logger.warning(
                    "Query spans tau in [%.0f, %.0f] M but the residual is only "
                    "modelled on [%.0f, %.0f] M (M=%.1f); outside it the surrogate "
                    "returns the bare reference with no waveform uncertainty.",
                    tau.min(), tau.max(), self.tau[0], self.tau[-1], mass,
                )
                self._warned_support = True

        res = {}
        qk = float(self._q_kernel(q))
        wt = self._warp(tau[inside])
        gp_mode = "factor" if mode == "factor" else mode
        for name, gp in self._gps.items():
            mean = np.zeros(len(tau))
            m_in, second_in = gp.posterior(qk, wt, mode=gp_mode, rank=rank)
            mean[inside] = m_in
            second = None
            if second_in is not None:
                if gp_mode == "diagonal":
                    second = np.zeros(len(tau))
                    second[inside] = second_in
                elif gp_mode == "factor":
                    second = np.zeros((len(tau), second_in.shape[1]))
                    second[inside] = second_in
                else:  # full
                    second = np.zeros((len(tau), len(tau)))
                    second[np.ix_(inside, inside)] = second_in
            res[name] = (mean, second)
        return times, scale, hp, hc, c, s, res

    def _mean_dict(self, parameters, times, scale, hp, hc, c, s, res):
        re, im = res["re"][0], res["im"][0]
        h_plus = scale * (hp + re * c + im * s)
        h_cross = scale * (hc + re * s - im * c)
        out = WaveformDict(parameters={k: v for k, v in parameters.items()
                                       if k not in ("time", "times")})
        return out, h_plus, h_cross

    def predict(self, parameters: dict, covariance: str = "full") -> WaveformDict:
        """Waveform with (exact-linear) residual uncertainty.

        ``parameters``: ``mass_ratio``; ``times`` (array, seconds relative to
        merger) or ``time`` ({lower, upper, number}); optional ``total_mass``
        [M_sun] and ``luminosity_distance`` [Mpc] (defaults: the reference
        values). ``covariance``: ``'full'`` | ``'diagonal'`` | ``'none'``.
        """
        if covariance not in ("full", "diagonal", "none"):
            raise ValueError(f"covariance must be 'full', 'diagonal' or 'none'; "
                             f"got {covariance!r}")
        times, scale, hp, hc, c, s, res = self._evaluate(parameters, covariance)
        out, h_plus, h_cross = self._mean_dict(parameters, times, scale, hp, hc, c, s, res)
        k = self.covariance_inflation * scale ** 2
        if covariance == "none":
            out["plus"] = Waveform(data=h_plus, times=times)
            out["cross"] = Waveform(data=h_cross, times=times)
        elif covariance == "diagonal":
            v_re, v_im = res["re"][1], res["im"][1]
            out["plus"] = Waveform(data=h_plus, times=times,
                                   variance=k * (c**2 * v_re + s**2 * v_im))
            out["cross"] = Waveform(data=h_cross, times=times,
                                    variance=k * (s**2 * v_re + c**2 * v_im))
        else:
            C_re, C_im = res["re"][1], res["im"][1]
            cc, ss, cs = np.outer(c, c), np.outer(s, s), np.outer(c, s)
            out["plus"] = Waveform(data=h_plus, times=times,
                                   covariance=k * (cc * C_re + ss * C_im))
            out["cross"] = Waveform(data=h_cross, times=times,
                                    covariance=k * (ss * C_re + cc * C_im))
            # Re/Im both feed h+ and hx, so they are correlated:
            # Cov(h+, hx) = cos sin^T . C_re - sin cos^T . C_im.
            out.cross_covariance = k * (cs * C_re - cs.T * C_im)
        return out

    def predict_factor(self, parameters: dict, rank: int | None = None):
        """Mean waveform plus low-rank plus/cross covariance factors.

        Returns ``(waveform_dict, {'plus': U_p, 'cross': U_c})`` with
        ``U_p, U_c`` of shape ``(N, m)`` sharing latent columns:
        ``Cov(h+) = U_p U_p^T``, ``Cov(hx) = U_c U_c^T`` and
        ``Cov(h+, hx) = U_p U_c^T``. ``rank`` (default ``factor_rank``) caps
        the modes kept *per* Re/Im GP, so ``m <= 2 * rank``.
        """
        rank = self.factor_rank if rank is None else rank
        times, scale, hp, hc, c, s, res = self._evaluate(parameters, "factor", rank=rank)
        out, h_plus, h_cross = self._mean_dict(parameters, times, scale, hp, hc, c, s, res)
        out["plus"] = Waveform(data=h_plus, times=times)
        out["cross"] = Waveform(data=h_cross, times=times)
        k = np.sqrt(self.covariance_inflation) * scale
        F_re, F_im = res["re"][1], res["im"][1]
        U_plus = k * np.hstack([c[:, None] * F_re, s[:, None] * F_im])
        U_cross = k * np.hstack([s[:, None] * F_re, -c[:, None] * F_im])
        return out, {"plus": U_plus, "cross": U_cross}

    def covariance_diagonal(self, parameters: dict) -> dict:
        """Per-polarisation predictive variance without forming N x N matrices."""
        wf = self.predict(parameters, covariance="diagonal")
        return {"plus": wf["plus"].variance, "cross": wf["cross"].variance}

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        from heron import __version__ as heron_version

        checkpoint = {
            "format_version": 1,
            "heron_version": heron_version,
            "model_class": type(self).__name__,
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
            "mass_ratios": self.mass_ratios,
            "tau": self.tau,
            "y_plus": self._y_plus,
            "y_cross": self._y_cross,
            "base_approximant": self._base_name,
            "oracle_approximant": self._oracle_name,
            "phase_correction": self.phase_correction,
            "reference_mass": self.reference_mass,
            "reference_distance": self.distance_factor,
            "f_low": self.f_low,
            "alpha": self.alpha,
            "tau_ref": self.tau_ref,
            "q_warping": self.q_warping,
            "nu": self.nu,
            "covariance_inflation": self.covariance_inflation,
            "factor_rank": self.factor_rank,
            "gps": {name: gp.state() for name, gp in self._gps.items()},
        }
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint to %s", path)

    @classmethod
    def load(cls, path: str | Path, base_approximant=None) -> "KroneckerDemodGPSurrogate":
        """Load a checkpoint. ``base_approximant`` overrides the stored name
        (pass an instance for non-registry references, e.g. test stubs)."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("model_class") not in (None, cls.__name__):
            logger.warning("Checkpoint was saved by %s", ck.get("model_class"))
        gp_states = ck["gps"]
        instance = cls(
            ck["mass_ratios"], ck["tau"], ck["y_plus"], ck["y_cross"],
            base_approximant=(base_approximant if base_approximant is not None
                              else ck["base_approximant"]),
            oracle_approximant=ck["oracle_approximant"],
            phase_correction=ck["phase_correction"],
            reference_mass=ck["reference_mass"],
            reference_distance=ck["reference_distance"],
            f_low=ck["f_low"], alpha=ck["alpha"], tau_ref=ck["tau_ref"],
            q_warping=ck["q_warping"], nu=ck["nu"],
            ls_min_q=gp_states["re"]["ls_min"][0],
            ls_min_time=gp_states["re"]["ls_min"][1],
            noise_floor_rel=gp_states["re"]["noise_floor_rel"],
            fit=False,
            covariance_inflation=ck["covariance_inflation"],
            factor_rank=ck["factor_rank"],
        )
        instance._gps = {name: KroneckerGP.from_state(state)
                         for name, state in gp_states.items()}
        return instance

    def __getstate__(self):
        """Pickle via the checkpoint format (the reference holds a SWIG lal.Dict)."""
        import io

        buf = io.BytesIO()
        self.save(buf)
        state = {"_heron_checkpoint": buf.getvalue()}
        # A registry reference is rebuilt by name on load; a non-registry one
        # (e.g. a test stub) has to travel with the pickle.
        gen = self._ref.generator
        if gen is not None:
            try:
                from heron.models import lalsimulation as lalsim_models
                is_registry = hasattr(lalsim_models, self._base_name)
            except ImportError:
                is_registry = False
            if not is_registry:
                state["_base_instance"] = gen
        return state

    def __setstate__(self, state):
        import io

        obj = type(self).load(io.BytesIO(state["_heron_checkpoint"]),
                              base_approximant=state.get("_base_instance"))
        self.__dict__.update(obj.__dict__)

    # -- interface ---------------------------------------------------------

    @property
    def parameter_names(self) -> list[str]:
        return ["mass_ratio", "total_mass"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "mass_ratio": (float(self.mass_ratios[0]), float(self.mass_ratios[-1])),
            "total_mass": (0.0, float("inf")),
        }

    @property
    def hyperparameters(self) -> dict:
        return {name: gp.hyperparameters for name, gp in self._gps.items()}
