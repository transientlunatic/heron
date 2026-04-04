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
        q = x[:, 0]
        t_input = x[:, -1]

        # Unwarp time if needed
        if self.warping is not None:
            t = self.warping.unwarp(t_input)
        else:
            t = t_input

        eta = self._symmetric_mass_ratio(q)

        # Time to coalescence (t < 0 during inspiral, t = 0 at merger)
        # τ = -t for t < 0
        tau = -t

        # Newtonian chirp: the waveform is non-zero only during inspiral
        # where τ > 0 (i.e., t < 0)
        inspiral_mask = tau > 1e-6

        # Dimensionless time parameter
        # Θ = η τ / (5 t_scale) — the PN expansion parameter
        theta = eta.unsqueeze(-1) if eta.dim() == 0 else eta
        tau_dimless = tau / (5.0 * self._t_scale)

        # Amplitude: h ∝ η^(2/5) Θ^(-1/4) * h_scale
        # with Θ = η τ / (5 t_scale)
        amplitude = torch.zeros_like(t)
        safe_tau = torch.where(inspiral_mask, tau_dimless, torch.ones_like(tau_dimless))
        amplitude = torch.where(
            inspiral_mask,
            4.0 * self._h_scale * eta * safe_tau ** (-0.25),
            torch.zeros_like(t),
        )

        # Phase: Φ(τ) = -2 (τ/(5 t_scale))^(5/8) / η^(3/8)
        phase = torch.where(
            inspiral_mask,
            -2.0 * safe_tau ** 0.625 / (eta ** 0.375 + 1e-30),
            torch.zeros_like(t),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x[:, 0]
        t_input = x[:, -1]

        if self.warping is not None:
            t = self.warping.unwarp(t_input)
        else:
            t = t_input

        eta = self._symmetric_mass_ratio(q)

        tau = -t
        inspiral_mask = tau > 1e-6

        tau_dimless = tau / (5.0 * self._t_scale)
        safe_tau = torch.where(inspiral_mask, tau_dimless, torch.ones_like(tau_dimless))

        # Newtonian amplitude
        amplitude = torch.where(
            inspiral_mask,
            4.0 * self._h_scale * eta * safe_tau ** (-0.25),
            torch.zeros_like(t),
        )

        # Phase with 1PN correction
        # Φ_0 = -2 Θ^(5/8) / η^(3/8)
        # Φ_1PN = (3715/8064 + 55η/96) Θ^(3/8) / η^(5/8)
        theta_5_8 = safe_tau ** 0.625
        theta_3_8 = safe_tau ** 0.375
        eta_safe = eta + 1e-30

        phase_0 = -2.0 * theta_5_8 / eta_safe ** 0.375
        phase_1pn = (3715.0 / 8064.0 + 55.0 * eta / 96.0) * theta_3_8 / eta_safe ** 0.625

        phase = torch.where(
            inspiral_mask,
            phase_0 + phase_1pn,
            torch.zeros_like(t),
        )

        h = amplitude * torch.cos(phase)
        return h * self.output_scale
