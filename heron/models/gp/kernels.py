"""Non-stationary kernels for waveform GP surrogates.

The inspiral is slowly-varying (long correlation length in time); merger
and ringdown are fast transients (short correlation length). A single
stationary Matern lengthscale cannot represent both regimes at once — see
the CLAUDE.md "Known Issues" merger/ringdown notes. `NonstationaryMaternKernel`
lets the time lengthscale itself vary smoothly with time, short near merger
and long away from it, instead of relying on a fixed-shape time warp to
compress that variation into a uniform lengthscale.
"""

from __future__ import annotations

import math

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Interval


class NonstationaryMaternKernel(gpytorch.kernels.Kernel):
    """Matern kernel with a time-varying lengthscale.

    Generalises the stationary Matern kernel via Paciorek & Schervish
    (2004): each point x has its own local lengthscale l(x), and

        k(x, x') = sqrt(2 l(x) l(x') / (l(x)^2 + l(x')^2))
                   * matern_corr(s(x, x'))

        s(x, x') = sqrt(2 (x - x')^2 / (l(x)^2 + l(x')^2))

    which reduces to the ordinary stationary Matern correlation when
    l(x) = l(x') = l, and is positive-definite for any smooth positive
    l(.) (Paciorek & Schervish, "Nonstationary covariance functions for
    Gaussian process regression", NeurIPS 2004).

    l(.) is parameterised as a sigmoid blend between two lengthscales,
    ``lengthscale_far`` (used away from merger) and ``lengthscale_near``
    (used at/after merger), transitioning around ``center`` over a scale
    ``width``:

        l(t) = lengthscale_near
               + (lengthscale_far - lengthscale_near) * sigmoid((center - t) / width)

    ``center`` is a fixed physical constant, not learnable: every warping
    in this codebase maps raw t=0 (merger, by the `mean.py` convention "t=0
    at merger") to warped 0 exactly, so there is nothing to learn — it is
    known a priori. Two earlier attempts at making it learnable (bounded to
    the full data range, then bounded to +-20*width with an informative
    prior) both still collapsed: L-BFGS consistently drove it to the edge
    of whatever range it was allowed, because a *shorter* lengthscale
    always improves in-sample MLL fit unless something stops it exploiting
    that (the same "collapse toward the floor" pathology CLAUDE.md
    documents for plain Matern lengthscales, playing out here through the
    transition location instead). Confirmed directly on real training data
    (dense10 grid, N=2000): with center bounded to +-0.4 (warped), it
    still pinned at -0.4, which unwarps to t=-0.92s — deep in the
    inspiral, nowhere near merger/ringdown. Fixing ``center`` removes that
    exploit entirely; ``width`` is bounded on *both* sides (a floor against
    collapsing to a hard step, a ceiling against expanding to swallow the
    inspiral) for the same reason.

    ``lengthscale_far``, ``lengthscale_near`` and ``width`` are learnable.
    Expects a single active input dimension (time).
    """

    has_lengthscale = False

    def __init__(
        self,
        nu: float = 2.5,
        ls_min_far: float = 0.0005,
        ls_min_near: float = 0.0005,
        min_width: float = 1e-3,
        max_width: float = 1.0,
        init_ls_far: float = 1.0,
        init_ls_near: float = 0.1,
        center: float = 0.0,
        init_width: float = 0.1,
        **kwargs,
    ):
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"nu must be 0.5, 1.5, or 2.5, got {nu}")
        if not (min_width < init_width < max_width):
            raise ValueError(
                f"init_width={init_width} must lie strictly within "
                f"(min_width={min_width}, max_width={max_width})"
            )
        super().__init__(**kwargs)
        self.nu = nu
        self.register_buffer("center", torch.as_tensor(float(center)))

        self.register_parameter("raw_lengthscale_far", torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter("raw_lengthscale_near", torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter("raw_width", torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        self.register_constraint("raw_lengthscale_far", GreaterThan(ls_min_far))
        self.register_constraint("raw_lengthscale_near", GreaterThan(ls_min_near))
        self.register_constraint("raw_width", Interval(min_width, max_width))

        self.lengthscale_far = init_ls_far
        self.lengthscale_near = init_ls_near
        self.width = init_width

    @property
    def lengthscale_far(self) -> torch.Tensor:
        return self.raw_lengthscale_far_constraint.transform(self.raw_lengthscale_far)

    @lengthscale_far.setter
    def lengthscale_far(self, value):
        value = torch.as_tensor(value, dtype=self.raw_lengthscale_far.dtype)
        self.initialize(raw_lengthscale_far=self.raw_lengthscale_far_constraint.inverse_transform(value))

    @property
    def lengthscale_near(self) -> torch.Tensor:
        return self.raw_lengthscale_near_constraint.transform(self.raw_lengthscale_near)

    @lengthscale_near.setter
    def lengthscale_near(self, value):
        value = torch.as_tensor(value, dtype=self.raw_lengthscale_near.dtype)
        self.initialize(raw_lengthscale_near=self.raw_lengthscale_near_constraint.inverse_transform(value))

    @property
    def width(self) -> torch.Tensor:
        return self.raw_width_constraint.transform(self.raw_width)

    @width.setter
    def width(self, value):
        value = torch.as_tensor(value, dtype=self.raw_width.dtype)
        self.initialize(raw_width=self.raw_width_constraint.inverse_transform(value))

    def local_lengthscale(self, t: torch.Tensor) -> torch.Tensor:
        """l(t), the local time lengthscale at each point in `t`."""
        width = self.width.squeeze(-1)
        l_far = self.lengthscale_far.squeeze(-1)
        l_near = self.lengthscale_near.squeeze(-1)
        gate = torch.sigmoid((self.center - t) / width)
        return l_near + (l_far - l_near) * gate

    def forward(self, x1, x2, diag=False, **params):
        t1 = x1[..., 0]
        t2 = x2[..., 0]
        l1 = self.local_lengthscale(t1)
        l2 = self.local_lengthscale(t2)

        if diag:
            diff = t1 - t2
        else:
            diff = t1.unsqueeze(-1) - t2.unsqueeze(-2)
            l1 = l1.unsqueeze(-1)
            l2 = l2.unsqueeze(-2)

        sumsq = l1.pow(2) + l2.pow(2)
        prefactor = torch.sqrt(2.0 * l1 * l2 / sumsq)
        # s(x,x') reduces to the ordinary |x-x'|/l when l1 == l2 == l.
        s = torch.sqrt(2.0 * diff.pow(2) / sumsq + 1e-12)

        if self.nu == 0.5:
            corr = torch.exp(-s)
        elif self.nu == 1.5:
            sqrt3 = math.sqrt(3.0)
            corr = (1.0 + sqrt3 * s) * torch.exp(-sqrt3 * s)
        else:  # nu == 2.5
            sqrt5 = math.sqrt(5.0)
            corr = (1.0 + sqrt5 * s + 5.0 * s.pow(2) / 3.0) * torch.exp(-sqrt5 * s)

        return prefactor * corr
