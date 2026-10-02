"""Non-stationary kernels for waveform GP surrogates.

The inspiral is slowly-varying (long correlation length in time); merger
and ringdown are fast transients (short correlation length). A single
stationary Matern lengthscale cannot represent both regimes at once — see
the CLAUDE.md "Known Issues" merger/ringdown notes. `NonstationaryMaternKernel`
lets the time lengthscale itself vary smoothly with time, short near merger
and long away from it, instead of relying on a fixed-shape time warp to
compress that variation into a uniform lengthscale.

`build_additive_floor_kernel` was written to address a different structural
issue in the mass-ratio dimension: any noiseless (or near-noiseless) GP's
posterior variance collapses to ~0 at its own training inputs by
construction, producing a training-grid-periodic oscillation in
K(mass_ratio) that biases the GW likelihood's log-det term toward training
nodes regardless of whether the mean is actually more accurate there (see
CLAUDE.md's "Log-det bias" notes and the `k_smoothing_offsets` runtime
patch in `heron/gw_likelihood.py`).

**Tested and found NOT to fix that specific problem** (see
`tests/test_gp_q_floor_kernel.py::test_flattens_variance_oscillation_relative_to_plain_kernel`
and CLAUDE.md's q_floor_kernel entry) — kept as a documented negative
result, not a recommended tool for the log-det bias. The posterior
variance at a training node is, to leading order,
``Var_post(x_i) ≈ σ²`` regardless of kernel shape (a general fact of exact
GP regression at low noise: conditioning on noiseless data collapses the
posterior to a point mass at that input no matter what prior kernel
produced it). Adding a bounded-amplitude long-lengthscale component does
not change this, and — because any lengthscale long enough to act as a
stable "floor" (not itself pinned to `ls_min` and prone to the same
collapse pathology) is also long enough to correlate at ~0.98-1.0 over a
*half* training-grid-spacing distance — it barely changes the antinode
value either. Measured directly: node/antinode variance ratio 822.8
(with) vs 822.5 (without) — no meaningful improvement. The only lever that
demonstrably raises the on-node floor is observation noise
(`noise_floor_rel`); kernel lengthscale changes (short OR long) mainly
trade off *how far* the collapse's footprint extends, not whether it
happens.
"""

from __future__ import annotations

import math

import torch
import gpytorch
from linear_operator.operators import DiagLinearOperator
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


def build_additive_floor_kernel(
    nu: float,
    active_dims: list[int],
    ls_min: float,
    init_ls: float,
    floor_lengthscale: float,
    floor_outputscale_min: float = 0.05,
    floor_outputscale_init: float | None = None,
) -> gpytorch.kernels.Kernel:
    """Short Matern kernel + a fixed-lengthscale 'floor' Matern component.

    ``k(x, x') = k_short(x, x') + floor_outputscale * k_long(x, x')``

    ``k_short`` is an ordinary learnable Matern kernel (``lengthscale``
    floored at ``ls_min``, same as every other dimension in this codebase) —
    it does the sharp local interpolation and is left free to shrink toward
    ``ls_min`` exactly as before, so the mean fit is essentially undisturbed.
    ``k_long``'s lengthscale is *fixed* (``floor_lengthscale``, intended to
    be several times the span of the training grid in this dimension) and
    NOT learned — every experiment in this codebase that let a kernel shape
    parameter float freely under L-BFGS/MLL has collapsed to whichever
    extreme improves in-sample fit the most (see `NonstationaryMaternKernel`
    and the "collapse toward the floor" pattern documented throughout
    CLAUDE.md); fixing it removes that failure mode entirely rather than
    trying to out-bound it. Only ``k_long``'s amplitude
    (``floor_outputscale``, via its own `ScaleKernel`) is learned, floored
    at ``floor_outputscale_min`` so MLL cannot optimise the floor away to
    zero and defeat the point of adding it.

    **Verified NOT to flatten the training-grid-periodic variance
    oscillation targeted by the log-det bias work** (see module docstring
    and `tests/test_gp_q_floor_kernel.py`) — because ``k_long`` varies
    negligibly over one training-grid spacing, it *also* varies negligibly
    over half a spacing, so it explains itself away almost as completely as
    the short component when conditioned on the same nearby training
    point. It does not raise the posterior variance at a training node
    (only nonzero observation noise does that) and does not meaningfully
    reduce the antinode value either. Kept in the codebase as a correctly-
    implemented, well-tested (PSD, bounded, gradient-safe) kernel
    component, but not currently recommended as a fix for K(mass_ratio)'s
    log-det bias.
    """
    from gpytorch.constraints import GreaterThan

    short = gpytorch.kernels.MaternKernel(
        nu=nu, active_dims=active_dims,
        lengthscale_constraint=GreaterThan(ls_min),
    )
    short.lengthscale = init_ls

    long = gpytorch.kernels.MaternKernel(nu=nu, active_dims=active_dims)
    long.lengthscale = floor_lengthscale
    long.raw_lengthscale.requires_grad_(False)

    long_scaled = gpytorch.kernels.ScaleKernel(
        long,
        outputscale_constraint=GreaterThan(floor_outputscale_min),
    )
    long_scaled.outputscale = (
        floor_outputscale_init if floor_outputscale_init is not None
        else floor_outputscale_min * 2.0
    )

    return short + long_scaled


class WarpedMaternKernel(gpytorch.kernels.MaternKernel):
    """Matern kernel evaluated on a fixed, monotonic warp of its 1-D input.

    ``k(x, x') = matern_corr(|phi(x) - phi(x')| / lengthscale)`` for a
    fixed (not learned) monotonic ``phi``. A single stationary lengthscale
    in phi-space corresponds to a varying *effective* lengthscale in raw
    x-space, ``ls_phi / |phi'(x)|`` -- long where phi is compressed
    (phi'(x) small), short where phi is stretched (phi'(x) large). This is
    a simple, closed-form alternative to `NonstationaryMaternKernel`'s
    explicit local-lengthscale parameterisation: it reuses ordinary
    `MaternKernel` machinery (lengthscale constraint, LogNormal prior,
    ``active_dims`` slicing) unchanged, at the cost of ``phi`` having to be
    chosen or fit externally rather than learned jointly with the rest of
    the model.

    ``phi`` is deliberately not learnable, for the same reason `center` in
    `NonstationaryMaternKernel` is fixed rather than free: CLAUDE.md
    documents this codebase's kernels repeatedly collapsing free shape
    parameters to whichever extreme improves in-sample fit, absent a hard
    reason not to.

    Only the input actually reaching this kernel is warped (i.e. whatever
    ``active_dims`` selects) -- the shared training tensor `x` passed to
    the mean function is untouched, so mean functions that need the
    physical (e.g. mass ratio) value -- like the LAL-approximant means in
    `mean.py` -- do not need any changes.
    """

    def __init__(self, warp_fn, **kwargs):
        super().__init__(**kwargs)
        self.warp_fn = warp_fn

    def forward(self, x1, x2, diag=False, **params):
        return super().forward(self.warp_fn(x1), self.warp_fn(x2), diag=diag, **params)


def symmetric_mass_ratio_warp(q: torch.Tensor) -> torch.Tensor:
    """eta(q) = q / (1+q)^2 -- monotonic on q in (0, 1], compresses q->1
    (eta'(q) -> 0) relative to q->0 (eta'(q) -> 1), so a stationary kernel
    on eta(q) has a longer effective lengthscale near q=1 and a shorter
    one near q=0. Same definition as `heron.models.gp.mean._symmetric_mass_ratio`
    (re-derived here, not imported, to keep this module's only dependency
    on `mean.py` being this one well-known closed-form expression, not the
    rest of that module's import surface).
    """
    return q / (1 + q) ** 2


Q_WARP_FUNCTIONS = {
    "eta": symmetric_mass_ratio_warp,
}


class TimeEnvelope(torch.nn.Module):
    """Learned positive amplitude envelope s(t) over (warped) time.

    ``log s`` is piecewise-linear through ``n_knots`` fixed knots spaced
    uniformly over the (warped) training times, so the sparsely sampled
    inspiral is resolved as well as the densely sampled merger and is held constant beyond the outer
    knots. The knot values are ordinary parameters, so the envelope is fitted
    by marginal likelihood with everything else. The raw knot values are
    soft-bounded to +-``max_log`` and centred to zero mean, which removes the
    flat direction against the kernel outputscale and keeps the line search
    from driving the kernel to an ill-conditioned extreme.
    """

    max_log = 5.0

    def __init__(self, times: torch.Tensor, n_knots: int = 12, max_log: float | None = None):
        super().__init__()
        if max_log is not None:
            self.max_log = float(max_log)
        t = times.detach().flatten()
        knots = torch.linspace(float(t.min()), float(t.max()), n_knots, dtype=t.dtype, device=t.device)
        self.register_buffer("knots", knots)
        self.log_s = torch.nn.Parameter(torch.zeros_like(knots))

    def effective_log_s(self) -> torch.Tensor:
        bounded = self.max_log * torch.tanh(self.log_s / self.max_log)
        return bounded - bounded.mean()

    def roughness(self) -> torch.Tensor:
        d2 = torch.diff(self.effective_log_s(), n=2)
        return (d2**2).sum()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        knots = self.knots
        log_s = self.effective_log_s()
        idx = torch.clamp(torch.searchsorted(knots, t.contiguous()) - 1, 0, len(knots) - 2)
        k0, k1 = knots[idx], knots[idx + 1]
        frac = torch.clamp((t - k0) / (k1 - k0), 0.0, 1.0)
        return torch.exp(log_s[idx] * (1.0 - frac) + log_s[idx + 1] * frac)


def _scale(envelope, q_envelope, x):
    s = envelope(x[..., -1])
    if q_envelope is not None:
        s = s * q_envelope(x[..., 0])
    return s


class EnvelopeKernel(gpytorch.kernels.Kernel):
    """k(x, x') = s(t) s(t') k_base(x, x'), with t the last input column.

    An optional ``q_envelope`` adds a second factor Q(q) Q(q') over the first
    input column, so the outputscale can differ between parameter regions.
    """

    def __init__(self, base_kernel: gpytorch.kernels.Kernel, envelope: TimeEnvelope,
                 q_envelope: TimeEnvelope | None = None):
        super().__init__()
        self.base_kernel = base_kernel
        self.envelope = envelope
        self.q_envelope = q_envelope

    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False, **params):
        s1 = _scale(self.envelope, self.q_envelope, x1)
        s2 = _scale(self.envelope, self.q_envelope, x2)
        if diag:
            return self.base_kernel(x1, x2, diag=True, **params) * s1 * s2
        k = self.base_kernel(x1, x2, **params).to_dense()
        return s1.unsqueeze(-1) * k * s2.unsqueeze(-2)


class EnvelopeNoise(gpytorch.likelihoods.noise_models.HomoskedasticNoise):
    """Observation noise sigma^2 s(t)^2, sharing the kernel's envelope so the
    signal-to-noise ratio is stationary."""

    def __init__(self, envelope: TimeEnvelope, q_envelope: TimeEnvelope | None = None, **kwargs):
        super().__init__(**kwargs)
        self.envelope = envelope
        self.q_envelope = q_envelope

    def forward(self, *params, shape=None, **kwargs):
        base = super().forward(*params, shape=shape, **kwargs)
        x = params[0] if params else None
        if x is None or not torch.is_tensor(x) or "noise" in kwargs:
            return base
        s = _scale(self.envelope, self.q_envelope, x)
        return DiagLinearOperator(base.diagonal(dim1=-1, dim2=-2) * s**2)
