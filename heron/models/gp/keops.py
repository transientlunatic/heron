"""KeOps-backed exact-GP prediction for the heron GP surrogates.

`ExactGPSurrogate.predict()` spends almost all of its time (and, at long
analysis segments, all of its memory) on the dense N x M test-train
cross-covariance ``K_*`` that gpytorch materialises inside
``exact_prediction``, and -- because prediction runs under
``fast_pred_var`` -- on the ``K_* @ R`` product with the M x M Cholesky
root ``R`` of ``(K + sigma^2 I)^{-1}``, which gpytorch computes eagerly even
when only the mean is wanted.

This module evaluates the same posterior without ever forming ``K_*``:

* **Mean** -- ``m(x_*) + K_* alpha`` is a single KeOps kernel reduction
  (``pykeops.torch.LazyTensor``): ``K_*`` stays symbolic, memory is
  O(N + M) and the O(N M) work runs in one fused kernel.
* **Diagonal variance** -- ``k(x_*, x_*) - rowsum((K_* R)^2)``. With the
  exact (Cholesky, full-rank) root that heron deliberately uses (raised
  ``max_cholesky_size``), ``K_* R`` is an O(N M^2) matrix product against a
  dense M x M matrix. That is a GEMM, not a kernel reduction, and is *not*
  profitably KeOps-able: a KeOps reduction against an M-wide right-hand side
  must be split into blocks and re-evaluates the kernel once per block
  (measured 2-3x slower than BLAS/cuBLAS on both CPU and a 2080 Ti). It is
  instead computed in row chunks -- ``K_*`` for a chunk of test points is
  built densely from the *same* formula, multiplied by ``R`` and reduced --
  so peak memory is bounded by the chunk size rather than N x M.

The symbolic kernel is built by walking the surrogate's gpytorch kernel
tree (`ScaleKernel`, `ProductKernel`, `AdditiveKernel`, `MaternKernel`,
`WarpedMaternKernel`, `NonstationaryMaternKernel`, honouring
``active_dims``) and re-expressing each node with the same hyperparameters,
warping and float64 arithmetic as its ``forward``. The same builder emits
either a KeOps ``LazyTensor`` or a dense torch tensor, so the mean and the
variance chunks share one formula. The ``alpha`` (``mean_cache``) and ``R``
(``covar_cache``) solves are taken from the model's own gpytorch
prediction strategy, so they are bit-for-bit the caches the dense path
uses. Any other kernel type raises `UnsupportedKernelError` rather than
silently guessing.
"""

from __future__ import annotations

import math
import warnings

import torch
import gpytorch

from .kernels import NonstationaryMaternKernel, WarpedMaternKernel

try:  # pragma: no cover - availability depends on the environment
    from pykeops.torch import LazyTensor as _LazyTensor

    KEOPS_AVAILABLE = True
except Exception:  # ImportError, or a broken compiler toolchain at import
    _LazyTensor = None
    KEOPS_AVAILABLE = False


#: Upper bound on the number of elements in one dense chunk of ``K_*`` used
#: for the diagonal variance (2**24 float64 = 128 MiB per temporary; the
#: formula needs a handful of temporaries plus the ``K_* R`` chunk).
DENSE_CHUNK_ELEMENTS = 2**24


class UnsupportedKernelError(NotImplementedError):
    """The kernel tree contains a node with no KeOps/dense re-expression."""


def require_keops() -> None:
    if not KEOPS_AVAILABLE:
        raise ImportError(
            "use_keops=True needs pykeops (pip install pykeops); it could not "
            "be imported in this environment."
        )


# --------------------------------------------------------------------------
# Backends: how a per-point column becomes an i- or j-indexed symbolic term.
# Both return objects supporting + - * / ** .abs() .exp() .sqrt().
# --------------------------------------------------------------------------


class _DenseBackend:
    @staticmethod
    def i(v: torch.Tensor):
        return v.reshape(-1, 1)

    @staticmethod
    def j(v: torch.Tensor):
        return v.reshape(1, -1)


class _KeOpsBackend:
    @staticmethod
    def i(v: torch.Tensor):
        return _LazyTensor(v.reshape(-1, 1, 1).contiguous())

    @staticmethod
    def j(v: torch.Tensor):
        return _LazyTensor(v.reshape(1, -1, 1).contiguous())


def _matern_from_distance(d, nu: float):
    """gpytorch's Matern correlation as a function of the scaled distance."""
    exp_component = (-math.sqrt(nu * 2) * d).exp()
    if nu == 0.5:
        return exp_component
    if nu == 1.5:
        return (math.sqrt(3) * d + 1.0) * exp_component
    if nu == 2.5:
        return (math.sqrt(5) * d + 1.0 + 5.0 / 3.0 * d**2) * exp_component
    raise UnsupportedKernelError(f"Matern nu={nu} not supported")


def _matern(kernel: gpytorch.kernels.MaternKernel, x1, x2, be):
    # Mirrors gpytorch.kernels.MaternKernel.forward: centre on x1's mean,
    # divide by the lengthscale, Euclidean distance.
    ls = kernel.lengthscale.reshape(-1)
    if ls.numel() not in (1, x1.shape[-1]):
        raise UnsupportedKernelError("lengthscale shape does not match inputs")
    centre = x1.mean(dim=-2, keepdim=True)
    u1 = (x1 - centre).div(ls)
    u2 = (x2 - centre).div(ls)
    if u1.shape[-1] == 1:
        d = (be.i(u1[:, 0]) - be.j(u2[:, 0])).abs()
    else:
        sq = None
        for k in range(u1.shape[-1]):
            term = (be.i(u1[:, k]) - be.j(u2[:, k])) ** 2
            sq = term if sq is None else sq + term
        d = sq.sqrt()
    return _matern_from_distance(d, kernel.nu)


def _nonstationary_matern(kernel: NonstationaryMaternKernel, x1, x2, be):
    # Mirrors NonstationaryMaternKernel.forward (Paciorek & Schervish).
    t1 = x1[:, 0]
    t2 = x2[:, 0]
    l1 = be.i(kernel.local_lengthscale(t1))
    l2 = be.j(kernel.local_lengthscale(t2))
    diff = be.i(t1) - be.j(t2)
    sumsq = l1**2 + l2**2
    prefactor = (2.0 * l1 * l2 / sumsq).sqrt()
    s = (2.0 * diff**2 / sumsq + 1e-12).sqrt()
    if kernel.nu == 0.5:
        corr = (-s).exp()
    elif kernel.nu == 1.5:
        sqrt3 = math.sqrt(3.0)
        corr = (1.0 + sqrt3 * s) * (-sqrt3 * s).exp()
    else:  # 2.5 (validated in the kernel's __init__)
        sqrt5 = math.sqrt(5.0)
        corr = (1.0 + sqrt5 * s + 5.0 * s**2 / 3.0) * (-sqrt5 * s).exp()
    return prefactor * corr


def _symbolic(kernel: gpytorch.kernels.Kernel, x1, x2, be):
    """Re-express ``kernel(x1, x2)`` (test x train) on backend ``be``."""
    if len(getattr(kernel, "batch_shape", torch.Size())) != 0:
        raise UnsupportedKernelError("batched kernels are not supported")
    if kernel.active_dims is not None:
        x1 = x1.index_select(-1, kernel.active_dims)
        x2 = x2.index_select(-1, kernel.active_dims)

    if isinstance(kernel, gpytorch.kernels.ScaleKernel):
        return float(kernel.outputscale.detach()) * _symbolic(kernel.base_kernel, x1, x2, be)
    if isinstance(kernel, gpytorch.kernels.ProductKernel):
        out = None
        for k in kernel.kernels:
            term = _symbolic(k, x1, x2, be)
            out = term if out is None else out * term
        return out
    if isinstance(kernel, gpytorch.kernels.AdditiveKernel):
        out = None
        for k in kernel.kernels:
            term = _symbolic(k, x1, x2, be)
            out = term if out is None else out + term
        return out
    if isinstance(kernel, NonstationaryMaternKernel):
        return _nonstationary_matern(kernel, x1, x2, be)
    if isinstance(kernel, WarpedMaternKernel):
        # Fixed pointwise warp: apply it to the (active-dims) inputs, then
        # the ordinary Matern -- exactly WarpedMaternKernel.forward.
        return _matern(kernel, kernel.warp_fn(x1), kernel.warp_fn(x2), be)
    if type(kernel) is gpytorch.kernels.MaternKernel:
        return _matern(kernel, x1, x2, be)
    raise UnsupportedKernelError(
        f"No KeOps expression for kernel type {type(kernel).__name__}"
    )


def dense_cross_covariance(kernel, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Dense ``kernel(x1, x2)`` from the same formula the KeOps path uses."""
    k = _symbolic(kernel, x1, x2, _DenseBackend)
    return k.expand(x1.shape[0], x2.shape[0])


def keops_cross_matmul(kernel, x1: torch.Tensor, x2: torch.Tensor,
                       rhs: torch.Tensor) -> torch.Tensor:
    """``kernel(x1, x2) @ rhs`` with the cross-covariance kept symbolic."""
    require_keops()
    k = _symbolic(kernel, x1, x2, _KeOpsBackend)
    return k @ rhs.contiguous()


# --------------------------------------------------------------------------
# Posterior
# --------------------------------------------------------------------------


def _prediction_strategy(model: gpytorch.models.ExactGP):
    """The model's gpytorch prediction strategy, creating it if needed.

    Created by one throw-away single-point prediction with posterior
    variances skipped, so building it never triggers the O(M^3) root-inverse
    (``covar_cache``) that only the variance needs. The caches it then
    exposes are exactly those the dense ``model(x)`` path uses.
    """
    if model.prediction_strategy is None:
        x0 = model.train_inputs[0][:1]
        with gpytorch.settings.skip_posterior_variances():
            model(x0)
    return model.prediction_strategy


def latent_posterior(
    model: gpytorch.models.ExactGP,
    x: torch.Tensor,
    *,
    cholesky_size: int,
    mean: bool = True,
    variance: bool = False,
    chunk_elements: int | None = None,
):
    """Latent posterior mean and/or diagonal variance at ``x``.

    Numerically equivalent to ``model(x).mean`` / ``model(x).variance``
    evaluated under ``fast_pred_var()`` and
    ``max_cholesky_size(cholesky_size)`` (the settings
    `ExactGPSurrogate.predict` uses), including gpytorch's clamp of tiny /
    negative variances to ``settings.min_variance``.

    Returns ``(mean or None, variance or None)``.
    """
    if chunk_elements is None:
        chunk_elements = DENSE_CHUNK_ELEMENTS
    kernel = model.covar_module
    out_mean = out_var = None
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.max_cholesky_size(cholesky_size):
        strategy = _prediction_strategy(model)
        train_x = model.train_inputs[0]
        x = x.to(dtype=train_x.dtype, device=train_x.device)

        if mean:
            alpha = strategy.mean_cache
            if alpha.dim() != 1:
                raise UnsupportedKernelError("batched/multitask models are not supported")
            cross = keops_cross_matmul(kernel, x, train_x, alpha.unsqueeze(-1))
            out_mean = model.mean_module(x) + cross.squeeze(-1)

        if variance:
            root = strategy.covar_cache  # R with R R^T = (K + sigma^2 I)^{-1}
            prior_var = kernel(x, diag=True)
            if not torch.is_tensor(prior_var):  # pragma: no cover
                prior_var = prior_var.to_dense()
            n, m = x.shape[0], train_x.shape[0]
            rows = max(1, chunk_elements // max(m, root.shape[-1]))
            reduction = torch.empty(n, dtype=train_x.dtype, device=train_x.device)
            for start in range(0, n, rows):
                stop = min(start + rows, n)
                k_chunk = dense_cross_covariance(kernel, x[start:stop], train_x)
                v = k_chunk @ root
                reduction[start:stop] = (v * v).sum(-1)
                del k_chunk, v
            out_var = prior_var - reduction
            min_variance = gpytorch.settings.min_variance.value(out_var.dtype)
            if out_var.lt(min_variance).any():
                warnings.warn(
                    "Negative variance values detected. This is likely due to "
                    "numerical instabilities. Rounding negative variances up "
                    f"to {min_variance}.",
                    gpytorch.utils.warnings.NumericalWarning,
                )
                out_var = out_var.clamp_min(min_variance)
    return out_mean, out_var
