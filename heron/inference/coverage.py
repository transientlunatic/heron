"""Coverage / PP-plot utilities for validating posterior calibration.

For a well-calibrated pipeline the *credible level* of the injected truth — the
fraction of posterior mass below it — is uniform on [0, 1] across many noise
realisations.  Aggregating credible levels over an injection campaign and
plotting their empirical CDF against the diagonal is the PP-plot; deviations
outside the binomial confidence bands flag mis-calibration.

These helpers are lifted from ``scripts/pp_plot_demod.py`` so the same
credible-level and plotting logic is reusable from any campaign script.  The
grid path is exact and cheap for low-dimensional (e.g. 2-D ``(q, tc)``)
problems; for higher dimensions compute credible levels from nested-sampling
posteriors instead and pass them to :func:`pp_plot` directly.
"""
from __future__ import annotations

import numpy as np


def credible_level_1d(log_density: np.ndarray, grid: np.ndarray, truth: float) -> float:
    """Posterior CDF at *truth* for a 1-D (unnormalised) log density on *grid*."""
    from scipy.integrate import trapezoid

    p = np.exp(log_density - np.max(log_density))
    p = p / trapezoid(p, grid)
    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))]
    )
    return float(np.interp(truth, grid, cdf))


def credible_levels_from_grid(
    logL_grid: np.ndarray,
    grids: list[np.ndarray],
    truths: list[float],
) -> list[float]:
    """Per-axis credible levels of the truth for an N-D log-likelihood grid.

    Each axis is marginalised (log-sum-exp over the others) under a uniform
    prior, then :func:`credible_level_1d` is evaluated at that axis's truth.

    Parameters
    ----------
    logL_grid : ndarray
        Log-likelihood evaluated on the tensor-product *grids*.
    grids : list[ndarray]
        The 1-D grid for each axis, in ``logL_grid``'s axis order.
    truths : list[float]
        Injected truth per axis.

    Returns
    -------
    list[float]
        Credible level of the truth for each axis.
    """
    from scipy.special import logsumexp

    ndim = logL_grid.ndim
    out = []
    for axis in range(ndim):
        other = tuple(a for a in range(ndim) if a != axis)
        log_marg = logsumexp(logL_grid, axis=other)
        out.append(credible_level_1d(log_marg, grids[axis], truths[axis]))
    return out


def ks_uniform_pvalue(credible_levels: np.ndarray) -> float:
    """KS-test p-value that *credible_levels* are uniform on [0, 1]."""
    from scipy.stats import kstest

    return float(kstest(np.asarray(credible_levels), "uniform").pvalue)


def pp_plot(
    credible_levels: dict[str, np.ndarray],
    output: str | None = None,
    title: str | None = None,
):
    """Draw a PP-plot from per-parameter credible-level arrays.

    Parameters
    ----------
    credible_levels : dict[str, ndarray]
        Mapping of label → array of credible levels (one per injection).
    output : str or None
        If given, save the figure there.
    title : str or None
        Optional plot title.

    Returns
    -------
    (fig, ax, pvalues)
        The matplotlib figure/axis and a dict of per-label KS p-values.
    """
    import matplotlib.pyplot as plt

    n = len(next(iter(credible_levels.values())))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    x = np.linspace(0, 1, 200)
    for z, alpha in ((1, 0.3), (2, 0.18), (3, 0.1)):
        band = z * np.sqrt(np.clip(x * (1 - x), 0, None) / n)
        ax.fill_between(x, np.clip(x - band, 0, 1), np.clip(x + band, 0, 1),
                        color="gray", alpha=alpha, lw=0)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)

    pvalues = {}
    yy = np.arange(1, n + 1) / n
    for i, (label, cl) in enumerate(credible_levels.items()):
        p = ks_uniform_pvalue(cl)
        pvalues[label] = p
        ax.plot(np.sort(cl), yy, color=f"C{i}", lw=1.8,
                label=f"{label}  (KS p={p:.2f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("credible level of truth")
    ax.set_ylabel("fraction of injections")
    ax.legend(loc="upper left", fontsize=9)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=130)
    return fig, ax, pvalues
