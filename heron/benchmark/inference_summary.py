"""NS-bias, PP-coverage and training-cost summaries (B0 measurements 4-6).

Pure functions on arrays/dicts, so they run on saved campaign output
(``results/*.npz``) without lalsuite, nested sampling or a GPU.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..inference.coverage import ks_uniform_pvalue


# --- 4. NS bias, with-K vs no-K -------------------------------------------

def ns_summary(samples, truth, bins: int = 60, nodes=None,
               node_tol: float = 0.003) -> dict:
    """One posterior: shift from truth in sigma, 68% CI, saved histogram,
    mode count, and (if ``nodes`` given) the training-node clustering ratio."""
    x = np.asarray(samples, float)
    med, std = float(np.median(x)), float(np.std(x))
    lo, hi = np.percentile(x, [16, 84])
    counts, edges = np.histogram(x, bins=bins)
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    peaks, _ = signal.find_peaks(smooth, prominence=0.15 * smooth.max())
    out = {
        "n": len(x), "median": med, "std": std,
        "ci68": (float(lo), float(hi)),
        "displacement_sigma": float((med - truth) / std) if std > 0 else float("inf"),
        "n_modes": int(len(peaks)),
        "hist_counts": counts.tolist(), "hist_edges": edges.tolist(),
    }
    if nodes is not None:
        out["node_clustering"] = node_clustering(x, nodes, node_tol)
    return out


def node_clustering(samples, nodes, tol: float = 0.003) -> float:
    """Node-snapping ratio: posterior mass within ``tol`` of a training node,
    divided by the mass within ``tol`` of a *midpoint* between nodes.

    ~1 for a posterior with no preference; >>1 when samples pile up on nodes
    (the grid-snap failure).  Comparing nodes with midpoints, rather than with a
    width-based expectation, stays meaningful when the posterior is narrower
    than ``tol``.  Meant for injections *off* the node grid -- a truth sitting
    on a node legitimately gives a large ratio.  A +1/n floor keeps it finite.
    """
    x = np.asarray(samples, float)
    nodes = np.sort(np.asarray(nodes, float))
    mids = 0.5 * (nodes[:-1] + nodes[1:])
    near_node = np.mean(np.min(np.abs(x[:, None] - nodes[None, :]), axis=1) <= tol)
    near_mid = np.mean(np.min(np.abs(x[:, None] - mids[None, :]), axis=1) <= tol)
    eps = 1.0 / len(x)
    return float((near_node + eps) / (near_mid + eps))


def compare_with_without_k(with_k, no_k, truth, nodes=None,
                           bias_slack_sigma: float = 1.0,
                           max_clustering: float = 2.0) -> dict:
    """B0 pass rule: with-K no more biased than no-K (within ``bias_slack_sigma``,
    measured in the *with-K* sigma) and no clustering on training nodes."""
    w, n = ns_summary(with_k, truth, nodes=nodes), ns_summary(no_k, truth, nodes=nodes)
    # no-K displacement re-expressed in with-K sigma so the two are comparable
    n_disp_in_w_sigma = (n["median"] - truth) / w["std"] if w["std"] > 0 else float("inf")
    not_more_biased = abs(w["displacement_sigma"]) <= abs(n_disp_in_w_sigma) + bias_slack_sigma
    clustering = w.get("node_clustering", float("nan"))
    no_snap = bool(np.isnan(clustering) or clustering < max_clustering)
    return {"with_k": w, "no_k": n,
            "width_ratio": w["std"] / n["std"] if n["std"] > 0 else float("inf"),
            "not_more_biased": bool(not_more_biased), "no_node_clustering": no_snap,
            "passed": bool(not_more_biased and no_snap)}


def multi_seed_shift(medians, sigmas, truth) -> dict:
    """Seed-averaged shift for the q~0.505 midpoint injection (5 noise seeds).

    Returns the mean displacement in sigma and its standard error across seeds,
    so a systematic ~2 sigma shift can be told from a single unlucky draw.
    """
    d = (np.asarray(medians, float) - truth) / np.asarray(sigmas, float)
    n = len(d)
    sem = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    mean = float(d.mean())
    return {"n_seeds": n, "per_seed_sigma": d.tolist(), "mean_sigma": mean,
            "sem": sem, "significant": bool(n > 1 and abs(mean) > 2 * sem)}


# --- 5. PP coverage ---------------------------------------------------------

def pp_summary(credible_levels, band_sigma: float = 2.0) -> dict:
    """KS p-value plus the pointwise binomial band check of the PP curve.

    ``within_band_fraction`` is the fraction of the ECDF evaluated on a grid of
    nominal levels that lies inside ``+-band_sigma`` binomial standard errors
    of the diagonal (well-calibrated: ~0.95 for 2 sigma, curve hugging 1).
    """
    cl = np.sort(np.asarray(credible_levels, float))
    n = len(cl)
    grid = np.linspace(0.01, 0.99, 99)
    ecdf = np.searchsorted(cl, grid, side="right") / n
    se = np.sqrt(grid * (1 - grid) / n)
    inside = np.abs(ecdf - grid) <= band_sigma * se
    return {"n": n, "ks_pvalue": ks_uniform_pvalue(cl),
            "within_band_fraction": float(inside.mean()),
            "max_abs_deviation": float(np.max(np.abs(ecdf - grid)))}


# --- 6. training cost --------------------------------------------------------

def bound_position(value: float, lower: float, upper: float | None = None,
                   stuck_tol: float = 0.02) -> dict:
    """Where a hyperparameter sits in its allowed range.

    Position is measured in log space when both bounds are positive (the
    natural scale for lengthscales).  ``stuck_at_lower`` flags a value within
    ``stuck_tol`` of the lower bound -- the overfitting signature in this
    codebase's history (floor-pinned lengthscales).
    """
    if upper is not None and lower > 0 and upper > lower and value > 0:
        pos = float((np.log(value) - np.log(lower)) / (np.log(upper) - np.log(lower)))
    elif upper is not None and upper > lower:
        pos = float((value - lower) / (upper - lower))
    else:
        pos = float("nan")
    rel = float((value - lower) / abs(lower)) if lower != 0 else float("nan")
    return {"value": float(value), "lower": float(lower), "position": pos,
            "rel_above_lower": rel,
            "stuck_at_lower": bool(np.isfinite(rel) and rel <= stuck_tol)}


def training_cost_summary(wall_seconds: float, hyperparameters: dict,
                          bounds: dict, gpu_memory_bytes: float | None = None) -> dict:
    """Wall time, peak GPU memory (if known) and each bounded hyperparameter's
    position; ``n_stuck`` counts those pinned at their lower bound."""
    pos = {}
    for name, (lo, hi) in bounds.items():
        if name in hyperparameters:
            pos[name] = bound_position(float(np.ravel(hyperparameters[name])[0]), lo, hi)
    return {"wall_seconds": float(wall_seconds), "gpu_memory_bytes": gpu_memory_bytes,
            "hyperparameters": pos,
            "n_stuck": int(sum(p["stuck_at_lower"] for p in pos.values()))}


# --- 3. likelihood cost -------------------------------------------------------

def scaling_benchmark(make_call, sizes, measure_up_to, repeats: int = 3,
                      exponent: float | None = None) -> list[dict]:
    """Wall time per call vs N; sizes above ``measure_up_to`` are extrapolated
    from a power-law fit to the measured points and **labelled as such**.

    ``make_call(n)`` returns a zero-argument callable to time.  ``exponent``
    overrides the fitted slope (e.g. 3 for a dense factorisation).
    """
    import time

    rows, ns, ts = [], [], []
    for n in sizes:
        if n <= measure_up_to:
            fn = make_call(n)
            fn()                                   # warm up
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            t = (time.perf_counter() - t0) / repeats
            ns.append(n), ts.append(t)
            rows.append({"N": n, "seconds": t, "extrapolated": False})
    if len(ns) >= 2:
        slope = exponent if exponent is not None else float(np.polyfit(np.log(ns), np.log(ts), 1)[0])
        ref_n, ref_t = ns[-1], ts[-1]
        for n in sizes:
            if n > measure_up_to:
                rows.append({"N": n, "seconds": float(ref_t * (n / ref_n) ** slope),
                             "extrapolated": True, "assumed_exponent": slope})
    return rows
