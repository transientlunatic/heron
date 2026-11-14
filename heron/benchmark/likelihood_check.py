"""Likelihood correctness: low-rank/Woodbury path vs the dense reference."""
from __future__ import annotations

import numpy as np
import torch

from ..likelihood import LowRankMarginalLogLikelihood, MarginalLogLikelihood


def likelihood_agreement(C, mu, U, d) -> dict:
    """Compare ``LowRankMarginalLogLikelihood(L_C, mu, U)`` with the dense
    ``MarginalLogLikelihood(C, mu, U U^T)`` on the same problem.

    Returns both log-likelihoods and the relative difference
    ``|a - b| / max(|a|, |b|)`` (B0 bar: ~1e-8).  ``U=None`` means K = 0.
    """
    C = np.asarray(C, float)
    n = C.shape[0]
    if U is None:
        K = np.zeros((n, n))
    else:
        U = np.asarray(U, float)
        K = U @ U.T
    dense = MarginalLogLikelihood(C, mu, K)(d)
    L_C = torch.linalg.cholesky(torch.as_tensor(C, dtype=torch.float64))
    low = LowRankMarginalLogLikelihood(L_C, mu, U)(d)
    rel = abs(dense - low) / max(abs(dense), abs(low), 1e-300)
    return {"dense": float(dense), "lowrank": float(low), "rel_diff": float(rel)}
