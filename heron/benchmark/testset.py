"""Held-out mass-ratio test set for the shared benchmark (B0 'Shared inputs').

Only the *parameters* are generated here (reproducible for a fixed seed);
oracle waveforms are produced from them downstream with
``peak_alignment="interpolated"`` so LAL's sample-snapped epoch (#399) does not
feed into the scores.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

# dense30 training nodes (examples/train_phenomd_dense30_demod_kron.yaml)
DENSE30_NODES = tuple(round(0.10 + 0.03 * i, 2) for i in range(30))

# B0 injections: SNR~20 at these q; SNR ladder at q=0.8 and the midpoint.
INJECTION_QS = (0.50, 0.60, 0.80, 0.505)
SNR_LADDER = (20, 75, 250)
LADDER_QS = (0.80, 0.505)
MIDPOINT_SEEDS = 5          # noise seeds for the q~0.505 midpoint injection


@dataclass(frozen=True)
class TestSet:
    __test__ = False        # not a pytest class despite the name

    nodes: tuple
    midpoints: tuple
    off_grid: tuple
    low_q: tuple            # recorded, not pass/fail (measurements 1-2 only)
    seed: int

    def digest(self) -> str:
        blob = json.dumps([self.nodes, self.midpoints, self.off_grid,
                           self.low_q, self.seed])
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def make_test_set(seed: int = 0, q_lo: float = 0.4, q_hi: float = 0.95,
                  n_off_grid: int = 4, n_low_q: int = 4,
                  grid=DENSE30_NODES, min_node_gap: float = 0.003) -> TestSet:
    """Nodes + midpoints inside ``[q_lo, q_hi]``, a few random off-grid q's
    (kept ``min_node_gap`` away from every node), and low-q extras in
    ``[0.15, q_lo)``.  Deterministic for a fixed ``seed``."""
    grid = np.asarray(grid, float)
    nodes = grid[(grid >= q_lo - 1e-9) & (grid <= q_hi + 1e-9)]
    mids = 0.5 * (nodes[:-1] + nodes[1:])
    rng = np.random.default_rng(seed)

    off = []
    while len(off) < n_off_grid:
        q = float(rng.uniform(q_lo, q_hi))
        if np.min(np.abs(grid - q)) > min_node_gap:
            off.append(round(q, 4))
    low = sorted(round(float(x), 4) for x in rng.uniform(0.15, q_lo, n_low_q))

    t = lambda x: tuple(round(float(v), 4) for v in x)  # noqa: E731
    return TestSet(t(nodes), t(mids), tuple(sorted(off)), tuple(low), seed)
