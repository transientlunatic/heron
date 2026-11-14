"""Synthetic oracle + harness builders (no lalsuite required).

``chirp_oracle`` is a smooth, q-dependent, merger-shaped chirp: oscillatory
carrier whose frequency rises towards t=0 and an envelope that peaks near
merger.  It exercises the same failure modes as real waveforms (carrier
oscillation, q-dependent phasing) while being analytic and instant.
"""
from __future__ import annotations

import numpy as np

from .base import Harness, Prediction
from .metrics import Thresholds

TIMES = np.linspace(-0.3, 0.02, 160)


def chirp_oracle(q: float, times: np.ndarray):
    t = np.asarray(times, float)
    tau = t + 0.3                                   # 0 at start, 0.3 at merger
    f0, k = 20.0 + 8.0 * q, 60.0 + 25.0 * q          # q-dependent phasing
    phase = 2 * np.pi * (f0 * tau + 0.5 * k * tau ** 2)
    amp = (0.5 + 0.5 * q) * np.exp(-(t / 0.12) ** 2)
    return amp * np.cos(phase), amp * np.sin(phase)


class OraclePlusError:
    """Analytic surrogate: oracle plus a controlled, *known* defect.

    Used (a) as the always-available harness so every level runs without
    torch, and (b) by the self-tests, which inject defects the levels must
    catch.  ``sigma`` is the variance it reports; ``noise`` is the error it
    actually makes, so calibration can be made right or wrong on purpose.
    """

    def __init__(self, oracle=chirp_oracle, *, noise=1e-4, sigma=1e-3,
                 phase_drift=0.0, amp_scale=1.0, time_shift=0.0, seed=0):
        self.oracle, self.noise, self.sigma = oracle, noise, sigma
        self.phase_drift, self.amp_scale, self.time_shift = phase_drift, amp_scale, time_shift
        self.rng = np.random.default_rng(seed)

    def predict(self, q, times):
        t = np.asarray(times, float)
        hp, hx = self.oracle(q, t - self.time_shift)
        if self.phase_drift:                         # secular error, 0 at start
            d = self.phase_drift * (t - t[0]) / (t[-1] - t[0])
            hp, hx = (hp * np.cos(d) - hx * np.sin(d), hp * np.sin(d) + hx * np.cos(d))
        hp, hx = self.amp_scale * hp, self.amp_scale * hx
        hp = hp + self.noise * self.rng.standard_normal(len(t))
        hx = hx + self.noise * self.rng.standard_normal(len(t))
        var = np.full(len(t), self.sigma ** 2)
        return Prediction(hp, hx, var, var)


TRAIN_QS = (0.2, 0.4, 0.6, 0.8, 1.0)
HOLDOUT_QS = (0.3, 0.5, 0.7, 0.9)


def make_analytic_harness(**defects) -> Harness:
    return Harness("analytic", OraclePlusError(**defects), chirp_oracle,
                   TRAIN_QS, HOLDOUT_QS, TIMES,
                   thresholds={n: Thresholds() for n in (0, 1, 2, 3)})


def make_gp_harness(iterations: int = 60) -> Harness:
    """Train a small ExactGPSurrogate on the synthetic oracle (needs torch)."""
    import torch
    from heron.models.gp.exact import ExactGPSurrogate
    from .base import WaveformSurrogateAdapter

    xs, yp, yc = [], [], []
    ttrain = np.linspace(-0.3, 0.02, 80)
    for q in TRAIN_QS:
        hp, hx = chirp_oracle(q, ttrain)
        xs.append(np.column_stack([np.full(len(ttrain), q), ttrain]))
        yp.append(hp), yc.append(hx)
    model = ExactGPSurrogate(
        train_x=torch.tensor(np.vstack(xs), dtype=torch.float32),
        train_y_plus=torch.tensor(np.concatenate(yp), dtype=torch.float32),
        train_y_cross=torch.tensor(np.concatenate(yc), dtype=torch.float32),
        warping="chirp", nu=2.5, output_scale=1.0, device="cpu",
        training_iterations=iterations, optimizer="adam",
        ls_min_q=0.2, ls_min_time=0.01, noise_floor_rel=1e-4,
    )
    # Measured baseline for this 5-node/80-sample GP: node mismatch <= ~7e-4,
    # phase error <= ~8 deg; between nodes mismatch ~0.3 (carrier phase varies
    # by cycles across q, the documented interpolation weakness).  Node limits
    # sit ~3x above the measured values; the held-out (level 3) limit is a
    # *ratchet* on today's behaviour -- tighten it as the model improves.
    node = Thresholds(mismatch=2e-3, correlation=0.99, amplitude_ratio_tol=0.1,
                      phase_error_deg=15.0)
    holdout = Thresholds(mismatch=0.5, correlation=0.0, amplitude_ratio_tol=0.5,
                         phase_error_deg=180.0)
    return Harness("gp-exact-synthetic", WaveformSurrogateAdapter(model), chirp_oracle,
                   TRAIN_QS, HOLDOUT_QS, TIMES,
                   thresholds={0: node, 1: node, 2: node, 3: holdout})
