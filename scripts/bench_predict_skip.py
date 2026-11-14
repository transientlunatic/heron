"""Benchmark ExactGPSurrogate.predict() with/without the skipped covariance correction.

"before" re-creates the old behaviour by turning gpytorch's
``skip_posterior_variances`` into a no-op, so both arms run the same code.
Works on a plain ExactGPSurrogate or a DemodGPSurrogate checkpoint.
"""
import argparse
import contextlib
import time

import numpy as np
import torch
import gpytorch

from heron.models.gp.demod import DemodGPSurrogate
from heron.models.gp.exact import ExactGPSurrogate


def load(path, device):
    try:
        return DemodGPSurrogate.load(path, device=device)
    except Exception:
        return ExactGPSurrogate.load(path, device=device)


def timed(fn, device, reps):
    fn()  # warm-up (float64 model clone, caches)
    ts = []
    for _ in range(reps):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--q", type=float, default=0.8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--variance-grid", type=int, default=512)
    a = ap.parse_args()

    s = load(a.checkpoint, a.device)
    gp = getattr(s, "_gp", s)
    n_train = gp._train_x_raw.shape[0]
    print(f"model={type(s).__name__} n_train={n_train} N={a.n} device={a.device}")
    params = {"mass_ratio": a.q, "times": np.linspace(-0.5, 0.02, a.n)}

    real = gpytorch.settings.skip_posterior_variances

    @contextlib.contextmanager
    def old_behaviour():
        # Forcing the flag off == the pre-fix behaviour. (Only the callable is
        # replaced; ``.on()`` is a classmethod looked up on the real class.)
        class _Off(real):
            def __init__(self, state=True):
                super().__init__(False)

        gpytorch.settings.skip_posterior_variances = _Off
        try:
            yield
        finally:
            gpytorch.settings.skip_posterior_variances = real

    cases = [("none", None), ("diagonal+grid", a.variance_grid), ("diagonal exact", None)]
    for label, grid in cases:
        cov = "none" if label == "none" else "diagonal"
        s.variance_grid = grid
        with old_behaviour():
            t_old, o_old = timed(lambda: s.predict(params, covariance=cov), a.device, a.reps)
        t_new, o_new = timed(lambda: s.predict(params, covariance=cov), a.device, a.reps)
        same = all(np.array_equal(o_old[p].data, o_new[p].data) for p in ("plus", "cross"))
        print(f"{label:16s} before={t_old:8.3f}s after={t_new:8.3f}s "
              f"speedup={t_old / t_new:6.2f}x mean_bit_identical={same}", flush=True)


if __name__ == "__main__":
    main()
