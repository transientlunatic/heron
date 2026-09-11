"""
Follow-up to the fine-scale instability diagnosis (see probe_peak_offset.py):
tests whether the jagged sub-lengthscale predictions are a float32 precision
artefact or an inherent ill-conditioning problem that survives exact
arithmetic.

Method: load a trained checkpoint, deep-copy the underlying GPyTorch model
to float64, and compare (a) condition number of K+noise*I and magnitude of
alpha = (K+noise*I)^-1(y-mean) between float32 and float64, and (b) the
roughness of the predictive mean over a fine q-scan at fixed warped time,
in both precisions.

Usage::

    python scripts/gp_precision_probe.py --checkpoint checkpoints/phenomd_nonspinning_dense30.pt
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import gpytorch

from heron.models.gp.exact import ExactGPSurrogate


def roughness(y: np.ndarray, n: int) -> tuple[float, float]:
    window = max(3, n // 40)
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="same")
    residual = y - smoothed
    variation = y.max() - y.min()
    return residual.std(), variation


def analyze(ckpt_path: str, pol: str, q_true: float, half_width: float, n_grid: int) -> None:
    surrogate = ExactGPSurrogate.load(ckpt_path)
    cholesky_size = surrogate.cholesky_size
    model32 = surrogate.models[pol]
    model32.eval()
    model64 = copy.deepcopy(model32).double()
    model64.eval()

    train_x = model32.train_inputs[0]
    q_train = torch.unique(surrogate._train_x_raw[:, 0])
    t_probe = train_x[:, 1].median().item()

    print(f"=== {ckpt_path} [{pol}] ===")
    print(f"N_train={train_x.shape[0]}, n_q={len(q_train)}, "
          f"q_train spacing~{float((q_train[1:] - q_train[:-1]).mean()):.4f}")

    print("\n-- Kernel conditioning at trained hyperparameters --")
    with gpytorch.settings.max_cholesky_size(cholesky_size):
        for name, model in [("float32", model32), ("float64", model64)]:
            Xtr = model.train_inputs[0]
            ytr = model.train_targets
            K = model.covar_module(Xtr).evaluate()
            noise = model.likelihood.noise.item()
            Kn = K + noise * torch.eye(K.shape[0], dtype=K.dtype)
            mean_tr = model.mean_module(Xtr)
            resid = ytr - mean_tr
            cond = torch.linalg.cond(Kn.detach()).item()
            alpha = torch.linalg.solve(Kn.detach(), resid.detach().unsqueeze(-1)).squeeze(-1)
            print(f"  [{name:7s}] cond(K+noise*I)={cond:.3e}  "
                  f"noise={noise:.3e}  "
                  f"|alpha|/|y-mean|={alpha.norm().item() / resid.norm().item():.3e}  "
                  f"max|alpha|={alpha.abs().max().item():.3e}  "
                  f"max|y-mean|={resid.abs().max().item():.3e}")

    print(f"\n-- Fine q-scan at fixed warped t={t_probe:.4f}, "
          f"q in [{q_true - half_width:.4f}, {q_true + half_width:.4f}], n={n_grid} --")
    q_grid = np.linspace(q_true - half_width, q_true + half_width, n_grid)
    means = {}
    with gpytorch.settings.max_cholesky_size(cholesky_size):
        for name, model, dtype in [
            ("float32", model32, torch.float32),
            ("float64", model64, torch.float64),
        ]:
            pts = torch.column_stack([
                torch.tensor(q_grid, dtype=dtype),
                torch.full((n_grid,), t_probe, dtype=dtype),
            ])
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mean = model(pts).mean.numpy()
            means[name] = mean
            rough, variation = roughness(mean, n_grid)
            print(f"  [{name:7s}] variation={variation:.3e}  roughness={rough:.3e}  "
                  f"roughness/variation={rough / (variation + 1e-300):.3f}")

    diff = np.abs(means["float32"] - means["float64"])
    scale = np.abs(means["float64"]).max() + 1e-300
    print(f"\nfloat32 vs float64 mean disagreement: max|diff|={diff.max():.3e} "
          f"({diff.max() / scale:.3e} relative to max|float64 mean|)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pol", default="plus", choices=["plus", "cross"])
    parser.add_argument("--q-true", type=float, default=0.8)
    parser.add_argument("--half-width", type=float, default=0.002)
    parser.add_argument("--n-grid", type=int, default=400)
    args = parser.parse_args()
    analyze(args.checkpoint, args.pol, args.q_true, args.half_width, args.n_grid)


if __name__ == "__main__":
    main()
