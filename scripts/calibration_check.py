"""
Check whether a trained GP's own reported posterior variance (K) is
calibrated against the TRUE squared residual (surrogate mean vs the true
oracle waveform), in the model's *native* representation space -- strain
(plus/cross) for ExactGPSurrogate, log-amplitude/phase for
PhaseAmplitudeGPSurrogate -- rather than the reconstructed-strain space
(which for the phase-amplitude model only approximates the true
covariance via delta-method propagation, and so isn't the right place to
check the underlying GPs' own calibration).

Motivated by the grid-snap-finding investigation (2026-07-16): on the
phase-amplitude XAS-mean checkpoint at q=0.50, K was found to be
1000-4000x LARGER than the true residual -- ruling out "K too small" as
the cause of the residual with-K nested-sampling bias, and pointing at
the log-det marginalisation term's own behaviour rather than a
calibration bug. This script generalises that one-off check into a
reusable scan across mass ratio, and -- now that ExactGPSurrogate also
supports an XAS mean via LALApproximantPlusMean/CrossMean -- lets the
same check be run on the plus/cross representation, to see whether the
same overestimation shows up there too (a property of fitting an XAS-mean
GP in general) or is specific to the phase-amplitude decomposition.

Usage::

    python scripts/calibration_check.py \\
        --checkpoint checkpoints/phenomd_nonspinning_dense30_phase_amplitude_xas.pt \\
        --oracle-approximant IMRPhenomD \\
        --mass-ratios 0.30 0.50 0.70 0.90
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import gpytorch


def load_surrogate(path: str, device: str = "cpu"):
    """Load a checkpoint into whichever WaveformSurrogate class trained it
    (self-described via the checkpoint's `model_class` field)."""
    model_class = torch.load(path, map_location="cpu", weights_only=False)["model_class"]
    if model_class == "PhaseAmplitudeGPSurrogate":
        from heron.models.gp.phase_amplitude import PhaseAmplitudeGPSurrogate
        return PhaseAmplitudeGPSurrogate.load(path, device=device)
    if model_class == "DeltaGPSurrogate":
        from heron.models.gp.delta import DeltaGPSurrogate
        return DeltaGPSurrogate.load(path, device=device)
    from heron.models.gp.exact import ExactGPSurrogate
    return ExactGPSurrogate.load(path, device=device)


def native_mean_variance(surrogate, q: float, times: np.ndarray) -> dict[str, tuple]:
    """GP mean/variance in the model's own native output space, at
    (mass_ratio=q, times).

    For ExactGPSurrogate this IS predict()'s output (plus/cross strain).
    For PhaseAmplitudeGPSurrogate, predict() reconstructs plus/cross via
    delta-method propagation from two independent (log-amplitude, phase)
    GPs -- to check each component GP's OWN calibration we need its
    native-space predictive distribution directly, not the propagated
    strain covariance, so this reaches into the same
    `_get_predict_models()` + warped-points machinery predict() itself
    uses (heron/models/gp/phase_amplitude.py `predict()`).
    """
    type_name = type(surrogate).__name__

    if type_name == "ExactGPSurrogate":
        wf = surrogate.predict({"mass_ratio": float(q), "times": times})
        return {
            "plus": (wf["plus"].data, wf["plus"].variance),
            "cross": (wf["cross"].data, wf["cross"].variance),
        }

    if type_name in ("PhaseAmplitudeGPSurrogate", "DeltaGPSurrogate"):
        times_t = torch.tensor(times, dtype=torch.float64)
        points = torch.column_stack([torch.full_like(times_t, float(q)), times_t])
        points_warped = points.clone()
        points_warped[:, -1] = surrogate.warping.warp(
            points_warped[:, -1], mass_ratio=points_warped[:, 0]
        )
        predict_models = surrogate._get_predict_models()
        out = {}
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.max_cholesky_size(surrogate.cholesky_size):
            for name in ("log_amplitude", "phase"):
                latent = predict_models[name](points_warped)
                out[name] = (latent.mean.cpu().numpy(), latent.variance.cpu().numpy())
        return out

    raise ValueError(f"Unsupported surrogate type for this check: {type_name}")


def true_native_values(
    surrogate, oracle_approximant: str, total_mass: float, distance: float,
    f_low: float, q: float, times: np.ndarray, channels: list[str],
) -> dict[str, np.ndarray]:
    """True channel values at `times`, generated directly from the oracle
    approximant.

    For the phase channel, the unwrap MUST be anchored the same way the
    checkpoint's own training targets were: if the checkpoint's phase mean
    is a full-IMR approximant mean (`_LALApproximantMeanBase`, e.g.
    LALApproximantPhaseMean(IMRPhenomXAS)), `strain_to_amplitude_phase`
    built the training phase target via *mean-referenced unwrapping*
    against that exact mean (see its docstring) -- a naive independent
    `np.unwrap(atan2(...))` of the oracle here can land on a different
    2*pi branch and produce a spurious ~2*pi*k "residual" that has
    nothing to do with real GP accuracy. Reusing the checkpoint's own
    (already-loaded) phase mean module as the unwrap reference keeps this
    script's truth consistent with whatever the GP was actually trained
    against, with no separate branch bookkeeping.
    """
    from astropy import units as u
    from scipy.interpolate import CubicSpline

    from heron.train import _get_approximant
    from heron.models.gp.mean import _LALApproximantMeanBase

    approximant = _get_approximant(oracle_approximant)
    params = {
        "mass_ratio": float(q),
        "total_mass": total_mass * u.solMass,
        "luminosity_distance": distance * u.Mpc,
        "f_min": f_low * u.Hertz,
        "delta_t": (1.0 / 4096) * u.second,
    }
    wf = approximant.time_domain(params)
    native_times = np.asarray(wf["plus"].times, dtype=np.float64)
    hp = np.asarray(wf["plus"].data, dtype=np.float64)
    hx = np.asarray(wf["cross"].data, dtype=np.float64)
    t_clamped = np.clip(times, native_times[0], native_times[-1])

    out = {}
    if "plus" in channels:
        out["plus"] = CubicSpline(native_times, hp)(t_clamped)
    if "cross" in channels:
        out["cross"] = CubicSpline(native_times, hx)(t_clamped)

    if "log_amplitude" in channels or "phase" in channels:
        amplitude = np.sqrt(hp**2 + hx**2)
        principal = np.arctan2(hx, hp)

        phase_mean = getattr(surrogate.models.get("phase"), "mean_module", None)
        if isinstance(phase_mean, _LALApproximantMeanBase):
            q_t = torch.full((len(native_times),), float(q), dtype=torch.float64)
            t_warped = surrogate.warping.warp(
                torch.tensor(native_times, dtype=torch.float64), mass_ratio=q_t
            )
            with torch.no_grad():
                ref_phase = phase_mean(torch.stack([q_t, t_warped], dim=1)).numpy()
            diff = np.unwrap(np.angle(np.exp(1j * (principal - ref_phase))))
            diff -= 2 * np.pi * np.round(np.median(diff) / (2 * np.pi))
            phase_native = ref_phase + diff
        else:
            phase_native = np.unwrap(principal)

        if "log_amplitude" in channels:
            out["log_amplitude"] = CubicSpline(
                native_times, np.log(amplitude + 1e-30)
            )(t_clamped)
        if "phase" in channels:
            out["phase"] = CubicSpline(native_times, phase_native)(t_clamped)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oracle-approximant", default="IMRPhenomD",
                         help="Reference approximant the checkpoint was trained against "
                              "(the training TARGET, not the mean -- IMRPhenomD for all "
                              "checkpoints in this repo so far).")
    parser.add_argument("--mass-ratios", type=float, nargs="+", default=[0.30, 0.50, 0.70, 0.90])
    parser.add_argument("--total-mass", type=float, default=60.0)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument("--f-low", type=float, default=20.0)
    parser.add_argument("--t-lo", type=float, default=-0.3)
    parser.add_argument("--t-hi", type=float, default=0.01)
    parser.add_argument("--n-times", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint} ...")
    surrogate = load_surrogate(args.checkpoint, device=args.device)
    print(f"  model_class = {type(surrogate).__name__}")

    times = np.linspace(args.t_lo, args.t_hi, args.n_times)

    for q in args.mass_ratios:
        native = native_mean_variance(surrogate, q, times)
        truth = true_native_values(
            surrogate, args.oracle_approximant, args.total_mass, args.distance,
            args.f_low, q, times, channels=list(native.keys()),
        )

        print(f"\n=== q = {q:.4f} ===")
        for channel, (mean, var) in native.items():
            resid2 = (np.asarray(mean).ravel() - truth[channel]) ** 2
            var = np.asarray(var).ravel()
            # Avoid divide-by-zero where both residual and variance are
            # numerically zero (e.g. deep in a zeroed tail).
            safe = resid2 > 0
            ratio = var[safe] / resid2[safe]
            print(
                f"  {channel:>13s}: "
                f"median resid={np.sqrt(np.median(resid2)):10.4e}  "
                f"median sqrt(K)={np.sqrt(np.median(var)):10.4e}  "
                f"median K/resid^2={np.median(ratio):10.4e}  "
                f"[{np.min(ratio):.2e}, {np.max(ratio):.2e}]"
            )


if __name__ == "__main__":
    main()
