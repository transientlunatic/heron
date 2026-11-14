"""Head-to-head: KroneckerDemodGPSurrogate vs the dense30 DemodGPSurrogate.

Trains (or loads) the Kronecker model from
examples/train_phenomd_dense30_demod_kron.yaml and runs the same mismatch
protocol as scripts/mismatch_vs_mass_ratio.py (512 samples over
[-0.5, +0.02] s, aLIGO design PSD, phase+time maximised fitting factor
against the true oracle) on both models. Also checks total-mass
self-similarity against fresh oracle waveforms at M != reference_mass,
which the physical-unit model cannot do without retraining.

Usage:
    python scripts/validate_demod_kron.py [--retrain] [--n-scan 150]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from heron.evaluation.mismatch import compute_mismatch
from heron.evaluation.psd import aligo_design_psd
from heron.train import _get_approximant, heron_train
from heron.utils import load_yaml

CONFIG = "examples/train_phenomd_dense30_demod_kron.yaml"
OLD = "checkpoints/phenomd_nonspinning_dense30_demod.pt"


def load_old_demod(path):
    """Load a DemodGPSurrogate checkpoint across the gpytorch LogNormal-prior
    buffer rename (_buffered_* <-> _transformed_*, see CLAUDE.md Known Issues)."""
    import io

    import torch
    from heron.models.gp.demod import DemodGPSurrogate

    try:
        return DemodGPSurrogate.load(path)
    except RuntimeError:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        swaps = (("_transformed_loc", "_buffered_loc"), ("_transformed_scale", "_buffered_scale"))
        for name, state in ck["model_states"].items():
            fixed = {}
            for k, v in state.items():
                for a, b in swaps:
                    if k.endswith(a):
                        k = k[: -len(a)] + b
                fixed[k] = v
            ck["model_states"][name] = fixed
        buf = io.BytesIO()
        torch.save(ck, buf)
        buf.seek(0)
        return DemodGPSurrogate.load(buf)


def oracle_plus(approximant, q, mass, times, distance=100.0):
    from astropy import units as u

    wf = approximant.time_domain({
        "mass_ratio": float(q), "total_mass": mass * u.solMass,
        "luminosity_distance": distance * u.Mpc,
    }, times=times)
    return np.asarray(wf["plus"].data)


def scan(surrogate, approximant, q_scan, times, psd, mass=60.0, extra=None):
    dt = times[1] - times[0]
    out = np.empty(len(q_scan))
    for i, q in enumerate(q_scan):
        params = {"mass_ratio": float(q), "times": times, **(extra or {})}
        try:
            h = surrogate.predict(params, covariance="none")["plus"].data
        except TypeError:
            h = surrogate.predict(params)["plus"].data
        out[i] = compute_mismatch(h, oracle_plus(approximant, q, mass, times), dt, psd)
    return out


def summary(mm):
    return {"median": float(np.median(mm)), "p90": float(np.percentile(mm, 90)),
            "worst": float(mm.max()), "frac_below_1e-3": float(np.mean(mm < 1e-3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--n-scan", type=int, default=150)
    ap.add_argument("--output", default="results/validate_demod_kron.json")
    args = ap.parse_args()

    from heron.models.gp.demod_kron import KroneckerDemodGPSurrogate

    cfg = load_yaml(CONFIG)
    ckpt = cfg["training"]["checkpoint"]
    t0 = time.time()
    if args.retrain:
        new = heron_train(cfg)
    else:
        try:
            new = KroneckerDemodGPSurrogate.load(ckpt)
        except FileNotFoundError:
            new = heron_train(cfg)
    train_s = time.time() - t0
    old = load_old_demod(OLD)
    approximant = _get_approximant(cfg["training"]["approximant"])

    times = np.linspace(-0.5, 0.02, 512)
    psd = aligo_design_psd(np.fft.rfftfreq(512, d=times[1] - times[0]))
    q_scan = np.linspace(0.12, 0.95, args.n_scan)

    t0 = time.time()
    mm_new = scan(new, approximant, q_scan, times, psd)
    t_new = time.time() - t0
    t0 = time.time()
    mm_old = scan(old, approximant, q_scan, times, psd)
    t_old = time.time() - t0

    # Total-mass checks at M != reference_mass (window scaled with M so the
    # same tau range is compared). Two references, to separate surrogate error
    # from oracle non-self-similarity:
    #   vs_oracle_at_M     -- LAL generated directly at M (f_min = 20 Hz);
    #   vs_rescaled_oracle -- LAL at reference_mass, rescaled to M (what a
    #                         geometric-unit model *should* reproduce);
    #   oracle_self_similarity -- LAL(M) vs rescaled LAL(reference_mass), no GP.
    # LAL's FD->TD conditioning (taper / high-pass at f_min, in physical Hz) is
    # not self-similar in M, so the first and third agree and set the floor.
    mass_checks = {}
    q_sub = q_scan[:: max(1, len(q_scan) // 15)]
    for mass in (40.0, 90.0):
        t_m = times * mass / 60.0
        dt_m = t_m[1] - t_m[0]
        psd_m = aligo_design_psd(np.fft.rfftfreq(512, d=dt_m))
        vs_m, vs_resc, self_sim = [], [], []
        for q in q_sub:
            h = new.predict({"mass_ratio": float(q), "times": t_m, "total_mass": mass},
                            covariance="none")["plus"].data
            truth = oracle_plus(approximant, q, mass, t_m)
            resc = oracle_plus(approximant, q, 60.0, t_m * 60.0 / mass) * mass / 60.0
            vs_m.append(compute_mismatch(h, truth, dt_m, psd_m))
            vs_resc.append(compute_mismatch(h, resc, dt_m, psd_m))
            self_sim.append(compute_mismatch(resc, truth, dt_m, psd_m))
        mass_checks[f"M={mass:g}"] = {
            "vs_oracle_at_M": summary(np.array(vs_m)),
            "vs_rescaled_oracle": summary(np.array(vs_resc)),
            "oracle_self_similarity": summary(np.array(self_sim)),
        }

    result = {
        "train_or_load_seconds": train_s,
        "kronecker": {**summary(mm_new), "scan_seconds": t_new},
        "dense30_demod": {**summary(mm_old), "scan_seconds": t_old},
        "kronecker_other_masses": mass_checks,
        "hyperparameters": new.hyperparameters,
    }
    print(json.dumps(result, indent=2))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    np.savez(args.output.replace(".json", ".npz"), q_scan=q_scan,
             mismatch_kronecker=mm_new, mismatch_dense30_demod=mm_old)


if __name__ == "__main__":
    main()
