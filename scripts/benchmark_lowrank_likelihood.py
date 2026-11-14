"""Per-call cost of the network likelihood: diagonal K (dense C+K) vs low-rank K.

Times one likelihood evaluation (single detector, fixed phase) at real-data
segment lengths, for the matched filter (K=0), the existing diagonal-K path
(dense O(N^3) factorisation of C + P diag(var) P^T per call) and the low-rank
Woodbury path (O(N^2 r)). The one-off noise Cholesky (O(N^3), shared by all
three) is reported separately. Uses a KroneckerDemodGPSurrogate checkpoint.

Usage:
    python scripts/benchmark_lowrank_likelihood.py \
        --checkpoint checkpoints/phenomd_nonspinning_dense30_demod_kron.pt \
        --sizes 2048 4096 8192 16384 --rank 16 --diag-max 8192
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from heron.evaluation.psd import aligo_design_psd
from heron.inference.detectors import Detector
from heron.inference.network import NetworkLikelihood
from heron.models.gp.demod_kron import KroneckerDemodGPSurrogate

TC = 1187008882.4


def time_call(fn, repeats):
    fn()  # warm (reference cache, FFT plans)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--fs", type=float, default=4096.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--diag-max", type=int, default=8192,
                    help="Skip the dense diagonal path above this N (memory).")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default="results/benchmark_lowrank_likelihood.json")
    args = ap.parse_args()

    surrogate = KroneckerDemodGPSurrogate.load(args.checkpoint)
    det = Detector.from_name("H1", psd_fn=aligo_design_psd)
    params = {"mass_ratio": 0.62, "tc": TC, "ra": 1.95, "dec": -1.27, "psi": 0.82,
              "coalescence_phase": 0.3}
    rows = []
    for n in args.sizes:
        # Merger 0.2 s before the end of the segment, like a GWOSC analysis.
        times = TC - (n / args.fs - 0.2) + np.arange(n) / args.fs
        data = {"H1": np.zeros(n)}
        row = {"N": n}
        t0 = time.perf_counter()
        base = NetworkLikelihood(data, times, det, surrogate, use_waveform_uncertainty=False)
        row["setup_noise_cholesky_s"] = time.perf_counter() - t0
        row["matched_filter_s"] = time_call(lambda: base(params), args.repeats)

        low = NetworkLikelihood.__new__(NetworkLikelihood)
        low.__dict__.update(base.__dict__)  # share the noise Cholesky
        low.use_waveform_uncertainty = True
        low._lowrank = True
        low._covariance_rank = args.rank
        row["lowrank_s"] = time_call(lambda: low(params), args.repeats)
        _, fac = low._predict_factor({"mass_ratio": 0.62, "times": times - TC})
        row["lowrank_columns"] = int(fac["plus"].shape[1])

        if n <= args.diag_max:
            diag = NetworkLikelihood.__new__(NetworkLikelihood)
            diag.__dict__.update(base.__dict__)
            diag.use_waveform_uncertainty = True
            row["diagonal_s"] = time_call(lambda: diag(params), 1)
        print(json.dumps(row))
        rows.append(row)
    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
