#!/usr/bin/env python
"""R1 go/no-go: SVD spectrum of the dense30 demod residuals (no GP training).

Question (manager task #642): is the demodulated oracle-reference residual
z(q, t) low-rank enough, with smooth coefficients c_k(q), for a reduced-basis
GP (per-coefficient GPs over q, Woodbury K = B Sigma B^T) to be worth building?

Everything is built from the FROZEN checkpoint's own definitions
(`DemodGPSurrogate._reference`, `.phase_correction`, `.warping`, the raw
oracle training strain stored in the checkpoint), so "demod residual" means
exactly what the model fits:

    z = (h_oracle - h_ref) * exp(+i (Phi_ref - phase_correction))
    Re z = rp cos + rc sin ,  Im z = rp sin - rc cos

Representations compared (all stacked one row per training q on a COMMON
merger-referenced grid, uniform in chirp-warped time, cubic-spline
interpolated -- never np.interp):

  demod_raw     Re z, Im z from the 200 training samples per q  (spec primary)
  demod_amp     Re z / a, Im z / a (a = floored ref amplitude; the frozen model's
                actual GP target if amplitude_normalise)
  demod_native  Re z, Im z evaluated at native 4096 Hz (cross-check: how much does
                the 200-sample training grid itself cost?)
  strain        oracle h_plus, h_cross (native)          -- contrast
  phase_logamp  oracle log A, unwrapped Phi - Phi(0)     -- contrast

Reconstruction quality = aLIGO-PSD fitting-factor mismatch (plus polarisation,
repo convention `compute_mismatch`) of  [reference + rank-r residual]  against
the native oracle strain, on a dense 4096 Hz grid. r=0 for demod reps is the bare
reference (XAS alone).  Two regimes:
  train  : basis from all 30 q, row of that q reconstructed at rank r
  loo-p  : basis from the other 29; held-out row PROJECTED on it (what the basis can express)
  loo-i  : basis from the other 29; coefficients CUBIC-SPLINE interpolated in q to q*
           (what a smooth per-coefficient surrogate could deliver; interior q only)
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import torch
from scipy.fft import rfftfreq
from scipy.interpolate import CubicSpline
from scipy.signal.windows import tukey

from heron.evaluation.mismatch import compute_mismatch
from heron.evaluation.psd import aligo_design_psd
from heron.models.gp.delta import _ApproximantEvaluator
from heron.models.gp.demod import DemodGPSurrogate

logging.basicConfig(level=logging.WARNING)
DT = 1.0 / 4096
NPAD = 16384
EDGE_MARGIN = 0.15  # s inside the shortest waveform's start (LAL conditioning taper)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class Warp:
    """Plain q-independent chirp warp (alpha, t_ref from the checkpoint)."""

    def __init__(self, alpha, t_ref):
        self.a, self.t0 = float(alpha), float(t_ref)

    def w(self, t):
        t = np.asarray(t, float)
        return np.sign(t) * np.abs(t / self.t0) ** self.a * self.t0

    def t(self, w):
        w = np.asarray(w, float)
        return np.sign(w) * np.abs(w / self.t0) ** (1 / self.a) * self.t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--output", required=True, help="prefix; writes .npz and .json")
    ap.add_argument("--ng", type=int, default=1536, help="common-grid points")
    a = ap.parse_args()

    model = DemodGPSurrogate.load(a.checkpoint, device="cpu")
    wcfg = model._warping_config()
    log("warping config", wcfg, "| amplitude_normalise", model.amplitude_normalise,
        "| phase_correction", model.phase_correction, "| q_warping", model._gp.q_warping,
        "| ls_min_q", model.ls_min_q, "| ls_min_time", model.ls_min_time)
    warp = Warp(wcfg.get("alpha", 0.625), wcfg.get("t_ref", 0.1))

    X = model._train_x_raw.numpy().astype(np.float64)
    Yp = model._train_y_plus_raw.numpy().astype(np.float64)
    Yc = model._train_y_cross_raw.numpy().astype(np.float64)

    # cross-check against the dense30 HDF5
    import h5py
    with h5py.File(a.h5, "r") as f:
        hx, hyp = f["x"][:].astype(np.float64), f["y_plus"][:].astype(np.float64)
    same_shape = hx.shape == X.shape
    h5_dx = float(np.max(np.abs(hx - X))) if same_shape else float("nan")
    h5_dy = (float(np.max(np.abs(hyp - Yp)) / np.max(np.abs(Yp))) if same_shape else float("nan"))
    log("h5 vs checkpoint training data: shape equal", same_shape, "max|dx|", h5_dx, "rel max|dy_plus|", h5_dy)

    qs = np.unique(np.round(X[:, 0], 9))
    nq = len(qs)
    log("n_q", nq, "q range", qs[0], qs[-1], "samples per q", [int((np.round(X[:, 0], 9) == q).sum()) for q in qs[:3]])

    oracle = _ApproximantEvaluator("IMRPhenomD", total_mass=model.mass_factor,
                                   distance=model.distance_factor, f_low=model.f_low)
    ref = model._ref

    # --- common window ------------------------------------------------------
    starts, ends = [], []
    for q in qs:
        t0o, t1o = oracle.support(q)
        t0r, t1r = ref.support(q)
        m = np.round(X[:, 0], 9) == q
        starts.append(max(t0o, t0r, X[m, 1].min()))
        ends.append(min(t1o, t1r, X[m, 1].max()))
    t_lo, t_hi = max(starts) + EDGE_MARGIN, min(ends)
    log(f"common physical window [{t_lo:.4f}, {t_hi:.4f}] s; shortest start q={qs[int(np.argmax(starts))]:.3f}; "
        f"per-q starts range {min(starts):.2f}..{max(starts):.2f}")
    wgrid = np.linspace(warp.w(t_lo), warp.w(t_hi), a.ng)
    tgrid = warp.t(wgrid)
    td = np.arange(t_lo, t_hi, DT)  # dense physical grid for mismatch
    wd = warp.w(td)
    taper = tukey(len(td), 0.1)
    freqs = rfftfreq(NPAD, DT)
    psd = aligo_design_psd(freqs)

    def mm(h_true, h_rec):
        A = np.zeros(NPAD); B = np.zeros(NPAD)
        A[: len(td)] = h_true * taper
        B[: len(td)] = h_rec * taper
        if not np.any(B):
            return 1.0
        return float(max(compute_mismatch(A, B, DT, psd), 0.0))

    # --- per-q data ---------------------------------------------------------
    R = {k: [] for k in ["demod_raw", "demod_amp", "demod_native", "strain", "phase_logamp"]}
    dense = []   # per q: dict of dense arrays for reconstruction
    spline_err = []
    for i, q in enumerate(qs):
        m = np.round(X[:, 0], 9) == q
        tq, Dp, Dc = X[m, 1], Yp[m], Yc[m]
        o = np.argsort(tq); tq, Dp, Dc = tq[o], Dp[o], Dc[o]
        hXp, hXc, cP, sP = model._reference(np.full_like(tq, q), tq)
        rp, rc = Dp - hXp, Dc - hXc
        re_s, im_s = rp * cP + rc * sP, rp * sP - rc * cP
        wq = warp.w(tq)
        re_g = CubicSpline(wq, re_s)(wgrid)
        im_g = CubicSpline(wq, im_s)(wgrid)
        amp_g = model._amplitude(np.full(a.ng, q), tgrid)
        R["demod_raw"].append(np.concatenate([re_g, im_g]))
        R["demod_amp"].append(np.concatenate([re_g / amp_g, im_g / amp_g]))

        # native-resolution quantities
        def native(tt):
            lA, ph = oracle.log_amplitude_phase(q, tt)
            Ah = np.exp(lA)
            return Ah * np.cos(ph), Ah * np.sin(ph), lA, ph
        Dp_g, Dc_g, lA_g, ph_g = native(tgrid)
        hXp_g, hXc_g, cP_g, sP_g = model._reference(np.full(a.ng, q), tgrid)
        rpg, rcg = Dp_g - hXp_g, Dc_g - hXc_g
        re_n, im_n = rpg * cP_g + rcg * sP_g, rpg * sP_g - rcg * cP_g
        R["demod_native"].append(np.concatenate([re_n, im_n]))
        R["strain"].append(np.concatenate([Dp_g, Dc_g]))
        _, ph0 = oracle.log_amplitude_phase(q, np.array([0.0]))
        R["phase_logamp"].append(np.concatenate([lA_g, ph_g - ph0[0]]))
        spline_err.append(float(np.linalg.norm(np.concatenate([re_g - re_n, im_g - im_n]))
                                / np.linalg.norm(np.concatenate([re_n, im_n]))))

        # dense arrays
        Dp_d, Dc_d, lA_d, ph_d = native(td)
        hXp_d, hXc_d, cP_d, sP_d = model._reference(np.full(len(td), q), td)
        dense.append(dict(hp=Dp_d, hXp=hXp_d, cP=cP_d, sP=sP_d,
                          amp=model._amplitude(np.full(len(td), q), td)))
        if i % 5 == 0:
            log(f"built q={q:.3f} ({i+1}/{nq})  spline-vs-native rel err of z: {spline_err[-1]:.2e}")

    R = {k: np.array(v) for k, v in R.items()}
    # the "truth" mismatch floor: reference alone, and the exact dense reconstruction check
    ref_only = np.array([mm(d["hp"], d["hXp"]) for d in dense])
    log("reference-only (r=0 demod) mismatch: median %.2e max %.2e" % (np.median(ref_only), ref_only.max()))

    # --- reconstruction -----------------------------------------------------
    def to_dense_hp(rep, i, vec):
        d = dense[i]
        ng = a.ng
        if rep in ("demod_raw", "demod_native", "demod_amp"):
            re = CubicSpline(wgrid, vec[:ng])(wd)
            im = CubicSpline(wgrid, vec[ng:])(wd)
            if rep == "demod_amp":
                re, im = re * d["amp"], im * d["amp"]
            return d["hXp"] + re * d["cP"] + im * d["sP"]
        if rep == "strain":
            return CubicSpline(wgrid, vec[:ng])(wd)
        lA = CubicSpline(wgrid, vec[:ng])(wd)
        ph = CubicSpline(wgrid, vec[ng:])(wd)
        return np.exp(lA) * np.cos(ph)

    def svd(M):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        return U, s, Vt

    ranks = list(range(0, nq))  # 0..nq-1 ; nq-1 is full rank for LOO
    out = {"qs": qs.tolist(), "ranks": ranks, "reps": {}}
    npz = {"qs": qs, "wgrid": wgrid, "tgrid": tgrid, "ref_only_mismatch": ref_only,
           "spline_vs_native_rel_err": np.array(spline_err)}

    for rep, M in R.items():
        t0 = time.time()
        U, s, Vt = svd(M)
        energy = np.cumsum(s**2) / np.sum(s**2)
        # sign convention: make each q-vector (U col) have positive sum of |.| weighted... use largest |.| positive
        for k in range(len(s)):
            j = int(np.argmax(np.abs(U[:, k])))
            sg = np.sign(U[j, k]) or 1.0
            U[:, k] *= sg; Vt[k] *= sg
        C = U * s  # coefficients c_k(q): (nq, K)
        # in-sample ("train") and LOO mismatches per rank
        mm_train = np.zeros((nq, len(ranks)))
        mm_loop = np.zeros((nq, len(ranks)))
        mm_looi = np.full((nq, len(ranks)), np.nan)
        relerr_loop = np.zeros((nq, len(ranks)))
        for r_i, r in enumerate(ranks):
            for i in range(nq):
                if r == 0:
                    zero = np.zeros(M.shape[1])
                    base = zero if rep in ("demod_raw", "demod_amp", "demod_native") else None
                if r == 0:
                    vec = np.zeros(M.shape[1])
                else:
                    vec = (C[i, :r] @ Vt[:r]) if r <= len(s) else M[i]
                mm_train[i, r_i] = mm(dense[i]["hp"], to_dense_hp(rep, i, vec))
        for i in range(nq):
            idx = np.array([j for j in range(nq) if j != i])
            Ul, sl, Vl = svd(M[idx])
            for r_i, r in enumerate(ranks):
                if r == 0:
                    vecp = np.zeros(M.shape[1])
                else:
                    vecp = (M[i] @ Vl[:r].T) @ Vl[:r]
                mm_loop[i, r_i] = mm(dense[i]["hp"], to_dense_hp(rep, i, vecp))
                relerr_loop[i, r_i] = (np.linalg.norm(M[i] - vecp) / np.linalg.norm(M[i]))
                if 0 < i < nq - 1:
                    if r == 0:
                        vecq = np.zeros(M.shape[1])
                    else:
                        co = M[idx] @ Vl[:r].T  # (nq-1, r)
                        cq = CubicSpline(qs[idx], co, axis=0)(qs[i])
                        vecq = cq @ Vl[:r]
                    mm_looi[i, r_i] = mm(dense[i]["hp"], to_dense_hp(rep, i, vecq))
        log(f"{rep}: SVD + mismatch tables done in {time.time()-t0:.0f}s")

        def first_rank(tab, tol, stat, rows=slice(None)):
            v = tab[rows]
            f = np.nanmedian(v, axis=0) if stat == "median" else np.nanmax(v, axis=0)
            ok = np.where(f <= tol)[0]
            return int(ranks[ok[0]]) if len(ok) else None

        interior = slice(1, nq - 1)
        tols = [1e-4, 1e-5, 1e-6]
        need = {}
        for name, tab, rows in [("train", mm_train, slice(None)), ("loo_project", mm_loop, slice(None)),
                                ("loo_interp_interior", mm_looi, interior)]:
            need[name] = {f"{t:.0e}": {"median": first_rank(tab, t, "median", rows),
                                       "max": first_rank(tab, t, "max", rows)} for t in tols}

        # smoothness of c_k(q)
        K = min(12, len(s))
        smooth = []
        rng = np.random.default_rng(0)
        for k in range(K):
            c = C[:, k]
            rough = np.linalg.norm(np.diff(c, 2)) / (np.linalg.norm(c) + 1e-300)
            sh = np.mean([np.linalg.norm(np.diff(rng.permutation(c), 2)) / (np.linalg.norm(c) + 1e-300)
                          for _ in range(300)])
            # LOO cubic-spline interpolation error of c_k (interior points)
            e = []
            for i in range(1, nq - 1):
                idx = np.array([j for j in range(nq) if j != i])
                e.append(CubicSpline(qs[idx], c[idx])(qs[i]) - c[i])
            loo_rel = float(np.sqrt(np.mean(np.square(e))) / (np.std(c) + 1e-300))
            d2 = np.abs(np.diff(c, 2))
            med = np.median(d2) + 1e-300
            j287 = int(np.argmin(np.abs(qs - 0.287)))
            # second-difference at interior node n sits at d2[n-1]
            smooth.append(dict(k=k, s=float(s[k]), roughness=float(rough), roughness_shuffled=float(sh),
                               rough_over_shuffled=float(rough / sh), loo_spline_rel_rmse=loo_rel,
                               d2_over_median_max=float(d2.max() / med),
                               d2_max_at_q=float(qs[1 + int(np.argmax(d2))]),
                               d2_over_median_near_q287=float(d2[max(j287 - 1, 0)] / med),
                               q_node_near_287=float(qs[j287])))
        out["reps"][rep] = dict(
            sing_vals=s.tolist(), cum_energy=energy.tolist(),
            mismatch_train_median=np.median(mm_train, 0).tolist(), mismatch_train_max=mm_train.max(0).tolist(),
            mismatch_loop_median=np.median(mm_loop, 0).tolist(), mismatch_loop_max=mm_loop.max(0).tolist(),
            mismatch_looi_median=np.nanmedian(mm_looi, 0).tolist(), mismatch_looi_max=np.nanmax(mm_looi, 0).tolist(),
            relerr_loop_median=np.median(relerr_loop, 0).tolist(),
            rank_needed=need, smoothness=smooth)
        npz[f"{rep}_s"] = s
        npz[f"{rep}_basis"] = Vt[:8]
        npz[f"{rep}_coef"] = C[:, :12]
        npz[f"{rep}_mm_train"] = mm_train
        npz[f"{rep}_mm_loop"] = mm_loop
        npz[f"{rep}_mm_looi"] = mm_looi

    out["meta"] = dict(checkpoint=a.checkpoint, h5=a.h5, warping=wcfg, ng=a.ng,
                       phase_correction=model.phase_correction,
                       amplitude_normalise=model.amplitude_normalise,
                       window=[float(t_lo), float(t_hi)], edge_margin=EDGE_MARGIN,
                       h5_max_dx=h5_dx, h5_rel_max_dy=h5_dy,
                       ref_only_median=float(np.median(ref_only)), ref_only_max=float(ref_only.max()),
                       spline_vs_native_rel_err_median=float(np.median(spline_err)),
                       spline_vs_native_rel_err_max=float(np.max(spline_err)))
    np.savez_compressed(a.output + ".npz", **npz)
    with open(a.output + ".json", "w") as f:
        json.dump(out, f, indent=1)
    log("wrote", a.output + ".npz/.json")
    for rep, d in out["reps"].items():
        log(rep, json.dumps(d["rank_needed"]))


if __name__ == "__main__":
    main()
