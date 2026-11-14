#!/usr/bin/env python
"""R1 post-processing: subset statistics (q split) + self-contained d3 HTML figure.

Reads results/r1_svd.{npz,json} written by scripts/r1_svd_gonogo.py.
Usage: r1_svd_report.py PREFIX OUT.html [--d3 path/to/d3.min.js]
"""
import argparse, json
import numpy as np
from scipy.interpolate import CubicSpline

ap = argparse.ArgumentParser()
ap.add_argument("prefix"); ap.add_argument("out"); ap.add_argument("--d3", default=None)
ap.add_argument("--qsplit", type=float, default=0.30)
a = ap.parse_args()
d = np.load(a.prefix + ".npz"); J = json.load(open(a.prefix + ".json"))
qs = d["qs"]; nq = len(qs); ranks = np.arange(nq)
REPS = ["demod_raw", "demod_amp", "demod_native", "strain", "phase_logamp"]
hi = qs > a.qsplit            # "mid/high q"
interior = np.arange(nq); interior = (interior > 0) & (interior < nq - 1)

def need(tab, rows, tol, stat):
    v = tab[rows]
    f = np.nanmedian(v, 0) if stat == "median" else np.nanmax(v, 0)
    ok = np.where(f <= tol)[0]
    return int(ok[0]) if len(ok) else None

stats = {}
for rep in REPS:
    st = {}
    for name, tab in [("train", d[f"{rep}_mm_train"]), ("loo_project", d[f"{rep}_mm_loop"]),
                      ("loo_interp", d[f"{rep}_mm_looi"])]:
        for sub, rows in [("all", np.ones(nq, bool) if name != "loo_interp" else interior),
                          (f"q>{a.qsplit}", hi & (interior if name == "loo_interp" else True)),
                          (f"q<={a.qsplit}", (~hi) & (interior if name == "loo_interp" else True))]:
            st[f"{name}|{sub}"] = {f"{t:.0e}": {s: need(tab, rows, t, s) for s in ("median", "max")}
                                   for t in (1e-4, 1e-5, 1e-6)}
    # smoothness per q-subset
    C = d[f"{rep}_coef"]; sm = {}
    rng = np.random.default_rng(0)
    for sub, rows in [(f"q>{a.qsplit}", hi), (f"q<={a.qsplit}", ~hi), ("all", np.ones(nq, bool))]:
        qq = qs[rows]; out = []
        for k in range(8):
            c = C[rows, k]
            rough = np.linalg.norm(np.diff(c, 2)) / np.linalg.norm(c)
            shuf = np.mean([np.linalg.norm(np.diff(rng.permutation(c), 2)) / np.linalg.norm(c) for _ in range(300)])
            e = [CubicSpline(np.delete(qq, i), np.delete(c, i))(qq[i]) - c[i] for i in range(1, len(qq) - 1)]
            out.append(dict(k=k, rough_over_shuffled=float(rough / shuf),
                            loo_rel_rmse=float(np.sqrt(np.mean(np.square(e))) / np.std(c))))
        sm[sub] = out
    st["smooth"] = sm
    # jump scan near q~0.287: robust z-score of 2nd difference at each interior node, modes 0-5
    z = []
    for k in range(6):
        d2 = np.diff(C[:, k], 2); z.append(np.abs(d2) / (np.median(np.abs(d2)) + 1e-300))
    zz = np.max(z, 0)  # per interior node 1..nq-2
    st["jump_scan"] = {f"{qs[1+i]:.2f}": float(zz[i]) for i in range(len(zz))}
    stats[rep] = st
json.dump(stats, open(a.prefix + "_subsets.json", "w"), indent=1)

# ---------------- figure data ----------------
tgrid = d["tgrid"]; ng = len(tgrid)
fig = {"qs": qs.tolist(), "t": tgrid.tolist(), "reps": {}, "meta": J["meta"], "ranks": ranks.tolist(),
       "ref_only_median": J["meta"]["ref_only_median"]}
for rep in REPS:
    r = J["reps"][rep]; B = d[f"{rep}_basis"][:6]
    fig["reps"][rep] = dict(
        s=r["sing_vals"], cum=r["cum_energy"], basis=[[b[:ng].tolist(), b[ng:].tolist()] for b in B],
        coef=d[f"{rep}_coef"][:, :6].T.tolist(),
        mm={"train": r["mismatch_train_median"], "loo_project": r["mismatch_loop_median"],
            "loo_interp": r["mismatch_looi_median"]},
        mmmax={"train": r["mismatch_train_max"], "loo_project": r["mismatch_loop_max"],
               "loo_interp": r["mismatch_looi_max"]},
        mm_loo_interp_by_q={str(rk): d[f"{rep}_mm_looi"][:, rk].tolist() for rk in (2, 4, 8, 16)})
html = open(__file__.replace("r1_svd_report.py", "r1_figure_template.html")).read()
d3src = open(a.d3).read() if a.d3 else None
html = html.replace("/*D3*/", d3src if d3src else "")
html = html.replace("<!--D3TAG-->", "" if d3src else '<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>')
html = html.replace("/*DATA*/null", json.dumps(fig, separators=(",", ":")))
open(a.out, "w").write(html)
print("wrote", a.out, len(html) // 1024, "KB; subsets ->", a.prefix + "_subsets.json")
