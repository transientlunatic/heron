# R1 — SVD go/no-go on the dense30 demod residuals (task #642)

**Verdict: scoped GO.** For q > 0.3 the demod residual is very low rank (r = 3 → 1e-5, r = 4 → 1e-6 median mismatch) and the coefficients c_k(q) are smooth enough to interpolate. For q ≤ 0.3 the subspace is still low-rank, but the coefficients are *not* smooth in q at the dense30 spacing (0.03). The literal criterion "smooth c_k(q) over the whole grid" is therefore **not met**; a reduced-basis track is justified only for the q ≥ ~0.35 domain (which is the PE prior, q ≥ 0.4), and R2 must not claim low q.

Run: frozen `phenomd_nonspinning_dense30_demod_env28u_eta.pt` (sha256 3feffb4e…, hash-checked, untouched), dense30 HDF5 identical to the checkpoint's stored training data (max|Δx| = 0, rel Δy = 5e-8). HTCondor clusters 37695 (SVD) and 37696 (report), CPU. Files: `scripts/r1_svd_gonogo.py`, `scripts/r1_svd_report.py` (+ `r1_figure_template.html`), `condor/r1/`, figure `docs/r1/r1_svd_figure.html`, numbers `docs/r1/*.json`.

## What was computed
z = (h_D − h_XAS)·e^{+i(Φ_XAS − c)}, c = −2.1475 rad, exactly as `DemodGPSurrogate` builds it (it uses the model's own `_reference`). Rows = 30 training q, columns = [Re z, Im z] on a common grid of 1536 points, uniform in chirp-warped time (α = 0.625, t_ref = 0.1), t ∈ [−1.62, +0.09] s about merger, all by `CubicSpline`. Window start is set by the shortest waveform (q = 0.97) plus a 0.15 s margin, so the early inspiral of low q (up to −5 s) is cut. Quality = aLIGO-PSD fitting-factor mismatch (plus polarisation, repo `compute_mismatch`, Tukey 0.1) of *reference + rank-r residual* against the native 4096 Hz IMRPhenomD strain. r = 0 is bare XAS: median 2.1e-3, max 3.1e-3.
Three regimes: **train** (basis from all 30), **LOO-project** (basis from the other 29, held-out row projected: what the subspace can express), **LOO-interp** (same basis, c_k cubic-spline-interpolated in q to the held-out q, interior q only: what a smooth per-coefficient surrogate could deliver).

## Numbers (demod, native-resolution z; rank needed for median mismatch ≤ tol)
| tol | train | LOO-project | LOO-interp, all interior q | LOO-interp, q > 0.3 |
|---|---|---|---|---|
| 1e-4 | 2 | 2 | 2 | 2 (max over q: 2) |
| 1e-5 | 3 | 3 | 3 | 3 (max: 4) |
| 1e-6 | 4 | 4 | 4 | 4 (max: not reached; q = 0.31 stuck at 4e-6, all q ≥ 0.34 ≤ 4e-7 at r = 8) |

Singular spectrum: σ_k/σ_0 = 1, 0.17, 0.059, 0.030, 0.013, 0.0065; 1 − cumulative energy = 3.1e-2 (r = 1), 1.1e-3 (3), 6.9e-5 (5), 5.5e-6 (8), 1.9e-7 (12). Worst-q *train* mismatch needs r = 5/6/8 for 1e-4/1e-5/1e-6.

**Low q (q ≤ 0.28) fails interpolation at any rank**: LOO-interp mismatch at r = 8 is 6e-4 (q = 0.13), 3e-4 (0.16), 3e-4 (0.19), 9e-4 (0.22), 2e-4 (0.25), 2e-5 (0.28); it does not improve at r = 16. LOO-project at those q is fine (≤1e-5, ≤6e-6 at r = 29), so the basis spans them; it is the coefficients that cannot be interpolated. Smoothness of c_0..c_5 (roughness ÷ shuffled-order roughness; LOO spline error ÷ std): q > 0.3 → 0.01-0.06 and ≈0.00-0.01 (very smooth); q ≤ 0.3 → 0.3-1.4 and 0.5-3.7 (no better than random order). Second-difference jump scan peaks at q = 0.19-0.22 and 0.13-0.16 (60-140× the median) — **no isolated jump at q ≈ 0.287**: 0.25/0.28/0.31 sit at 6-30× the median, a gradual transition, not a single-node snap. (If an epoch-snap artefact exists it is not distinguishable here from the general low-q roughness.)

## Caveats that change how to read this
1. **Training-grid floor (not a rank issue).** Rows built from the frozen model's own 200 samples/q (spec's "spline the training data") floor at ~1.5e-5 mismatch for *every* r — the 200-sample grid under-resolves z (spline vs native z: 5% median relative error, 9% max). So 1e-5 and 1e-6 are unreachable from the training rows at any rank; this matches the frozen model's own median 1.2e-5. `demod_raw`/`demod_amp` rows reach 1e-4 at r = 2 only. R2 should build targets from native-resolution z (oracle and reference are cheap to evaluate), not the 200-sample grid.
2. LOO-interp uses a cubic spline on coefficients as a stand-in for a GP; no GP was fit. Edge q (0.10, 0.97) are excluded from LOO-interp (extrapolation).
3. This tests the **mean** residual only. Whether K = BΣBᵀ is an adequate *covariance* (R3) is untested; likewise cross polarisation was not scored and SNR-weighted likelihood effects were not tested.
4. Window/grid choices (above) are mine; the low-q truncation and the 0.03 q spacing both could contribute to the low-q failure — not separated.
5. The SVD script was untracked when run, so `provenance.json` shows `dirty: false` misleadingly; the script is committed unchanged afterwards. The figure was generated and JS-syntax-reviewed by eye but **not opened in a browser** (no browser on the host) — please check it renders.

## Contrast (same pipeline, native resolution)
* **Raw strain**: needs r = 14 (1e-4, 1e-5 train median) and 15 (1e-6); LOO-project/interp max never reach 1e-4 below r ≈ 17. 1 − cumulative energy at r = 8 is 3e-2. Coefficients are less obviously jagged than predicted for q > 0.3 (roughness ratio 0.24-0.45), but jump scan is strongly peaked at q = 0.25/0.28 (350×/310× median) — the one representation with a spike right at the epoch-snap q.
* **log A + Φ**: median r = 5/8/10 for 1e-4/1e-5/1e-6 (train), LOO-interp max never gets there (≥1e-3 at q ≈ 0.13-0.22). c_0, c_1 are extremely smooth (LOO spline error 0.8-6%), c_2+ are not (26-63%). Spectrum falls fastest (σ_1/σ_0 = 0.022) because the unwrapped phase is dominated by one mode, but that mode is not what the mismatch needs.
* Demod is the clear winner: ~4× fewer modes than log A + Φ and ~4× fewer than strain at 1e-6, with smooth coefficients above q ≈ 0.3.

## Recommendation
Proceed to R2 restricted to q ≥ 0.35, rank r ≈ 8 (headroom over r = 4 for 1e-6), native-resolution targets. For q < 0.3 either exclude it or first study why c_k(q) is irregular there (denser q or the documented low-q Nyquist/timing issues) before spending GPs on it.
