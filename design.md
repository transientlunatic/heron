This document is a scratch-pad for future development and experiment design.

# Things to try

Most of these will be bad, but I think it's good to be able to show the logical progression of these developments.

## Non-spinning baseline

+ A very simple model
    - Using PhenomD as an oracle waveform
    - No mean function (or constant mean function)
    - Also try the spectral mixture kernel here — it's a one-line change and directly
      addresses the oscillatory-function problem with Matérn

+ PhenomD with time warping

+ PhenomD with a physically-informed mean function (PN inspiral)

**Architectural fork:** at this point, choose the waveform representation for all
subsequent work. Options:
- Time-domain h(t) directly (current approach)
- Amplitude-phase decomposition: model A(t) and Φ(t) separately, reconstruct h = A·cos(Φ)
  — both components are smooth, Matérn is appropriate
- Frequency-domain: GP on |h̃(f)| and arg(h̃(f)) — naturally connects to matched filtering
  and avoids the oscillatory time-domain problem entirely

**RESOLVED (2026-07-27): the demodulated-residual representation
(`DemodGPSurrogate`) is the current answer for PE.** With a full-IMR reference
approximant (IMRPhenomXAS) as the thing we fit the residual *against*, heterodyne
the oracle−reference strain residual by the reference phase — z = (h_oracle −
h_ref)·e^{+iΦ_ref} — so Re(z), Im(z) are smooth and Cartesian/bounded (no log, no
unwrapping). This is a strict improvement on both the raw-strain and the
amplitude-phase forks: the Matérn kernel sees de-oscillated targets (its
mass-ratio lengthscale finally trains *off* its floor), the reconstruction h =
h_ref + Re·cosΦ + Im·sinΦ is linear so the covariance is *exact* (no delta
method), and it removes the two things that wrecked amplitude-phase's K (the log
and the unwrap). Result: mismatch ~1.2e-5 (best of all reps), and — crucially —
NS- and PP-plot-validated log-det-bias-free, well-calibrated PE at SNR~20 with no
k-smoothing needed. See CLAUDE.md's DemodGPSurrogate + log-det-bias close-out. The
frequency-domain fork remains genuinely unexplored and is the natural next
representation experiment if demod ever hits a wall.

## Architectural improvements

+ A reduced-basis model
    - SVD-decompose the waveform matrix to get ~10–20 basis vectors
    - Fit one 1D GP per basis coefficient over parameter space
    - Fixes O(N³) scaling; makes multi-dimensional extensions tractable

## Precessing systems

+ PhenomPv2 with a physically-informed mean function
    - The key question: can we use PhenomD as the mean function and have the GP learn
      only the precessing residual? Or can we get away with something even faster and more
      naive (e.g. a simple PN waveform)?
    - Cheaper mean function → smaller GP residual → easier problem

+ PhenomPv2 trained using active learning and a physically-informed mean function

## Multi-fidelity

+ A multi-fidelity model with IMRPhenomXAS and an appropriate SEOB model as the oracles
    - IMRPhenomXAS as the cheap low-fidelity model / mean function
    - SEOB as the expensive high-fidelity correction
    - GP learns the systematic difference between them (the delta surrogate idea)

+ A model with NR waveforms as the oracle
    - NR only available at specific pre-ordained parameters
    - IMRPhenomXAS as the mean function or low-fidelity model
    - This is the full realisation of the multi-fidelity approach above

## Alternative paradigms

+ A model with a spectral mixture kernel (if not already tried above)

+ Using neural operators instead?
    - Fourier Neural Operator maps parameters → waveform as a function
    - No calibrated uncertainty by default; pair with ensembles

## Full physics

+ Working up to a fully precessing model

+ Adding higher modes

## Known limitation, deferred: low mass-ratio (q ≲ 0.4)

The current `ExactGPSurrogate` mean develops a secular (monotonically
accumulating) phase-evolution-rate error below q≈0.4, peaking worst around
q≈0.27-0.30 (mismatch 55-65% there vs ~2% at q≳0.7). Root-caused (2026-07-14,
see CLAUDE.md Known Issues and the `mean_function_low_q_defect` memory) to
two compounding structural issues, not a config/training bug:

1. `ChirpTimeWarping` applies one global `(alpha, t_ref)` to every mass
   ratio, but the Newtonian chirp timescale scales as η⁻¹ at fixed total
   mass (η varies ~3x across a typical q=0.1-1.0 grid) — a single global
   warping can't be the self-similar transform for every q at once.
2. The product/separable Matérn kernel has no q×time interaction term, so
   one global time lengthscale must serve the whole q range — and it's
   confirmed pinned at its regularisation floor (`ls_min_time`) in every
   checkpoint trained so far, meaning the optimizer wants finer resolution
   and structurally can't get it with this kernel.

Tried and **ruled out as quick fixes**: more training density (dense10 →
dense45, no change — this rules out interpolation/grid-spacing as the
cause), a mass-ratio-adaptive warping (`MassRatioChirpTimeWarping`,
`heron/models/warping.py`, `type: chirp_adaptive`) at two different anchor
points — both real (phase drift in the targeted region drops 3-10x) but
net-negative in aggregate (the fix trades hump-region improvement for
upper-range regression, since it redistributes the same floor-limited
kernel capacity rather than adding any).

**Decision (2026-07-14): accept this as a hard limitation for now.**
Restrict any PE/inference prior to q≥0.4 (`scripts/injection_nested_sampling.py`
now defaults `--q-bounds` to `[0.4, 0.95]`) rather than chase a fix further
in the near term. This mirrors a difficulty other time-domain waveform
surrogate frameworks also face at extreme mass ratios (sparse/expensive
oracle coverage and fast phase evolution both bite harder there) — it's
not unique to this GP formulation, so it's reasonable to scope it out as
follow-on work rather than a blocker for q≥0.4 science.

**Real fix, future work (not attempted yet):** pair the adaptive warping
with either a smaller `ls_min_time` (+ higher `noise_floor_rel` to keep
`K+σ²I` conditioned) or a non-separable/per-q kernel so capacity is added
rather than redistributed; alternatively the amplitude-phase decomposition
or reduced-basis architectural changes above may sidestep the problem
entirely by construction. Low priority relative to the higher-priority
architectural items above unless a science case specifically needs q<0.4.
