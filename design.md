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
