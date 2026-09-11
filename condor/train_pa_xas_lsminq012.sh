#!/bin/bash
# HTCondor job wrapper: train the phase-amplitude+XAS-mean checkpoint with
# ls_min_q raised to 0.12, grounded in the measured empirical q-correlation
# length (see examples/train_phenomd_hf_dense30_phase_amplitude_xas_lsminq012.yaml
# and memory manifold_correlation_length.md). Tests whether this helps the
# residual log-det bias at q=0.50/0.60/0.80 without regressing the
# separately-measured, much-shorter low-q (q<0.3) correlation length.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/heron train \
  --settings examples/train_phenomd_hf_dense30_phase_amplitude_xas_lsminq012.yaml
