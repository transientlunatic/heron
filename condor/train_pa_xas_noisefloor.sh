#!/bin/bash
# HTCondor job wrapper: train the phase-amplitude+XAS-mean checkpoint with
# noise_floor_rel raised further (1e-2 -> 5e-2) on top of the already-
# validated ls_min_q=0.06/noise_floor_rel=1e-2 grid-snap fix (see
# examples/train_phenomd_hf_dense30_phase_amplitude_xas_noisefloor.yaml and
# memory logdet_bias_isolated.md). Tests whether raising the noise floor
# directly reduces the residual log-det bias found on the lsminq006
# checkpoint (q=0.50/0.60/0.80: -6.2/+9.8/-16.0 sigma with-K, unsmoothed).
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/heron train \
  --settings examples/train_phenomd_hf_dense30_phase_amplitude_xas_noisefloor.yaml
