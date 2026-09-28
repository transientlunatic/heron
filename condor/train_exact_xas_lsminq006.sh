#!/bin/bash
# HTCondor job wrapper: retrain the plus/cross ExactGPSurrogate+XAS-mean
# checkpoint with the ls_min_q=0.06/noise_floor_rel=1e-2 fix already
# validated for the phase-amplitude XAS checkpoint (see
# examples/train_phenomd_hf_dense30_exact_xas_lsminq006.yaml and memory
# grid_snap_finding.md). Parent node of the matched-injection NS jobs for
# the exact representation in dag_matched_lsminq006.dag -- they need this
# checkpoint to exist before they can run.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/heron train \
  --settings examples/train_phenomd_hf_dense30_exact_xas_lsminq006.yaml
