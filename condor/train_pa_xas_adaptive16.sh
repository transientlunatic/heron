#!/bin/bash
# HTCondor job wrapper: train the phase-amplitude+XAS-mean checkpoint on
# the adaptive 16-point mass-ratio grid (dense for q<0.3, sparse above --
# see examples/train_phenomd_adaptive16_phase_amplitude_xas.yaml and memory
# manifold_correlation_length.md). Tests whether density matched to the
# manifold's measured local correlation length holds up as well as the
# uniform 30-point dense30 grid, at roughly half the training data.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/heron train \
  --settings examples/train_phenomd_adaptive16_phase_amplitude_xas.yaml
