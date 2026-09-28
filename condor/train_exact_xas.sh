#!/bin/bash
# HTCondor job wrapper: retrain the base (ls_min_q=0.03) plus/cross
# ExactGPSurrogate+XAS-mean checkpoint, now with the phase_correction fix
# (heron/models/gp/mean.py::compute_phase_correction, wired automatically
# through heron/train.py). Supersedes the first exact_xas checkpoint,
# which was trained before this fix and had a ~-2.15 rad D-vs-XAS phase
# offset baked into its mean, making the strain-domain residual worse
# than ZeroMean's.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/heron train \
  --settings examples/train_phenomd_hf_dense30_exact_xas.yaml
