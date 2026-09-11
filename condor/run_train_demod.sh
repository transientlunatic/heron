#!/bin/bash
# Train the DemodGPSurrogate on the dense30 IMRPhenomD targets (GPU, wiay).
# Heterodyne the IMRPhenomD-vs-IMRPhenomXAS strain residual by the XAS phase
# -> smooth Re/Im GPs -> exact linear covariance. Trains the real
# WaveformSurrogate class (heron.models.gp.demod.DemodGPSurrogate), the
# successor to the scripts/demod_proto.py prototype. See scripts/train_demod.py.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/train_demod.py \
  --source checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt \
  --output checkpoints/phenomd_nonspinning_dense30_demod.pt \
  --device cuda \
  --iterations 200 \
  --mismatch-scan
