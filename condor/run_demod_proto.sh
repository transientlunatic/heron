#!/bin/bash
# Demodulated-residual ExactGP prototype (heterodyne the D-XAS residual by the
# XAS reference phase -> smooth Re/Im GPs -> exact linear covariance to strain).
# Trains 2 GPs on the dense30 targets from the exact-XAS checkpoint, then reports
# mismatch-vs-D and K-calibration (var/err^2). See scripts/demod_proto.py.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/demod_proto.py \
  checkpoints/phenomd_nonspinning_dense30_demod_prototype.pt \
  results/demod_prototype_mismatch.npz
