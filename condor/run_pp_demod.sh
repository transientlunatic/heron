#!/bin/bash
# PP (coverage) test for the DemodGPSurrogate at SNR~20 -- grid posteriors, no
# nested sampling. One GPU job: the (q,tc) template grid is precomputed ONCE
# (data-independent, ~0.1 s/template on GPU) and every injection is batched
# linear algebra against it, so a full ~300-injection campaign is one ~25-min
# job, not days of NS. Produces both no-K (matched-filter) and with-K
# (GP-marginalised) PP curves. Priors tightened (q in [0.45,0.90], tc +-3 ms)
# so the fixed shared grid resolves the narrow SNR~20 posteriors. Args:
# SEED N_INJ TAG. See scripts/pp_plot_demod.py.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
SEED="$1"
NINJ="$2"
TAG="$3"
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/pp_plot_demod.py \
  --checkpoint checkpoints/phenomd_nonspinning_dense30_demod.pt \
  --device cuda \
  --n-injections "$NINJ" \
  --q-bounds 0.45 0.90 \
  --tc-half-ms 3.0 \
  --approximant IMRPhenomD \
  --distance 1500 \
  --n-q 225 --n-tc 61 \
  --seed "$SEED" \
  --output "results/pp_demod_snr20_${TAG}.npz" \
  --plot
