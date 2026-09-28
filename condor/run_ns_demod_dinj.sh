#!/bin/bash
# HTCondor job wrapper: clean IMRPhenomD self-consistency injection against the
# DemodGPSurrogate checkpoint (heterodyned D-vs-XAS residual, mismatch ~1e-5).
#
# PURPOSE: the demod representation reproduces IMRPhenomD to ~1e-5 mismatch and
# reports a very small K (K << C), so the log-det term -1/2 log|C+K(theta)| is
# nearly flat and should NOT grid-snap toward training nodes even with-K -- the
# failure mode the exact-GP+XAS checkpoint shows at realistic SNR (memory
# exact_xas_no_logdet_bias). This is the direct with-K test of that hypothesis.
# Compare against results/injection_ns_exact_xas_lsminq006_dinj_*_snr20_* .
#
# Mirrors run_ns_exact_xas_lsminq006_dinj.sh exactly (same q-bounds, IMRPhenomD
# injection, same seed/noise) except the checkpoint, so the two are directly
# comparable. Distance is passed by the .sub (1500 Mpc -> SNR~20).
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
Q="$1"
TAG="$2"
shift 2
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/injection_nested_sampling.py \
  --checkpoint checkpoints/phenomd_nonspinning_dense30_demod.pt \
  --device cuda \
  --q-true "$Q" \
  --q-bounds 0.15 0.97 \
  --approximant IMRPhenomD \
  --output "results/injection_ns_demod_dinj_${TAG}.png" \
  "$@"
