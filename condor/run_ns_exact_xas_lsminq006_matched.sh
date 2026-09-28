#!/bin/bash
# HTCondor job wrapper: matched-family injection (IMRPhenomXAS) against
# the ls_min_q=0.06-fixed plus/cross ExactGPSurrogate+XAS checkpoint --
# the exact-representation counterpart of
# run_ns_pa_xas_lsminq006_matched.sh, for the first apples-to-apples
# (both representations, both q-kernel-fixed, both matched-family
# injected) nested-sampling comparison this session.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
Q="$1"
TAG="$2"
shift 2
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/injection_nested_sampling.py \
  --checkpoint checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt \
  --device cuda \
  --q-true "$Q" \
  --q-bounds 0.15 0.97 \
  --approximant IMRPhenomXAS \
  --output "results/injection_ns_exact_xas_lsminq006_matched_${TAG}.png" \
  "$@"
