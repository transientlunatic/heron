#!/bin/bash
# SELF-INJECTION control: inject the surrogate's OWN prediction at q_true
# (mismatch=0 by construction) and recover. Must return truth at ~0 sigma if the
# inference pipeline is unbiased. Isolates machinery/likelihood bias from
# surrogate-vs-D template mismatch.
#
# Compare directly with run_ns_exact_xas_lsminq006_dinj.sh (IMRPhenomD injection),
# which shows +6-16 sigma offsets at SNR~300 -- present IDENTICALLY with and
# without K, hence a template-mismatch systematic (surrogate targets D, tiny
# residual interpolation error amplified by SNR~300's tiny statistical width),
# not a log-det/K effect. If self-injection here lands at ~0 sigma while D-
# injection is offset, that confirms the offset is 100% surrogate-vs-D mismatch
# and the pipeline itself is unbiased.
#
# distance defaults to 100 Mpc (SNR~300) to match the D-injection arm.
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
  --self-inject \
  --output "results/injection_ns_exact_xas_lsminq006_selfinj_${TAG}.png" \
  "$@"
