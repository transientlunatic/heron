#!/bin/bash
# HTCondor job wrapper: CLEAN self-consistency injection (IMRPhenomD -- the
# surrogate's actual training target) against the ls_min_q=0.06 plus/cross
# ExactGPSurrogate+XAS checkpoint.
#
# WHY D, NOT XAS: run_ns_exact_xas_lsminq006_matched.sh injects IMRPhenomXAS,
# which is CONFOUNDED -- these checkpoints use XAS only as the prior MEAN; the
# GP correction is fit against IMRPhenomD training targets, so the surrogate's
# actual reproduction target is D. Injecting D means the surrogate mismatch is
# ~1e-4 (self-consistency) and any residual with-K/no-K difference is genuinely
# the GP-marginalisation (log-det) effect, not a D-vs-XAS approximant systematic
# (see memory logdet_bias_isolated for that confound).
#
# PURPOSE: confirm, in a full nested-sampling posterior, the local profile-scan
# finding (memory exact_xas_no_logdet_bias) that exact-GP+XAS shows NO with-K
# peak shift at q=0.50/0.60/0.80 -- where the phase-amplitude representation, at
# byte-for-byte identical hyperparameters, was -3sigma/-15sigma biased.
#
# distance defaults to 100 Mpc (SNR~300), matching how the phase-amplitude fixes
# were validated. Requires the injection_nested_sampling.py luminosity_distance
# fix synced to wiay (a no-op at 100 Mpc, but sync it anyway for correctness).
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
  --approximant IMRPhenomD \
  --output "results/injection_ns_exact_xas_lsminq006_dinj_${TAG}.png" \
  "$@"
