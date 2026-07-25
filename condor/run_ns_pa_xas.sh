#!/bin/bash
# HTCondor job wrapper: nested-sampling injection recovery against the
# XAS-mean phase-amplitude checkpoint (see condor/scan_pa_xas{,_nounc}.sub).
# Uses a single wide mass-ratio prior [0.15, 0.97] for every injection --
# unlike the earlier lsmin0075 scans (default [0.4, 0.95]), because the
# XAS-mean model has no low-q defect to hide from, and a wide prior lets
# the sampler expose any spurious modes (cf. the SNR~100 q~0.41 attractor).
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
Q="$1"
TAG="$2"
shift 2
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/injection_nested_sampling.py \
  --checkpoint checkpoints/phenomd_nonspinning_dense30_phase_amplitude_xas.pt \
  --device cuda \
  --q-true "$Q" \
  --q-bounds 0.15 0.97 \
  --output "results/injection_ns_pa_xas_${TAG}.png" \
  "$@"
