#!/bin/bash
# HTCondor job wrapper: nested-sampling injection recovery against the
# eta(q)-warped checkpoint (see condor/scan_pa_xas_qwarp_eta.sub).
# Unsmoothed (no --k-smoothing-grid-spacing), same three q's as every
# other run in this family (0.50/0.60/0.80), for direct comparability
# against the lsminq006 baseline (-6.2/+9.8/-16.0 sigma).
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
Q="$1"
TAG="$2"
shift 2
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/injection_nested_sampling.py \
  --checkpoint checkpoints/phenomd_nonspinning_dense30_phase_amplitude_xas_qwarp_eta.pt \
  --device cuda \
  --q-true "$Q" \
  --q-bounds 0.15 0.97 \
  --output "results/injection_ns_pa_xas_qwarp_eta_${TAG}.png" \
  "$@"
