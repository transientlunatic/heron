#!/bin/bash
# HTCondor job wrapper: nested-sampling injection recovery against the
# adaptive16-grid checkpoint (see condor/scan_pa_xas_adaptive16.sub).
# Unsmoothed (no --k-smoothing-grid-spacing), same three q's as every
# other run in this family (0.50/0.60/0.80) -- none of which are training
# nodes on this grid (nearest: 0.49, 0.58, 0.76/0.85) -- for direct
# comparability against the lsminq006 baseline (-6.2/+9.8/-16.0 sigma).
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
Q="$1"
TAG="$2"
shift 2
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/injection_nested_sampling.py \
  --checkpoint checkpoints/phenomd_nonspinning_adaptive16_phase_amplitude_xas.pt \
  --device cuda \
  --q-true "$Q" \
  --q-bounds 0.15 0.97 \
  --output "results/injection_ns_pa_xas_adaptive16_${TAG}.png" \
  "$@"
