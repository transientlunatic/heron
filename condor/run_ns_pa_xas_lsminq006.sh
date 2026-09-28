#!/bin/bash
# HTCondor job wrapper: nested-sampling injection recovery against the
# ls_min_q=0.06/noise_floor_rel=1e-2 retrain of the XAS-mean phase-amplitude
# checkpoint (see condor/scan_pa_xas_lsminq006.sub). This is the
# confirmatory test for the grid-snap fix -- see memory grid_snap_finding.md:
# raising ls_min_q removed the ~450-500x posterior-variance collapse at
# training nodes found in the original checkpoint (scan_pa_xas.sub); this
# batch re-runs with-K at the same three off-grid q's that snapped onto
# their nearest training node before (0.50/0.60/0.80), same wide prior and
# default (IMRPhenomD) injection as the original run, to see whether they
# now recover cleanly instead of pinning to a node.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
Q="$1"
TAG="$2"
shift 2
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/injection_nested_sampling.py \
  --checkpoint checkpoints/phenomd_nonspinning_dense30_phase_amplitude_xas_lsminq006.pt \
  --device cuda \
  --q-true "$Q" \
  --q-bounds 0.15 0.97 \
  --output "results/injection_ns_pa_xas_lsminq006_${TAG}.png" \
  "$@"
