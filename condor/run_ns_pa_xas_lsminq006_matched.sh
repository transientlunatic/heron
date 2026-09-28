#!/bin/bash
# HTCondor job wrapper: matched-family injection (IMRPhenomXAS, matching
# the surrogate's own mean approximant) against the ls_min_q=0.06-fixed
# phase-amplitude XAS checkpoint. Unlike condor/run_ns_pa_xas.sh's
# scan_pa_xas_matched.sub (which ran this same matched-injection test
# against the UNFIXED ls_min_q=0.03 checkpoint and got a 51-400 sigma
# failure), this isolates matched-family injection on a checkpoint that
# doesn't have the known q-kernel overfitting/variance-collapse pathology
# -- the untested combination flagged in memory grid_snap_finding.md.
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
  --approximant IMRPhenomXAS \
  --output "results/injection_ns_pa_xas_lsminq006_matched_${TAG}.png" \
  "$@"
