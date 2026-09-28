#!/bin/bash
# HTCondor job wrapper: nested-sampling injection recovery against the
# noise_floor_rel=5e-2 retrain of the ls_min_q=0.06 XAS-mean phase-amplitude
# checkpoint (see condor/scan_pa_xas_noisefloor.sub). Deliberately UNSMOOTHED
# (no --k-smoothing-grid-spacing) and run at the same three q's as the
# unsmoothed lsminq006 baseline (q=0.50/0.60/0.80: -6.2/+9.8/-16.0 sigma,
# see memory logdet_bias_isolated.md) so this isolates whether raising the
# noise floor alone reduces the log-det bias, directly comparable to that
# baseline and to the k_smoothing_offsets runtime fix.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
Q="$1"
TAG="$2"
shift 2
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/injection_nested_sampling.py \
  --checkpoint checkpoints/phenomd_nonspinning_dense30_phase_amplitude_xas_noisefloor.pt \
  --device cuda \
  --q-true "$Q" \
  --q-bounds 0.15 0.97 \
  --output "results/injection_ns_pa_xas_noisefloor_${TAG}.png" \
  "$@"
