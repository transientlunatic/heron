#!/bin/bash
# HTCondor job wrapper: train the phase-amplitude+XAS-mean checkpoint with
# the mass ratio input warped by eta(q)=q/(1+q)^2 before the kernel sees
# it (see examples/train_phenomd_dense30_phase_amplitude_xas_qwarp_eta.yaml
# and memory manifold_correlation_length.md). Third alternative tried for
# the log-det bias's K(mass_ratio) oscillation, after the disproven
# additive-kernel idea and the density-matched adaptive grid.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/heron train \
  --settings examples/train_phenomd_dense30_phase_amplitude_xas_qwarp_eta.yaml
