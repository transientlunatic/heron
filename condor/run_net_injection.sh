#!/bin/bash
# HTCondor wrapper: network-injection PE test on the DemodGPSurrogate checkpoint
# using the new heron.inference PE layer (NetworkLikelihood + analytic extrinsic
# projection + PriorDict + NessaiSampler).
#
# Runs SERIAL on CPU (the surrogate holds an unpicklable lal.Dict, so nessai
# n_pool is unavailable; throughput comes from concurrent independent condor
# jobs -- wiay has 88 cores). Each job is capped to a few threads so several
# coexist. The campaign is keyed by TAG (below) so the .sub only passes a
# comma-free tag -- condor's queue-from splits on commas, which the detector /
# parameter lists contain.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45

TAG="$1"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

# Per-tag configuration: DETECTORS, SAMPLE (params to sample), and EXTRA flags.
EXTRA=""
case "$TAG" in
  net2d_h1l1_withK)
    DET="H1,L1"; SAMPLE="mass_ratio,tc"; NLIVE=250 ;;
  net2d_h1l1_noK)
    DET="H1,L1"; SAMPLE="mass_ratio,tc"; NLIVE=250; EXTRA="--no-uncertainty" ;;
  net5d_h1l1_withK)
    DET="H1,L1"
    SAMPLE="mass_ratio,tc,luminosity_distance,inclination,coalescence_phase"
    NLIVE=250 ;;
  net8d_h1l1v1_withK)
    DET="H1,L1,V1"
    SAMPLE="mass_ratio,tc,ra,dec,psi,luminosity_distance,inclination,coalescence_phase"
    NLIVE=250 ;;
  net8d_h1l1v1_noK)
    DET="H1,L1,V1"
    SAMPLE="mass_ratio,tc,ra,dec,psi,luminosity_distance,inclination,coalescence_phase"
    NLIVE=250; EXTRA="--no-uncertainty" ;;
  *)
    echo "unknown TAG: $TAG" >&2; exit 2 ;;
esac

exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python \
  scripts/network_injection_ns.py \
  --checkpoint checkpoints/phenomd_nonspinning_dense30_demod.pt \
  --device cpu \
  --approximant IMRPhenomD \
  --q-true 0.8 \
  --distance 1500 \
  --sampler nessai \
  --seed 42 \
  --detectors "$DET" \
  --sample "$SAMPLE" \
  --nlive "$NLIVE" \
  --output "results/net_${TAG}" \
  $EXTRA
