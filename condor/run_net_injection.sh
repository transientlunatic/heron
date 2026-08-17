#!/bin/bash
# HTCondor wrapper: network-injection PE test on the DemodGPSurrogate checkpoint
# using the new heron.inference PE layer (NetworkLikelihood + analytic extrinsic
# projection + PriorDict + NessaiSampler).
#
# Parallelism: nessai n_pool (2nd arg). The surrogate/approximant are now
# picklable, so the whole likelihood is sent to worker processes; each worker is
# single-threaded so request_cpus ~ n_pool + slack. The campaign is keyed by TAG
# (below) so the .sub only passes a comma-free tag -- condor's queue-from splits
# on commas, which the detector / parameter lists contain.
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45

TAG="$1"
NPOOL="${2:-1}"          # nessai likelihood-pool workers; 1 = serial

if [ "$NPOOL" -gt 1 ]; then THR=1; POOL_ARGS="--n-pool $NPOOL"; else THR=4; POOL_ARGS=""; fi
export OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=$THR

# Per-tag config: DETECTORS, SAMPLE, NLIVE, DIST (Mpc -> SNR), EXTRA flags
# (--no-uncertainty for no-K, --self-inject for the mismatch=0 SNR ladders),
# and CHECKPOINT/APPROXIMANT (representation under test -- default demod).
DET="H1,L1"; NLIVE=250; DIST=1500; EXTRA=""
CHECKPOINT="checkpoints/phenomd_nonspinning_dense30_demod.pt"; APPROXIMANT="IMRPhenomD"
case "$TAG" in
  # --- wave 1/2: IMRPhenomD injection at SNR~20 (machinery) ----------------
  net2d_h1l1_withK)   SAMPLE="mass_ratio,tc" ;;
  net2d_h1l1_noK)     SAMPLE="mass_ratio,tc"; EXTRA="--no-uncertainty" ;;
  net5d_h1l1_withK)
    SAMPLE="mass_ratio,tc,luminosity_distance,inclination,coalescence_phase" ;;
  net8d_h1l1v1_withK)
    DET="H1,L1,V1"
    SAMPLE="mass_ratio,tc,ra,dec,psi,luminosity_distance,inclination,coalescence_phase" ;;
  net8d_h1l1v1_noK)
    DET="H1,L1,V1"
    SAMPLE="mass_ratio,tc,ra,dec,psi,luminosity_distance,inclination,coalescence_phase"
    EXTRA="--no-uncertainty" ;;
  # --- SNR ladder: demod SELF-injection (mismatch=0) isolates the K effect --
  # d = 1400 / 370 / 110 Mpc  ->  net SNR ~ 20 / 74 / 250 (H1+L1).
  netladder_snr20_withK)   SAMPLE="mass_ratio,tc"; DIST=1400; EXTRA="--self-inject" ;;
  netladder_snr20_noK)     SAMPLE="mass_ratio,tc"; DIST=1400; EXTRA="--self-inject --no-uncertainty" ;;
  netladder_snr75_withK)   SAMPLE="mass_ratio,tc"; DIST=370;  EXTRA="--self-inject" ;;
  netladder_snr75_noK)     SAMPLE="mass_ratio,tc"; DIST=370;  EXTRA="--self-inject --no-uncertainty" ;;
  netladder_snr250_withK)  SAMPLE="mass_ratio,tc"; DIST=110;  EXTRA="--self-inject" ;;
  netladder_snr250_noK)    SAMPLE="mass_ratio,tc"; DIST=110;  EXTRA="--self-inject --no-uncertainty" ;;
  # --- SNR ladder, exact+XAS (production exact_xas_lsminq006 checkpoint):
  # demod's K is negligible by design (SNR ladder above), so it can't show
  # the log-det bias appearing. exact+XAS's K is 3-8 orders larger and is the
  # representation the k_smoothing/q_warping/noise-floor fix stack was built
  # against -- same distances/self-injection/SNR targets as the demod ladder,
  # unsmoothed (no --k-smoothing-grid-spacing) to see the RAW bias, for a
  # direct, controlled comparison against demod's clean result at the same
  # SNRs. IMRPhenomXAS is irrelevant to --self-inject (source is the
  # surrogate's own mean) but set for consistency/metadata.
  exactxasladder_snr20_withK)
    SAMPLE="mass_ratio,tc"; DIST=1400; EXTRA="--self-inject"
    CHECKPOINT="checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt"; APPROXIMANT="IMRPhenomXAS" ;;
  exactxasladder_snr20_noK)
    SAMPLE="mass_ratio,tc"; DIST=1400; EXTRA="--self-inject --no-uncertainty"
    CHECKPOINT="checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt"; APPROXIMANT="IMRPhenomXAS" ;;
  exactxasladder_snr75_withK)
    SAMPLE="mass_ratio,tc"; DIST=370; EXTRA="--self-inject"
    CHECKPOINT="checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt"; APPROXIMANT="IMRPhenomXAS" ;;
  exactxasladder_snr75_noK)
    SAMPLE="mass_ratio,tc"; DIST=370; EXTRA="--self-inject --no-uncertainty"
    CHECKPOINT="checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt"; APPROXIMANT="IMRPhenomXAS" ;;
  exactxasladder_snr250_withK)
    SAMPLE="mass_ratio,tc"; DIST=110; EXTRA="--self-inject"
    CHECKPOINT="checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt"; APPROXIMANT="IMRPhenomXAS" ;;
  exactxasladder_snr250_noK)
    SAMPLE="mass_ratio,tc"; DIST=110; EXTRA="--self-inject --no-uncertainty"
    CHECKPOINT="checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt"; APPROXIMANT="IMRPhenomXAS" ;;
  *)
    echo "unknown TAG: $TAG" >&2; exit 2 ;;
esac

exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python \
  scripts/network_injection_ns.py \
  --checkpoint "$CHECKPOINT" \
  --device cpu \
  --approximant "$APPROXIMANT" \
  --q-true 0.8 \
  --distance "$DIST" \
  --sampler nessai \
  --seed 42 \
  --detectors "$DET" \
  --sample "$SAMPLE" \
  --nlive "$NLIVE" \
  --output "results/net_${TAG}" \
  $POOL_ARGS $EXTRA
