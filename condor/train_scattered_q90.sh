#!/bin/bash
# HTCondor job wrapper: train the first SCATTERED / quasi-random mass-ratio
# ExactGPSurrogate+XAS-mean checkpoint (90 Sobol q over [0.10,0.97] x 65 t
# = 5850, mode: scattered -- generates the training data on the fly from
# the IMRPhenomD oracle). This breaks the tensor-product training grid (30
# distinct q's) that produces the log-det grid-snap comb -- see
# examples/train_phenomd_scattered_q90_exact_xas.yaml and memory
# scattered_manifold_sampling_idea.md.
#
# Unlike train_exact_xas_lsminq006.sh (mode: data, loads a pre-made HDF5),
# this uses mode: scattered and generates 90 IMRPhenomD waveforms at
# runtime, so lalsuite must be in the env (it is). The checkpoint + its
# saved training HDF5 are written under checkpoints/.
#
# IMPORTANT: this job needs the NEW scattered-sampling code
# (heron/train.py generate_training_data_scattered, mode: scattered;
# heron/training/sampling.py jittered_grid_sample) and the new example
# config -- rsync heron/ and examples/ to this checkout BEFORE submitting
# (the /scratch checkout source tree can be stale -- see
# wiay_remote_training.md).
set -euo pipefail
cd /scratch/wiay/daniel/heron-dense45
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/heron train \
  --settings examples/train_phenomd_scattered_q90_exact_xas.yaml
