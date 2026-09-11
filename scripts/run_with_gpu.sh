#!/bin/bash
# Wrapper script to ensure CUDA is available for HTCondor jobs

# Use local development version of heron instead of installed package
export PYTHONPATH=/home/daniel/repositories/ligo/heron:$PYTHONPATH

# Set up CUDA environment - use system CUDA for KeOps compilation
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/data/wiay/conda_envs/heron2026/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:/data/wiay/conda_envs/heron2026/bin:$PATH

# Make GPU visible (HTCondor sets CUDA_VISIBLE_DEVICES)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# Force use of system FFTW instead of LAL's bundled version
export LD_PRELOAD=/data/wiay/conda_envs/heron2026/lib/libfftw3.so

# Print GPU info for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LD_PRELOAD: $LD_PRELOAD"

# Run the Python script with all arguments
exec /data/wiay/conda_envs/heron2026/bin/python "$@"
