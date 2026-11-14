#!/bin/bash
set -euo pipefail
WT=/home/daniel/claude-sessions/heron-registry-9ac2b20ed9da5bc4/r1-svd-gonogo-77b923
cd "$WT"; export PYTHONPATH="$WT"
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/r1_svd_report.py /scratch/wiay/daniel/heron-frozen/results/r1_svd /scratch/wiay/daniel/heron-frozen/results/r1_svd_figure.html --d3 /scratch/wiay/daniel/heron-frozen/results/d3.min.js
