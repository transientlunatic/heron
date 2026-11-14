#!/bin/bash
# R1 SVD go/no-go on the FROZEN checkpoint (read-only; hash-checked). Usage: run_r1.sh OUTPUT_PREFIX [ARGS...]
# Installed heron in the env is an unversioned copy, so PYTHONPATH forces this worktree.
set -euo pipefail
WT=/home/daniel/claude-sessions/heron-registry-9ac2b20ed9da5bc4/r1-svd-gonogo-77b923
CKPT=/scratch/wiay/daniel/heron-frozen/phenomd_nonspinning_dense30_demod_env28u_eta.pt
H5=/scratch/wiay/daniel/heron-dense45/checkpoints/phenomd_nonspinning_dense30_data.h5
SHA=3feffb4e269e62564038462f32b971eb295ce9a4da02de2f974f56d91b1b94ff
OUT="$1"; shift
[ "$(sha256sum "$CKPT" | cut -d' ' -f1)" = "$SHA" ] || { echo "checkpoint hash mismatch" >&2; exit 3; }
cd "$WT"; export PYTHONPATH="$WT"
cat > "${OUT}.provenance.json" <<JSON
{"checkpoint": "$CKPT", "sha256": "$SHA", "h5": "$H5", "heron_commit": "$(git rev-parse HEAD)",
 "dirty": $(git diff --quiet -- heron scripts && echo false || echo true),
 "host": "$(hostname)", "started": "$(date -Is)", "script": "scripts/r1_svd_gonogo.py", "args": "$*"}
JSON
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python scripts/r1_svd_gonogo.py --checkpoint "$CKPT" --h5 "$H5" --output "$OUT" "$@"
