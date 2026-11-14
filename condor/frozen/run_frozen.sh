#!/bin/bash
# Run a heron script from the git worktree with the FROZEN checkpoint, and write
# a provenance sidecar. Usage: run_frozen.sh OUTPUT_PREFIX SCRIPT [ARGS...]
# The env's installed heron is an unversioned copy, so PYTHONPATH forces this worktree.
set -euo pipefail
WT=/home/daniel/repositories/ligo/heron-registry
CKPT=/scratch/wiay/daniel/heron-frozen/phenomd_nonspinning_dense30_demod_env28u_eta.pt
SHA=3feffb4e269e62564038462f32b971eb295ce9a4da02de2f974f56d91b1b94ff
OUT="$1"; shift; SCRIPT="$1"; shift
[ "$(sha256sum "$CKPT" | cut -d' ' -f1)" = "$SHA" ] || { echo "checkpoint hash mismatch" >&2; exit 3; }
cd "$WT"; export PYTHONPATH="$WT"
cat > "${OUT}.provenance.json" <<JSON
{"checkpoint": "$CKPT", "sha256": "$SHA", "heron_commit": "$(git rev-parse HEAD)",
 "dirty": $(git diff --quiet -- heron scripts && echo false || echo true),
 "host": "$(hostname)", "started": "$(date -Is)", "script": "$SCRIPT", "args": "$*"}
JSON
exec /scratch/wiay/daniel/micromamba_envs/heron-dense45/bin/python "$SCRIPT" --checkpoint "$CKPT" --output "$OUT" "$@"
