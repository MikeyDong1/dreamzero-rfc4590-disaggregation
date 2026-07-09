#!/usr/bin/env bash
# Run the DreamZero parallel-encoder POC inside the container on gnr17408.
# Encoders ONLY (UMT5 text + Wan VAE) — the DiT is never built/loaded.
#
# Usage (inside container, via docker exec):
#   pe_run_encoders.sh <mode> <out_json> [extra args...]
#     <mode> = serial | one_card_stream | two_card | all
set -Eeuo pipefail

MODE="${1:?mode required}"
OUT="${2:?out json path required}"
shift 2 || true

# BMG oneCCL fix + local libs (per proven gnr17408 recipe).
export LD_LIBRARY_PATH="/opt/intel/oneapi/ccl/2021.15/lib:${LD_LIBRARY_PATH:-}:/usr/local/lib/"

MODEL_PATH="/workspace/model_repo/snapshots/96ad344138c66e82536422432ad742f015784942"
STITCHED="/workspace/pe/test_data/model_input_stitched.npz"
PROMPT="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"

cd /workspace/pe/scripts
echo "=== pe_run_encoders: mode=$MODE out=$OUT ==="
python -u dreamzero_parallel_encoders.py \
  --model-path "$MODEL_PATH" \
  --parallel-encoder-mode "$MODE" \
  --stitched-npz "$STITCHED" \
  --prompt "$PROMPT" \
  --num-warmup-runs 3 --num-benchmark-runs 10 \
  --verify-against-pipeline \
  --out "$OUT" \
  "$@"
echo "=== pe_run_encoders DONE (exit $?) ==="
