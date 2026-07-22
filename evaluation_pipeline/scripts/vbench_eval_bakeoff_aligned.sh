#!/usr/bin/env bash
# VBench 8-dim eval for the bakeoff. Runs custom_input on each model's videos.
# Dimensions: i2v_subject, subject_consistency, background_consistency,
#   temporal_flickering, motion_smoothness, dynamic_degree,
#   aesthetic_quality, imaging_quality
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${FORGEWM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BAKEOFF="${BAKEOFF:-$SCRIPT_DIR/../data/main_table/constant_action}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT_DIR/VBench}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"
VBENCH_LOCAL_CKPT="${VBENCH_LOCAL_CKPT:-0}"
GPU="${GPU:-0}"
# dynamic_degree is computed in custom_metrics_eval.py (Farneback-based) since
# VBench's RAFT weights (raft-things.pth) are not downloadable on this network.
DIMS="${DIMS:-i2v_subject subject_consistency background_consistency temporal_flickering motion_smoothness aesthetic_quality imaging_quality}"
MODELS="${MODELS:-hyworld_aligned}"

export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$VBENCH_ROOT/.cache}"

cd "$ROOT_DIR" || exit 1

for m in $MODELS; do
  mdir="$BAKEOFF/$m"
  vdir="$mdir/videos"
  if compgen -G "$mdir/eval_videos/*.mp4" > /dev/null; then
    vdir="$mdir/eval_videos"
  fi
  idir="$mdir/images"
  odir="$mdir/eval"
  [ -d "$vdir" ] || { echo "skip $m (no videos dir)"; continue; }
  n=$(find -L "$vdir" -maxdepth 2 -type f -name '*.mp4' | wc -l)
  echo "=== VBench eval: $m ($n videos) ==="
  mkdir -p "$odir"
  local_ckpt_args=()
  [[ "$VBENCH_LOCAL_CKPT" == "1" ]] && local_ckpt_args=(--load_ckpt_from_local True)
  PYTHONPATH="$EXTRA_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON_BIN" "$VBENCH_ROOT/evaluate_i2v.py" \
    --mode custom_input \
    --videos_path "$vdir" \
    --custom_image_folder "$idir" \
    --dimension $DIMS \
    --ratio 16-9 \
    "${local_ckpt_args[@]}" \
    --output_path "$odir" 2>&1 | tail -30
  echo "=== $m done ==="
done

echo "ALL VBENCH EVAL DONE"
