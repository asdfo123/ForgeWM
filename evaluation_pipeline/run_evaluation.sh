#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FORGEWM_ROOT="${FORGEWM_ROOT:-$(cd "$PIPELINE_DIR/.." && pwd)}"
CONFIG="${EVAL_CONFIG:-$PIPELINE_DIR/config/paper_main.env}"
STAGE="${1:-all}"

[[ -f "$CONFIG" ]] || { echo "Missing evaluation config: $CONFIG" >&2; exit 2; }
# shellcheck source=/dev/null
source "$CONFIG"
export GAMEWORLD_ROOT

SCRIPTS="$PIPELINE_DIR/scripts"
LOG_ROOT="$METRIC_ROOT/logs"
read -r -a PAIRED_ARRAY <<< "$PAIRED_MODELS"
read -r -a CONSTANT_ARRAY <<< "$CONSTANT_MODELS"
read -r -a GPU_ARRAY <<< "$GPU_LIST"
[[ ${#PAIRED_ARRAY[@]} -eq ${#CONSTANT_ARRAY[@]} ]] || {
  echo "PAIRED_MODELS and CONSTANT_MODELS must have matching order and length" >&2
  exit 2
}
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "GPU_LIST is empty" >&2; exit 2; }
mkdir -p "$LOG_ROOT" "$METRIC_ROOT/depth" "$METRIC_ROOT/perceptual"
cd "$FORGEWM_ROOT"

wait_all() {
  local failed=0 pid
  for pid in "$@"; do wait "$pid" || failed=1; done
  return "$failed"
}

validate_inputs() {
  "$PYTHON_GPU" "$SCRIPTS/validate_main_table.py" \
    --main-root "$MAIN_ROOT" --visual-root "$VISUAL_ROOT" \
    --gt-manifest "$GT_MANIFEST"
}

run_temporal() {
  "$PYTHON_GPU" "$SCRIPTS/paired_temporal_action_metrics.py" \
    --dataset-manifest "$GT_MANIFEST" \
    --eval-root "$PAIRED_ROOT" \
    --models "${PAIRED_ARRAY[@]}" \
    --output-tsv "$METRIC_ROOT/paired_temporal_action_metrics.tsv" \
    --workers "$CPU_WORKERS" \
    >"$LOG_ROOT/paired_temporal_action.log" 2>&1
}

run_depth() {
  local pids=() model gpu i out
  for i in "${!PAIRED_ARRAY[@]}"; do
    model="${PAIRED_ARRAY[$i]}"
    gpu="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
    out="$METRIC_ROOT/depth/$model/depth.tsv"
    mkdir -p "$(dirname "$out")"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_GPU" "$SCRIPTS/paired_depth_metrics.py" \
      --dataset-manifest "$GT_MANIFEST" \
      --model-manifest "$PAIRED_ROOT/$model/manifest.tsv" \
      --output "$out" --da3-model "$DA3_MODEL" --device cuda:0 \
      >"$LOG_ROOT/${model}_depth.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
}

run_vbench() {
  FORGEWM_ROOT="$FORGEWM_ROOT" ROOT_DIR="$FORGEWM_ROOT" \
    VBENCH_ROOT="$VBENCH_ROOT" VBENCH_LOCAL_CKPT="$VBENCH_LOCAL_CKPT" \
    BAKEOFF="$VISUAL_ROOT" MODELS="$CONSTANT_MODELS" GPU_LIST="$GPU_LIST" \
    PYTHON_BIN="$PYTHON_GPU" bash "$SCRIPTS/run_vbench_parallel.sh" \
    >"$LOG_ROOT/constant_vbench_driver.log" 2>&1
}

run_perceptual() {
  local pids=() model gpu i out
  for i in "${!PAIRED_ARRAY[@]}"; do
    model="${PAIRED_ARRAY[$i]}"
    gpu="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
    out="$METRIC_ROOT/perceptual/$model"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_IDM" "$SCRIPTS/evaluate_missing_perceptual.py" \
      --dataset_manifest "$GT_MANIFEST" \
      --model_manifest "$PAIRED_ROOT/$model/manifest.tsv" \
      --out_dir "$out" --rank 0 --world_size 1 \
      --clip_checkpoint "$CLIP_CHECKPOINT" \
      >"$LOG_ROOT/${model}_perceptual.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
}

run_control() {
  local pids=() paired_model constant_model video_dir gpu i
  for i in "${!CONSTANT_ARRAY[@]}"; do
    paired_model="${PAIRED_ARRAY[$i]}"
    constant_model="${CONSTANT_ARRAY[$i]}"
    gpu="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
    video_dir="$CONSTANT_ROOT/$constant_model/videos"
    [[ "$constant_model" == "hyworld_aligned" ]] && \
      video_dir="$CONSTANT_ROOT/$constant_model/eval_videos"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_GPU" "$SCRIPTS/action_consistency_eval.py" \
      --videos-dir "$video_dir" --model-name "$paired_model" \
      --output-dir "$CONSTANT_ROOT/action_trajectory/$constant_model" \
      --da3-model "$DA3_MODEL" \
      --device cuda:0 --stride 4 \
      >"$LOG_ROOT/${paired_model}_trajectory.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"

  pids=()
  for i in "${!CONSTANT_ARRAY[@]}"; do
    paired_model="${PAIRED_ARRAY[$i]}"
    constant_model="${CONSTANT_ARRAY[$i]}"
    gpu="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
    video_dir="$CONSTANT_ROOT/$constant_model/videos"
    [[ "$constant_model" == "hyworld_aligned" ]] && \
      video_dir="$CONSTANT_ROOT/$constant_model/eval_videos"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_IDM" "$SCRIPTS/evaluate_constant_action_idm.py" \
      --videos-dir "$video_dir" --model-name "$paired_model" \
      --output-dir "$CONSTANT_ROOT/action_idm/$constant_model" \
      --model "$IDM_MODEL" --weights "$IDM_WEIGHTS" \
      --device cuda:0 --frame-count 76 \
      >"$LOG_ROOT/${paired_model}_idm.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
}

run_summary() {
  "$PYTHON_GPU" "$SCRIPTS/summarize_main_table.py" \
    --main-root "$MAIN_ROOT" --visual-root "$VISUAL_ROOT" \
    --metric-root "$METRIC_ROOT" \
    | tee "$LOG_ROOT/main_table_summary.log"
}

case "$STAGE" in
  validate) validate_inputs ;;
  temporal) validate_inputs; run_temporal ;;
  depth) validate_inputs; run_depth ;;
  vbench) validate_inputs; run_vbench ;;
  perceptual) validate_inputs; run_perceptual ;;
  control) validate_inputs; run_control ;;
  summarize) run_summary ;;
  all)
    validate_inputs
    run_temporal
    run_depth
    run_vbench
    run_perceptual
    run_control
    run_summary
    ;;
  *)
    echo "Usage: $0 {validate|temporal|depth|vbench|perceptual|control|summarize|all}" >&2
    exit 2
    ;;
esac
