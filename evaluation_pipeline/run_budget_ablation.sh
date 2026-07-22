#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FORGEWM_ROOT="${FORGEWM_ROOT:-$(cd "$PIPELINE_DIR/.." && pwd)}"
CONFIG="${EVAL_CONFIG:-$PIPELINE_DIR/config/budget_ablation.env}"
STAGE="${1:-all}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing evaluation config: $CONFIG" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$CONFIG"
export GAMEWORLD_ROOT

SCRIPTS="$PIPELINE_DIR/scripts"
FULL_ROOT="$EVAL_ROOT/full1000"
CONST_ROOT="$EVAL_ROOT/constant"
METRIC_ROOT="$EVAL_ROOT/metrics"
LOG_ROOT="$METRIC_ROOT/logs"
read -r -a MODEL_ARRAY <<< "$MODELS"
read -r -a GPU_ARRAY <<< "$GPU_LIST"

if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "GPU_LIST must contain at least one GPU id" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$METRIC_ROOT/full1000/custom_metrics" \
  "$METRIC_ROOT/full1000/perceptual" \
  "$METRIC_ROOT/constant/action_trajectory" \
  "$METRIC_ROOT/constant/action_idm"
cd "$FORGEWM_ROOT"

wait_all() {
  local failed=0 pid
  for pid in "$@"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

validate_inputs() {
  [[ -s "$GT_MANIFEST" ]] || { echo "Missing GT manifest: $GT_MANIFEST" >&2; return 1; }
  local model
  for model in "${MODEL_ARRAY[@]}"; do
    "$PYTHON_GPU" "$SCRIPTS/validate_native_budget_output.py" \
      --validation "$FULL_ROOT/$model/validation.json" \
      --expected "$FULL1000_EXPECTED"
    "$PYTHON_GPU" "$SCRIPTS/validate_native_budget_output.py" \
      --validation "$CONST_ROOT/$model/validation.json" \
      --expected "$CONSTANT_EXPECTED"
    [[ -s "$FULL_ROOT/$model/manifest.tsv" ]] || return 1
    [[ -d "$CONST_ROOT/$model/eval_videos" ]] || return 1
    [[ -d "$CONST_ROOT/$model/images" ]] || return 1
  done
}

run_paired() {
  local pids=() model gpu i
  "$PYTHON_GPU" "$SCRIPTS/paired_temporal_action_metrics.py" \
    --dataset-manifest "$GT_MANIFEST" \
    --eval-root "$FULL_ROOT" \
    --models "${MODEL_ARRAY[@]}" \
    --output-tsv "$METRIC_ROOT/full1000/paired_temporal_action_metrics.tsv" \
    --workers "$CPU_WORKERS" \
    >"$LOG_ROOT/paired_temporal_action.log" 2>&1 &
  pids+=("$!")
  for i in "${!MODEL_ARRAY[@]}"; do
    model="${MODEL_ARRAY[$i]}"
    gpu="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_GPU" "$SCRIPTS/paired_custom_metrics.py" \
      --dataset_manifest "$GT_MANIFEST" \
      --model_manifest "$FULL_ROOT/$model/manifest.tsv" \
      --out_dir "$METRIC_ROOT/full1000/custom_metrics/$model" \
      --device cuda:0 --da3-model "$DA3_MODEL" --vjepa-model "$VJEPA_MODEL" \
      --dino-repo "$DINO_REPO" --dino-checkpoint "$DINO_CHECKPOINT" \
      >"$LOG_ROOT/${model}_custom.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
}

run_vbench() {
  ROOT_DIR="$FORGEWM_ROOT" FORGEWM_ROOT="$FORGEWM_ROOT" \
    VBENCH_ROOT="$VBENCH_ROOT" VBENCH_LOCAL_CKPT="$VBENCH_LOCAL_CKPT" \
    BAKEOFF="$CONST_ROOT" MODELS="$MODELS" GPU_LIST="$GPU_LIST" \
    PYTHON_BIN="$PYTHON_GPU" bash "$SCRIPTS/run_vbench_parallel.sh" \
    >"$LOG_ROOT/constant_vbench_driver.log" 2>&1
}

run_perceptual() {
  local pids=() model gpu task=0 rank
  for model in "${MODEL_ARRAY[@]}"; do
    mkdir -p "$METRIC_ROOT/full1000/perceptual/$model"
    for rank in 0 1; do
      gpu="${GPU_ARRAY[$((task % ${#GPU_ARRAY[@]}))]}"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_IDM" "$SCRIPTS/evaluate_missing_perceptual.py" \
        --dataset_manifest "$GT_MANIFEST" \
        --model_manifest "$FULL_ROOT/$model/manifest.tsv" \
        --out_dir "$METRIC_ROOT/full1000/perceptual/$model" \
        --rank "$rank" --world_size 2 \
        --clip_checkpoint "$CLIP_CHECKPOINT" \
        >"$LOG_ROOT/${model}_perceptual_rank${rank}.log" 2>&1 &
      pids+=("$!")
      task=$((task + 1))
    done
  done
  wait_all "${pids[@]}"
}

run_control() {
  local pids=() model gpu i task=0
  for i in "${!MODEL_ARRAY[@]}"; do
    model="${MODEL_ARRAY[$i]}"
    gpu="${GPU_ARRAY[$((task % ${#GPU_ARRAY[@]}))]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_GPU" "$SCRIPTS/action_consistency_eval.py" \
      --videos-dir "$CONST_ROOT/$model/eval_videos" \
      --model-name "$model" \
      --output-dir "$METRIC_ROOT/constant/action_trajectory/$model" \
      --da3-model "$DA3_MODEL" \
      --device cuda:0 --stride 4 \
      >"$LOG_ROOT/${model}_trajectory.log" 2>&1 &
    pids+=("$!")
    task=$((task + 1))
  done
  for model in "${MODEL_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$((task % ${#GPU_ARRAY[@]}))]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_IDM" "$SCRIPTS/evaluate_constant_action_idm.py" \
      --videos-dir "$CONST_ROOT/$model/eval_videos" \
      --model-name "$model" \
      --output-dir "$METRIC_ROOT/constant/action_idm/$model" \
      --model "$IDM_MODEL" --weights "$IDM_WEIGHTS" \
      --device cuda:0 --frame-count 76 \
      >"$LOG_ROOT/${model}_idm.log" 2>&1 &
    pids+=("$!")
    task=$((task + 1))
  done
  wait_all "${pids[@]}"
}

run_summary() {
  "$PYTHON_GPU" "$SCRIPTS/summarize_native_budget_metrics.py" \
    --root "$EVAL_ROOT" | tee "$LOG_ROOT/summarize.log"
}

case "$STAGE" in
  validate) validate_inputs ;;
  paired) validate_inputs; run_paired ;;
  vbench) validate_inputs; run_vbench ;;
  perceptual) validate_inputs; run_perceptual ;;
  control) validate_inputs; run_control ;;
  summarize) run_summary ;;
  all)
    validate_inputs
    run_paired
    run_vbench
    run_perceptual
    run_control
    run_summary
    ;;
  *)
    echo "Usage: $0 {validate|paired|vbench|perceptual|control|summarize|all}" >&2
    exit 2
    ;;
esac
