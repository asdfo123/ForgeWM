#!/usr/bin/env bash
# Run one VBench process per model on separate GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${FORGEWM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BAKEOFF="${BAKEOFF:?BAKEOFF is required}"
MODELS_STR="${MODELS:-forgewm mg2 hyworld_aligned}"
GPU_LIST_STR="${GPU_LIST:-0 1 2}"
LOG_DIR="$BAKEOFF/pipeline_logs"

read -r -a MODELS_ARR <<< "$MODELS_STR"
read -r -a GPUS <<< "$GPU_LIST_STR"
mkdir -p "$LOG_DIR"

pids=()
for i in "${!MODELS_ARR[@]}"; do
  model="${MODELS_ARR[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  ROOT_DIR="$ROOT_DIR" BAKEOFF="$BAKEOFF" MODELS="$model" GPU="$gpu" \
    bash "$SCRIPT_DIR/vbench_eval_bakeoff_aligned.sh" \
    > "$LOG_DIR/${model}_vbench.log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  echo "Started VBench $model on GPU $gpu (pid=$pid)"
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "At least one VBench worker failed. See $LOG_DIR/*_vbench.log" >&2
  exit 1
fi
