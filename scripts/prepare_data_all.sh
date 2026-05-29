#!/bin/bash
# Prepare all 10 shards of GF-Minecraft data in parallel (one per GPU).
#
# Prerequisites:
#   1. Download GF-Minecraft dataset: https://github.com/KlingAIResearch/GameFactory
#      Expected structure: data_2003/video/*.mp4 + data_2003/metadata/*.json
#   2. MG2 base model downloaded (for VAE): bash scripts/download_models.sh
#
# Usage:
#   GF_DATA=/path/to/GF-Minecraft/data_2003 bash scripts/prepare_data_all.sh
#
# Output: ./data/action_lmdb/shard00..shard09 (each ~9 GB, total ~89 GB)

set -e

GF_DATA="${GF_DATA:?Set GF_DATA to path of GF-Minecraft/data_2003 (contains video/ and metadata/)}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/action_lmdb}"
VAE_PATH="${VAE_PATH:-./ckpts/MG2-base/Wan2.1_VAE.pth}"
NUM_GPUS="${NUM_GPUS:-8}"
VIDEOS_PER_SHARD=200
NUM_SHARDS=10
CLIPS_PER_VIDEO=20

echo "=== ForgeWM Data Preparation ==="
echo "  GF-Minecraft data: $GF_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  VAE: $VAE_PATH"
echo "  GPUs: $NUM_GPUS"
echo "  Shards: $NUM_SHARDS × $VIDEOS_PER_SHARD videos × $CLIPS_PER_VIDEO clips"
echo ""

if [ ! -f "$VAE_PATH" ]; then
    echo "ERROR: VAE not found at $VAE_PATH. Run scripts/download_models.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_name=$(printf "shard%02d" $shard)
    start=$((shard * VIDEOS_PER_SHARD))
    end=$((start + VIDEOS_PER_SHARD))
    gpu=$((shard % NUM_GPUS))

    if [ -d "$OUTPUT_DIR/$shard_name" ] && [ -f "$OUTPUT_DIR/$shard_name/data.mdb" ]; then
        echo "[$shard_name] Already exists, skipping (delete to re-generate)"
        continue
    fi

    echo "[$shard_name] GPU=$gpu videos[$start:$end] → $OUTPUT_DIR/$shard_name"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/prepare_data.py \
        --data_dir "$GF_DATA" \
        --output_dir "$OUTPUT_DIR/$shard_name" \
        --vae_path "$VAE_PATH" \
        --video_start $start \
        --video_end $end \
        --num_clips_per_video $CLIPS_PER_VIDEO \
        --flush_every 50 &

    # Limit parallelism to NUM_GPUS
    if [ $(( (shard + 1) % NUM_GPUS )) -eq 0 ]; then
        wait
    fi
done

wait
echo ""
echo "=== Done. Shards at $OUTPUT_DIR/ ==="
ls -lh "$OUTPUT_DIR"/
