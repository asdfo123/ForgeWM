#!/bin/bash
# Upload ForgeWM data and checkpoints to HuggingFace.
# 
# Prerequisites:
#   1. Create repos on HuggingFace:
#      - https://huggingface.co/new-dataset → asdfo123/ForgeWM-data
#      - https://huggingface.co/new → asdfo123/ForgeWM
#   2. Generate a WRITE token at https://huggingface.co/settings/tokens
#   3. Run: huggingface-cli login --token <YOUR_WRITE_TOKEN>
#
# Usage:
#   bash scripts/upload_to_hf.sh

set -e

HF_USER="asdfo123"
DATA_REPO="${HF_USER}/ForgeWM-data"
MODEL_REPO="${HF_USER}/ForgeWM"

DATA_DIR="/apdcephfs_gy6/share_302641016/victorxyli/datasets/action_data_2003_360p"
STAGE0_CKPT="/apdcephfs_gy6/share_302641016/victorxyli/logs_new/mg2_stage0_bid_sft_360p/checkpoint_model_004000/model.pt"
STAGE3_CKPT="/apdcephfs_gy6/share_302641016/victorxyli/logs_new/mg2_stage3_dmd_v5_sft_teacher/checkpoint_model_002400/model.pt"

echo "=== Upload Training Data (89 GB) ==="
echo "Repo: ${DATA_REPO}"
echo "This will take a while..."
huggingface-cli upload "$DATA_REPO" "$DATA_DIR" . --repo-type dataset

echo ""
echo "=== Upload Checkpoints ==="
echo "Repo: ${MODEL_REPO}"

# Create temp structure for upload
UPLOAD_DIR=$(mktemp -d)
mkdir -p "$UPLOAD_DIR/stage0" "$UPLOAD_DIR/stage3"
cp "$STAGE0_CKPT" "$UPLOAD_DIR/stage0/model.pt"
cp "$STAGE3_CKPT" "$UPLOAD_DIR/stage3/model.pt"

# Add a model card
cat > "$UPLOAD_DIR/README.md" << 'CARD'
---
license: apache-2.0
tags:
  - world-model
  - minecraft
  - video-generation
  - causal-forcing
---

# ForgeWM Checkpoints

Training checkpoints for [ForgeWM](https://github.com/asdfo123/ForgeWM).

## Available Checkpoints

| File | Stage | Description |
|------|-------|-------------|
| `stage0/model.pt` | Stage 0 | Bidirectional SFT (domain adaptation, 4000 steps) |
| `stage3/model.pt` | Stage 3 | DMD final model (4-step real-time inference, 2400 steps) |

## Usage

```bash
huggingface-cli download asdfo123/ForgeWM --local-dir ./ckpts
```

Then run inference:
```bash
python inference.py --checkpoint_path ckpts/stage3/model.pt --image_path demo_images/forest.png --action_type forward
```

## 🚧 Coming Soon
- Stage 1 (Teacher-Forcing AR) checkpoint
- Stage 2 (Consistency Distillation) checkpoint
CARD

huggingface-cli upload "$MODEL_REPO" "$UPLOAD_DIR" . --repo-type model
rm -rf "$UPLOAD_DIR"

echo ""
echo "=== Done! ==="
echo "Data:   https://huggingface.co/datasets/${DATA_REPO}"
echo "Models: https://huggingface.co/${MODEL_REPO}"
