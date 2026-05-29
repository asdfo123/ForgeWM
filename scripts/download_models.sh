#!/bin/bash
# Download Matrix-Game 2 base model weights for ForgeWM.
#
# Pulls weights from HuggingFace (Skywork/Matrix-Game-2.0) into ./ckpts/MG2-base/
# where the training configs expect them.
#
# Required files (total ~9 GB):
#   - base_model/diffusion_pytorch_model.safetensors  (DiT weights, ~3.5 GB)
#   - base_model/base_config.json                      (architecture)
#   - Wan2.1_VAE.pth                                   (VAE, ~485 MB)
#   - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  (CLIP, ~4.5 GB)
#   - xlm-roberta-large/                               (tokenizer)
#
# Usage:
#   bash scripts/download_models.sh
#
# Requires: huggingface_hub  (pip install huggingface_hub)
#
# After running, ckpts/MG2-base/ will look like:
#   ckpts/MG2-base/
#   ├── base_config.json
#   ├── diffusion_pytorch_model.safetensors
#   ├── Wan2.1_VAE.pth
#   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
#   └── xlm-roberta-large/

set -e

CKPT_DIR="${CKPT_DIR:-./ckpts}"
MG2_DIR="${CKPT_DIR}/MG2-base"
HF_REPO="Skywork/Matrix-Game-2.0"

if [ -f "${MG2_DIR}/diffusion_pytorch_model.safetensors" ] \
   && [ -f "${MG2_DIR}/Wan2.1_VAE.pth" ] \
   && [ -f "${MG2_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ] \
   && [ -d "${MG2_DIR}/xlm-roberta-large" ] \
   && [ -f "${MG2_DIR}/base_config.json" ]; then
    echo "MG2 base model already present at ${MG2_DIR}. Skipping download."
    exit 0
fi

if ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: huggingface-cli is required. Install with:"
    echo "  pip install huggingface_hub"
    exit 1
fi

echo "=== Downloading Matrix-Game 2 base model from HuggingFace ==="
echo "Repo: ${HF_REPO}"
echo "Target: ${MG2_DIR}"
echo "Total download size: ~9 GB"
echo ""

mkdir -p "${MG2_DIR}"

# Download only the files we actually need into a staging dir, then move them
# into the flat layout the training configs expect.
STAGE_DIR=$(mktemp -d)
trap 'rm -rf "$STAGE_DIR"' EXIT

huggingface-cli download "$HF_REPO" \
    --local-dir "$STAGE_DIR" \
    --include "base_model/diffusion_pytorch_model.safetensors" \
              "base_model/base_config.json" \
              "Wan2.1_VAE.pth" \
              "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
              "xlm-roberta-large/*"

echo ""
echo "Moving files into flat layout at ${MG2_DIR}/ ..."
mv "$STAGE_DIR/base_model/diffusion_pytorch_model.safetensors" "${MG2_DIR}/"
mv "$STAGE_DIR/base_model/base_config.json"                    "${MG2_DIR}/"
mv "$STAGE_DIR/Wan2.1_VAE.pth"                                 "${MG2_DIR}/"
mv "$STAGE_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "${MG2_DIR}/"
rm -rf "${MG2_DIR}/xlm-roberta-large"
mv "$STAGE_DIR/xlm-roberta-large"                              "${MG2_DIR}/"

echo ""
echo "=== Done. Files in ${MG2_DIR}/: ==="
ls -lh "${MG2_DIR}/"
echo ""
echo "Next: prepare training data (see README #training-pipeline)."
