#!/bin/bash
# Download Matrix-Game 2 base model weights for ForgeWM.
#
# This script clones the MG2 repo (with git-lfs) and symlinks the required
# files into ./ckpts/MG2-base/ where the training configs expect them.
#
# Required files (total ~8.5 GB):
#   - diffusion_pytorch_model.safetensors (DiT weights, 3.5 GB)
#   - base_config.json (model architecture config)
#   - Wan2.1_VAE.pth (VAE encoder/decoder, 484 MB)
#   - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth (CLIP, 4.5 GB)
#   - xlm-roberta-large/ (tokenizer)
#
# Usage:
#   bash scripts/download_models.sh
#
# After running, you should have:
#   ckpts/MG2-base/
#   ├── base_config.json
#   ├── diffusion_pytorch_model.safetensors
#   ├── Wan2.1_VAE.pth
#   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
#   └── xlm-roberta-large/

set -e

CKPT_DIR="${CKPT_DIR:-./ckpts}"
MG2_DIR="${CKPT_DIR}/MG2-base"

if [ -f "${MG2_DIR}/diffusion_pytorch_model.safetensors" ]; then
    echo "MG2 base model already exists at ${MG2_DIR}. Skipping download."
    exit 0
fi

echo "=== Downloading Matrix-Game 2 base model ==="
echo "This will clone ~8.5 GB of model weights via git-lfs."
echo ""

# Clone MG2 repo (git-lfs required for large files)
if ! command -v git-lfs &> /dev/null; then
    echo "ERROR: git-lfs is required. Install with:"
    echo "  apt install git-lfs && git lfs install"
    exit 1
fi

TEMP_DIR=$(mktemp -d)
echo "Cloning SkyworkAI/Matrix-Game into temporary directory..."
git clone --depth 1 https://github.com/SkyworkAI/Matrix-Game.git "$TEMP_DIR/Matrix-Game"

# Create ckpts directory and copy/link required files
mkdir -p "$MG2_DIR"

MG2_SRC="$TEMP_DIR/Matrix-Game/Matrix-Game-2.0"

if [ ! -f "$MG2_SRC/base_model/diffusion_pytorch_model.safetensors" ]; then
    echo "ERROR: LFS files not downloaded. Run:"
    echo "  cd $TEMP_DIR/Matrix-Game && git lfs pull"
    exit 1
fi

echo "Copying model files to ${MG2_DIR}/ ..."
cp "$MG2_SRC/base_model/diffusion_pytorch_model.safetensors" "$MG2_DIR/"
cp "$MG2_SRC/base_model/base_config.json" "$MG2_DIR/"
cp "$MG2_SRC/Wan2.1_VAE.pth" "$MG2_DIR/"
cp "$MG2_SRC/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "$MG2_DIR/"
cp -r "$MG2_SRC/xlm-roberta-large" "$MG2_DIR/"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=== Done! Model files at ${MG2_DIR}/ ==="
ls -lh "$MG2_DIR/"
echo ""
echo "Next: prepare training data (see scripts/prepare_data.py)"
