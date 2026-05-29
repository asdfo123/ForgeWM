#!/bin/bash
# Download Matrix-Game 2 base model and VAE weights.
# Requires: huggingface-cli (pip install huggingface_hub)

set -e

CKPT_DIR="${CKPT_DIR:-./ckpts}"
mkdir -p "$CKPT_DIR"

echo "=== Downloading MG2 base model ==="
echo "Please download the Matrix-Game 2 base model from Skywork's release"
echo "and place it at: $CKPT_DIR/MG2-base/"
echo ""
echo "Expected structure:"
echo "  $CKPT_DIR/MG2-base/"
echo "  ├── base_config.json"
echo "  └── diffusion_pytorch_model.safetensors"
echo ""
echo "For Wan2.1 VAE and CLIP:"
echo "  $CKPT_DIR/Wan2.1-T2V-1.3B/"
echo "  ├── Wan2.1_VAE.pth"
echo "  ├── models_t5_umt5-xxl-enc-bf16.pth"
echo "  └── xlm_roberta_tokenizer/"
echo ""
echo "You can download Wan2.1-T2V-1.3B from HuggingFace:"
echo "  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir $CKPT_DIR/Wan2.1-T2V-1.3B"
