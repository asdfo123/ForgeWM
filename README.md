<p align="center">
  <img src="assets/banner.png" width="600">
</p>

<h1 align="center">ForgeWM</h1>

<p align="center">
  <b>The first publicly reproducible training recipe for the Matrix-Game 2 lineage.</b>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#training-pipeline">Training</a> •
  <a href="#results">Results</a> •
  <a href="#acknowledgements">Acknowledgements</a>
</p>

---

## About

**ForgeWM is an integration project.** We don't introduce new methods, models, or data. We connect [Matrix-Game 2](https://github.com/skywork-ai/matrix-game) (Skywork), the [GameFactory](https://github.com/GameFactory-Minecraft/GameFactory) open dataset, and the [Causal Forcing](https://arxiv.org/abs/2602.02214) distillation paradigm into the first publicly reproducible training recipe for the MG2 lineage — runnable on 8 GPUs.

### Why does this exist?

Matrix-Game 2 released open weights but not training code or data. Causal Forcing released a general-purpose framework but not a game-specific recipe. GameFactory released Minecraft gameplay data but not a full training pipeline. **ForgeWM connects these three pieces into a working end-to-end system.**

---

## Comparison

| Project | Base Model | Control | Action Injection | Paradigm | Data Open | Code Open |
|---------|-----------|---------|-----------------|----------|-----------|-----------|
| **ForgeWM** | MG2 (Wan2.1-1.3B) | Keyboard + Mouse | Cross-Attn (kbd) + Concat (mouse) | Causal Forcing | ✅ GameFactory | ✅ |
| minWM | HY1.5 / Wan2.1 | Camera pose | PRoPE (attention-bias) | Causal Forcing | ❌ | ✅ |
| MG2 (Skywork) | Wan2.1-1.3B | Keyboard + Mouse | Cross-Attn + Concat | Self Forcing | ❌ | ❌ |
| HY-GameCraft | HunyuanVideo | Unified camera | Token-add | Phased Consistency | ❌ | Partial |

---

## Results

### ForgeWM vs Matrix-Game 2 Base

ForgeWM's 4-stage Causal Forcing pipeline produces action-controllable video while maintaining visual quality comparable to MG2's original Self Forcing distillation:

> 🚧 Visual comparisons will be uploaded with checkpoint release.

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Download Models

```bash
bash scripts/download_models.sh
```

### Inference (Single GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config_path configs/stage3_dmd.yaml \
    --checkpoint_path ckpts/stage3/model.pt \
    --image_path demo_images/cave.png \
    --action_type forward \
    --output_path output/demo.mp4
```

---

## Training Pipeline

| Stage | Method | Input | Output | Time (8×H20) |
|-------|--------|-------|--------|--------------|
| 0 | Bidirectional SFT | MG2 base | Domain-adapted base | ~10h |
| 1 | Teacher-Forcing Causal AR | Stage 0 | Causal AR model | ~30h |
| 2 | Consistency Distillation | Stage 1 | Few-step model | ~18h |
| 3 | DMD | Stage 2 | 4-step real-time model | ~32h |

```bash
# Stage 0
torchrun --nproc_per_node=8 train.py --config_path configs/stage0_bid_sft.yaml --logdir logs/stage0

# Stage 1
torchrun --nproc_per_node=8 train.py --config_path configs/stage1_teacher_forcing.yaml --logdir logs/stage1

# Stage 2
torchrun --nproc_per_node=8 train.py --config_path configs/stage2_consistency_distillation.yaml --logdir logs/stage2

# Stage 3
torchrun --nproc_per_node=8 train.py --config_path configs/stage3_dmd.yaml --logdir logs/stage3
```

---

## Architecture

Matrix-Game 2's hybrid action injection:

- **Keyboard (discrete)**: Cross-attention into each transformer block
- **Mouse (continuous)**: Concatenation with sliding-window grouping
- **History frame**: Channel-concat (I2V cond_concat + CLIP visual context)
- **Attention**: Block-wise causal + sliding window (local_attn_size=6)

---

## Roadmap

- ✅ Stage 0–3 training pipeline
- 🚧 Checkpoint release (HuggingFace)
- 🚧 Interactive demo
- 🚧 Tech report

---

## Acknowledgements

ForgeWM integrates work from multiple research groups:

| Component | Source | Contribution |
|-----------|--------|-------------|
| Base model | [Matrix-Game 2](https://github.com/skywork-ai/matrix-game) | I2V backbone + hybrid action module |
| Training data | [GameFactory](https://github.com/GameFactory-Minecraft/GameFactory) | Open Minecraft data + balanced actions |
| Distillation | [Causal Forcing](https://arxiv.org/abs/2602.02214) | AR diffusion distillation paradigm |

We also thank the authors of:
- [Self-Forcing](https://arxiv.org/abs/2406.00893) — Autoregressive generation via self-forcing
- [CausVid](https://arxiv.org/abs/2412.07772) — Causal video diffusion distillation
- [Wan 2.1](https://github.com/Wan-AI/Wan) — Video generation foundation model
- [minWM](https://github.com/MIN-Lab/minWM) — Causal Forcing pipeline reference
- [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) — Video generation backbone

---

## Citation

```bibtex
@misc{forgewm2025,
  title={ForgeWM: A Reproducible Training Recipe for Matrix-Game 2 World Models},
  author={ForgeWM Team},
  year={2025},
  url={https://github.com/asdfo123/ForgeWM}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
