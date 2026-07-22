# ForgeWM paper main-table evaluation pipeline

This directory reproduces the paper's main comparison table for **ForgeWM,
Matrix-Game 2.0, and HY-WorldPlay**. It is intentionally self-contained inside
the ForgeWM repository. Model inference is not part of the default entry point:
the pipeline starts from frozen generated videos and recomputes the metrics.

The reference inputs are versioned separately on Hugging Face. Generated model
outputs are not in that reference repository and must be copied into the layout
below before running the complete evaluation.

## Main-table protocol

| Metric | Input | Reference | Samples per model |
|---|---|---|---:|
| Imaging Quality | constant single-action videos | no GT video | 462 |
| Aesthetic Quality | constant single-action videos | no GT video | 462 |
| LPIPS | shared action rollouts | paired GT video | 1,000 |
| Flow Profile Cosine | shared action rollouts | paired GT video | 1,000 |
| Depth Similarity | shared action rollouts | paired GT video | 1,000 |
| Action Sign Accuracy | shared action rollouts | paired GT/action | 1,000 |
| KCtrl | opposite constant-action pairs | commanded action | 462 |
| Mouse Accuracy | constant turn videos | commanded action | 462 |

The constant-action suite contains 77 initial frames and six controls:
`forward`, `back`, `left`, `right`, `turn_left`, and `turn_right`. Thus each
model has 77 × 6 = 462 clips. The paired suite uses the same 1,000 GT
scene/action trajectories for all three models. Clips contain 77 frames at
640×352, are encoded at 12 fps, and use seed 0.

## Data layout

```text
evaluation_pipeline/data/
├── reference/                           # downloaded from Hugging Face
│   ├── gt_1000/                         # paired GT videos/actions/frames
│   └── constant_action_77/              # independent initial frames
├── main_table/
│   ├── paired_full1000/
│   │   ├── forgewm/                     # 1,000 generated clips
│   │   ├── mg2/                         # 1,000 generated clips
│   │   ├── hyworld/                     # 1,000 generated clips
│   │   ├── custom_metrics/              # frozen paired depth results
│   │   └── supplementary_metrics/       # frozen flow/sign results
│   ├── constant_action/
│   │   ├── forgewm/                     # 462 generated clips
│   │   ├── mg2/                         # 462 generated clips
│   │   ├── hyworld_aligned/             # 462 generated clips
│   │   ├── action_trajectory/           # KCtrl results
│   │   └── action_idm/                  # mouse-control results
│   ├── constant_action_aligned77/        # 77-frame VBench visual inputs/results
│   ├── perceptual/                      # frozen Full-1000 LPIPS results
│   └── metrics/                         # clean reruns and final main table
└── native_budget_eval/                  # optional 1/2-step budget ablation
```

Paths inside manifests may be absolute, relative to the manifest, or relative
to the ForgeWM repository. Manifest-relative paths are recommended for shared
bundles.

## Reproduction setup

Create one Python environment with a CUDA-compatible PyTorch build, then install
the lightweight metric dependencies:

```bash
python3 -m pip install -r evaluation_pipeline/requirements.txt
```

Fetch the exact external source revisions and the public VPT IDM weights:

```bash
bash evaluation_pipeline/setup_external_repos.sh
```

The VBench and GameWorld projects have additional dependencies. Install their
requirements into the same environment before running `vbench` or `control`:

```bash
python3 -m pip install -r evaluation_pipeline/external/VBench/requirements.txt
python3 -m pip install -r \
  evaluation_pipeline/external/Matrix-Game/Matrix-Game-1/GameWorldScore/requirements.txt
```

The setup script pins these source revisions:

- VBench: `45e79ec14e69a2187202c675d2dbce1a71843d53`
- Matrix-Game: `71c3cd7f741311f8100f6cf9cde942b6c1378d11`

Log in to Hugging Face (the dataset is private) and download the frozen
reference bundle:

```bash
huggingface-cli login
python3 evaluation_pipeline/download_reference_data.py
```

The downloader pins dataset revision
`5a7071587e641ad130c7dbf6018cdb81631af9ac`. The repository is
`ForgeWM/ForgeWM-GT-1000`; grant the reproducer read access before sharing
these commands.

Finally, copy the generated-video bundle into
`evaluation_pipeline/data/main_table/`. It must contain all three paired
Full-1000 outputs and all three 77x6 constant-action outputs shown in the data
layout. These outputs are required even if only `summarize` is run, unless the
complete frozen metric files are supplied instead.

## Run the main table

Validate every model and input set without launching metric jobs:

```bash
bash evaluation_pipeline/run_evaluation.sh validate
```

Reproduce the final table from the frozen per-video metric outputs:

```bash
bash evaluation_pipeline/run_evaluation.sh summarize
```

Recompute all metrics from the generated videos:

```bash
bash evaluation_pipeline/run_evaluation.sh all
```

Before the first VBench run, create the strictly frame-aligned visual suite:

```bash
bash evaluation_pipeline/scripts/prepare_constant_action_aligned77.sh
```

This uses the first 77 frames of every model output without re-encoding the
H.264 video stream. VBench Aesthetic and Imaging Quality read every input frame,
so they must use this directory rather than mixing 81-frame ForgeWM/Matrix-Game
outputs with 77-frame HY-WorldPlay outputs.

Stages can be launched or resumed independently:

```bash
bash evaluation_pipeline/run_evaluation.sh temporal
bash evaluation_pipeline/run_evaluation.sh depth
bash evaluation_pipeline/run_evaluation.sh vbench
bash evaluation_pipeline/run_evaluation.sh perceptual
bash evaluation_pipeline/run_evaluation.sh control
bash evaluation_pipeline/run_evaluation.sh summarize
```

The final artifacts are:

- `data/main_table/metrics/main_table_summary.json`
- `data/main_table/metrics/main_table_summary.md`
- `data/main_table/metrics/logs/`

Portable defaults for paths, models, GPUs, and checkpoints are in
`config/paper_main.env`. Override variables in the shell, or copy the file and
set `EVAL_CONFIG` to the copied configuration. In particular, `PYTHON_GPU`,
`GPU_LIST`, and local model paths may need adjustment.

## Optional budget ablation

The ForgeWM-1/2 native-budget evaluation is retained but is not the default
pipeline:

```bash
bash evaluation_pipeline/run_budget_ablation.sh validate
bash evaluation_pipeline/run_budget_ablation.sh summarize
```

Its settings are isolated in `config/budget_ablation.env`.

## Versioning and data policy

`data/`, `external/`, generated videos, checkpoints, and caches are ignored by
Git. Share a Git commit SHA for this directory together with the pinned HF
dataset revision above. Do not describe the Full-1000 trajectories as a
verified held-out benchmark; training-set exclusion has not been established.
