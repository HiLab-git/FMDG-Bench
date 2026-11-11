# Installation

## Requirements

- Linux (the scripts were developed on Ubuntu servers), NVIDIA GPU, CUDA driver
  compatible with CUDA 12.6 wheels.
- [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html).
- About 15 GB of disk for the environment, plus space for datasets and checkpoints.

## 1. Create the environment

```bash
git clone https://github.com/HiLab-git/FMDG-Bench.git
cd FMDG-Bench

conda env create -f envs/environment.yml
conda activate fmdg-bench
```

The environment is a cleaned export of the lab server environment used for the
SAM3 experiments (PyTorch 2.7.0, CUDA 12.6 wheels, Python 3.12, numpy 1.26.4).

## 2. Self-check

```bash
python tools/check_env.py
```

This prints the PyTorch/CUDA status, GPU capabilities (including whether bfloat16
is supported) and a per-family coverage table. `--strict` returns a non-zero exit
code when a required family is missing — useful before publishing or in CI.

## 3. Optional dependencies

| Feature | Install |
|---|---|
| CLIP / MedCLIP (`models/clip/*_new.py`) | `pip install git+https://github.com/openai/CLIP.git "transformers>=4.40"` |
| W&B logging (`models/sam3/*`, `models/baselines/trainer/train.py`) | `pip install wandb` |
| nnU-Net (`models/baselines/nnunet`) | `pip install nnunetv2` (recommended: separate environment) |

If `transformers` downgrades `huggingface-hub`, verify that SAM3 still imports.
A safe workflow is:

```bash
conda create -n fmdg-test --clone fmdg-bench
conda activate fmdg-test
pip install "transformers>=4.40" wandb
python tools/check_env.py
```

and only then update the main environment.

## 4. Weights

No checkpoints are shipped. Download the ViT-B backbones from the official
releases and pass their paths through the corresponding flags:

| Model | Flag | Where to get it |
|---|---|---|
| DINO / DINOv2 / DINOv3 | `--dino_ckpt` | official DINO / DINOv2 / DINOv3 release pages |
| SAM | `--checkpoint` (sam_vit_b) | Segment Anything release |
| SAM2 | `--checkpoint` (sam2.1_hiera_b+) | SAM 2 release |
| SAM3 | `--checkpoint` | SAM 3 release |
| MedSAM / SAM-Med2D | `--checkpoint` | MedSAM / SAM-Med2D releases |
| CLIP / MedCLIP | `--model_path` / package cache | OpenAI CLIP, MedCLIP, PubMedCLIP |

See each `models/*/README.md` for the exact flag names.

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no kernel image is available` | The PyTorch wheel has no kernels for your GPU (e.g. sm_61 for GTX 1080Ti). Install a CUDA 11.8 build of torch. |
| `bf16 is not supported` | GTX 1080Ti has no bfloat16. Use fp16/fp32 (the code falls back automatically in most scripts). |
| `ModuleNotFoundError: clip` | Install the OpenAI CLIP package (see above). |
| `ModuleNotFoundError: wandb` | `pip install wandb`, or set `WANDB_MODE=disabled` after installing. |
| `wandb.login()` asks for a key at runtime | Run `wandb login` once, or `export WANDB_MODE=disabled`. |
| `ModuleNotFoundError: seg_head` | Run from inside the model family folder (each family ships its own `seg_head/`). |
| OOM on 24 GB GPUs | Reduce `--batch_size`; SAM-family training uses batch size 1 in the provided scripts. |
