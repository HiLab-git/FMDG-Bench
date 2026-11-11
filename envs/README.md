# Environment

One conda environment is shipped for the SAM3 / SAM2 / SAM / MedSAM / DINO /
baseline pipelines:

```bash
conda env create -f envs/environment.yml
conda activate fmdg-bench
python tools/check_env.py
```

`environment.yml` was exported from the lab server environment `Fsam3` (PyTorch
2.7.0 + CUDA 12.6 wheels, Python 3.12, numpy 1.26.4, timm, MONAI, SimpleITK,
medpy, `sam3`, `loralib`, TensorBoard) and cleaned for release: the `prefix:` line
was removed and the environment was renamed to `fmdg-bench`.

## Coverage

| Model family | Covered by this env | Notes |
|---|---|---|
| SAM3 (`models/sam3`) | yes | `sam3==0.1.0` installed; `seg_head/` and `Datasets/` are vendored in the repo |
| SAM2 (`models/sam2`) | yes | `sam2` package is vendored; weights must be downloaded |
| SAM / MedSAM (`models/medsam`) | yes | `segment_anything` is vendored |
| DINOv1 / v2 / v3 (`models/dino`) | yes | `timm`; DINOv3 is loaded through `torch.hub` from `models/dino/dinov3` |
| Baselines: UNet, UNet++, Attention UNet, TransUNet | yes | `monai` for the UNet baseline |
| Baselines: nnU-Net (`models/baselines/nnunet`) | **no** | needs `nnunetv2`; historically validated in a separate environment |
| CLIP / MedCLIP (`models/clip`) | **no** | needs the OpenAI `clip` package and `transformers` |
| Logging | partial | TensorBoard is installed; `wandb` is imported by `models/sam3/*` and `models/baselines/trainer/train.py` |

This is why the environment is documented as *"validated on the SAM3 pipeline"*
rather than as a universal environment. Two optional installs close most of the gap:

```bash
pip install git+https://github.com/openai/CLIP.git
pip install "transformers>=4.40" wandb
```

Installing `transformers` may downgrade `huggingface-hub` (this environment ships
1.8.0, while stable transformers releases usually require `<1.0`). Test in a clone
first:

```bash
conda create -n fmdg-test --clone fmdg-bench
conda activate fmdg-test
pip freeze > /tmp/before.txt
pip install "transformers>=4.40" wandb
python tools/check_env.py
```

After a successful test, freeze only the additions:

```bash
pip freeze | grep -E '^(transformers|peft|wandb|clip)==' > envs/requirements-extra.txt
```

## Hardware notes

- **CUDA 12.6 wheels** require a reasonably recent driver (`nvidia-smi` should
  report CUDA Version >= 12.6, or at least a driver new enough for CUDA 12.x).
- **GTX 1080Ti (sm_61)** has no bfloat16 support: run with fp16/fp32. If PyTorch
  reports `no kernel image is available`, the wheel does not contain sm_61 kernels
  and a CUDA 11.8 build is needed.
- The paper reports experiments on 4x GTX 1080Ti and 1x RTX 3090Ti.

## Adding a second environment (if needed)

If nnU-Net or MedCLIP cannot coexist with the main environment on your machine,
install them in separate environments and document the mapping in the root
`README.md`. Keep `environment.yml` as the single *default* environment so that
new users have one obvious entry point.
