# Models

Every folder is self-contained: `cd` into it and run the training/inference scripts.
Shared helpers (`segment_anything/`, `seg_head/`, `Datasets/`, `naso_dataset*.py`)
are vendored per family on purpose — see [../docs/repository.md](../docs/repository.md).

| Folder | Models | Entry point |
|---|---|---|
| [`sam3/`](sam3/README.md) | SAM3 | `cnn_sam3_*.py`, `DPT_sam3_*.py`, `run.sh` |
| [`sam2/`](sam2/README.md) | SAM2 | `cnn_sam2_*.py`, `DPT_sam2_*.py`, `run.sh` |
| [`medsam/`](medsam/README.md) | SAM, MedSAM, SAM-Med2D | `*_16.py`, `run_16.sh` |
| [`clip/`](clip/README.md) | CLIP, MedCLIP | `*_new.py` |
| [`dino/`](dino/README.md) | DINO, DINOv2, DINOv3 | `train_dinov*_*.py`, `dinov3_cnn_*.py` |
| [`baselines/`](baselines/README.md) | UNet, UNet++, Attention UNet, TransUNet, nnU-Net | `train.py`, `UNet.py` |

Common environment variables:

```bash
export FMDG_DATA_ROOT=/path/to/FMDG-data
export FMDG_WORK_DIR=/path/to/outputs
```
