# Baselines

Five segmentation networks trained from scratch (full fine-tuning).

| Folder | Model | Entry point |
|---|---|---|
| `unet/` | UNet (MONAI implementation, full-parameter training) | `UNet.py` |
| `trainer/` | UNet++, Attention UNet, TransUNet, nnU-Net wrapper | `train.py --model_type {unetpp,attention_unet,transunet,nnunet}` |
| `nnunet/` | nnU-Net training pipeline | `train.py`, `run_all.sh` |

`trainer/models/transunet.py` is the **authoritative TransUNet implementation** used
for the paper; the copies that existed inside the SAM3 working directory were
removed.

## UNet

```bash
cd models/baselines/unet
python UNet.py --data_root $FMDG_DATA_ROOT/Fundus --trainid 1,3 --testid 2,4 \
    --work_dir $FMDG_WORK_DIR --num_epochs 50
```

## UNet++ / Attention UNet / TransUNet

```bash
cd models/baselines/trainer
python train.py --data_root $FMDG_DATA_ROOT/Fundus --trainid 1,3 --testid 2,4 \
    --model_type transunet --work_dir $FMDG_WORK_DIR --num_epochs 50
```

`run_Baseline.sh` and `run_all.sh` contain the full grids.

## nnU-Net

```bash
cd models/baselines/nnunet
bash run_all.sh
```

Requires `nnunetv2`; a separate environment is recommended because nnU-Net pins
`numpy<2` and other constraints that may conflict with the foundation-model stack.

## Notes

- The baseline trainer imports `wandb`; install it or patch the import.
- Per-domain Dice/ASSD evaluation is in `unet/utils/metrics.py` and
  `trainer/models/` (see `docs/reproduce.md`).
