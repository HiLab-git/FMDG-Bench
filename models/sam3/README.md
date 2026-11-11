# SAM3

Adaptation of SAM3 with a CNN or DPT decoder, with or without LoRA on the image
encoder.

## Files

| File | Purpose |
|---|---|
| `cnn_sam3_{Fundus,NPC,PolypDB}.py` | CNN decoder; frozen encoder |
| `DPT_sam3_{Fundus,NPC,PolypDB}.py` | DPT decoder; frozen encoder |
| `predsam3.py` | inference / prediction export |
| `run.sh`, `run_base.sh`, `pred.sh` | full run grids (3 datasets x 2 splits x 4 strategies) |
| `naso_dataset_16.py` | NPC dataset |
| `Datasets/`, `seg_head/` | dataloaders and decoders (shared) |
| `sam3/` | vendored SAM3 package |
| `README_TRAIN.md` | upstream SAM3 fine-tuning documentation |

## LoRA switch

The encoder is frozen by default. If `--task_name` contains `lora` (for example
`sam3_cnnlora_fu` or `sam3_DPTlora_na`), all linear layers of the encoder are
replaced by LoRA layers (`loralib`, r = 4, alpha = 4).

## Training

```bash
cd models/sam3
mkdir -p logs

python cnn_sam3_Fundus.py --data_root $FMDG_DATA_ROOT/Fundus \
    --trainid 1,3 --testid 2,4 --work_dir $FMDG_WORK_DIR \
    --task_name sam3_cnn_fu --num_epochs 50 --iternum 3000000 --device cuda:0
```

SAM3 weights: pass `--checkpoint /path/to/sam3_vitb.pth`. The 3.3 GB checkpoint from
the working copy is **not** shipped.

## Notes

- `import wandb` is unconditional in these scripts; `pip install wandb`.
- For the SAM3 baseline on 1080Ti, fp16 is used instead of bf16.
