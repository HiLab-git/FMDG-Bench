# SAM / MedSAM / SAM-Med2D

The `*_16.py` scripts are the versions used for the paper. SAM, MedSAM and
SAM-Med2D share the same code path and differ only in the checkpoint passed via
`--checkpoint`.

## Files

| File | Purpose |
|---|---|
| `cnn_{fundus,npc,poly}_16.py` | CNN decoder (frozen encoder, optional LoRA) |
| `sam_DPT_{fundus,npc,poly}_16.py` | DPT decoder (optional LoRA) |
| `naso_dataset_16.py` | NPC dataset |
| `run_16.sh` | full run grid (SAM / MedSAM x CNN / DPT x ± LoRA) |
| `segment_anything/` | vendored Segment Anything code |
| `seg_head/` | CNN and DPT decoders |
| `Datasets/` | Fundus / PolypDB dataloaders |

## Training

```bash
cd models/medsam
python sam_DPT_fundus_16.py --data_root $FMDG_DATA_ROOT/Fundus \
    --trainid 1,3 --testid 2,4 --work_dir $FMDG_WORK_DIR \
    --task_name sam_DPT_fu --num_epochs 50 --device cuda:0
```

Or run the whole grid:

```bash
mkdir -p logsss
bash run_16.sh
```

## Checkpoints

| Model | Checkpoint |
|---|---|
| SAM ViT-B | `sam_vit_b_01ec64.pth` |
| MedSAM ViT-B | `medsam_vit_b.pth` |
| SAM-Med2D | `SAM-Med2D-B.pth` |

LoRA is enabled when `--task_name` contains `lora` (for example `medsam_DPTlora_fu`).

## Notes

- The earlier variants (`*_another.py`, `save_files/`, `savefiles/`, runs) were
  removed; only the `_16` versions are kept.
- `run_16.sh` expects `$FMDG_DATA_ROOT` and `$FMDG_WORK_DIR` to be exported.
