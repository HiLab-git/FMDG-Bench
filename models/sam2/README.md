# SAM2

Adaptation of SAM2 with a CNN or DPT decoder, with or without LoRA.

## Files

| File | Purpose |
|---|---|
| `cnn_sam2_{Fundus,NPC,PolypDB}.py` | CNN decoder |
| `DPT_sam2_{Fundus,NPC,PolypDB}.py` | DPT decoder |
| `predsam2.py` | inference |
| `test.py`, `test_sam2_Fundus.py` | evaluation |
| `naso_dataset.py`, `naso_dataset_16.py` | NPC datasets (two preprocessing variants) |
| `run.sh` | full run grid |
| `sam2/`, `seg_head/`, `Datasets/` | vendored model code and helpers |

## Training

```bash
cd models/sam2
python cnn_sam2_NPC.py --data_root $FMDG_DATA_ROOT/NPC/preprocessed \
    --trainid 1,4 --testid 2,3 --work_dir $FMDG_WORK_DIR \
    --task_name sam2_cnn_na --num_epochs 50 --device cuda:0
```

Weights: download `sam2.1_hiera_b+.pt` from the official release and pass it with
`--checkpoint`.

## Notes

- LoRA is switched on by the `lora` substring in `--task_name`, as in the SAM3 and
  MedSAM pipelines.
- The upstream `demo/`, `sav_dataset/` and `training/` folders were removed; they are
  not used by the benchmark.
