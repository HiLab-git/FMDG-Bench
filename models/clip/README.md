# CLIP / MedCLIP

Only the `*_new.py` training scripts are shipped — they are the versions used for
the paper. The older duplicates without `_new` were removed.

## Files

| File | Purpose |
|---|---|
| `clip_cnn_{Fundus,NPC,PolypDB}_new.py` | CLIP + CNN decoder |
| `clip_DPT_{Fundus,NPC,PolypDB}_new.py` | CLIP + DPT decoder |
| `medclip_cnn_*_new.py`, `medclip_DPT_*_new.py` | MedCLIP variants (PubMedCLIP vision tower) |
| `dataset.py` | folder datasets: `FolderDataset_new`, `FolderDataset_NPC`, `FolderDataset_PolypDB` |
| `blocks.py`, `dpt_head.py`, `dpt_clip.py`, `dpt_clip_full_compatible.py`, `dpt_any_clip.py` | decoder implementations |
| `clip_feature_extractor.py`, `clip_feature_extractor_pubmed.py` | feature extractors |

## Dependencies

`import clip` requires the OpenAI CLIP package (not vendored), and the MedCLIP
variants require `transformers`:

```bash
pip install git+https://github.com/openai/CLIP.git "transformers>=4.40"
```

## Training

```bash
cd models/clip
python clip_cnn_Fundus_new.py --data_dir $FMDG_DATA_ROOT/Fundus --dataset fundus \
    --trainid 1,3 --testid 2,4 --epochs 50 --batch_size 4 --num_classes 3
```

LoRA is enabled when `--task_name` contains `lora` (for example
`--task_name clip_cnnlora_fu`); the rank and scale are set with `--lora_r` and
`--lora_alpha` (default 4 and 4).

## Notes

- `--num_classes 3` for Fundus (background, OC, OD), `--num_classes 1` (or 2,
  depending on the criterion) for NPC and PolypDB.
- MedCLIP weights are not shipped; download them and point the scripts to the local
  path.
