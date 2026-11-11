# DINO / DINOv2 / DINOv3

Self-supervised foundation models adapted with a CNN or DPT decoder, with or
without LoRA.

## Files

| File | Purpose |
|---|---|
| `train_dinov1_D_{Fundus,NPC,PolypDB}.py` | DINOv1 + DPT decoder |
| `train_dinov2_D_{Fundus,NPC,PolypDB}.py` | DINOv2 + DPT decoder |
| `train_segdinov3_{Fundus,NPC,PolypDB}_D.py` | DINOv3 (loaded via `torch.hub`) |
| `dinov{1,2}_cnn_lora_*.py` | CNN decoder + LoRA |
| `dinov3_cnn_*.py` | DINOv3 + CNN decoder |
| `dpt.py`, `dpt_D*.py`, `blocks.py`, `dataset.py` | decoders and datasets |
| `dino/`, `dinov2/`, `dinov3/` | vendored upstream repositories |
| `run*.sh` | run grids |

## Training

```bash
cd models/dino
python train_dinov1_D_Fundus.py --data_dir $FMDG_DATA_ROOT/Fundus \
    --dino_ckpt /path/to/dinov1_vitb16.pth --dino_size b \
    --trainid 1,3 --testid 2,4 --epochs 50
```

For DINOv3:

```bash
python dinov3_cnn_Fundus.py --data_dir $FMDG_DATA_ROOT/Fundus \
    --dino_ckpt /path/to/dinov3_vitb16_pretrain.pth --repo_dir ./dinov3 \
    --trainid 1,3 --testid 2,4
```

`download_all_dino.sh` lists the official weight URLs.

## Notes

- DINOv3 is loaded with `torch.hub.load('./dinov3', 'dinov3_vitb16', source='local')`,
  which is why the vendored `dinov3/` folder must stay where it is.
- `requirements.txt` is the upstream SegDINO list; the shipped conda environment
  already covers it (`timm`, `torch`, `monai`, `medpy`).
