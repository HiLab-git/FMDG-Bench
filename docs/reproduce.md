# Reproducing the experiments

All commands assume `conda activate fmdg-bench`, `FMDG_DATA_ROOT` and
`FMDG_WORK_DIR` are exported, and the working directory is the model family folder.

```bash
export FMDG_DATA_ROOT=/path/to/FMDG-data
export FMDG_WORK_DIR=/path/to/outputs
```

## Naming convention

- `task_name` selects the experiment and is also the output folder name.
- **LoRA is enabled by the task name**: if `task_name` contains `lora`
  (case-insensitive), every linear layer of the image encoder is replaced by a
  LoRA layer (`loralib`, r = 4, alpha = 4). Otherwise the encoder stays frozen and
  only the decoder is trained.
- `cnn` / `DPT` in the script name selects the decoder.

Examples: `sam3_cnn_fu` = SAM3 + CNN decoder, frozen encoder; `sam3_cnnlora_fu` =
SAM3 + CNN decoder + LoRA; `sam_DPT_na` / `medsam_DPTlora_na` = SAM / MedSAM with
DPT decoder, with or without LoRA.

## Experiment matrix

| Paper component | Model family | Scripts | Adaptation strategies |
|---|---|---|---|
| Foundation models (general) | DINO / DINOv2 / DINOv3 | `models/dino/train_*` | CNN, DPT, +LoRA (`*_lora_*` scripts) |
| Foundation models (vision-language) | CLIP / MedCLIP | `models/clip/*_new.py` | CNN, DPT, +LoRA (separate scripts) |
| Foundation models (promptable) | SAM3 | `models/sam3/cnn_sam3_*.py`, `DPT_sam3_*.py` | CNN, DPT, +LoRA (`task_name`) |
| | SAM2 | `models/sam2/cnn_sam2_*.py`, `DPT_sam2_*.py` | CNN, DPT, +LoRA (`task_name`) |
| | SAM / MedSAM | `models/medsam/*_16.py` | CNN, DPT, +LoRA (`task_name`) |
| | SAM-Med2D | `models/medsam/*_16.py` with the SAM-Med2D checkpoint | CNN, DPT, +LoRA |
| Baselines | UNet | `models/baselines/unet/UNet.py` | full fine-tuning |
| | UNet++ / Attention UNet / TransUNet | `models/baselines/trainer/train.py --model_type` | full fine-tuning |
| | nnU-Net | `models/baselines/nnunet/train.py` | full fine-tuning |

## SAM3

```bash
cd models/sam3
mkdir -p logs

# CNN decoder, frozen encoder
python cnn_sam3_Fundus.py --data_root $FMDG_DATA_ROOT/Fundus \
    --trainid 1,3 --testid 2,4 --work_dir $FMDG_WORK_DIR \
    --task_name sam3_cnn_fu --num_epochs 50 --iternum 3000000 --device cuda:0

# CNN decoder + LoRA (the "lora" substring switches LoRA on)
python cnn_sam3_Fundus.py --data_root $FMDG_DATA_ROOT/Fundus \
    --trainid 1,3 --testid 2,4 --work_dir $FMDG_WORK_DIR \
    --task_name sam3_cnnlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0
```

`run.sh`, `run_base.sh` and `pred.sh` contain the full grid (3 datasets x 2 splits x
4 strategies) and the inference commands.

## SAM2

```bash
cd models/sam2
python cnn_sam2_NPC.py --data_root $FMDG_DATA_ROOT/NPC/preprocessed \
    --trainid 1,4 --testid 2,3 --work_dir $FMDG_WORK_DIR \
    --task_name sam2_cnn_na --num_epochs 50 --device cuda:0
```

## SAM / MedSAM (`*_16`)

```bash
cd models/medsam
bash run_16.sh          # all 12 runs; edit the paths at the top or export the variables
```

`run_16.sh` uses `$FMDG_DATA_ROOT` and `$FMDG_WORK_DIR`; the SAM-Med2D variants only
differ in the checkpoint passed via `--checkpoint`.

## CLIP / MedCLIP (`*_new`)

```bash
cd models/clip
python clip_cnn_Fundus_new.py --data_dir $FMDG_DATA_ROOT/Fundus --dataset fundus \
    --trainid 1,3 --testid 2,4 --epochs 50 --batch_size 4 --num_classes 3 \
    --task_name clip_cnn_fu
```

The `_new` scripts are the versions used for the paper. `medclip_*` variants use the
PubMedCLIP-style vision tower instead of OpenAI CLIP.

## DINO family

```bash
cd models/dino
python train_dinov1_D_Fundus.py --data_dir $FMDG_DATA_ROOT/Fundus \
    --dino_ckpt /path/to/dinov1_vitb16.pth --dino_size b \
    --trainid 1,3 --testid 2,4 --epochs 50
```

`dinov3_cnn_*` and `train_segdinov3_*` load DINOv3 through `torch.hub` from the
vendored `models/dino/dinov3` repository (`--repo_dir ./dinov3`).

## Baselines

```bash
cd models/baselines/trainer
python train.py --data_root $FMDG_DATA_ROOT/Fundus --trainid 1,3 --testid 2,4 \
    --model_type transunet --work_dir $FMDG_WORK_DIR --num_epochs 50

cd ../unet
python UNet.py --data_root $FMDG_DATA_ROOT/Fundus --trainid 1,3 --testid 2,4 \
    --work_dir $FMDG_WORK_DIR --num_epochs 50

cd ../nnunet
python train.py --help        # see run_all.sh for the full grid
```

## Inference and metrics

- SAM3: `predsam3.py`; SAM2: `predsam2.py`; MedSAM/SAM: inference utilities are in
  the corresponding `_16` scripts; CLIP/DINO: `test.py` / `test_segdino.py`.
- Metrics: Dice and ASSD (Average Symmetric Surface Distance). ASSD is computed with
  `medpy.metric.binary.assd`; the UNet baseline also ships `utils/metrics.py`.
- Report per-domain Dice/ASSD, then average over the two splits to obtain
  `Mean_in`, `Mean_out` and `gap = Mean_in - Mean_out`.
