# Repository notes

This repository was assembled from the two lab servers used for the experiments.
This page documents what was kept, what was removed, and the known limitations, so
that future contributors do not have to reverse-engineer the decisions.

## Provenance

| Folder | Source |
|---|---|
| `models/sam3/` | the SAM3 working copy (full upstream repository + experiment scripts) |
| `models/sam2/` | the SAM2 working copy (upstream package + adapted training scripts) |
| `models/medsam/` | the SAM / MedSAM pipeline, files ending in `_16` |
| `models/clip/` | the CLIP / MedCLIP pipeline, files containing `_new` |
| `models/dino/` | the SegDINO pipeline with vendored DINO, DINOv2 and DINOv3 |
| `models/baselines/trainer/` | the baseline trainer (UNet++, Attention UNet, TransUNet, nnU-Net wrapper) |
| `models/baselines/nnunet/` | the nnU-Net training pipeline |
| `models/baselines/unet/` | the UNet baseline (`UNet.py`, MONAI UNet, full fine-tuning) |

## Authoritative copies (deduplication rules)

Several models existed in more than one place. Only one implementation of each was
kept, as agreed:

- **CLIP / MedCLIP** — only the `*_new.py` training scripts are shipped; the older
  duplicates without `_new` were dropped.
- **TransUNet** — the implementation under `models/baselines/` is authoritative.
  The copies inside the SAM3 working directory were removed.
- **UNet** — the implementation from `wjx_sam` is authoritative
  (`models/baselines/unet/UNet.py`).
- **SAM / MedSAM** — only the `*_16.py` scripts are shipped; the earlier `_another`,
  `_16`-less and `save_files/` variants were dropped.
- **SAM2 / SAM3 / DINO** — the adapted training scripts are shipped together with the
  upstream model code they import, because the experiments relied on local
  modifications and on `torch.hub` loading from the vendored repository.

## Removed from the servers

| Removed | Reason |
|---|---|
| `*.mp4`, `*.avi`, `*.gif` demo videos | sample media of the upstream repos, not needed to train |
| `*.ipynb` notebooks | demos only; the benchmark uses the `.py` scripts |
| `runs/`, `work_dir/`, `logs/`, `logss/`, `logsss/`, `logsu/`, `save*/` | training outputs, duplicated scripts, TensorBoard events |
| `*.pth`, `*.pt`, `*.safetensors`, `sam3.pt` (3.3 GB) | checkpoints; obtain from official releases |
| `Datasets/` raw data, `dataset_cache/`, `pretrain_model/` | datasets and caches; not redistributable |
| `models/*/assets/`, demo frontends, `sav_dataset/` | upstream demo assets |
| `.git/` of the SAM3 working copy | server-side history is not part of this repository |
| absolute paths (`/media/ubuntu/...`, `/data1/...`) | replaced with `$FMDG_DATA_ROOT`, `$FMDG_WORK_DIR` or `./data` |

## Deliberate duplication

Shared modules (`segment_anything/`, `seg_head/`, `Datasets/`, `naso_dataset*.py`)
are copied into several model folders. This keeps each family runnable with
`cd models/<family> && python ...` and avoids a fragile global `PYTHONPATH`.
The duplicated files are small (a few hundred KB in total).

## Known limitations / TODO

- The checkpoint download script of SAM2 (`checkpoints/download_ckpts.sh`) was not
  copied; download the ViT-B weights manually and pass `--checkpoint`.
- `models/sam3/*` and `models/baselines/trainer/train.py` import `wandb`
  unconditionally; install it (see `envs/README.md`) or patch the import if you do
  not want experiment tracking.
- The run scripts (`run.sh`, `run_16.sh`, `run_all.sh`) were written for the lab
  servers. Paths are parameterised through `$FMDG_DATA_ROOT` / `$FMDG_WORK_DIR`, but
  review the flags before launching a full grid.
- The `run*.sh` scripts write to `logs/` and `logsss/`; create those directories
  (`mkdir -p logs logsss`) before running.
- CLIP / MedCLIP and nnU-Net were validated in separate environments; see
  `envs/README.md`. A single unified environment is not claimed.
- No results tables are stored here; they belong to the paper. Predictions and
  checkpoints are produced under `$FMDG_WORK_DIR`.
