# Data preparation

The datasets are **not** included in this repository. They are subject to their own
licenses and data-use agreements; please obtain them from the original sources and
follow their terms (in particular for the NPC MRI data, which was collected under
an institutional collaboration and must not be redistributed).

## Sources

| Dataset | Task | Domains | Where to obtain |
|---|---|---|---|
| Fundus | Optic cup (OC) and optic disc (OD) segmentation | D1 Drishti-GS, D2 RIM-ONE-r3, D3 REFUGE train, D4 REFUGE validation | Drishti-GS, RIM-ONE, REFUGE challenge websites |
| NPC | Nasopharyngeal carcinoma GTV segmentation (2D slices) | D1 T1w, D2 Gd-T1w, D3 T1-water, D4 T2-water | institutional collaboration (contact the authors) |
| PolypDB | Colonoscopy polyp segmentation | D1 BLI, D2 FICE, D3 LCI, D4 NBI | PolypDB release |

## Expected layout

```bash
export FMDG_DATA_ROOT=/path/to/FMDG-data
```

```
$FMDG_DATA_ROOT/
├── Fundus/
│   ├── Domain1/
│   │   ├── train/          # images + masks, 80% of the domain
│   │   └── test/           # images + masks, 20% of the domain
│   ├── Domain2/
│   ├── Domain3/
│   └── Domain4/
├── NPC/
│   └── preprocessed/       # 2D slices and masks for D1-D4
└── PolypDB/
    └── PolypDB_modality_wise/
        ├── BLI/
        ├── FICE/
        ├── LCI/
        └── NBI/
```

The exact fields consumed by each pipeline are defined in the corresponding
dataloaders:

- `models/sam3/Datasets/new_dataloader.py`, `models/sam3/naso_dataset_16.py`
- `models/sam2/Datasets/new_dataloader.py`, `models/sam2/naso_dataset.py`
- `models/medsam/Datasets/new_dataloader.py`, `models/medsam/naso_dataset_16.py`
- `models/clip/dataset.py` (folder datasets for Fundus / NPC / PolypDB)
- `models/dino/dataset.py`
- `models/baselines/*/Datasets/`

## Splits and domain groups

Domain identifiers are passed as comma-separated lists:

```bash
# train on D1 + D3, test on D2 + D4 (out-of-domain) and on D1 + D3 (in-domain)
--trainid 1,3 --testid 2,4
```

The paper uses two complementary splits per dataset:

| Dataset | Split A | Split B |
|---|---|---|
| Fundus | train 1,3 / test 2,4 | train 2,4 / test 1,3 |
| NPC | train 1,4 / test 2,3 | train 2,3 / test 1,4 |
| PolypDB | train 1,2 / test 3,4 | train 3,4 / test 1,2 |

Each domain is split 80/20 into train/test before the domain groups are formed, so
the "in-domain" test images are the held-out 20% of the training domains and the
"out-of-domain" images come from the two unseen domains.

## Baseline input format

The baseline trainer (`models/baselines/trainer`, `models/baselines/unet`) reads
pre-extracted ROI arrays:

```bash
python train.py -i $FMDG_DATA_ROOT/Fundus/Domain1/train/ROIs/ \
                -j $FMDG_DATA_ROOT/Fundus/Domain1/test/ROIs/ \
                --data_root $FMDG_DATA_ROOT/Fundus --trainid 1,3 --testid 2,4
```

If your preprocessing produces a different layout, adapt the paths or write a small
conversion script; the loaders only require image/mask pairs plus a dataset list.

## Sanity check before training

1. `ls $FMDG_DATA_ROOT/Fundus/Domain1/train` returns both images and masks.
2. The number of image/mask pairs matches for every domain.
3. Mask values are `{0, 1}` (binary) or `{0, 1, 2}` for the 3-class Fundus setup
   (background / OC / OD) — check `--num_classes` accordingly.
