# Third-party code and licenses

This repository contains code adapted from, or vendored copies of, the following
projects. Each retains its original license file inside the corresponding folder.
If you use this benchmark, please also follow the upstream licenses.

| Directory | Upstream project | License |
|---|---|---|
| `models/sam3/` | SAM 3 (Meta) | SAM License — see `models/sam3/LICENSE` |
| `models/sam2/` | SAM 2 (Meta) | Apache License 2.0 — see `models/sam2/LICENSE` |
| `models/medsam/segment_anything/` | Segment Anything (Meta) / SAM-Med2D | Apache License 2.0 (upstream) |
| `models/dino/dino/` | DINO (Meta) | Apache License 2.0 |
| `models/dino/dinov2/` | DINOv2 (Meta) | Apache License 2.0; some model weights are covered by additional research licenses (`LICENSE_CELL_DINO_MODELS`, `LICENSE_XRAY_DINO_MODEL`) |
| `models/dino/dinov3/` | DINOv3 (Meta) | DINOv3 License — see `models/dino/dinov3/LICENSE.md` |
| `models/baselines/nnunet/` | nnU-Net | Apache License 2.0 |
| `models/baselines/trainer/models/transunet.py` | TransUNet | see the header of the file / upstream repository |
| `models/clip/` | CLIP (OpenAI) and MedCLIP | CLIP is MIT; MedCLIP follows its upstream license. The `clip` package is **not** vendored — install it separately. |

Notes:

- The **datasets** (Fundus, NPC, PolypDB) are not distributed here and are governed
  by their own licenses and data-use agreements. Obtain them from the original
  sources listed in [docs/data.md](docs/data.md).
- Pretrained **checkpoints** are not distributed here either; download them from the
  official release pages of each foundation model.
- Files copied from the upstream repositories keep their original copyright headers.
