#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_env.py — check whether the current Python environment can run the model
families of FMDG-Bench.

It also puts the self-contained family directories on sys.path, so the vendored
modules (`segment_anything`, `sam3`, `sam2`, `seg_head`) are found when the script
is executed from the repository root.

Usage:
    python tools/check_env.py            # report only
    python tools/check_env.py --strict   # non-zero exit if a required group is missing
"""
import argparse
import importlib
import importlib.metadata as md
import platform
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
for sub in ("models/sam3", "models/sam2", "models/medsam",
            "models/baselines/unet", "models/baselines/trainer"):
    p = ROOT / sub
    if p.is_dir():
        sys.path.insert(0, str(p))

# (group, any-of-ok, optional, [(import name, distribution candidates)])
GROUPS = [
    ("core",          False, False, [("torch", "torch"), ("torchvision", "torchvision"),
                                     ("numpy", "numpy"), ("scipy", "scipy"),
                                     ("PIL", "Pillow"), ("yaml", "PyYAML"), ("tqdm", "tqdm")]),
    ("SAM3",          False, False, [("sam3", "sam3")]),
    ("SAM2",          False, False, [("sam2", "SAM-2")]),
    ("SAM / MedSAM",  False, False, [("segment_anything", "segment-anything"),
                                     ("seg_head", None)]),
    ("CLIP / MedCLIP", True, False, [("clip", "clip"), ("transformers", "transformers")]),
    ("LoRA",          True,  False, [("loralib", "loralib"), ("peft", "peft")]),
    ("DINO family",   False, False, [("timm", "timm")]),
    ("Baselines",     True,  False, [("monai", "monai"), ("nnunetv2", "nnunetv2"),
                                     ("batchgenerators", "batchgenerators")]),
    ("Logging",       True,  True,  [("wandb", "wandb"), ("tensorboard", "tensorboard")]),
]


def probe(import_name, dist_name):
    try:
        importlib.import_module(import_name)
    except Exception as exc:
        return False, f"import failed: {type(exc).__name__}"
    for name in (dist_name, import_name):
        if not name:
            continue
        try:
            return True, md.version(name)
        except Exception:
            continue
    return True, "installed (version unknown)"


def torch_report():
    try:
        import torch
    except Exception as exc:
        return [f"torch: not importable ({exc})"]
    lines = [f"torch {torch.__version__} | CUDA build {torch.version.cuda} | "
             f"cudnn {torch.backends.cudnn.version()}"]
    if not torch.cuda.is_available():
        lines.append("CUDA not available (CPU only)")
        return lines
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        bf16 = torch.cuda.is_bf16_supported()
        lines.append(f"gpu{i}: {torch.cuda.get_device_name(i)} sm_{cap[0]}{cap[1]} "
                     f"bf16={'yes' if bf16 else 'no'}")
        if cap[0] < 8:
            lines.append("  note: this GPU has no bfloat16 support, use fp16/fp32")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit with non-zero status if a required group is missing")
    args = ap.parse_args()

    print("=" * 74)
    print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()}")
    print(f"repo   {ROOT}")
    print("=" * 74)
    print("-- hardware / pytorch --")
    for line in torch_report():
        print("  " + line)

    missing = []
    print("\n-- groups --")
    width = max(len(g[0]) for g in GROUPS)
    for name, any_ok, optional, mods in GROUPS:
        results = [(m, d) + probe(m, d) for m, d in mods]
        ok_n = sum(1 for r in results if r[2])
        group_ok = (ok_n > 0) if any_ok else (ok_n == len(results))
        if group_ok and any_ok:
            status = "OK*"
        elif group_ok:
            status = "OK"
        elif optional:
            status = "optional"
        else:
            status = "MISSING"
            missing.append(name)
        print(f"[{status:8s}] {name:<{width}}")
        for mod, dist, ok, ver in results:
            print(f"             {'ok' if ok else '--'}  {mod:18s} {ver}")

    print()
    if missing:
        print("missing groups:", ", ".join(missing))
        print("-> this environment does not cover every model family; see envs/README.md")
    else:
        print("all groups satisfied: the environment covers every model family.")
        print("note: import success does not prove a training step works.")
    return 1 if (args.strict and missing) else 0


if __name__ == "__main__":
    sys.exit(main())
