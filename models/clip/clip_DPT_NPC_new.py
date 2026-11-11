# train_clip_final.py
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import clip
import argparse
import random

# 添加LoRA相关导入
try:
    import loralib as lora
except ImportError:
    print("请安装loralib：pip install loralib")
    sys.exit(1)


# 添加LoRA替换函数（与第一个程序一致）
def replace_linear_with_lora(module, lora_r=4, lora_alpha=4):
    """递归替换模型中的线性层为LoRA线性层"""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            lora_module = lora.Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                r=lora_r,
                lora_alpha=lora_alpha,
                bias=child.bias is not None
            )
            lora_module.weight = child.weight
            if child.bias is not None:
                lora_module.bias = child.bias
            setattr(module, name, lora_module)
        else:
            replace_linear_with_lora(child, lora_r=lora_r, lora_alpha=lora_alpha)


# 原有的辅助函数保持不变
def tensor_to_rgb(img_t: torch.Tensor, mean=None, std=None) -> np.ndarray:
    img = img_t.detach().cpu().float()
    img = img.clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    m = mask_t.detach().cpu().float()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected mask tensor shape: {m.shape}")
    if m.max() > 1.0 or m.min() < 0.0:
        m = torch.sigmoid(m)
    m_bin = (m > thr).float()
    m_img = (m_bin * 255.0).round().clamp(0, 255).byte().numpy()
    return m_img


def save_train_visuals(epoch, inputs, logits, targets, out_dir, max_save=8, thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
    b = min(inputs.size(0), max_save)
    for i in range(b):
        img_bgr = tensor_to_rgb(inputs[i])
        pred_gray = mask_to_gray(logits[i], thr)
        gt_gray = mask_to_gray(targets[i], thr)
        base = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}")
        cv2.imwrite(base + "_img.png", img_bgr)
        cv2.imwrite(base + "_pred.png", pred_gray)
        cv2.imwrite(base + "_gt.png", gt_gray)


@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, thr=0.5, fname_prefix="val"):
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = tensor_to_rgb(inputs)
    pred_gray = mask_to_gray(logits, thr)
    gt_gray = mask_to_gray(targets, thr)
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png", img_bgr)
    cv2.imwrite(base + "_pred.png", pred_gray)
    cv2.imwrite(base + "_gt.png", gt_gray)


def iou_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    iou = (inter + eps) / union
    return iou.view(-1)


def dice_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / union
    return dice

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, eps=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        """
        logits: [B, 1, H, W]
        target: [B, 1, H, W]
        """
        target = target.float().clamp(0, 1)

        bce_loss = self.bce(logits, target)

        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_score = (2.0 * inter + self.eps) / (union + self.eps)
        dice_loss = 1.0 - dice_score.mean()

        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total_loss, dice_loss, bce_loss

def train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes=1, dice_thr=0.5, vis_dir=None, epoch=0):
    model.train()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    dice_scores = []
    iou_scores = []
    first_batch_logged = False
    pbar = train_loader

    for step, (inputs, targets, _) in enumerate(pbar):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()
        logits = model(inputs)

        if num_classes == 1:
            total_batch_loss, dice_loss, bce_loss = criterion(logits, targets)
        else:
            raise NotImplementedError("当前这个NPC脚本是二分类版，组合损失按二分类实现。")

        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_dice_loss += dice_loss.item()
        total_bce_loss += bce_loss.item()

        with torch.no_grad():
            dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            iou = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            dice_scores.append(dice)
            iou_scores.append(iou)


        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals(epoch, inputs, logits, targets, out_dir=vis_dir, max_save=8, thr=dice_thr)
            first_batch_logged = True

    avg_loss = total_loss / max(1, len(train_loader))
    avg_dice_loss = total_dice_loss / max(1, len(train_loader))
    avg_bce_loss = total_bce_loss / max(1, len(train_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0

    print(
        f"[Train Epoch {epoch}] "
        f"loss={avg_loss:.4f}  bce={avg_bce_loss:.4f}  diceloss={avg_dice_loss:.4f}  "
        f"dice={avg_dice:.4f}  iou={avg_iou:.4f}"
    )
    return avg_loss, avg_dice_loss, avg_bce_loss, avg_dice, avg_iou


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, num_classes=1, dice_thr=0.5, vis_dir=None, split_name="Val"):
    model.eval()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    dice_scores = []
    iou_scores = []
    idx_global = 0
    pbar = data_loader

    for inputs, targets, _ in pbar:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        logits = model(inputs)

        if num_classes == 1:
            total_batch_loss, dice_loss, bce_loss = criterion(logits, targets)
        else:
            raise NotImplementedError("当前这个NPC脚本是二分类版，组合损失按二分类实现。")

        total_loss += total_batch_loss.item()
        total_dice_loss += dice_loss.item()
        total_bce_loss += bce_loss.item()

        dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        iou = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        dice_scores.append(dice)
        iou_scores.append(iou)


        if vis_dir is not None:
            os.makedirs(vis_dir, exist_ok=True)
            B = inputs.size(0)
            for b in range(B):
                save_eval_visuals(
                    idx_global,
                    inputs[b],
                    logits[b],
                    targets[b],
                    out_dir=vis_dir,
                    thr=dice_thr,
                    fname_prefix=split_name.lower()
                )
                idx_global += 1

    avg_loss = total_loss / max(1, len(data_loader))
    avg_dice_loss = total_dice_loss / max(1, len(data_loader))
    avg_bce_loss = total_bce_loss / max(1, len(data_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0

    print(
        f"[{split_name}] "
        f"loss={avg_loss:.4f}  bce={avg_bce_loss:.4f}  diceloss={avg_dice_loss:.4f}  "
        f"dice={avg_dice:.4f}  iou={avg_iou:.4f}"
    )
    return avg_loss, avg_dice_loss, avg_bce_loss, avg_dice, avg_iou


def build_npc_datasets(args, train_transform, val_transform, test_transform):
    from dataset import FolderDataset_NPC

    data_root = args.data_dir
    train_domains = [int(domain_id) for domain_id in args.trainid.split(',')]
    test_domains = [int(domain_id) for domain_id in args.testid.split(',')]

    print(f"✅ Training domains: {train_domains}")
    print(f"✅ Testing domains: {test_domains}")

    train_datasets = []
    val_datasets = []
    test_domain_datasets = {}

    # train: trainid_train
    for domain_id in train_domains:
        path = os.path.join(data_root, f"Domain{domain_id}_train/")
        if os.path.exists(path):
            dataset = FolderDataset_NPC(
                root=path,
                split="train",
                img_dir_name=args.img_dir_name,
                label_dir_name=args.label_dir_name,
                mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
                transform=train_transform,
                use_pkl=True,
                pkl_dir=path
            )
            train_datasets.append(dataset)
            print(f"✅ Loaded train dataset from Domain{domain_id}_train: {len(dataset)} samples")
        else:
            print(f"⚠️ Warning: train path not found: {path}")

    # val: trainid_test
    for domain_id in train_domains:
        path = os.path.join(data_root, f"Domain{domain_id}_test/")
        if os.path.exists(path):
            dataset = FolderDataset_NPC(
                root=path,
                split="test",
                img_dir_name=args.img_dir_name,
                label_dir_name=args.label_dir_name,
                mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
                transform=val_transform,
                use_pkl=True,
                pkl_dir=path
            )
            val_datasets.append(dataset)
            print(f"✅ Loaded val dataset from Domain{domain_id}_test: {len(dataset)} samples")
        else:
            print(f"⚠️ Warning: val path not found: {path}")

    # test: 每个 testid 单独构建
    for domain_id in test_domains:
        domain_subsets = []

        for split_name, split_arg in [("train", "train"), ("test", "test")]:
            path = os.path.join(data_root, f"Domain{domain_id}_{split_name}/")
            if os.path.exists(path):
                dataset = FolderDataset_NPC(
                    root=path,
                    split=split_arg,
                    img_dir_name=args.img_dir_name,
                    label_dir_name=args.label_dir_name,
                    mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
                    transform=test_transform,
                    use_pkl=True,
                    pkl_dir=path
                )
                domain_subsets.append(dataset)
                print(f"✅ Loaded test dataset from Domain{domain_id}_{split_name}: {len(dataset)} samples")
            else:
                print(f"⚠️ Warning: test path not found: {path}")

        if len(domain_subsets) == 0:
            print(f"⚠️ Warning: Domain{domain_id} has no test data!")
        elif len(domain_subsets) == 1:
            test_domain_datasets[domain_id] = domain_subsets[0]
        else:
            test_domain_datasets[domain_id] = torch.utils.data.ConcatDataset(domain_subsets)

    if not train_datasets:
        raise ValueError("❌ No training datasets loaded successfully!")
    if not val_datasets:
        raise ValueError("❌ No validation datasets loaded successfully!")
    if not test_domain_datasets:
        raise ValueError("❌ No test datasets loaded successfully!")

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    return train_dataset, val_dataset, test_domain_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="tn3k")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                       choices=["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
                               "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"])
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--trainid", type=str, default="1,2,3")
    parser.add_argument("--testid", type=str, default="4")
    parser.add_argument("--task_name", type=str, default="NPC_D")
    parser.add_argument("--features", type=int, default=128, help="DPT features dimension")
    parser.add_argument("--out_channels", type=str, default="96,192,384,768",
                       help="Comma-separated list of output channels for DPT head")
    parser.add_argument("--use_bn", action="store_true", help="Use batch normalization in DPT head")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=4, help="LoRA的alpha值")

    args = parser.parse_args()
    use_lora = "lora" in args.task_name.lower()

    if use_lora:
        print(f"✅ 使用LoRA，秩(r)={args.lora_r}, alpha={args.lora_alpha}")

    out_channels = [int(x) for x in args.out_channels.split(',')]

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, will use CPU instead!")
        device = "cpu"
    else:
        cuda_count = torch.cuda.device_count()
        if args.cuda_id >= cuda_count or args.cuda_id < 0:
            raise ValueError(f"Invalid CUDA ID {args.cuda_id}! Only {cuda_count} GPU(s) available")
        torch.cuda.set_device(args.cuda_id)
        device = f"cuda:{args.cuda_id}"
        print(f"Using CUDA device ID: {args.cuda_id} (device: {device})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_root = f"./runs/dpt_clip_{args.clip_model.replace('/', '_')}_{args.input_h}_{args.task_name}_train{args.trainid}_test{args.testid}"
    os.makedirs(save_root, exist_ok=True)
    train_vis_dir = os.path.join(save_root, "train_vis")
    val_vis_dir = os.path.join(save_root, "val_vis")
    test_vis_dir = os.path.join(save_root, "test_vis")
    ckpt_dir = os.path.join(save_root, "ckpts")
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(val_vis_dir, exist_ok=True)
    os.makedirs(test_vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    latest_model_path = os.path.join(ckpt_dir, "latest.pth")

    print(f"Loading CLIP model: {args.clip_model}")
    clip_model_full, preprocess = clip.load(args.clip_model, device=device)

    clip_model_full = clip_model_full.float()
    backbone = clip_model_full.visual
    backbone.eval()

    if use_lora:
        for param in backbone.parameters():
            param.requires_grad = False
        replace_linear_with_lora(backbone, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        print("✅ LoRA已应用到CLIP backbone")

    print(f"✅ CLIP {args.clip_model} loaded successfully!")

    from dpt_clip import DPT_CLIP_Full
    model = DPT_CLIP_Full(
        encoder_size='b',
        nclass=args.num_classes,
        features=args.features,
        out_channels=out_channels,
        use_bn=args.use_bn,
        clip_model=backbone,
        model_name=args.clip_model,
        freeze_backbone=not use_lora
    )
    model = model.to(device)

    if use_lora:
        trainable_params = []
        trainable_params.extend(model.head.parameters())
        for _, param in backbone.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params_count:,}")
        print(f"✅ Trainable ratio: {trainable_params_count / total_params:.2%}")
    else:
        optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    criterion = DiceBCELoss(
        dice_weight=0.5,
        bce_weight=0.5,
    )

    from dataset import ResizeAndNormalize_NPC

    train_transform = ResizeAndNormalize_NPC(size=(args.input_h, args.input_w))
    val_transform = ResizeAndNormalize_NPC(size=(args.input_h, args.input_w))
    test_transform = ResizeAndNormalize_NPC(size=(args.input_h, args.input_w))

    train_dataset, val_dataset, test_domain_datasets = build_npc_datasets(
        args=args,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
    )

    print("\n📊 Dataset Summary:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    total_test_samples = sum(len(ds) for ds in test_domain_datasets.values())
    print(f"   Total test samples: {total_test_samples}")
    for domain_id, ds in test_domain_datasets.items():
        print(f"   Domain{domain_id} test samples: {len(ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    test_loaders = {}
    for domain_id, test_dataset in test_domain_datasets.items():
        test_loaders[domain_id] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )


    best_val_dice = -1.0
    best_val_dice_epoch = -1
    best_val_iou = -1.0
    best_val_iou_epoch = -1

    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice_loss, train_bce_loss, train_dice, train_iou = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=args.num_classes,
            dice_thr=0.5,
            vis_dir=train_vis_dir,
            epoch=epoch
        )

        val_loss, val_dice_loss, val_bce_loss, val_dice, val_iou = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=args.num_classes,
            dice_thr=0.5,
            vis_dir=val_vis_dir,
            split_name="Val"
        )

        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
                "best_val_dice": best_val_dice,
                "best_val_iou": best_val_iou,
                "train_loss": train_loss,
                "train_dice_loss": train_dice_loss,
                "train_bce_loss": train_bce_loss,
                "val_loss": val_loss,
                "val_dice_loss": val_dice_loss,
                "val_bce_loss": val_bce_loss,
            },
            latest_model_path
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_dice_epoch = epoch
            best_val_iou = val_iou
            best_val_iou_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": args,
                    "best_val_dice": best_val_dice,
                    "best_val_iou": best_val_iou,
                    "train_loss": train_loss,
                    "train_dice_loss": train_dice_loss,
                    "train_bce_loss": train_bce_loss,
                    "val_loss": val_loss,
                    "val_dice_loss": val_dice_loss,
                    "val_bce_loss": val_bce_loss,
                },
                best_model_path
            )
            print(f"[Save] New best ckpt: {best_model_path}")
        elif val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_iou_epoch = epoch

    print("=" * 60)
    print(f"[Summary] Best Val Dice = {best_val_dice:.4f} @ epoch {best_val_dice_epoch}")
    print(f"[Summary] Best Val IoU = {best_val_iou:.4f} @ epoch {best_val_iou_epoch}")
    print("=" * 60)

    print("\n🧪 Loading best model for final testing...")
    best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    print("=" * 60)
    print("✅ Final Test Results (Best Model, Per Domain):")
    print("=" * 60)

    all_domain_results = {}

    for domain_id, test_loader in test_loaders.items():
        domain_vis_dir = os.path.join(test_vis_dir, f"Domain{domain_id}")
        os.makedirs(domain_vis_dir, exist_ok=True)

        print(f"\n📌 Testing on Domain{domain_id} ...")

        test_loss, test_dice_loss, test_bce_loss, test_dice, test_iou = evaluate(
            model,
            test_loader,
            criterion,
            device,
            num_classes=args.num_classes,
            dice_thr=0.5,
            vis_dir=domain_vis_dir,
            split_name=f"Test_Domain{domain_id}"
        )

        all_domain_results[domain_id] = {
            "loss": test_loss,
            "dice_loss": test_dice_loss,
            "bce_loss": test_bce_loss,
            "dice": test_dice,
            "iou": test_iou,
        }

        print("-" * 60)
        print(f"📍 Domain{domain_id} Results:")
        print(f"[Final Test][Domain{domain_id}] Loss     = {test_loss:.4f}")
        print(f"[Final Test][Domain{domain_id}] BCE      = {test_bce_loss:.4f}")
        print(f"[Final Test][Domain{domain_id}] DiceLoss = {test_dice_loss:.4f}")
        print(f"[Final Test][Domain{domain_id}] Dice     = {test_dice:.4f}")
        print(f"[Final Test][Domain{domain_id}] IoU      = {test_iou:.4f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("✅ Summary of Per-Domain Test Results:")
    print("=" * 60)
    for domain_id, result in all_domain_results.items():
        print(
            f"Domain{domain_id} | "
            f"Loss={result['loss']:.4f}  "
            f"BCE={result['bce_loss']:.4f}  "
            f"DiceLoss={result['dice_loss']:.4f}  "
            f"Dice={result['dice']:.4f}  "
            f"IoU={result['iou']:.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
