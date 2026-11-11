# train_segdinov3_multi_class_train_val_test.py
import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import argparse
import torch.utils.data as data

sys.path.append('./dinov3')

# 添加LoRA相关导入
try:
    import loralib as lora
except ImportError:
    print("请安装loralib：pip install loralib")
    sys.exit(1)


def replace_linear_with_lora(module, lora_r=4, lora_alpha=4):
    """递归替换模型中的线性层为LoRA线性层"""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            lora_module = lora.Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                r=lora_r,
                lora_alpha=lora_alpha,
                bias=child.bias is not None,
            )
            lora_module.weight = child.weight
            if child.bias is not None:
                lora_module.bias = child.bias
            setattr(module, name, lora_module)
        else:
            replace_linear_with_lora(child, lora_r=lora_r, lora_alpha=lora_alpha)


def tensor_to_rgb(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).round().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def mask_to_color(mask_t: torch.Tensor, num_classes=3) -> np.ndarray:
    if mask_t.ndim == 4:
        m = mask_t[0]
    else:
        m = mask_t

    if m.shape[0] > 1:
        pred = torch.argmax(m, dim=0)
    else:
        pred = m.squeeze(0).long()

    pred_np = pred.cpu().numpy()
    h, w = pred_np.shape

    color_map = {
        0: [0, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255],
    }

    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        if class_id in color_map:
            color_mask[pred_np == class_id] = color_map[class_id]
    return color_mask


def save_train_visuals(epoch, inputs, logits, targets, out_dir, max_save=4, num_classes=3):
    os.makedirs(out_dir, exist_ok=True)
    b = min(inputs.size(0), max_save)
    for i in range(b):
        img_bgr = tensor_to_rgb(inputs[i])
        pred_color = mask_to_color(logits[i], num_classes)
        gt_color = mask_to_color(targets[i], num_classes)
        base = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}")
        cv2.imwrite(base + "_img.png", img_bgr)
        cv2.imwrite(base + "_pred.png", pred_color)
        cv2.imwrite(base + "_gt.png", gt_color)
        overlay_pred = cv2.addWeighted(img_bgr, 0.7, pred_color, 0.3, 0)
        overlay_gt = cv2.addWeighted(img_bgr, 0.7, gt_color, 0.3, 0)
        cv2.imwrite(base + "_overlay_pred.png", overlay_pred)
        cv2.imwrite(base + "_overlay_gt.png", overlay_gt)


@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, num_classes=3, fname_prefix="val"):
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = tensor_to_rgb(inputs)
    pred_color = mask_to_color(logits, num_classes)
    gt_color = mask_to_color(targets, num_classes)
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png", img_bgr)
    cv2.imwrite(base + "_pred.png", pred_color)
    cv2.imwrite(base + "_gt.png", gt_color)
    overlay_pred = cv2.addWeighted(img_bgr, 0.7, pred_color, 0.3, 0)
    overlay_gt = cv2.addWeighted(img_bgr, 0.7, gt_color, 0.3, 0)
    cv2.imwrite(base + "_overlay_pred.png", overlay_pred)
    cv2.imwrite(base + "_overlay_gt.png", overlay_gt)


def multi_class_dice(pred, target, num_classes=3):
    probs = F.softmax(pred, dim=1)
    dice_scores = []
    for c in range(1, num_classes):
        pred_class = probs[:, c, :, :]
        target_class = target[:, c, :, :]
        intersection = (pred_class * target_class).sum(dim=[1, 2])
        union = pred_class.sum(dim=[1, 2]) + target_class.sum(dim=[1, 2])
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice)
    return torch.stack(dice_scores)


def multi_class_iou(pred, target, num_classes=3):
    probs = F.softmax(pred, dim=1)
    pred_onehot = torch.argmax(probs, dim=1, keepdim=True)
    pred_onehot_full = torch.zeros_like(probs)
    pred_onehot_full.scatter_(1, pred_onehot, 1)

    iou_scores = []
    for c in range(1, num_classes):
        pred_class = pred_onehot_full[:, c, :, :]
        target_class = target[:, c, :, :]
        intersection = (pred_class * target_class).sum(dim=[1, 2])
        union = pred_class.sum(dim=[1, 2]) + target_class.sum(dim=[1, 2]) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_scores.append(iou)
    return torch.stack(iou_scores)

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, num_classes=3, ce_class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes

        if ce_class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        pred:   [B, C, H, W] logits
        target: [B, C, H, W] one-hot
        """
        target_indices = torch.argmax(target, dim=1).long()
        ce_loss = self.ce_loss(pred, target_indices)

        probs = F.softmax(pred, dim=1)
        dice_scores = []
        for c in range(1, self.num_classes):
            pred_class = probs[:, c, :, :]
            target_class = target[:, c, :, :]
            intersection = (pred_class * target_class).sum(dim=(1, 2))
            union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            dice_scores.append(dice)

        dice_scores = torch.stack(dice_scores)
        dice_loss = 1.0 - dice_scores.mean()

        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return total_loss, dice_loss, ce_loss

def train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes=3, vis_dir=None, epoch=0):
    model.train()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_ce_loss = 0.0
    dice_scores_disc = []
    dice_scores_cup = []
    iou_scores_disc = []
    iou_scores_cup = []
    first_batch_logged = False

    for step, (inputs, targets, _) in enumerate(train_loader):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()
        logits = model(inputs)

        total_batch_loss, dice_loss, ce_loss = criterion(logits, targets)
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_dice_loss += dice_loss.item()
        total_ce_loss += ce_loss.item()

        with torch.no_grad():
            dice = multi_class_dice(logits, targets, num_classes=num_classes)
            dice_disc = dice[0, :].mean().item()
            dice_cup = dice[1, :].mean().item()

            iou = multi_class_iou(logits, targets, num_classes=num_classes)
            iou_disc = iou[0, :].mean().item()
            iou_cup = iou[1, :].mean().item()

            dice_scores_disc.append(dice_disc)
            dice_scores_cup.append(dice_cup)
            iou_scores_disc.append(iou_disc)
            iou_scores_cup.append(iou_cup)

        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals(
                epoch,
                inputs,
                logits,
                targets,
                out_dir=vis_dir,
                max_save=4,
                num_classes=num_classes
            )
            first_batch_logged = True

    avg_loss = total_loss / max(1, len(train_loader))
    avg_dice_loss = total_dice_loss / max(1, len(train_loader))
    avg_ce_loss = total_ce_loss / max(1, len(train_loader))
    avg_dice_disc = float(np.mean(dice_scores_disc)) if dice_scores_disc else 0.0
    avg_dice_cup = float(np.mean(dice_scores_cup)) if dice_scores_cup else 0.0
    avg_iou_disc = float(np.mean(iou_scores_disc)) if iou_scores_disc else 0.0
    avg_iou_cup = float(np.mean(iou_scores_cup)) if iou_scores_cup else 0.0

    print(
        f"[Train Epoch {epoch}] total={avg_loss:.4f}  dice_loss={avg_dice_loss:.4f}  ce_loss={avg_ce_loss:.4f} | "
        f"Disc Dice={avg_dice_disc:.4f} | Cup Dice={avg_dice_cup:.4f} | "
        f"Disc IoU={avg_iou_disc:.4f} | Cup IoU={avg_iou_cup:.4f}"
    )

    return avg_loss, avg_dice_disc, avg_dice_cup, avg_iou_disc, avg_iou_cup


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, num_classes=3, epoch=None, split_name="Val", vis_dir=None):
    model.eval()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_ce_loss = 0.0
    dice_scores_disc = []
    dice_scores_cup = []
    iou_scores_disc = []
    iou_scores_cup = []
    idx_global = 0

    for (inputs, targets, _) in data_loader:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        logits = model(inputs)
        total_batch_loss, dice_loss, ce_loss = criterion(logits, targets)

        total_loss += total_batch_loss.item()
        total_dice_loss += dice_loss.item()
        total_ce_loss += ce_loss.item()

        dice = multi_class_dice(logits, targets, num_classes=num_classes)
        dice_disc = dice[0, :].mean().item()
        dice_cup = dice[1, :].mean().item()

        iou = multi_class_iou(logits, targets, num_classes=num_classes)
        iou_disc = iou[0, :].mean().item()
        iou_cup = iou[1, :].mean().item()

        dice_scores_disc.append(dice_disc)
        dice_scores_cup.append(dice_cup)
        iou_scores_disc.append(iou_disc)
        iou_scores_cup.append(iou_cup)

        if vis_dir is not None:
            B = inputs.size(0)
            for b in range(B):
                save_eval_visuals(
                    idx_global,
                    inputs[b],
                    logits[b],
                    targets[b],
                    out_dir=vis_dir,
                    num_classes=num_classes,
                    fname_prefix=split_name.lower(),
                )
                idx_global += 1

    avg_loss = total_loss / max(1, len(data_loader))
    avg_dice_loss = total_dice_loss / max(1, len(data_loader))
    avg_ce_loss = total_ce_loss / max(1, len(data_loader))
    avg_dice_disc = float(np.mean(dice_scores_disc)) if dice_scores_disc else 0.0
    avg_dice_cup = float(np.mean(dice_scores_cup)) if dice_scores_cup else 0.0
    avg_iou_disc = float(np.mean(iou_scores_disc)) if iou_scores_disc else 0.0
    avg_iou_cup = float(np.mean(iou_scores_cup)) if iou_scores_cup else 0.0

    if epoch is not None:
        print(
            f"[{split_name} Epoch {epoch:3d}] total={avg_loss:.4f}  dice_loss={avg_dice_loss:.4f}  ce_loss={avg_ce_loss:.4f} | "
            f"Disc Dice={avg_dice_disc:.4f} | Cup Dice={avg_dice_cup:.4f} | "
            f"Disc IoU={avg_iou_disc:.4f} | Cup IoU={avg_iou_cup:.4f}"
        )
    else:
        print(
            f"[{split_name}] total={avg_loss:.4f}  dice_loss={avg_dice_loss:.4f}  ce_loss={avg_ce_loss:.4f} | "
            f"Disc Dice={avg_dice_disc:.4f} | Cup Dice={avg_dice_cup:.4f} | "
            f"Disc IoU={avg_iou_disc:.4f} | Cup IoU={avg_iou_cup:.4f}"
        )

    return avg_loss, avg_dice_disc, avg_dice_cup, avg_iou_disc, avg_iou_cup




def build_fundus_datasets(args, train_transform, val_transform, test_transform):
    from dataset import FolderDataset_new

    train_domains = [int(domain_id) for domain_id in args.trainid.split(',')]
    test_domains = [int(domain_id) for domain_id in args.testid.split(',')]

    print(f"✅ Training domains: {train_domains}")
    print(f"✅ Testing domains: {test_domains}")

    train_datasets = []
    val_datasets = []
    test_domain_datasets = {}

    # train: trainid/train
    for domain_id in train_domains:
        train_path = os.path.join(args.data_dir, f"Domain{domain_id}/train/ROIs/")
        if os.path.exists(train_path):
            dataset = FolderDataset_new(
                root=train_path,
                img_dir_name=args.img_dir_name,
                label_dir_name=args.label_dir_name,
                mask_ext=args.mask_ext,
                transform=train_transform,
                strict_pair=True,
                num_classes=args.num_classes,
            )
            train_datasets.append(dataset)
            print(f"✅ Loaded training dataset from Domain{domain_id}/train: {len(dataset)} samples")
        else:
            print(f"⚠️ Warning: Training path not found: {train_path}")

    # val: trainid/test
    for domain_id in train_domains:
        val_path = os.path.join(args.data_dir, f"Domain{domain_id}/test/ROIs/")
        if os.path.exists(val_path):
            dataset = FolderDataset_new(
                root=val_path,
                img_dir_name=args.img_dir_name,
                label_dir_name=args.label_dir_name,
                mask_ext=args.mask_ext,
                transform=val_transform,
                strict_pair=True,
                num_classes=args.num_classes,
            )
            val_datasets.append(dataset)
            print(f"✅ Loaded validation dataset from Domain{domain_id}/test: {len(dataset)} samples")
        else:
            print(f"⚠️ Warning: Validation path not found: {val_path}")

    # test: 每个 testid 单独构建
    for domain_id in test_domains:
        domain_subsets = []

        for split in ["train", "test"]:
            test_path = os.path.join(args.data_dir, f"Domain{domain_id}/{split}/ROIs/")
            if os.path.exists(test_path):
                dataset = FolderDataset_new(
                    root=test_path,
                    img_dir_name=args.img_dir_name,
                    label_dir_name=args.label_dir_name,
                    mask_ext=args.mask_ext,
                    transform=test_transform,
                    strict_pair=True,
                    num_classes=args.num_classes,
                )
                domain_subsets.append(dataset)
                print(f"✅ Loaded test dataset from Domain{domain_id}/{split}: {len(dataset)} samples")
            else:
                print(f"⚠️ Warning: Test path not found: {test_path}")

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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, required=True,
                        help="Path to the pretrained DINO checkpoint (.pth). Use ViT-B/16 checkpoint for --dino_size b, or ViT-S/16 checkpoint for --dino_size s.")
    parser.add_argument("--dino_size", type=str, default="b", choices=["b", "s"],
                        help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--last_layer_idx", type=int, default=-1)
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--cuda_id", type=int, default=1)
    parser.add_argument("--trainid", type=str, default="1,2,3")
    parser.add_argument("--testid", type=str, default="4",
                        help="Comma-separated list of domain IDs to include in testing")
    parser.add_argument("--task_name", type=str, default="Fundus_D")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=4, help="LoRA的alpha值")
    args = parser.parse_args()

    use_lora = "lora" in args.task_name.lower()
    if use_lora:
        print(f"✅ 使用LoRA，秩(r)={args.lora_r}, alpha={args.lora_alpha}")

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, will use CPU instead!")
        device = "cpu"
    else:
        cuda_count = torch.cuda.device_count()
        if args.cuda_id >= cuda_count or args.cuda_id < 0:
            raise ValueError(f"Invalid CUDA ID {args.cuda_id}! Only {cuda_count} GPU(s) available (ID: 0~{cuda_count-1})")
        torch.cuda.set_device(args.cuda_id)
        device = f"cuda:{args.cuda_id}"
        print(f"Using CUDA device ID: {args.cuda_id} (device: {device})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_root = f"./runs/segdino3_{args.dino_size}_{args.input_h}_{args.task_name}_train{args.trainid}_test{args.testid}_multi_class"
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

    try:
        from dataset import ResizeAndNormalize_new, FolderDataset_new
    except ImportError as e:
        print(f"❌ 无法导入dataset模块: {e}")
        print("💡 请确保dataset.py在正确路径，并且包含了ResizeAndNormalize_new和FolderDataset_new类")
        sys.exit(1)

    try:
        if args.dino_size == "b":
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
        else:
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)
        print(f"✅ 成功加载DINOv3-{args.dino_size}模型")
    except Exception as e:
        print(f"❌ 无法加载DINOv3模型: {e}")
        print("💡 请确保DINOv3代码在正确路径，并且模型文件存在")
        sys.exit(1)

    backbone.eval()

    if use_lora:
        for param in backbone.parameters():
            param.requires_grad = False
        replace_linear_with_lora(backbone, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        print("✅ LoRA已应用到DINOv3 backbone")

    try:
        from dpt_D import DPT
    except ImportError:
        print("❌ 无法导入DPT模块，请确保dpt_D.py在正确路径")
        sys.exit(1)

    model = DPT(
        nclass=args.num_classes,
        backbone=backbone,
        encoder_size='small',
        freeze_backbone=not use_lora,
    ).to(device)

    if use_lora:
        trainable_params = []
        trainable_params.extend(model.head.parameters())
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params_count:,}")
        print(f"✅ Trainable ratio: {trainable_params_count/total_params:.2%}")
    else:
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ce_class_weights = torch.tensor([0.2, 1.0, 1.0], device=device)
    criterion = DiceCELoss(
        dice_weight=0.5,
        ce_weight=0.5,
        num_classes=args.num_classes,
        ce_class_weights=ce_class_weights,
    )

    train_transform = ResizeAndNormalize_new(size=(args.input_h, args.input_w))
    val_transform = ResizeAndNormalize_new(size=(args.input_h, args.input_w))
    test_transform = ResizeAndNormalize_new(size=(args.input_h, args.input_w))

    train_dataset, val_dataset, test_domain_datasets = build_fundus_datasets(
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
    print(f"   Number of classes: {args.num_classes}")
    print(f"   Input size: {args.input_h}x{args.input_w}")
    print(f"   Using DINOv3 {args.dino_size}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loaders = {}
    for domain_id, test_dataset in test_domain_datasets.items():
        test_loaders[domain_id] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )


    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    best_val_disc_dice = 0.0
    best_val_disc_dice_epoch = 0
    best_val_cup_dice = 0.0
    best_val_cup_dice_epoch = 0
    best_val_disc_iou = 0.0
    best_val_disc_iou_epoch = 0
    best_val_cup_iou = 0.0
    best_val_cup_iou_epoch = 0
    best_val_mean_dice = 0.0
    best_val_mean_dice_epoch = 0
    best_val_mean_iou = 0.0
    best_val_mean_iou_epoch = 0

    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_disc_dice, train_cup_dice, train_disc_iou, train_cup_iou = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=args.num_classes,
            vis_dir=train_vis_dir,
            epoch=epoch,
        )


        val_loss, val_disc_dice, val_cup_dice, val_disc_iou, val_cup_iou = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=args.num_classes,
            epoch=epoch,
            split_name="Val",
            vis_dir=val_vis_dir,
        )

        val_mean_dice = (val_disc_dice + val_cup_dice) / 2
        val_mean_iou = (val_disc_iou + val_cup_iou) / 2

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch

        if val_disc_dice > best_val_disc_dice:
            best_val_disc_dice = val_disc_dice
            best_val_disc_dice_epoch = epoch

        if val_cup_dice > best_val_cup_dice:
            best_val_cup_dice = val_cup_dice
            best_val_cup_dice_epoch = epoch

        if val_disc_iou > best_val_disc_iou:
            best_val_disc_iou = val_disc_iou
            best_val_disc_iou_epoch = epoch

        if val_cup_iou > best_val_cup_iou:
            best_val_cup_iou = val_cup_iou
            best_val_cup_iou_epoch = epoch

        if val_mean_dice > best_val_mean_dice:
            best_val_mean_dice = val_mean_dice
            best_val_mean_dice_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_disc_dice": val_disc_dice,
                    "val_cup_dice": val_cup_dice,
                    "val_disc_iou": val_disc_iou,
                    "val_cup_iou": val_cup_iou,
                    "val_mean_dice": val_mean_dice,
                    "val_mean_iou": val_mean_iou,
                },
                best_model_path,
            )
            print(f"💾 [Save] New best model with mean Dice: {val_mean_dice:.4f} at epoch {epoch}")

        if val_mean_iou > best_val_mean_iou:
            best_val_mean_iou = val_mean_iou
            best_val_mean_iou_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_disc_dice": val_disc_dice,
                "val_cup_dice": val_cup_dice,
                "val_disc_iou": val_disc_iou,
                "val_cup_iou": val_cup_iou,
                "val_mean_dice": val_mean_dice,
                "val_mean_iou": val_mean_iou,
            },
            latest_model_path,
        )

    print("=" * 80)
    print("🏆 Training Complete! Best Results:")
    print("=" * 80)
    print(f"📊 Best Validation Loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch}")
    print("-" * 80)
    print(f"🎯 Best Disc Dice: {best_val_disc_dice:.4f} at epoch {best_val_disc_dice_epoch}")
    print(f"🎯 Best Cup Dice:  {best_val_cup_dice:.4f} at epoch {best_val_cup_dice_epoch}")
    print(f"🎯 Best Mean Dice: {best_val_mean_dice:.4f} at epoch {best_val_mean_dice_epoch}")
    print("-" * 80)
    print(f"🎯 Best Disc IoU:  {best_val_disc_iou:.4f} at epoch {best_val_disc_iou_epoch}")
    print(f"🎯 Best Cup IoU:   {best_val_cup_iou:.4f} at epoch {best_val_cup_iou_epoch}")
    print(f"🎯 Best Mean IoU:  {best_val_mean_iou:.4f} at epoch {best_val_mean_iou_epoch}")
    print("=" * 80)

    final_path = os.path.join(ckpt_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"💾 Final model saved to: {final_path}")

    print("\n🧪 Loading best model for final testing...")
    best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    print("=" * 80)
    print("✅ Final Test Results (Best Model, Per Domain):")
    print("=" * 80)

    all_domain_results = {}

    for domain_id, test_loader in test_loaders.items():
        domain_vis_dir = os.path.join(test_vis_dir, f"Domain{domain_id}")
        os.makedirs(domain_vis_dir, exist_ok=True)

        print(f"\n📌 Testing on Domain{domain_id} ...")

        test_loss, test_disc_dice, test_cup_dice, test_disc_iou, test_cup_iou = evaluate(
            model,
            test_loader,
            criterion,
            device,
            num_classes=args.num_classes,
            epoch=None,
            split_name=f"Test_Domain{domain_id}",
            vis_dir=domain_vis_dir,
        )

        test_mean_dice = (test_disc_dice + test_cup_dice) / 2
        test_mean_iou = (test_disc_iou + test_cup_iou) / 2

        all_domain_results[domain_id] = {
            "loss": test_loss,
            "disc_dice": test_disc_dice,
            "cup_dice": test_cup_dice,
            "mean_dice": test_mean_dice,
            "disc_iou": test_disc_iou,
            "cup_iou": test_cup_iou,
            "mean_iou": test_mean_iou,
        }

        print(
            f"Domain{domain_id} | "
            f"Loss={test_loss:.4f} | "
            f"Disc Dice={test_disc_dice:.4f} | Cup Dice={test_cup_dice:.4f} | Mean Dice={test_mean_dice:.4f} | "
            f"Disc IoU={test_disc_iou:.4f} | Cup IoU={test_cup_iou:.4f} | Mean IoU={test_mean_iou:.4f}"
        )

    print("\n" + "=" * 80)
    print("✅ Summary of Per-Domain Test Results:")
    print("=" * 80)
    for domain_id, result in all_domain_results.items():
        print(
            f"Domain{domain_id} | "
            f"Loss={result['loss']:.4f} | "
            f"Disc Dice={result['disc_dice']:.4f} | Cup Dice={result['cup_dice']:.4f} | Mean Dice={result['mean_dice']:.4f} | "
            f"Disc IoU={result['disc_iou']:.4f} | Cup IoU={result['cup_iou']:.4f} | Mean IoU={result['mean_iou']:.4f}"
        )
    print("=" * 80)

    results = {
        "dino_size": args.dino_size,
        "best_val_loss": float(best_val_loss),
        "best_val_disc_dice": float(best_val_disc_dice),
        "best_val_cup_dice": float(best_val_cup_dice),
        "best_val_mean_dice": float(best_val_mean_dice),
        "best_val_disc_iou": float(best_val_disc_iou),
        "best_val_cup_iou": float(best_val_cup_iou),
        "best_val_mean_iou": float(best_val_mean_iou),
        "best_epoch_loss": best_val_loss_epoch,
        "best_epoch_disc_dice": best_val_disc_dice_epoch,
        "best_epoch_cup_dice": best_val_cup_dice_epoch,
        "best_epoch_mean_dice": best_val_mean_dice_epoch,
        "best_epoch_disc_iou": best_val_disc_iou_epoch,
        "best_epoch_cup_iou": best_val_cup_iou_epoch,
        "best_epoch_mean_iou": best_val_mean_iou_epoch,
        "train_id": args.trainid,
        "test_id": args.testid,
        "input_size": f"{args.input_h}x{args.input_w}",
        "num_classes": args.num_classes,
        "per_domain_test_results": {
            str(domain_id): {
                "loss": float(result["loss"]),
                "disc_dice": float(result["disc_dice"]),
                "cup_dice": float(result["cup_dice"]),
                "mean_dice": float(result["mean_dice"]),
                "disc_iou": float(result["disc_iou"]),
                "cup_iou": float(result["cup_iou"]),
                "mean_iou": float(result["mean_iou"]),
            }
            for domain_id, result in all_domain_results.items()
        }
    }

    results_path = os.path.join(save_root, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to {results_path}")


if __name__ == "__main__":
    main()
