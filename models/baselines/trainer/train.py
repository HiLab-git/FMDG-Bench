# -*- coding: utf-8 -*-
"""
多模型统一训练脚本
支持: UNet++ (unetpp), Attention UNet (attention_unet), nnU-Net (nnunet)

基于 UNet.py 的训练/评估流程, 添加:
- 模型工厂动态加载
- Deep supervision 感知的 loss 计算
- 模型特定 checkpoint 路径
- 清理了 SAM 残留代码

用法:
    python train.py --model_type unetpp --data_root /path/to/Fundus/ \
        --trainid 1,3 --testid 2,4 --task_name UNetPP-Baseline \
        --num_epochs 100 --device cuda:0
"""

import math
import os
import glob
import shutil
import random
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# ---------- 数据集导入 (与 UNet.py 保持一致) ----------
from Datasets.naso_dataset import nasopharyngealDataset


try:
    from Datasets.new_dataloader import PolypDataset, RetinaDataset_prototype
except ModuleNotFoundError:
    from new_dataloader import PolypDataset, RetinaDataset_prototype

# ---------- 模型工厂 ----------
from models import get_model, list_models

# ---------- monai ----------
import monai
from monai.losses import DiceLoss

from medpy.metric.binary import assd


# ============================================================
# 基础设置
# ============================================================
torch.cuda.empty_cache()
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

join = os.path.join


# ============================================================
# 工具函数
# ============================================================
def is_na_task(task_name: str) -> bool:
    return "na" in task_name.lower()


def is_poly_task(task_name: str) -> bool:
    return "poly" in task_name.lower()


def is_fundus_task(task_name: str) -> bool:
    return not is_na_task(task_name) and not is_poly_task(task_name)


def get_num_classes(task_name: str) -> int:
    return 2 if (is_na_task(task_name) or is_poly_task(task_name)) else 3


# ============================================================
# ASSD 计算 (仅使用 medpy, 与参考脚本一致)
# ============================================================
def compute_assd(pred_mask, gt_mask):
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0.5).astype(np.uint8)

    # 两个都空 → 完美一致
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0
    # 一个空 → 无法计算, 跳过
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return np.nan

    return float(assd(pred_mask, gt_mask))


def multi_class_assd(pred, target, num_classes):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    assd_scores = []
    for c in range(1, num_classes):
        class_assd = []
        if pred.ndim == 4:
            pred_class = pred[:, c, :, :]
            target_class = target[:, c, :, :]
            for b in range(pred.shape[0]):
                class_assd.append(compute_assd(pred_class[b], target_class[b]))
        elif pred.ndim == 3:
            for b in range(pred.shape[0]):
                class_assd.append(compute_assd(
                    (pred[b] == c).astype(np.uint8),
                    (target[b] == c).astype(np.uint8)))
        assd_scores.append(np.array(class_assd))
    return np.stack(assd_scores) if assd_scores else np.array([])


# ============================================================
# 可视化 & 指标工具
# ============================================================
def create_color_mask(mask):
    color_map = {1: [0, 255, 0], 2: [0, 0, 255]}
    color_mask = torch.zeros((mask.shape[1], mask.shape[2], 3), dtype=torch.uint8)
    for class_id in range(1, mask.shape[0]):
        class_mask = mask[class_id, :, :] > 0.5
        color = torch.tensor(color_map.get(class_id, [255, 0, 0]), dtype=torch.uint8)
        color_mask[class_mask] = color
    return color_mask


def visualize_on_tensorboard(image, gt_mask, pred_mask, step, writer, image_name, phase="Train"):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axs[0].axis("off")
    gt_color = create_color_mask(gt_mask.cpu().numpy())
    axs[1].imshow(image.permute(1, 2, 0).cpu().numpy())
    axs[1].imshow(gt_color, alpha=0.5)
    axs[1].axis("off")
    pred_color = create_color_mask(pred_mask.cpu().detach().numpy())
    axs[2].imshow(image.permute(1, 2, 0).cpu().numpy())
    axs[2].imshow(pred_color, alpha=0.5)
    axs[2].axis("off")
    plt.tight_layout(pad=0.2)
    writer.add_figure(f"{phase}/Visualizations/{image_name}", fig, global_step=step)
    plt.close(fig)


def multi_class_dice(pred, target, num_classes):
    dice_scores = []
    for c in range(1, num_classes):
        pred_class = pred[:, c, :, :]
        target_class = target[:, c, :, :]
        intersection = (pred_class * target_class).sum(dim=[1, 2])
        union = pred_class.sum(dim=[1, 2]) + target_class.sum(dim=[1, 2])
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice)
    return torch.stack(dice_scores)


def multi_class_iou(pred, target, num_classes):
    iou_scores = []
    for c in range(1, num_classes):
        pred_class = pred[:, c, :, :]
        target_class = target[:, c, :, :]
        intersection = (pred_class * target_class).sum(dim=[1, 2])
        union = pred_class.sum(dim=[1, 2]) + target_class.sum(dim=[1, 2]) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_scores.append(iou)
    return torch.stack(iou_scores)


def probs_to_onehot(probs: torch.Tensor) -> torch.Tensor:
    pred = torch.zeros_like(probs)
    pred.scatter_(1, torch.argmax(probs, dim=1, keepdim=True), 1.0)
    return pred


def normalize_target(gt2D: torch.Tensor, num_classes: int) -> torch.Tensor:
    if gt2D.ndim == 3:
        if num_classes == 2:
            gt2D = gt2D.float()
            fg = (gt2D > 0.5).float()
            bg = 1.0 - fg
            return torch.stack([bg, fg], dim=1)
        # 多类别: [B, H, W] long → [B, C, H, W] one-hot
        return F.one_hot(gt2D.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    if gt2D.ndim == 4:
        if gt2D.shape[1] == 1 and num_classes == 2:
            fg = (gt2D[:, 0] > 0.5).float()
            bg = 1.0 - fg
            return torch.stack([bg, fg], dim=1)
        if gt2D.shape[1] == num_classes:
            return gt2D.float()
    raise ValueError(
        f"Target shape {gt2D.shape} is incompatible with num_classes={num_classes}")


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError(f"Unexpected batch format: len={len(batch)}")
        image = batch[0]
        gt2D = batch[1]
        meta = batch[2:] if len(batch) > 2 else ()
        return image, gt2D, meta
    raise ValueError(f"Unexpected batch type: {type(batch)}")


# ============================================================
# Deep Supervision 感知的 Loss 计算
# ============================================================
class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, target_onehot):
        probs = F.softmax(logits, dim=1)
        if not self.include_background:
            probs = probs[:, 1:, :, :]
            target_onehot = target_onehot[:, 1:, :, :]
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


ce_loss_fn = nn.CrossEntropyLoss()
dice_loss_fn = MultiClassDiceLoss(include_background=False)
CE_WEIGHT = 0.5
DICE_WEIGHT = 0.5


def _interpolate_to_match(logits, target):
    """将 logits 插值到与 target 相同的空间尺寸"""
    if logits.shape[2:] != target.shape[2:]:
        return F.interpolate(logits, size=target.shape[2:], mode="bilinear",
                             align_corners=False)
    return logits


def compute_loss(model_output, gt2D_onehot):
    """
    统一的 loss 计算, 兼容单一输出和 deep supervision (list/tuple)。

    参数:
        model_output:  单一 Tensor [B, C, H, W] 或 list of Tensors
        gt2D_onehot:   [B, C, H, W] 真实标签 one-hot

    返回:
        total_loss:  标量 loss
        main_output: 主输出 Tensor (用于计算指标)
    """
    target_labels = torch.argmax(gt2D_onehot, dim=1)

    if isinstance(model_output, (list, tuple)):
        # Deep supervision: 对各层分别计算 loss, 加权求和
        # 主输出 (最高分辨率) 权重更高
        weights = [1.0]
        for i in range(1, len(model_output)):
            decay = 1.0 / (2 ** i)
            weights.append(decay)
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        total_loss = 0.0
        for out, w in zip(model_output, weights):
            out = _interpolate_to_match(out, gt2D_onehot)
            ce = ce_loss_fn(out, target_labels)
            dc = dice_loss_fn(out, gt2D_onehot)
            total_loss += w * (CE_WEIGHT * ce + DICE_WEIGHT * dc)

        main_output = model_output[0]
    else:
        ce = ce_loss_fn(model_output, target_labels)
        dc = dice_loss_fn(model_output, gt2D_onehot)
        total_loss = CE_WEIGHT * ce + DICE_WEIGHT * dc
        main_output = model_output

    return total_loss, main_output


# ============================================================
# 数据路径 & 数据集构建
# ============================================================
def get_task_paths(train_domains, test_domains, data_root, task_name):
    if is_na_task(task_name):
        train_paths = [os.path.join(data_root, f"Domain{d}/train/") for d in train_domains]
        val_paths = [os.path.join(data_root, f"Domain{d}/test/") for d in train_domains]
        test_paths = (
            [os.path.join(data_root, f"Domain{d}/train/") for d in test_domains] +
            [os.path.join(data_root, f"Domain{d}/test/") for d in test_domains])
    elif is_poly_task(task_name):
        train_paths = [os.path.join(data_root, f"Domain{d}/split_dataset/train/") for d in train_domains]
        val_paths = [os.path.join(data_root, f"Domain{d}/split_dataset/test/") for d in train_domains]
        test_paths = (
            [os.path.join(data_root, f"Domain{d}/split_dataset/train/") for d in test_domains] +
            [os.path.join(data_root, f"Domain{d}/split_dataset/test/") for d in test_domains])
    else:
        train_paths = [os.path.join(data_root, f"Domain{d}/train/ROIs/") for d in train_domains]
        val_paths = [os.path.join(data_root, f"Domain{d}/test/ROIs/") for d in train_domains]
        test_paths = (
            [os.path.join(data_root, f"Domain{d}/train/ROIs/") for d in test_domains] +
            [os.path.join(data_root, f"Domain{d}/test/ROIs/") for d in test_domains])
    return train_paths, val_paths, test_paths


def build_datasets(train_paths, val_paths, test_paths, task_name):
    if is_na_task(task_name):
        train_datasets = [nasopharyngealDataset(p) for p in train_paths]
        val_datasets = [nasopharyngealDataset(p) for p in val_paths]
        test_datasets = [nasopharyngealDataset(p) for p in test_paths]
    elif is_poly_task(task_name):
        train_datasets = [PolypDataset(p) for p in train_paths]
        val_datasets = [PolypDataset(p) for p in val_paths]
        test_datasets = [PolypDataset(p) for p in test_paths]
    else:
        train_datasets = [RetinaDataset_prototype(p) for p in train_paths]
        val_datasets = [RetinaDataset_prototype(p) for p in val_paths]
        test_datasets = [RetinaDataset_prototype(p) for p in test_paths]
    return train_datasets, val_datasets, test_datasets


def find_best_model(checkpoint_dir):
    matches = glob.glob(os.path.join(checkpoint_dir, "**", "*_best_model.pth"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No best model found under {checkpoint_dir}")
    return max(matches, key=os.path.getmtime)


# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser(description="Multi-model segmentation training")

# 数据
parser.add_argument("-i", "--tr_npy_path", type=str,
                    default="./data/Fundus/Domain1/train/ROIs/")
parser.add_argument("-j", "--ts_npy_path", type=str,
                    default="./data/Fundus/Domain1/test/ROIs/")
parser.add_argument("--data_root", type=str, default="./data/Fundus/")
parser.add_argument("--trainid", type=str, default="1,2,3")
parser.add_argument("--testid", type=str, default="4")
parser.add_argument("--task_name", type=str, default="MultiModel-Baseline")

# 模型
parser.add_argument("--model_type", type=str, default="unetpp",
                    choices=["unetpp", "attention_unet", "nnunet","transunet"],
                    help="Model architecture to train")
parser.add_argument("--iternum", type=int, default=20000)
parser.add_argument("--checkpoint", type=str, default="best_model.pth")

# 训练
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--warmup_epochs", type=int, default=5)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--grad_clip", type=float, default=1.0)

# 其他
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--multi_gpu", type=str, default="",
                    help="逗号分隔的 GPU ID, 用于 DataParallel (如 '0,1,2,3'), 空字符串=不使用")
parser.add_argument("--use_wandb", type=bool, default=False)
parser.add_argument("--use_amp", action="store_true", default=False)
parser.add_argument("--resume", type=str, default="")

args = parser.parse_args()


# ============================================================
# wandb
# ============================================================
if args.use_wandb:
    import wandb
    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )


# ============================================================
# 路径设置 (包含模型名)
# ============================================================
run_id = datetime.now().strftime("%Y%m%d-%H%M")

if is_na_task(args.task_name):
    task_category = "naso_auto"
else:
    task_category = "fundus_auto"

model_save_path = join(
    args.work_dir, task_category,
    f"{args.model_type}-domain{args.trainid}_{args.iternum}",
    run_id,
)

device = torch.device(args.device)

log_dir = join(
    args.work_dir, "logs", args.task_name,
    f"domain{args.trainid}_{args.iternum}", run_id,
)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


# ============================================================
# Dataloader & 模型 设置
# ============================================================
def setup_dataloaders(train_domains, test_domains, data_root, task_name,
                      batch_size, num_workers):
    train_paths, val_paths, _ = get_task_paths(train_domains, test_domains,
                                               data_root, task_name)
    train_datasets, val_datasets, _ = build_datasets(train_paths, val_paths, [], task_name)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    generator = torch.Generator(device="cpu")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, generator=generator,
    )

    # 每个 test domain 单独建立 dataloader
    test_loaders = {}
    test_dataset_sizes = {}
    for domain_id in test_domains:
        if is_na_task(task_name):
            domain_test_paths = [
                os.path.join(data_root, f"Domain{domain_id}/train/"),
                os.path.join(data_root, f"Domain{domain_id}/test/"),
            ]
            domain_test_datasets = [nasopharyngealDataset(p) for p in domain_test_paths]
        elif is_poly_task(task_name):
            domain_test_paths = [
                os.path.join(data_root, f"Domain{domain_id}/split_dataset/train/"),
                os.path.join(data_root, f"Domain{domain_id}/split_dataset/test/"),
            ]
            domain_test_datasets = [PolypDataset(p) for p in domain_test_paths]
        else:
            domain_test_paths = [
                os.path.join(data_root, f"Domain{domain_id}/train/ROIs/"),
                os.path.join(data_root, f"Domain{domain_id}/test/ROIs/"),
            ]
            domain_test_datasets = [RetinaDataset_prototype(p) for p in domain_test_paths]

        domain_test_dataset = ConcatDataset(domain_test_datasets)
        test_dataset_sizes[domain_id] = len(domain_test_dataset)
        test_loaders[domain_id] = DataLoader(
            domain_test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, generator=generator,
        )

    return train_loader, val_loader, test_loaders, test_dataset_sizes


def setup_optimizer_scheduler(model, args):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )
    warmup_epochs = args.warmup_epochs
    total_epochs = args.num_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return max(args.min_lr / args.lr, cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def unwrap_model(model):
    """如果模型被 DataParallel 包裹, 返回内部模型; 否则返回原模型"""
    return model.module if isinstance(model, nn.DataParallel) else model


def setup_model(args):
    """使用模型工厂创建模型, 可选 DataParallel 多卡"""
    num_classes = get_num_classes(args.task_name)
    model = get_model(
        args.model_type,
        in_channels=3,
        out_channels=num_classes,
    ).to(device)

    # DataParallel 多卡支持
    if args.multi_gpu:
        gpu_ids = [int(x.strip()) for x in args.multi_gpu.split(",")]
        print(f"DataParallel enabled: GPUs {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    for param in model.parameters():
        param.requires_grad = True
    return model


# ============================================================
# 训练 / 验证 / 测试
# ============================================================
def train_epoch(model, dataloader, optimizer, device, iter_num, log_interval,
                writer, scaler=None):
    model.train()
    num_classes = get_num_classes(args.task_name)
    num_fg = num_classes - 1

    epoch_loss = 0.0
    epoch_dice = torch.zeros(num_fg)

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        image, gt2D, _ = unpack_batch(batch)
        image, gt2D = image.to(device), gt2D.to(device)
        gt2D_onehot = normalize_target(gt2D, num_classes)

        if args.use_amp and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output = model(image)
                loss, main_output = compute_loss(model_output, gt2D_onehot)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            model_output = model(image)
            loss, main_output = compute_loss(model_output, gt2D_onehot)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        epoch_loss += loss.item()

        probs = F.softmax(main_output, dim=1)
        pred_onehot = probs_to_onehot(probs)
        dice_score = multi_class_dice(pred_onehot, gt2D_onehot, num_classes)
        for i in range(num_fg):
            epoch_dice[i] += dice_score[i, :].mean().item()

        iter_num += 1

        if iter_num % log_interval == 0:
            writer.add_scalar("Train/Loss_total", loss.item(), iter_num)
            for i in range(num_fg):
                writer.add_scalar(f"Train/Dice_{i}",
                                  dice_score[i, :].mean().item(), iter_num)
            print(f"[Iter {iter_num:5d}] Loss: {loss.item():.4f}")

    epoch_loss /= len(dataloader)
    epoch_dice /= len(dataloader)
    return epoch_loss, epoch_dice, iter_num


def validate(model, dataloader, device, writer, epoch):
    model.eval()
    num_classes = get_num_classes(args.task_name)
    num_fg = num_classes - 1

    val_loss = 0.0
    epoch_val_dice = torch.zeros(num_fg)
    epoch_val_iou = torch.zeros(num_fg)
    epoch_val_assd = torch.zeros(num_fg)

    with torch.no_grad():
        for batch in dataloader:
            image, gt2D, _ = unpack_batch(batch)
            image, gt2D = image.to(device), gt2D.to(device)
            gt2D_onehot = normalize_target(gt2D, num_classes)

            model_output = model(image)
            loss, main_output = compute_loss(model_output, gt2D_onehot)
            val_loss += loss.item()

            probs = F.softmax(main_output, dim=1)
            pred_onehot = probs_to_onehot(probs)
            val_dice = multi_class_dice(pred_onehot, gt2D_onehot, num_classes)
            val_iou = multi_class_iou(pred_onehot, gt2D_onehot, num_classes)
            val_assd = multi_class_assd(pred_onehot, gt2D_onehot, num_classes)

            for i in range(num_fg):
                epoch_val_dice[i] += val_dice[i, :].mean().item()
                epoch_val_iou[i] += val_iou[i, :].mean().item()
                if val_assd.size > 0:
                    v = np.nanmean(val_assd[i, :])
                    if not np.isnan(v):
                        epoch_val_assd[i] += v

    denom = len(dataloader)
    val_loss /= denom
    epoch_val_dice /= denom
    epoch_val_iou /= denom
    epoch_val_assd /= denom

    val_dice_mean = epoch_val_dice.mean().item()
    val_assd_mean = epoch_val_assd.mean().item() if num_fg > 0 else 0.0

    writer.add_scalar("Validation/Loss_total", val_loss, epoch)
    writer.add_scalar("Validation/Dice_mean", val_dice_mean, epoch)
    writer.add_scalar("Validation/ASSD_mean", val_assd_mean, epoch)
    for i in range(num_fg):
        writer.add_scalar(f"Validation/Dice_{i}", epoch_val_dice[i], epoch)
        writer.add_scalar(f"Validation/IoU_{i}", epoch_val_iou[i], epoch)
        writer.add_scalar(f"Validation/ASSD_{i}", epoch_val_assd[i], epoch)

    return val_loss, epoch_val_dice, epoch_val_iou, epoch_val_assd, \
        val_dice_mean, val_assd_mean


def test(model, dataloader, device, writer):
    model.eval()
    num_classes = get_num_classes(args.task_name)
    num_fg = num_classes - 1

    dice_scores = [[] for _ in range(num_fg)]
    iou_scores = [[] for _ in range(num_fg)]
    assd_scores = [[] for _ in range(num_fg)]

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            image, gt2D, _ = unpack_batch(batch)
            image, gt2D = image.to(device), gt2D.to(device)
            gt2D_onehot = normalize_target(gt2D, num_classes)

            model_output = model(image)
            # 推理时取主输出
            if isinstance(model_output, (list, tuple)):
                main_output = model_output[0]
            else:
                main_output = model_output

            probs = F.softmax(main_output, dim=1)
            pred = probs_to_onehot(probs)

            test_dice = multi_class_dice(pred, gt2D_onehot, num_classes)
            test_iou = multi_class_iou(pred, gt2D_onehot, num_classes)
            test_assd = multi_class_assd(pred, gt2D_onehot, num_classes)

            for i in range(num_fg):
                dice_scores[i].append(test_dice[i, :].mean().item())
                iou_scores[i].append(test_iou[i, :].mean().item())
                if test_assd.size > 0:
                    v = np.nanmean(test_assd[i, :])
                    if not np.isnan(v):
                        assd_scores[i].append(v)

            if step % 10 == 0:
                image_resized = F.interpolate(
                    image[0].unsqueeze(0), size=(800, 800),
                    mode="bilinear", align_corners=False,
                )
                visualize_on_tensorboard(
                    image_resized[0], gt2D_onehot[0], pred[0],
                    step, writer, f"Test_Step_{step}", phase="Test",
                )

    avg_dice = torch.tensor([np.mean(v) for v in dice_scores])
    std_dice = torch.tensor([np.std(v) for v in dice_scores])
    avg_iou = torch.tensor([np.mean(v) for v in iou_scores])
    std_iou = torch.tensor([np.std(v) for v in iou_scores])
    avg_assd = torch.tensor([np.mean(v) if v else 0.0 for v in assd_scores])
    std_assd = torch.tensor([np.std(v) if v else 0.0 for v in assd_scores])

    for i in range(num_fg):
        writer.add_scalar(f"Test/Average_Dice_{i}", avg_dice[i].item())
        writer.add_scalar(f"Test/Average_IoU_{i}", avg_iou[i].item())
        writer.add_scalar(f"Test/Average_ASSD_{i}", avg_assd[i].item())

    print("\n=== Test Results ===")
    if num_classes == 2:
        print(f"Foreground Dice: {avg_dice[0]:.4f} +- {std_dice[0]:.4f}")
        print(f"Foreground IoU:  {avg_iou[0]:.4f} +- {std_iou[0]:.4f}")
        print(f"Foreground ASSD: {avg_assd[0]:.4f} +- {std_assd[0]:.4f}")
        return (avg_dice[0], std_dice[0], 0, 0,
                avg_iou[0], std_iou[0], 0, 0,
                avg_assd[0], std_assd[0], 0, 0)
    else:
        print(f"Disc Dice: {avg_dice[0]:.4f} +- {std_dice[0]:.4f}")
        print(f"Cup Dice:  {avg_dice[1]:.4f} +- {std_dice[1]:.4f}")
        print(f"Disc IoU:  {avg_iou[0]:.4f} +- {std_iou[0]:.4f}")
        print(f"Cup IoU:   {avg_iou[1]:.4f} +- {std_iou[1]:.4f}")
        print(f"Disc ASSD: {avg_assd[0]:.4f} +- {std_assd[0]:.4f}")
        print(f"Cup ASSD:  {avg_assd[1]:.4f} +- {std_assd[1]:.4f}")
        return (avg_dice[0], std_dice[0], avg_dice[1], std_dice[1],
                avg_iou[0], std_iou[0], avg_iou[1], std_iou[1],
                avg_assd[0], std_assd[0], avg_assd[1], std_assd[1])


def test_per_domain(model, test_loaders, device, writer):
    all_domain_results = {}
    for domain_id, loader in test_loaders.items():
        print(f"\n=== Testing on Domain{domain_id} ===")
        result = test(model, loader, device, writer)

        num_classes = get_num_classes(args.task_name)
        if num_classes == 2:
            (avg_dice0, std_dice0, _, _,
             avg_iou0, std_iou0, _, _,
             avg_assd0, std_assd0, _, _) = result
            all_domain_results[domain_id] = {
                "fg_dice_mean": float(avg_dice0),
                "fg_dice_std": float(std_dice0),
                "fg_iou_mean": float(avg_iou0),
                "fg_iou_std": float(std_iou0),
                "fg_assd_mean": float(avg_assd0),
                "fg_assd_std": float(std_assd0),
            }
        else:
            (avg_dice0, std_dice0, avg_dice1, std_dice1,
             avg_iou0, std_iou0, avg_iou1, std_iou1,
             avg_assd0, std_assd0, avg_assd1, std_assd1) = result
            all_domain_results[domain_id] = {
                "disc_dice_mean": float(avg_dice0),
                "disc_dice_std": float(std_dice0),
                "cup_dice_mean": float(avg_dice1),
                "cup_dice_std": float(std_dice1),
                "disc_iou_mean": float(avg_iou0),
                "disc_iou_std": float(std_iou0),
                "cup_iou_mean": float(avg_iou1),
                "cup_iou_std": float(std_iou1),
                "disc_assd_mean": float(avg_assd0),
                "disc_assd_std": float(std_assd0),
                "cup_assd_mean": float(avg_assd1),
                "cup_assd_std": float(std_assd1),
            }
    return all_domain_results


# ============================================================
# 主函数
# ============================================================
def main():
    print(f"\n{'='*60}")
    print(f"Model: {args.model_type}")
    print(f"Task:  {args.task_name}")
    print(f"Train domains: {args.trainid}  |  Test domains: {args.testid}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, join(model_save_path, run_id + "_train.py"))

    train_domains = [int(x) for x in args.trainid.split(",")]
    test_domains = [int(x) for x in args.testid.split(",")]

    train_loader, val_loader, test_loaders, test_dataset_sizes = setup_dataloaders(
        train_domains, test_domains, args.data_root, args.task_name,
        args.batch_size, args.num_workers,
    )

    print(f"Training samples:   {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Total test samples: {sum(test_dataset_sizes.values())}")
    for domain_id in test_domains:
        print(f"  Domain{domain_id} test samples: {test_dataset_sizes[domain_id]}")

    # ---------- 构建模型 ----------
    model = setup_model(args)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio:      {trainable_params / total_params:.2%}")

    optimizer, scheduler = setup_optimizer_scheduler(model, args)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    num_epochs = args.num_epochs
    iter_num = 0
    log_interval = 100

    num_classes = get_num_classes(args.task_name)
    num_fg = num_classes - 1

    best_val_dice = torch.zeros(num_fg)
    best_val_mean_dice = -1.0
    best_model_path = join(model_save_path, "_best_model.pth")

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Resumed from epoch {start_epoch}")

    print(f"\n=== Starting training for {num_epochs} epochs ===\n")

    for epoch in range(start_epoch, num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs}  |  "
              f"LR: {scheduler.get_last_lr()[0]:.6f} ---")

        # Train
        epoch_loss, epoch_dice, iter_num = train_epoch(
            model, train_loader, optimizer, device, iter_num,
            log_interval, writer, scaler,
        )
        scheduler.step()

        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch + 1)
        writer.add_scalar("Epoch/Loss_total", epoch_loss, epoch + 1)
        for i in range(num_fg):
            writer.add_scalar(f"Epoch/Dice_{i}", epoch_dice[i].item(), epoch + 1)

        print(f"[Train Epoch {epoch + 1}] Loss: {epoch_loss:.4f}")
        if num_classes == 2:
            print(f"  Foreground Dice: {epoch_dice[0].item():.4f} "
                  f"(mean: {epoch_dice.mean().item():.4f})")
        else:
            print(f"  Disc Dice: {epoch_dice[0].item():.4f}, "
                  f"Cup Dice: {epoch_dice[1].item():.4f}, "
                  f"Mean: {epoch_dice.mean().item():.4f}")

        # Validate
        val_loss, val_dice, val_iou, val_assd, val_dice_mean, val_assd_mean = validate(
            model, val_loader, device, writer, epoch + 1,
        )

        print(f"[Val Epoch {epoch + 1}] Loss: {val_loss:.4f}  |  "
              f"Mean Dice: {val_dice_mean:.4f}  |  Mean ASSD: {val_assd_mean:.4f}")

        if num_classes == 2:
            print(f"  FG Dice: {val_dice[0].item():.4f}, "
                  f"FG IoU: {val_iou[0].item():.4f}, "
                  f"FG ASSD: {val_assd[0].item():.4f}")
        else:
            print(f"  Disc Dice: {val_dice[0].item():.4f}, "
                  f"Cup Dice: {val_dice[1].item():.4f}")
            print(f"  Disc ASSD: {val_assd[0].item():.4f}, "
                  f"Cup ASSD: {val_assd[1].item():.4f}")

        # 更新最佳指标
        for i in range(num_fg):
            if val_dice[i] > best_val_dice[i]:
                best_val_dice[i] = val_dice[i]

        if val_dice_mean > best_val_mean_dice:
            best_val_mean_dice = val_dice_mean
            torch.save(unwrap_model(model).state_dict(), best_model_path)
            print(f"  -> Saved new best model (Mean Dice: {val_dice_mean:.4f})")

        # 定期保存 checkpoint
        checkpoint = {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "model_latest.pth"))

    print("\n=== Training completed ===")
    print(f"Best validation Mean Dice: {best_val_mean_dice:.4f}")

    # ---------- 加载 best model 测试 ----------
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}, skip testing.")
        return

    print(f"\n=== Loading best model for testing: {best_model_path} ===")
    unwrap_model(model).load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
    model.eval()

    all_domain_results = test_per_domain(model, test_loaders, device, writer)

    print("\n=== Final Per-Domain Test Results ===")
    for domain_id, result in all_domain_results.items():
        print(f"\n[Domain{domain_id}]")
        if num_classes == 2:
            print(f"  FG Dice: {result['fg_dice_mean']:.4f} +- {result['fg_dice_std']:.4f}")
            print(f"  FG IoU:  {result['fg_iou_mean']:.4f} +- {result['fg_iou_std']:.4f}")
            print(f"  FG ASSD: {result['fg_assd_mean']:.4f} +- {result['fg_assd_std']:.4f}")
        else:
            print(f"  Disc Dice: {result['disc_dice_mean']:.4f} +- {result['disc_dice_std']:.4f}")
            print(f"  Cup Dice:  {result['cup_dice_mean']:.4f} +- {result['cup_dice_std']:.4f}")
            print(f"  Disc IoU:  {result['disc_iou_mean']:.4f} +- {result['disc_iou_std']:.4f}")
            print(f"  Cup IoU:   {result['cup_iou_mean']:.4f} +- {result['cup_iou_std']:.4f}")
            print(f"  Disc ASSD: {result['disc_assd_mean']:.4f} +- {result['disc_assd_std']:.4f}")
            print(f"  Cup ASSD:  {result['cup_assd_mean']:.4f} +- {result['cup_assd_std']:.4f}")

    print(f"\nResults saved to: {model_save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
