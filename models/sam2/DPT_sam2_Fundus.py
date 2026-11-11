# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""
# Dice损失

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
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# 移除原有的SAM导入，改为SAM2
# from segment_anything import sam_model_registry
from seg_head.SamFeatSeg import DPTDecoderForSAM
from naso_dataset_16 import nasopharyngealDataset


from Datasets.new_dataloader import PolypDataset, RetinaDataset_prototype


# LoRA 改成按需导入，避免环境里没装时直接崩
try:
    import loralib as lora
except ModuleNotFoundError:
    lora = None
try:
    from medpy.metric.binary import assd
    MEDPY_AVAILABLE = True
except ImportError:
    MEDPY_AVAILABLE = False
    print("Warning: medpy not available, falling back to custom ASSD implementation")
# 添加SAM2相关的导入
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2
# from sam2.sam2_image_encoder import SAM2ImageEncoder
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.backbones.image_encoder import FpnNeck

# -----------------------------
# 基础设置
# -----------------------------
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


# -----------------------------
# 工具函数
# -----------------------------
def is_na_task(task_name: str) -> bool:
    return "na" in task_name.lower()


def is_poly_task(task_name: str) -> bool:
    return "poly" in task_name.lower()


def is_fundus_task(task_name: str) -> bool:
    return not is_na_task(task_name) and not is_poly_task(task_name)


def get_num_classes(task_name: str) -> int:
    # na / poly: 背景+前景 = 2 类
    # fundus: 背景+disc+cup = 3 类
    return 2 if (is_na_task(task_name) or is_poly_task(task_name)) else 3


def is_medsam_task(task_name: str) -> bool:
    name = task_name.lower()
    return ("medsam" in name) or ("sam-med" in name) or ("sam_med" in name)


def is_sam2_task(task_name: str) -> bool:
    name = task_name.lower()
    return "sam2" in name


def replace_linear_with_lora(module, lora_r=4, lora_alpha=4):
    if lora is None:
        raise ModuleNotFoundError(
            "当前 task_name 启用了 LoRA，但环境中未安装 loralib。请先安装 loralib。"
        )

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

def compute_assd(pred_mask, gt_mask, spacing_mm=(1.0, 1.0)):
    """
    计算Average Symmetric Surface Distance (ASSD)
    
    参数:
        pred_mask: 预测掩码 [H, W]
        gt_mask: 真实掩码 [H, W]
        spacing_mm: 像素间距（毫米）
    
    返回:
        assd_value: ASSD值
    """
    # 确保是二值掩码
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    
    # 如果两个掩码都是空的，返回0
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0
    
    # 如果只有一个是空的，返回最大可能距离（对角线）
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        h, w = pred_mask.shape
        return float(np.sqrt(h**2 + w**2) * max(spacing_mm))
    
    if MEDPY_AVAILABLE:
        try:
            # 使用medpy计算ASSD
            assd_value = assd(pred_mask, gt_mask, voxelspacing=spacing_mm)
            return float(assd_value)
        except Exception as e:
            print(f"Warning: medpy assd failed with error: {e}, using fallback")
            # 如果medpy失败，使用备用方法
            return compute_assd_fallback(pred_mask, gt_mask, spacing_mm)
    else:
        return compute_assd_fallback(pred_mask, gt_mask, spacing_mm)


def compute_assd_fallback(pred_mask, gt_mask, spacing_mm=(1.0, 1.0)):
    """备用的ASSD计算方法"""
    try:
        from scipy.spatial.distance import cdist
        from scipy.ndimage import distance_transform_edt
        
        # 使用距离变换
        pred_dt = distance_transform_edt(1 - pred_mask)
        gt_dt = distance_transform_edt(1 - gt_mask)
        
        # 获取表面点
        kernel = np.ones((3, 3), np.uint8)
        pred_surface = pred_mask - cv2.erode(pred_mask, kernel)
        gt_surface = gt_mask - cv2.erode(gt_mask, kernel)
        
        if pred_surface.sum() == 0 or gt_surface.sum() == 0:
            return 0.0
        
        # 计算表面点到对方表面的距离
        pred_dist = pred_dt[pred_surface > 0] * spacing_mm[0]
        gt_dist = gt_dt[gt_surface > 0] * spacing_mm[1]
        
        # 合并距离
        all_distances = np.concatenate([pred_dist, gt_dist])
        
        return float(np.mean(all_distances))
    except:
        # 最简单的备用方法
        pred_points = np.argwhere(pred_mask > 0) * spacing_mm
        gt_points = np.argwhere(gt_mask > 0) * spacing_mm
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return 0.0
        
        # 计算平均欧氏距离
        total_dist = 0.0
        count = 0
        for p in pred_points:
            dists = np.sqrt(np.sum((gt_points - p) ** 2, axis=1))
            total_dist += np.min(dists)
            count += 1
        
        return float(total_dist / count) if count > 0 else 0.0


def multi_class_assd(pred, target, num_classes, spacing_mm=(1.0, 1.0)):
    """
    计算多类ASSD
    
    参数:
        pred: 预测掩码 [B, C, H, W] 或 [B, H, W]
        target: 真实掩码 [B, C, H, W] 或 [B, H, W]
        num_classes: 类别数
        spacing_mm: 像素间距
    
    返回:
        assd_scores: [num_classes-1, B] ASSD分数
    """
    # 确保输入是numpy数组
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    assd_scores = []
    
    for c in range(1, num_classes):  # 跳过背景类
        class_assd = []
        
        if pred.ndim == 4:  # [B, C, H, W]
            pred_class = pred[:, c, :, :]
            target_class = target[:, c, :, :]
            
            for b in range(pred.shape[0]):
                assd_val = compute_assd(pred_class[b], target_class[b], spacing_mm)
                class_assd.append(assd_val)
                
        elif pred.ndim == 3:  # [B, H, W]
            for b in range(pred.shape[0]):
                assd_val = compute_assd((pred[b] == c).astype(np.uint8), 
                                       (target[b] == c).astype(np.uint8), 
                                       spacing_mm)
                class_assd.append(assd_val)
        
        assd_scores.append(np.array(class_assd))
    
    return np.stack(assd_scores) if assd_scores else np.array([])

def create_color_mask(mask):
    """
    mask: [C, H, W]
    """
    color_map = {
        1: [0, 255, 0],   # class 1
        2: [0, 0, 255],   # class 2
    }

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

    gt_mask_np = gt_mask.cpu().numpy()
    gt_color = create_color_mask(gt_mask_np)
    axs[1].imshow(image.permute(1, 2, 0).cpu().numpy())
    axs[1].imshow(gt_color, alpha=0.5)
    axs[1].axis("off")

    pred_mask_np = pred_mask.cpu().detach().numpy()
    pred_color = create_color_mask(pred_mask_np)
    axs[2].imshow(image.permute(1, 2, 0).cpu().numpy())
    axs[2].imshow(pred_color, alpha=0.5)
    axs[2].axis("off")

    plt.tight_layout(pad=0.2)
    writer.add_figure(f"{phase}/Visualizations/{image_name}", fig, global_step=step)
    plt.close(fig)


def multi_class_dice(pred, target, num_classes):
    """
    pred/target: [B, C, H, W]
    返回: [num_classes-1, B]
    """
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
    """
    pred/target: [B, C, H, W]
    返回: [num_classes-1, B]
    """
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
    """
    probs: [B, C, H, W]
    return onehot: [B, C, H, W]
    """
    pred = torch.zeros_like(probs)
    pred.scatter_(1, torch.argmax(probs, dim=1, keepdim=True), 1.0)
    return pred


def normalize_target(gt2D: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    把不同格式的标签统一成 [B, C, H, W]
    
    支持：
    1) 二分类单通道: [B, H, W]
    2) 二分类单通道: [B, 1, H, W]
    3) 已经是 one-hot: [B, C, H, W]
    """
    if gt2D.ndim == 3:
        # [B, H, W]
        if num_classes != 2:
            raise ValueError(f"Expected {num_classes} classes, but got single-channel target with shape {gt2D.shape}")
        gt2D = gt2D.float()
        fg = (gt2D > 0.5).float()
        bg = 1.0 - fg
        gt2D = torch.stack([bg, fg], dim=1)
        return gt2D

    if gt2D.ndim == 4:
        # [B, 1, H, W] -> 二分类转 one-hot
        if gt2D.shape[1] == 1 and num_classes == 2:
            fg = (gt2D[:, 0] > 0.5).float()
            bg = 1.0 - fg
            gt2D = torch.stack([bg, fg], dim=1)
            return gt2D

        # 已经是 [B, C, H, W]
        if gt2D.shape[1] == num_classes:
            return gt2D.float()

    raise ValueError(
        f"Target shape {gt2D.shape} is incompatible with num_classes={num_classes}"
    )


def unpack_batch(batch):
    """
    兼容不同 dataset 的返回格式。
    只取 image 和 gt2D，其他元素忽略。
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError(f"Unexpected batch format: len={len(batch)}")
        image = batch[0]
        gt2D = batch[1]
        meta = batch[2:] if len(batch) > 2 else ()
        return image, gt2D, meta
    raise ValueError(f"Unexpected batch type: {type(batch)}")


def get_task_paths(train_domains, test_domains, data_root, task_name):
    if is_na_task(task_name):
        train_paths = [os.path.join(data_root, f"Domain{d}/train/") for d in train_domains]
        val_paths = [os.path.join(data_root, f"Domain{d}/test/") for d in train_domains]
        test_paths = (
            [os.path.join(data_root, f"Domain{d}/train/") for d in test_domains] +
            [os.path.join(data_root, f"Domain{d}/test/") for d in test_domains]
        )
    elif is_poly_task(task_name):
        train_paths = [os.path.join(data_root, f"Domain{d}/split_dataset/train/") for d in train_domains]
        val_paths = [os.path.join(data_root, f"Domain{d}/split_dataset/test/") for d in train_domains]
        test_paths = (
            [os.path.join(data_root, f"Domain{d}/split_dataset/train/") for d in test_domains] +
            [os.path.join(data_root, f"Domain{d}/split_dataset/test/") for d in test_domains]
        )
    else:
        train_paths = [os.path.join(data_root, f"Domain{d}/train/ROIs/") for d in train_domains]
        val_paths = [os.path.join(data_root, f"Domain{d}/test/ROIs/") for d in train_domains]
        test_paths = (
            [os.path.join(data_root, f"Domain{d}/train/ROIs/") for d in test_domains] +
            [os.path.join(data_root, f"Domain{d}/test/ROIs/") for d in test_domains]
        )
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


# -----------------------------
# 参数
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--tr_npy_path", type=str, default="./data/Fundus/Domain1/train/ROIs/")
parser.add_argument("-j", "--ts_npy_path", type=str, default="./data/Fundus/Domain1/test/ROIs/")
parser.add_argument("--data_root", type=str, default="./data/Fundus/")
parser.add_argument("--trainid", type=str, default="1,2,3")
parser.add_argument("--testid", type=str, default="4")
parser.add_argument("--task_name", type=str, default="SAM2-ViT-B")
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--iternum", type=int, default=20000)

# test 阶段用来指向最佳模型目录
parser.add_argument("--checkpoint", type=str, default="sam2_vit_b.pth")

# SAM2 权重路径
parser.add_argument("--sam2_ckpt", type=str, default="./sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt")

parser.add_argument("--load_pretrain", type=bool, default=True)
parser.add_argument("--pretrain_model_path", type=str, default="")
parser.add_argument("--work_dir", type=str, default="./work_dir")

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup_epochs", type=int, default=5)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--grad_clip", type=float, default=1.0)

parser.add_argument("--use_wandb", type=bool, default=False)
parser.add_argument("--use_amp", action="store_true", default=False)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()


# -----------------------------
# wandb
# -----------------------------
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


# -----------------------------
# 路径
# -----------------------------
run_id = datetime.now().strftime("%Y%m%d-%H%M")

if is_na_task(args.task_name):
    model_save_path = join(
        args.work_dir,
        "naso_auto",
        args.task_name + f"-domain{args.trainid}_{args.iternum}",
        run_id,
    )
else:
    model_save_path = join(
        args.work_dir,
        "fundus_auto",
        args.task_name + f"-domain{args.trainid}_{args.iternum}",
        run_id,
    )

device = torch.device(args.device)

log_dir = join(
    args.work_dir,
    "logs",
    args.task_name,
    f"domain{args.trainid}_{args.iternum}",
    run_id,
)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


# -----------------------------
# 模型
# -----------------------------
class SamFeatSeg(nn.Module):
    def __init__(self, image_encoder, mask_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder

    def forward(self, image):
        original_size = image.shape[2:]
        
        # SAM2图像编码器返回字典，提取vision_features
        encoder_output = self.image_encoder(image)
        image_embedding = encoder_output["vision_features"]
        
        # 如果vision_features是(B, H*W, C)格式，重塑为(B, C, H, W)
        if len(image_embedding.shape) == 3:
            b, n, c = image_embedding.shape
            h = w = int(n**0.5)
            image_embedding = image_embedding.permute(0, 2, 1).reshape(b, c, h, w)
        
        low_res_masks = self.mask_decoder(image_embedding)
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        ori_res_masks = F.softmax(ori_res_masks, dim=1)
        return ori_res_masks


def setup_optimizer_scheduler(model, args):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
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


def setup_dataloaders(train_domains, test_domains, data_root, task_name, batch_size, num_workers):
    train_paths, val_paths, _ = get_task_paths(train_domains, test_domains, data_root, task_name)
    train_datasets, val_datasets, _ = build_datasets(train_paths, val_paths, [], task_name)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    generator = torch.Generator(device="cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
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
            domain_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=generator,
        )

    return train_loader, val_loader, test_loaders, test_dataset_sizes



def setup_sam2_image_encoder(ckpt_path, device):
    """设置SAM2图像编码器"""
    print(f"=== Loading SAM2 Model ===")
    print(f"Checkpoint path: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found at: {ckpt_path}")
    
    # 加载检查点
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Invalid checkpoint format. Expected dict with 'model' key.")
    
    model_state_dict = checkpoint["model"]
    
    # 构建完整的SAM2图像编码器
    print("\nBuilding complete SAM2 image encoder...")
    
    # 从权重推断参数
    patch_embed_key = "image_encoder.trunk.patch_embed.proj.weight"
    if patch_embed_key in model_state_dict:
        weight = model_state_dict[patch_embed_key]
        embed_dim = weight.shape[0]
        kernel_size = weight.shape[2]
        print(f"Inferred from patch_embed weight: embed_dim={embed_dim}, kernel_size={kernel_size}")
    else:
        embed_dim = 112
    
    # 构建Hiera骨干网络
    print(f"Building Hiera backbone with embed_dim={embed_dim}")
    
    # 构建Hiera骨干网络
    trunk = Hiera(
        embed_dim=embed_dim,
        num_heads=1,
        drop_path_rate=0.0,
        q_pool=3,
        q_stride=(2, 2),
        stages=(2, 3, 16, 3),  # hiera_base_plus的默认配置
        dim_mul=2.0,
        head_mul=2.0,
        window_pos_embed_bkg_spatial_size=(14, 14),
        window_spec=(8, 4, 14, 7),
        global_att_blocks=(12, 16, 20),
        weights_path=None,
        return_interm_layers=True,
    )
    
    print("✓ Created Hiera trunk")
    print(f"Trunk channel list: {trunk.channel_list}")
    
    # 构建位置编码
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=128,
        normalize=True,
    )
    
    # 构建FPN颈部
    backbone_channel_list = trunk.channel_list
    print(f"Using trunk channel list for neck: {backbone_channel_list}")
    
    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=backbone_channel_list,
        kernel_size=1,
        stride=1,
        padding=0,
        fpn_interp_model="bilinear",
        fuse_type="sum",
    )
    
    # 构建图像编码器
    image_encoder = ImageEncoder(
        trunk=trunk,
        neck=neck,
        scalp=0,
    )
    
    # 提取图像编码器权重
    image_encoder_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith("image_encoder."):
            new_key = k[14:]  # 移除"image_encoder."前缀
            image_encoder_state_dict[new_key] = v
    
    if not image_encoder_state_dict:
        raise ValueError("No image encoder weights found")
    
    print(f"Extracted {len(image_encoder_state_dict)} image encoder weights")
    
    # 加载权重
    missing_keys, unexpected_keys = image_encoder.load_state_dict(
        image_encoder_state_dict, strict=False
    )
    
    print(f"✓ Loaded image encoder weights")
    print(f"  Missing keys: {len(missing_keys)}")
    print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys and len(missing_keys) < 10:
        print(f"  First 5 missing keys: {missing_keys[:5]}")
    if unexpected_keys and len(unexpected_keys) < 10:
        print(f"  First 5 unexpected keys: {unexpected_keys[:5]}")
    
    return image_encoder


def setup_model_and_optimizer(args, device):
    num_classes = get_num_classes(args.task_name)

    seg_decoder = DPTDecoderForSAM(
        num_classes=num_classes,
        embed_dim=256,
        features=128,
        out_channels=[256, 512, 1024, 2048],
        use_bn=True,
    )

    if is_sam2_task(args.task_name):
        image_encoder = setup_sam2_image_encoder(args.sam2_ckpt, device)
    else:
        raise ValueError(f"Current script expects a SAM2 task, but got task_name={args.task_name}")

    for param in image_encoder.parameters():
        param.requires_grad = False

    if "lora" in args.task_name.lower():
        replace_linear_with_lora(image_encoder)

    medsam_model = SamFeatSeg(
        image_encoder=image_encoder,
        mask_decoder=seg_decoder,
    ).to(device)

    return medsam_model


# -----------------------------
# 训练 / 验证 / 测试
# -----------------------------
def train_epoch(model, dataloader, optimizer, device, iter_num, log_interval, writer, args, scaler=None):
    model.train()
    num_classes = get_num_classes(args.task_name)
    num_fg = num_classes - 1

    epoch_loss = 0.0
    epoch_dice = torch.zeros(num_fg)
    epoch_iou = torch.zeros(num_fg)

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        image, gt2D, _ = unpack_batch(batch)
        image, gt2D = image.to(device), gt2D.to(device)
        gt2D = normalize_target(gt2D, num_classes)

        if args.use_amp and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                medsam_pred = model(image)
                multi_dice = multi_class_dice(medsam_pred, gt2D, num_classes)

                if num_classes == 2:
                    loss = 1 - multi_dice[0].mean()
                else:
                    # 归一化加权 Dice： (1 * disc + 2 * cup) / 3
                    loss = 1 - ((1.0 * multi_dice[0] + 2.0 * multi_dice[1]) / 3.0).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            medsam_pred = model(image)
            multi_dice = multi_class_dice(medsam_pred, gt2D, num_classes)

            if num_classes == 2:
                loss = 1 - multi_dice[0].mean()
            else:
                # 归一化加权 Dice： (1 * disc + 2 * cup) / 3
                loss = 1 - ((1.0 * multi_dice[0] + 2.0 * multi_dice[1]) / 3.0).mean()

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=args.grad_clip,
                )
            optimizer.step()

        epoch_loss += loss.item()

        pred = probs_to_onehot(medsam_pred)
        dice_score = multi_class_dice(pred, gt2D, num_classes)
        iou_score = multi_class_iou(pred, gt2D, num_classes)

        for i in range(num_fg):
            epoch_dice[i] += dice_score[i, :].mean().item()
            epoch_iou[i] += iou_score[i, :].mean().item()

        iter_num += 1

        if iter_num % log_interval == 0:
            writer.add_scalar("Train/Loss", loss.item(), iter_num)
            for i in range(num_fg):
                writer.add_scalar(f"Train/Dice_{i}", dice_score[i, :].mean().item(), iter_num)
                writer.add_scalar(f"Train/IoU_{i}", iou_score[i, :].mean().item(), iter_num)

            if num_classes == 2:
                fg_dice = dice_score[0, :].mean().item()
                fg_iou = iou_score[0, :].mean().item()
                print(
                    f"[Iter {iter_num:5d}] Loss: {loss.item():.4f} | "
                    f"Foreground Dice: {fg_dice:.4f} | Foreground IoU: {fg_iou:.4f}"
                )
            else:
                disc_dice = dice_score[0, :].mean().item()
                cup_dice = dice_score[1, :].mean().item()
                disc_iou = iou_score[0, :].mean().item()
                cup_iou = iou_score[1, :].mean().item()
                mean_dice = (disc_dice + cup_dice) / 2
                mean_iou = (disc_iou + cup_iou) / 2
                print(
                    f"[Iter {iter_num:5d}] Loss: {loss.item():.4f} | "
                    f"Disc Dice: {disc_dice:.4f} | Cup Dice: {cup_dice:.4f} | Mean Dice: {mean_dice:.4f} | "
                    f"Disc IoU: {disc_iou:.4f} | Cup IoU: {cup_iou:.4f} | Mean IoU: {mean_iou:.4f}"
                )

    epoch_loss /= len(dataloader)
    epoch_dice /= len(dataloader)
    epoch_iou /= len(dataloader)

    return epoch_loss, epoch_dice, epoch_iou, iter_num


def validate(model, dataloader, device, writer, epoch, args):
    model.eval()
    num_classes = get_num_classes(args.task_name)
    num_fg = num_classes - 1

    val_loss = 0.0
    epoch_val_dice = torch.zeros(num_fg)
    epoch_val_iou = torch.zeros(num_fg)
    epoch_val_assd = torch.zeros(num_fg)  # 新增ASSD

    with torch.no_grad():
        for val_step, batch in enumerate(dataloader):
            image, gt2D, _ = unpack_batch(batch)
            image, gt2D = image.to(device), gt2D.to(device)
            gt2D = normalize_target(gt2D, num_classes)

            medsam_pred = model(image)
            val_dice_probs = multi_class_dice(medsam_pred, gt2D, num_classes)

            if num_classes == 2:
                loss = 1 - val_dice_probs[0].mean()
            else:
                # 归一化加权 Dice： (1 * disc + 2 * cup) / 3
                loss = 1 - ((1.0 * val_dice_probs[0] + 2.0 * val_dice_probs[1]) / 3.0).mean()

            val_loss += loss.item()

            pred = probs_to_onehot(medsam_pred)
            val_dice = multi_class_dice(pred, gt2D, num_classes)
            val_iou = multi_class_iou(pred, gt2D, num_classes)
            
            # 计算ASSD（每10个batch计算一次，避免太慢）
            if MEDPY_AVAILABLE and val_step % 1 == 0:
                val_assd_batch = multi_class_assd(pred, gt2D, num_classes)
                for i in range(num_fg):
                    if val_assd_batch.size > 0:
                        epoch_val_assd[i] += val_assd_batch[i, :].mean()

            for i in range(num_fg):
                epoch_val_dice[i] += val_dice[i, :].mean().item()
                epoch_val_iou[i] += val_iou[i, :].mean().item()

    denom = len(dataloader)
    val_loss /= denom
    epoch_val_dice /= denom
    epoch_val_iou /= denom
    epoch_val_assd /= (denom // 10) if denom > 10 else 1  # 调整分母

    val_dice_mean = epoch_val_dice.mean().item()
    val_iou_mean = epoch_val_iou.mean().item()
    val_assd_mean = epoch_val_assd.mean().item() if num_fg > 0 else 0.0

    print(f"[Validation Epoch {epoch}] Loss: {val_loss:.4f}")
    

    writer.add_scalar("Validation/Loss", val_loss, epoch)
    for i in range(num_fg):
        writer.add_scalar(f"Validation/Dice_{i}", epoch_val_dice[i], epoch)
        writer.add_scalar(f"Validation/IoU_{i}", epoch_val_iou[i], epoch)
        writer.add_scalar(f"Validation/ASSD_{i}", epoch_val_assd[i], epoch)  # 新增
    writer.add_scalar("Validation/Dice_mean", val_dice_mean, epoch)
    writer.add_scalar("Validation/IoU_mean", val_iou_mean, epoch)
    writer.add_scalar("Validation/ASSD_mean", val_assd_mean, epoch)  # 新增

    return val_loss, epoch_val_dice, epoch_val_iou, epoch_val_assd, val_dice_mean, val_iou_mean, val_assd_mean  # 修改返回值


def test(model, dataloader, device, writer, args):
    model.eval()
    num_classes = get_num_classes(args.task_name)
    num_fg = num_classes - 1

    with torch.no_grad():
        total_dice = torch.zeros(num_fg)
        total_iou = torch.zeros(num_fg)
        total_assd = torch.zeros(num_fg)  # 新增ASSD
        dice_scores = [[] for _ in range(num_fg)]
        iou_scores = [[] for _ in range(num_fg)]
        assd_scores = [[] for _ in range(num_fg)]  # 新增ASSD

        num_batches = len(dataloader)

        for step, batch in enumerate(dataloader):
            image, gt2D, _ = unpack_batch(batch)
            image, gt2D = image.to(device), gt2D.to(device)
            gt2D = normalize_target(gt2D, num_classes)

            medsam_pred = model(image)
            pred = probs_to_onehot(medsam_pred)

            test_dice = multi_class_dice(pred, gt2D, num_classes)
            test_iou = multi_class_iou(pred, gt2D, num_classes)
            
            # 计算ASSD
            if MEDPY_AVAILABLE:
                test_assd = multi_class_assd(pred, gt2D, num_classes)

            for i in range(num_fg):
                v_d = test_dice[i, :].mean().item()
                v_i = test_iou[i, :].mean().item()
                total_dice[i] += v_d
                total_iou[i] += v_i
                dice_scores[i].append(v_d)
                iou_scores[i].append(v_i)
                
                # 添加ASSD
                if MEDPY_AVAILABLE and test_assd.size > 0:
                    v_a = test_assd[i, :].mean().item()
                    total_assd[i] += v_a
                    assd_scores[i].append(v_a)

            if step % 10 == 0:
                image1 = F.interpolate(
                    image[0].unsqueeze(0),
                    size=(800, 800),
                    mode="bilinear",
                    align_corners=False,
                )
                visualize_on_tensorboard(image1[0], gt2D[0], pred[0], step, writer, f"Test_Step_{step}", phase="Test")

        avg_dice = total_dice / num_batches
        std_dice = torch.tensor([torch.std(torch.tensor(v)).item() for v in dice_scores])
        avg_iou = total_iou / num_batches
        std_iou = torch.tensor([torch.std(torch.tensor(v)).item() for v in iou_scores])
        avg_assd = total_assd / num_batches if MEDPY_AVAILABLE else torch.zeros(num_fg)  # 新增
        std_assd = torch.tensor([torch.std(torch.tensor(v)).item() for v in assd_scores]) if MEDPY_AVAILABLE else torch.zeros(num_fg)  # 新增

        for i in range(num_fg):
            writer.add_scalar(f"Test/Average_Dice_{i}", avg_dice[i].item())
            writer.add_scalar(f"Test/Std_Dice_{i}", std_dice[i].item())
            writer.add_scalar(f"Test/Average_IoU_{i}", avg_iou[i].item())
            writer.add_scalar(f"Test/Std_IoU_{i}", std_iou[i].item())
            if MEDPY_AVAILABLE:
                writer.add_scalar(f"Test/Average_ASSD_{i}", avg_assd[i].item())  # 新增
                writer.add_scalar(f"Test/Std_ASSD_{i}", std_assd[i].item())  # 新增

        print("\n=== Test Results ===")
        if num_classes == 2:
            print(f"Foreground Dice: {avg_dice[0]:.4f} ± {std_dice[0]:.4f}")
            print(f"Foreground IoU: {avg_iou[0]:.4f} ± {std_iou[0]:.4f}")
            if MEDPY_AVAILABLE:
                print(f"Foreground ASSD: {avg_assd[0]:.4f} ± {std_assd[0]:.4f}")
        else:
            print(f"Disc Dice: {avg_dice[0]:.4f} ± {std_dice[0]:.4f}")
            print(f"Cup Dice: {avg_dice[1]:.4f} ± {std_dice[1]:.4f}")
            print(f"Disc IoU: {avg_iou[0]:.4f} ± {std_iou[0]:.4f}")
            print(f"Cup IoU: {avg_iou[1]:.4f} ± {std_iou[1]:.4f}")
            if MEDPY_AVAILABLE:
                print(f"Disc ASSD: {avg_assd[0]:.4f} ± {std_assd[0]:.4f}")
                print(f"Cup ASSD: {avg_assd[1]:.4f} ± {std_assd[1]:.4f}")

        if num_classes == 2:
            return avg_dice[0], std_dice[0], 0, 0, avg_iou[0], std_iou[0], 0, 0, avg_assd[0].item() if MEDPY_AVAILABLE else 0, std_assd[0].item() if MEDPY_AVAILABLE else 0, 0, 0
        return (
            avg_dice[0], std_dice[0],
            avg_dice[1], std_dice[1],
            avg_iou[0], std_iou[0],
            avg_iou[1], std_iou[1],
            avg_assd[0].item() if MEDPY_AVAILABLE else 0, std_assd[0].item() if MEDPY_AVAILABLE else 0,  # 新增ASSD
            avg_assd[1].item() if MEDPY_AVAILABLE else 0, std_assd[1].item() if MEDPY_AVAILABLE else 0,  # 新增ASSD
        )

def test_per_domain(model, test_loaders, device, writer, args):
    all_domain_results = {}

    for domain_id, loader in test_loaders.items():
        print(f"\n=== Testing on Domain{domain_id} ===")
        result = test(model, loader, device, writer, args)

        if get_num_classes(args.task_name) == 2:
            avg_dice0, std_dice0, _, _, avg_iou0, std_iou0, _, _, avg_assd0, std_assd0, _, _ = result
            all_domain_results[domain_id] = {
                "fg_dice_mean": float(avg_dice0),
                "fg_dice_std": float(std_dice0),
                "fg_iou_mean": float(avg_iou0),
                "fg_iou_std": float(std_iou0),
                "fg_assd_mean": float(avg_assd0) if MEDPY_AVAILABLE else 0.0,  # 新增
                "fg_assd_std": float(std_assd0) if MEDPY_AVAILABLE else 0.0,   # 新增
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
                "disc_assd_mean": float(avg_assd0) if MEDPY_AVAILABLE else 0.0,  # 新增
                "disc_assd_std": float(std_assd0) if MEDPY_AVAILABLE else 0.0,   # 新增
                "cup_assd_mean": float(avg_assd1) if MEDPY_AVAILABLE else 0.0,   # 新增
                "cup_assd_std": float(std_assd1) if MEDPY_AVAILABLE else 0.0,    # 新增
            }

    return all_domain_results

# -----------------------------
# 主函数
# -----------------------------
def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, join(model_save_path, run_id + "_" + os.path.basename(__file__)))

    device = torch.device(args.device)

    train_domains = [int(x) for x in args.trainid.split(",")]
    test_domains = [int(x) for x in args.testid.split(",")]

    print(f"Training domains: {train_domains}")
    print(f"Testing domains: {test_domains}")

    train_loader, val_loader, test_loaders, test_dataset_sizes = setup_dataloaders(
        train_domains,
        test_domains,
        args.data_root,
        args.task_name,
        args.batch_size,
        args.num_workers,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Total test samples: {sum(test_dataset_sizes.values())}")
    for domain_id in test_domains:
        print(f"Domain{domain_id} test samples: {test_dataset_sizes[domain_id]}")

    # -------------------------
    # 1) 构建模型
    # -------------------------
    medsam_model = setup_model_and_optimizer(args, device)
    medsam_model.train()

    total_params = sum(p.numel() for p in medsam_model.parameters())
    trainable_params = sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params / total_params:.2%}")

    optimizer, scheduler = setup_optimizer_scheduler(medsam_model, args)

    print("\n=== Learning Rate Strategy ===")
    print(f"Initial learning rate: {args.lr}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Minimum learning rate: {args.min_lr}")
    print(f"Total epochs: {args.num_epochs}")
    print(f"Gradient clipping: {args.grad_clip}")
    # print(f"Loss = {ce_weight:.2f} * CE + {dice_weight:.2f} * Dice")

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    num_epochs = args.num_epochs
    iter_num = 0
    log_interval = 100

    num_classes = get_num_classes(args.task_name)
    num_fg = num_classes - 1

    best_val_dice = torch.zeros(num_fg)
    best_val_iou = torch.zeros(num_fg)
    best_val_assd = torch.zeros(num_fg)  # 新增ASSD
    best_val_epoch = torch.zeros(num_fg, dtype=torch.long)

    best_val_mean_dice = -1.0
    best_val_mean_dice_epoch = 0
    best_val_mean_assd = float('inf')  # 新增ASSD，越小越好
    best_val_mean_assd_epoch = 0  # 新增ASSD
    best_model_path = join(model_save_path, "_best_model.pth")

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Resumed from epoch {start_epoch}")

    print(f"\n=== Starting training for {num_epochs} epochs ===")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # -------------------------
        # Train
        # -------------------------
        epoch_loss, epoch_dice, epoch_iou, iter_num = train_epoch(
            medsam_model,
            train_loader,
            optimizer,
            device,
            iter_num,
            log_interval,
            writer,
            args,
            scaler,
        )

        scheduler.step()

        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch + 1)
        writer.add_scalar("Epoch/Loss_total", epoch_loss, epoch + 1)
        
        for i in range(num_fg):
            writer.add_scalar(f"Epoch/Dice_{i}", epoch_dice[i].item(), epoch + 1)
            writer.add_scalar(f"Epoch/IoU_{i}", epoch_iou[i].item(), epoch + 1)

        epoch_mean_dice = epoch_dice.mean().item()
        epoch_mean_iou = epoch_iou.mean().item()

        print(
            f"[Train Epoch {epoch + 1}] "
            f"Total Loss: {epoch_loss:.4f}"
        )

        if num_classes == 2:
            print(
                f"  Foreground Dice: {epoch_dice[0].item():.4f}, "
                f"Foreground IoU: {epoch_iou[0].item():.4f}"
            )
        else:
            print(
                f"  Disc Dice: {epoch_dice[0].item():.4f}, "
                f"Cup Dice: {epoch_dice[1].item():.4f}, Mean Dice: {epoch_mean_dice:.4f}"
            )
            print(
                f"  Disc IoU: {epoch_iou[0].item():.4f}, "
                f"Cup IoU: {epoch_iou[1].item():.4f}, Mean IoU: {epoch_mean_iou:.4f}"
            )

        # -------------------------
        # Validate
        # -------------------------
        print("Validating...")
        val_loss, val_dice, val_iou, val_assd, val_dice_mean, val_iou_mean, val_assd_mean = validate(
            medsam_model,
            val_loader,
            device,
            writer,
            epoch + 1,
            args,
        )

        print(
            f"[Val Epoch {epoch + 1}] "
            f"Total Loss: {val_loss:.4f}"
        )

        if num_classes == 2:
            print(
                f"  Foreground Dice: {val_dice[0].item():.4f}, "
                f"Foreground IoU: {val_iou[0].item():.4f},"
                f"Foreground ASSD: {val_assd[0].item():.4f},"
            )
        else:
            print(
                f"  Disc Dice: {val_dice[0].item():.4f}, "
                f"Cup Dice: {val_dice[1].item():.4f}, Mean Dice: {val_dice_mean:.4f}"
            )
            print(
                f"  Disc IoU: {val_iou[0].item():.4f}, "
                f"Cup IoU: {val_iou[1].item():.4f}, Mean IoU: {val_iou_mean:.4f}"
            )
            print(
                f"  Disc ASSD: {val_assd[0].item():.4f}, "
                f"Cup ASSD: {val_assd[1].item():.4f}, Mean ASSD: {val_assd_mean:.4f}"
            )

        # 更新单类最佳指标
        for i in range(num_fg):
            if val_dice[i] > best_val_dice[i]:
                best_val_dice[i] = val_dice[i]
                best_val_epoch[i] = epoch + 1

            if val_iou[i] > best_val_iou[i]:
                best_val_iou[i] = val_iou[i]
            if val_assd[i] < best_val_assd[i]:  # ASSD越小越好
                best_val_assd[i] = val_assd[i]

        # 更新最佳均值 Dice 模型
        if val_dice_mean > best_val_mean_dice:
            best_val_mean_dice = val_dice_mean
            best_val_mean_dice_epoch = epoch + 1
            torch.save(medsam_model.state_dict(), best_model_path)
            print(f"Saved new best model with Mean Dice: {val_dice_mean:.4f}")
        if val_assd_mean < best_val_mean_assd:  # ASSD越小越好
            best_val_mean_assd = val_assd_mean
            best_val_mean_assd_epoch = epoch + 1
        # 保存 latest checkpoint
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))

    print("\n=== Training completed ===")

    if num_classes == 2:
        print(f"Best validation Foreground Dice: {best_val_dice[0]:.4f} at epoch {best_val_epoch[0]}")
        print(f"Best validation Foreground IoU: {best_val_iou[0]:.4f}")
        print(f"Best validation Mean Dice: {best_val_mean_dice:.4f} at epoch {best_val_mean_dice_epoch}")
        print(f"Best validation Foreground ASSD: {best_val_assd[0]:.4f}")  # 新增
    else:
        print("Best validation results:")
        print(f"  Disc Dice: {best_val_dice[0]:.4f} at epoch {best_val_epoch[0]}")
        print(f"  Cup Dice: {best_val_dice[1]:.4f} at epoch {best_val_epoch[1]}")
        print(f"  Disc IoU: {best_val_iou[0]:.4f}")
        print(f"  Cup IoU: {best_val_iou[1]:.4f}")
        print(f"  Mean Dice: {best_val_mean_dice:.4f} at epoch {best_val_mean_dice_epoch}")
        print(f"  Disc ASSD: {best_val_assd[0]:.4f}")  # 新增
        print(f"  Cup ASSD: {best_val_assd[1]:.4f}")  # 新增
    # -------------------------
    # 2) 加载 best model 做测试
    # -------------------------
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}, skip testing.")
        return

    print(f"\n=== Loading best model for testing: {best_model_path} ===")
    medsam_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
    medsam_model.eval()

    all_domain_results = test_per_domain(medsam_model, test_loaders, device, writer, args)

    print("\n=== Final Per-Domain Test Results ===")
    for domain_id, result in all_domain_results.items():
        print(f"\n[Domain{domain_id}]")
        if get_num_classes(args.task_name) == 2:
            print(f"Foreground Dice: {result['fg_dice_mean']:.4f} ± {result['fg_dice_std']:.4f}")
            print(f"Foreground IoU:  {result['fg_iou_mean']:.4f} ± {result['fg_iou_std']:.4f}")
        else:
            print(f"Disc Dice: {result['disc_dice_mean']:.4f} ± {result['disc_dice_std']:.4f}")
            print(f"Cup Dice:  {result['cup_dice_mean']:.4f} ± {result['cup_dice_std']:.4f}")
            print(f"Disc IoU:  {result['disc_iou_mean']:.4f} ± {result['disc_iou_std']:.4f}")
            print(f"Cup IoU:   {result['cup_iou_mean']:.4f} ± {result['cup_iou_std']:.4f}")

if __name__ == "__main__":
    main()