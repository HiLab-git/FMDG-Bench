import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import random
import glob  # 新增：segdinov3中用到的库

# 添加DINOv2路径（保留原路径）
sys.path.append('./dino3/SegDINO/dinov2')

# ===================== 替换1：segdinov3的可视化/指标计算函数 =====================
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
        # 处理多类别预测（segdinov3的多类别逻辑）
        if logits.shape[1] > 1:  # 多类别
            pred = torch.argmax(logits[i], dim=0, keepdim=True)  # 获取预测类别
            # 创建两个二值掩码
            pred_disc = (pred == 1).float()  # 视盘
            pred_cup = (pred == 2).float()   # 视杯
            gt_disc = targets[i, 1]  # 视盘真实标签
            gt_cup = targets[i, 2]   # 视杯真实标签
            
            # 保存视盘预测
            base_disc = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}_disc")
            cv2.imwrite(base_disc + "_img.png", img_bgr)
            cv2.imwrite(base_disc + "_pred.png", (pred_disc[0].cpu().numpy() * 255).astype(np.uint8))
            cv2.imwrite(base_disc + "_gt.png", (gt_disc.cpu().numpy() * 255).astype(np.uint8))
            
            # 保存视杯预测
            base_cup = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}_cup")
            cv2.imwrite(base_cup + "_img.png", img_bgr)
            cv2.imwrite(base_cup + "_pred.png", (pred_cup[0].cpu().numpy() * 255).astype(np.uint8))
            cv2.imwrite(base_cup + "_gt.png", (gt_cup.cpu().numpy() * 255).astype(np.uint8))
        else:  # 二分类（保留兼容逻辑）
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
    if logits.dim() > 2 and logits.shape[0] > 1:  # 多类别
        pred = torch.argmax(logits, dim=0, keepdim=True)  # 获取预测类别
        # 创建两个二值掩码
        pred_disc = (pred == 1).float()  # 视盘
        pred_cup = (pred == 2).float()   # 视杯
        gt_disc = targets[1]  # 视盘真实标签
        gt_cup = targets[2]   # 视杯真实标签
        
        # 保存视盘预测
        base_disc = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}_disc")
        cv2.imwrite(base_disc + "_img.png", img_bgr)
        cv2.imwrite(base_disc + "_pred.png", (pred_disc[0].cpu().numpy() * 255).astype(np.uint8))
        cv2.imwrite(base_disc + "_gt.png", (gt_disc.cpu().numpy() * 255).astype(np.uint8))
        
        # 保存视杯预测
        base_cup = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}_cup")
        cv2.imwrite(base_cup + "_img.png", img_bgr)
        cv2.imwrite(base_cup + "_pred.png", (pred_cup[0].cpu().numpy() * 255).astype(np.uint8))
        cv2.imwrite(base_cup + "_gt.png", (gt_cup.cpu().numpy() * 255).astype(np.uint8))
    else:  # 二分类
        pred_gray = mask_to_gray(logits, thr)
        gt_gray = mask_to_gray(targets, thr)
        base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
        cv2.imwrite(base + "_img.png", img_bgr)
        cv2.imwrite(base + "_pred.png", pred_gray)
        cv2.imwrite(base + "_gt.png", gt_gray)

def dice_coefficient(pred_mask, gt_mask, eps=1e-6):
    """计算Dice系数（segdinov3版本）"""
    pred_mask_flat = pred_mask.view(-1)
    gt_mask_flat = gt_mask.view(-1)
    intersection = torch.sum(pred_mask_flat * gt_mask_flat)
    union = torch.sum(pred_mask_flat) + torch.sum(gt_mask_flat)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.item()

def iou_coefficient(pred_mask, gt_mask, eps=1e-6):
    """计算IoU系数（segdinov3版本）"""
    pred_mask_flat = pred_mask.view(-1)
    gt_mask_flat = gt_mask.view(-1)
    intersection = torch.sum(pred_mask_flat * gt_mask_flat)
    union = torch.sum(pred_mask_flat) + torch.sum(gt_mask_flat) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.item()

def multi_class_dice(pred, target, num_classes, eps=1e-6):
    """计算多类别Dice系数（segdinov3核心函数）"""
    dice_scores = []
    for c in range(1, num_classes):  # 跳过背景
        pred_class = pred[:, c, :, :]
        target_class = target[:, c, :, :]
        intersection = (pred_class * target_class).sum(dim=[1, 2])
        union = pred_class.sum(dim=[1, 2]) + target_class.sum(dim=[1, 2])
        dice = (2. * intersection + eps) / (union + eps)
        dice_scores.append(dice)
    return torch.stack(dice_scores)  # shape: [num_classes-1, batch]

def multi_class_iou(pred, target, num_classes, eps=1e-6):
    """计算多类别IoU系数（segdinov3核心函数）"""
    iou_scores = []
    for c in range(1, num_classes):  # 跳过背景
        pred_class = pred[:, c, :, :]
        target_class = target[:, c, :, :]
        intersection = (pred_class * target_class).sum(dim=[1, 2])
        union = pred_class.sum(dim=[1, 2]) + target_class.sum(dim=[1, 2]) - intersection
        iou = (intersection + eps) / (union + eps)
        iou_scores.append(iou)
    return torch.stack(iou_scores)  # shape: [num_classes-1, batch]

# ===================== 替换2：segdinov3的训练函数 =====================
def train_one_epoch(model, train_loader, optimizer, device, num_classes=3, dice_thr=0.5, vis_dir=None, epoch=0):
    model.train()
    total_loss = 0.0
    dice_scores_disc = []  # 视盘Dice
    dice_scores_cup = []   # 视杯Dice
    iou_scores_disc = []   # 视盘IoU
    iou_scores_cup = []    # 视杯IoU
    
    # 替换为CrossEntropyLoss（多类别）
    criterion = nn.CrossEntropyLoss()
    
    first_batch_logged = False
    pbar = tqdm(train_loader, desc=f"[Train e{epoch}]")
    
    for step, (inputs, targets, _) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 获取单通道标签（用于CrossEntropyLoss）
        if targets.dim() == 4 and targets.shape[1] == 3:
            target_labels = torch.argmax(targets, dim=1).long()  # [B, H, W]
        else:
            target_labels = targets.squeeze(1).long()
        
        # 维度校验（segdinov3新增）
        assert target_labels.dim() == 3, f"target_labels维度错误，期望3D，实际{target_labels.dim()}D，shape={target_labels.shape}"
        assert inputs.shape[2:] == target_labels.shape[1:], f"输入图像尺寸{inputs.shape[2:]}与标签尺寸{target_labels.shape[1:]}不匹配"
        
        optimizer.zero_grad()
        logits = model(inputs)  # 输出应该是[B, 3, H, W]
        
        # 计算损失
        loss = criterion(logits, target_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            # 应用softmax获取概率
            probs = torch.softmax(logits, dim=1)
            
            # 计算每个类别的Dice和IoU
            dice_scores = multi_class_dice(probs, targets, num_classes=num_classes)
            iou_scores = multi_class_iou(probs, targets, num_classes=num_classes)
            
            # 获取每个batch的视盘和视杯指标
            disc_dice = dice_scores[0, :].mean().item()  # 第一个前景类别
            cup_dice = dice_scores[1, :].mean().item()   # 第二个前景类别
            disc_iou = iou_scores[0, :].mean().item()
            cup_iou = iou_scores[1, :].mean().item()
            
            dice_scores_disc.append(disc_dice)
            dice_scores_cup.append(cup_dice)
            iou_scores_disc.append(disc_iou)
            iou_scores_cup.append(cup_iou)
        
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            disc_dice=f"{disc_dice:.4f}",
            cup_dice=f"{cup_dice:.4f}",
            disc_iou=f"{disc_iou:.4f}",
            cup_iou=f"{cup_iou:.4f}"
        )
        
        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals(epoch, inputs, probs, targets, out_dir=vis_dir, max_save=8, thr=dice_thr)
            first_batch_logged = True
    
    avg_loss = total_loss / max(1, len(train_loader))
    avg_dice_disc = float(np.mean(dice_scores_disc)) if dice_scores_disc else 0.0
    avg_dice_cup = float(np.mean(dice_scores_cup)) if dice_scores_cup else 0.0
    avg_iou_disc = float(np.mean(iou_scores_disc)) if iou_scores_disc else 0.0
    avg_iou_cup = float(np.mean(iou_scores_cup)) if iou_scores_cup else 0.0
    
    print(f"[Train Epoch {epoch}] loss={avg_loss:.4f}")
    print(f"  Disc: Dice={avg_dice_disc:.4f}, IoU={avg_iou_disc:.4f}")
    print(f"  Cup:  Dice={avg_dice_cup:.4f}, IoU={avg_iou_cup:.4f}")
    
    return avg_loss, avg_dice_disc, avg_dice_cup, avg_iou_disc, avg_iou_cup

# ===================== 替换3：segdinov3的评估函数 =====================
@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=3, dice_thr=0.5, vis_dir=None):
    model.eval()
    total_loss = 0.0
    dice_scores_disc = []
    dice_scores_cup = []
    iou_scores_disc = []
    iou_scores_cup = []
    
    criterion = nn.CrossEntropyLoss()
    idx_global = 0
    pbar = tqdm(val_loader, desc="[Eval]")
    
    for (inputs, targets, _) in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 获取单通道标签
        if targets.dim() == 4 and targets.shape[1] == 3:
            target_labels = torch.argmax(targets, dim=1).long()
        else:
            target_labels = targets.squeeze(1).long()
        
        logits = model(inputs)
        loss = criterion(logits, target_labels)
        total_loss += loss.item()
        
        # 应用softmax获取概率
        probs = torch.softmax(logits, dim=1)
        
        # 计算每个类别的Dice和IoU
        dice_scores = multi_class_dice(probs, targets, num_classes=num_classes)
        iou_scores = multi_class_iou(probs, targets, num_classes=num_classes)
        
        # 获取每个batch的视盘和视杯指标
        disc_dice = dice_scores[0, :].mean().item()
        cup_dice = dice_scores[1, :].mean().item()
        disc_iou = iou_scores[0, :].mean().item()
        cup_iou = iou_scores[1, :].mean().item()
        
        dice_scores_disc.append(disc_dice)
        dice_scores_cup.append(cup_dice)
        iou_scores_disc.append(disc_iou)
        iou_scores_cup.append(cup_iou)
        
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            disc_dice=f"{disc_dice:.4f}",
            cup_dice=f"{cup_dice:.4f}",
            disc_iou=f"{disc_iou:.4f}",
            cup_iou=f"{cup_iou:.4f}"
        )
        
        if vis_dir is not None:
            os.makedirs(vis_dir, exist_ok=True)
            B = inputs.size(0)
            for b in range(B):
                save_eval_visuals(
                    idx_global,
                    inputs[b],
                    probs[b],
                    targets[b],
                    out_dir=vis_dir,
                    thr=dice_thr,
                    fname_prefix="val"
                )
                idx_global += 1
    
    avg_loss = total_loss / max(1, len(val_loader))
    avg_dice_disc = float(np.mean(dice_scores_disc)) if dice_scores_disc else 0.0
    avg_dice_cup = float(np.mean(dice_scores_cup)) if dice_scores_cup else 0.0
    avg_iou_disc = float(np.mean(iou_scores_disc)) if iou_scores_disc else 0.0
    avg_iou_cup = float(np.mean(iou_scores_cup)) if iou_scores_cup else 0.0
    
    print(f"[Eval] loss={avg_loss:.4f}")
    print(f"  Disc: Dice={avg_dice_disc:.4f}, IoU={avg_iou_disc:.4f}")
    print(f"  Cup:  Dice={avg_dice_cup:.4f}, IoU={avg_iou_cup:.4f}")
    
    return avg_loss, avg_dice_disc, avg_dice_cup, avg_iou_disc, avg_iou_cup

# ===================== 保留原代码的main函数框架，仅修改关键参数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="tn3k")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=518)  # 保留原输入尺寸518
    parser.add_argument("--input_w", type=int, default=518)  # 保留原输入尺寸518
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    # 修改1：分类数改为3（背景、视盘、视杯）
    parser.add_argument("--num_classes", type=int, default=3)
    # 修改2：输入通道改为1（segdinov3的配置）
    parser.add_argument("--in_ch", type=int, default=1)
    parser.add_argument("--repo_dir", type=str, default="./dino3/SegDINO/dinov2")
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"])  # 保留原s配置
    parser.add_argument("--last_layer_idx", type=int, default=-1)
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--cuda_id", type=int, default=0)  # 保留原cuda_id=0
    parser.add_argument("--trainid", type=str, default="1,2,3")
    parser.add_argument("--task_name", type=str, default="Fundus_D")
    args = parser.parse_args()

    # 保留原CUDA配置逻辑
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

    # 保留原随机种子配置
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 保留原保存路径逻辑
    save_root = f"./runs/segdino2_{args.dino_size}_{args.input_h}_{args.task_name}_{args.trainid}"
    os.makedirs(save_root, exist_ok=True)
    train_vis_dir = os.path.join(save_root, "train_vis")
    val_vis_dir   = os.path.join(save_root, "val_vis")
    ckpt_dir      = os.path.join(save_root, "ckpts")
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(val_vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 保留原DINOv2加载逻辑（核心不变）
    # 检查输入尺寸是否能被14整除
    patch_size = 14
    if args.input_h % patch_size != 0 or args.input_w % patch_size != 0:
        print(f"WARNING: Input size {args.input_h}x{args.input_w} is not divisible by {patch_size}!")
        new_h = (args.input_h // patch_size) * patch_size
        new_w = (args.input_w // patch_size) * patch_size
        args.input_h = new_h if new_h > 0 else patch_size
        args.input_w = new_w if new_w > 0 else patch_size
        print(f"Adjusted to {args.input_h}x{args.input_w}")

    # 加载DINOv2模型 - 使用原代码的加载逻辑
    print("Loading DINOv2 small model with patch_size=14 and img_size=518...")
    try:
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False, source='github')
        print("Model loaded via torch.hub")
    except Exception as e:
        print(f"torch.hub failed: {e}")
        print("Trying alternative method...")
        from dinov2.models.vision_transformer import vit_small
        backbone = vit_small(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        print("Model created via direct import")
    
    # 保留原权重加载逻辑
    print(f"Loading weights from {args.dino_ckpt}")
    checkpoint = torch.load(args.dino_ckpt, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"pos_embed shape in checkpoint: {state_dict.get('pos_embed', 'Not found').shape if 'pos_embed' in state_dict else 'Not found'}")
    print(f"Model pos_embed shape: {backbone.pos_embed.shape}")
    
    if 'pos_embed' in state_dict and backbone.pos_embed.shape != state_dict['pos_embed'].shape:
        print(f"Shape mismatch: checkpoint {state_dict['pos_embed'].shape} vs model {backbone.pos_embed.shape}")
        checkpoint_patches = state_dict['pos_embed'].shape[1] - 1
        model_patches = backbone.pos_embed.shape[1] - 1
        
        if checkpoint_patches == 1369 and model_patches == 256:
            print("Interpolating pos_embed from 518x518 to 224x224...")
            cls_token = state_dict['pos_embed'][:, :1, :]
            patch_tokens = state_dict['pos_embed'][:, 1:, :]
            patch_tokens = patch_tokens.reshape(1, 37, 37, -1)
            patch_tokens = patch_tokens.permute(0, 3, 1, 2)
            patch_tokens = nn.functional.interpolate(
                patch_tokens, size=(16, 16), mode='bicubic', align_corners=False
            )
            patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, 256, -1)
            new_pos_embed = torch.cat([cls_token, patch_tokens], dim=1)
            state_dict['pos_embed'] = new_pos_embed
        elif checkpoint_patches == 256 and model_patches == 1369:
            print("Interpolating pos_embed from 224x224 to 518x518...")
            pass
    
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0:
        print(f"Missing keys (first 10): {missing_keys[:10]}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys (first 10): {unexpected_keys[:10]}")
    print(f"Final model pos_embed shape: {backbone.pos_embed.shape}")
    
    # 保留原backbone冻结逻辑
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    print("✅ DINOv2 model loaded successfully!")
    
    # 修改3：模型初始化nclass改为3（segdinov3配置）
    from dpt_D_dinov2 import DPT
    model = DPT(nclass=3, backbone=backbone, encoder_size=args.dino_size)  # nclass=3
    model = model.to(device)
    
    # 保留原优化器配置
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 修改4：数据加载替换为FolderDataset_two（segdinov3的数据集类）
    from dataset import ResizeAndNormalize, FolderDataset_two  # 替换FolderDataset为FolderDataset_two
    
    data_root = args.data_dir
    domain_ids = [int(domain_id) for domain_id in args.trainid.split(',')]
    train_paths = [os.path.join(data_root, f"Domain{domain_id}/train/ROIs/") for domain_id in domain_ids]
    val_paths = [os.path.join(data_root, f"Domain{domain_id}/test/ROIs/") for domain_id in domain_ids]
    
    train_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    val_transform   = ResizeAndNormalize(size=(args.input_h, args.input_w))

    # 替换为FolderDataset_two
    train_datasets=[FolderDataset_two(
        root=path,
        split="train",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=train_transform,
    )for path in train_paths]
    
    val_datasets=[FolderDataset_two(
        root=path,
        split="test",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=val_transform,
    )for path in val_paths]
    
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets) 
    
    print("ok-------------------------------------------------------------")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Using DINOv2 small (s) with patch_size=14")
    print(f"Input size: {args.input_h}x{args.input_w}")
    print(f"Backbone embed_dim: {backbone.embed_dim}")
    
    # 保留原DataLoader配置
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

    # 修改5：保存逻辑替换为segdinov3的视盘/视杯双指标保存
    best_val_dice_disc = -1.0
    best_val_dice_cup = -1.0
    best_val_dice_disc_epoch = -1
    best_val_dice_cup_epoch = -1

    for epoch in range(1, args.epochs + 1):
        # 修改6：调用segdinov3的train_one_epoch（返回5个值）
        train_loss, train_dice_disc, train_dice_cup, train_iou_disc, train_iou_cup = train_one_epoch(
            model, train_loader, optimizer, device,
            num_classes=args.num_classes, dice_thr=0.5,
            vis_dir=train_vis_dir, epoch=epoch
        )
        # 修改7：调用segdinov3的evaluate（返回5个值）
        val_loss, val_dice_disc, val_dice_cup, val_iou_disc, val_iou_cup = evaluate(
            model, val_loader, device,
            num_classes=args.num_classes, dice_thr=0.5,
            vis_dir=val_vis_dir
        )

        # 保留原最新模型保存逻辑
        latest_path = os.path.join(ckpt_dir, "latest.pth")
        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()},
            latest_path
        )

        # 修改8：按视盘Dice保存最佳模型
        if val_dice_disc > best_val_dice_disc:
            best_val_dice_disc = val_dice_disc
            best_val_dice_disc_epoch = epoch
            best_path = os.path.join(ckpt_dir, f"best_disc_ep{epoch:03d}_dice{val_dice_disc:.4f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[Save] New best disc ckpt: {best_path}")
        
        # 修改9：按视杯Dice保存最佳模型
        if val_dice_cup > best_val_dice_cup:
            best_val_dice_cup = val_dice_cup
            best_val_dice_cup_epoch = epoch
            best_path = os.path.join(ckpt_dir, f"best_cup_ep{epoch:03d}_dice{val_dice_cup:.4f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[Save] New best cup ckpt: {best_path}")

    # 修改10：总结信息替换为视盘/视杯双指标
    print("=" * 60)
    print(f"[Summary] Best Val Disc Dice = {best_val_dice_disc:.4f} @ epoch {best_val_dice_disc_epoch}")
    print(f"[Summary] Best Val Cup Dice  = {best_val_dice_cup:.4f} @ epoch {best_val_dice_cup_epoch}")
    print("=" * 60)

if __name__ == "__main__":
    main()