# train_dinov1_D_Fundus_CNN.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import argparse
import json
import torch.utils.data as data

sys.path.append('./dino')

try:
    import loralib as lora
except ImportError:
    print("请安装loralib：pip install loralib")
    sys.exit(1)


# ==================== CNN解码器定义 ====================
class SegDecoderCNN(nn.Module):
    """与第一个程序完全相同的CNN解码器"""
    def __init__(self, num_classes=1, embed_dim=256, num_depth=2, top_channel=64):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(embed_dim, top_channel * 2 ** num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_channel * 2 ** num_depth, top_channel * 2 ** num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        for i in range(num_depth):
            if num_depth > 2 > i:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        top_channel * 2 ** (num_depth - i),
                        top_channel * 2 ** (num_depth - i - 1),
                        2,
                        stride=2,
                    ),
                )
            self.blocks.append(block)

        self.final = nn.Sequential(
            nn.Conv2d(top_channel, top_channel, 3, padding=1),
            nn.Conv2d(top_channel, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.input_block(x)
        for blk in self.blocks:
            x = blk(x)
        return self.final(x)


class DINOFeatSeg(nn.Module):
    """DINOv1 + CNN解码器的组合模型"""
    def __init__(self, image_encoder, mask_decoder, img_size=224, patch_size=16, dino_size="b"):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder

        if dino_size == "b":
            self.dino_dim = 768
        else:
            self.dino_dim = 384

        self.num_patches = (img_size // patch_size) ** 2
        self.feature_projection = nn.Sequential(
            nn.Conv2d(self.dino_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.upsample_to_decoder = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)

    def extract_patch_tokens(self, x):
        self.image_encoder.eval()
        x = self.image_encoder.prepare_tokens(x)
        for blk in self.image_encoder.blocks:
            x = blk(x)
        x = self.image_encoder.norm(x)
        patch_tokens = x[:, 1:, :]
        return patch_tokens

    def forward(self, x):
        original_size = x.shape[-1]
        patch_tokens = self.extract_patch_tokens(x)
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        features = self.feature_projection(patch_tokens)
        features = self.upsample_to_decoder(features)
        out = self.mask_decoder(features)
        if out.shape[-1] != original_size:
            out = F.interpolate(out, (original_size, original_size), mode="bilinear", align_corners=False)
        return out


# ==================== LoRA替换函数 ====================
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


# ==================== 可视化辅助函数 ====================
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


# ==================== 多分类指标 ====================
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
        target_indices = torch.argmax(target, dim=1).long()   # [B, H, W]
        ce_loss = self.ce_loss(pred, target_indices)

        dice_scores = multi_class_dice(pred, target, self.num_classes)
        dice_loss = 1.0 - dice_scores.mean()

        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return total_loss, dice_loss, ce_loss

# ==================== 训练/验证/测试 ====================
def train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes=3, epoch=0, vis_dir=None):
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
        f"[Train Epoch {epoch}] "
        f"loss={avg_loss:.4f}  diceloss={avg_dice_loss:.4f}  celoss={avg_ce_loss:.4f}  "
        f"disc_dice={avg_dice_disc:.4f}  cup_dice={avg_dice_cup:.4f}  "
        f"disc_iou={avg_iou_disc:.4f}  cup_iou={avg_iou_cup:.4f}"
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

    for inputs, targets, _ in data_loader:
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

    print(
        f"[{split_name}] "
        f"loss={avg_loss:.4f}  diceloss={avg_dice_loss:.4f}  celoss={avg_ce_loss:.4f}  "
        f"disc_dice={avg_dice_disc:.4f}  cup_dice={avg_dice_cup:.4f}  "
        f"disc_iou={avg_iou_disc:.4f}  cup_iou={avg_iou_cup:.4f}"
    )

    return avg_loss, avg_dice_disc, avg_dice_cup, avg_iou_disc, avg_iou_cup



# ==================== 数据集构建 ====================
def build_fundus_datasets(args, train_transform, val_transform, test_transform):
    from dataset import FolderDataset_new

    train_domains = [int(id_) for id_ in args.trainid.split(',')]
    test_domains = [int(id_) for id_ in args.testid.split(',')]

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


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/Fundus/")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--dino_ckpt", type=str, required=True, help="Path to the pretrained DINO checkpoint (.pth).")
    parser.add_argument("--dino_size", type=str, default="b", choices=["b", "s"], help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--cuda_id", type=int, default=0, help="Specify CUDA device ID to use")
    parser.add_argument("--trainid", type=str, default="1,2,3", help="Comma-separated list of domain IDs to include in training")
    parser.add_argument("--testid", type=str, default="4", help="Comma-separated list of domain IDs to include in testing")
    parser.add_argument("--task_name", type=str, default="Fundus_D_CNN")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=4, help="LoRA的alpha值")

    args = parser.parse_args()
    use_lora = "lora" in args.task_name.lower()

    if use_lora:
        print(f"✅ 使用LoRA，秩(r)={args.lora_r}, alpha={args.lora_alpha}")

    # 设置设备
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, will use CPU instead!")
        device = "cpu"
    else:
        cuda_count = torch.cuda.device_count()
        if args.cuda_id >= cuda_count or args.cuda_id < 0:
            print(f"⚠️ Invalid CUDA ID {args.cuda_id}! Using CUDA:0")
            device = "cuda:0"
        else:
            torch.cuda.set_device(args.cuda_id)
            device = f"cuda:{args.cuda_id}"
            print(f"✅ Using CUDA device ID: {args.cuda_id} (device: {device})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_root = f"./runs/segdino1_cnn_{args.dino_size}_{args.input_h}_{args.task_name}_train{args.trainid}_test{args.testid}"
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

    # 导入DINOv1模块
    try:
        from vision_transformer import vit_base, vit_small
    except ImportError:
        print("❌ 无法导入DINOv1模块，请确保DINOv1代码在正确路径")
        print("💡 建议: git clone https://github.com/facebookresearch/dino.git 到当前目录")
        sys.exit(1)

    # 创建DINOv1模型
    if args.dino_size == "b":
        backbone = vit_base(patch_size=16, num_classes=0, img_size=[args.input_h, args.input_w])
    else:
        backbone = vit_small(patch_size=16, num_classes=0, img_size=[args.input_h, args.input_w])

    try:
        state_dict = torch.load(args.dino_ckpt, map_location="cpu", weights_only=True)
    except Exception:
        state_dict = torch.load(args.dino_ckpt, map_location="cpu")

    if "teacher" in state_dict:
        state_dict = state_dict["teacher"]

    msg = backbone.load_state_dict(state_dict, strict=False)
    print(f"✅ 加载DINOv1-{ 'base' if args.dino_size == 'b' else 'small' }权重: {msg}")

    backbone.eval()

    if use_lora:
        for param in backbone.parameters():
            param.requires_grad = False
        replace_linear_with_lora(backbone, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        print("✅ LoRA已应用到DINOv1 backbone")

    seg_decoder = SegDecoderCNN(
        num_classes=args.num_classes,
        embed_dim=256,
        num_depth=2,
        top_channel=64,
    )

    model = DINOFeatSeg(
        image_encoder=backbone,
        mask_decoder=seg_decoder,
        img_size=args.input_h,
        patch_size=16,
        dino_size=args.dino_size,
    ).to(device)

    if use_lora:
        trainable_params = []
        trainable_params.extend(model.mask_decoder.parameters())
        trainable_params.extend(model.feature_projection.parameters())
        trainable_params.extend(model.upsample_to_decoder.parameters())
        for _, param in backbone.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
    else:
        for param in backbone.parameters():
            param.requires_grad = False
        trainable_params = (
            list(model.mask_decoder.parameters()) +
            list(model.feature_projection.parameters()) +
            list(model.upsample_to_decoder.parameters())
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # ==================== 新增：Dice + CE 组合损失 ====================
    ce_class_weights = torch.tensor([0.2, 1.0, 1.0], device=device)
    criterion = DiceCELoss(
        dice_weight=0.5,
        ce_weight=0.5,
        num_classes=args.num_classes,
        ce_class_weights=ce_class_weights,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params_count:,}")
    print(f"✅ Trainable ratio: {trainable_params_count / total_params:.2%}")

    try:
        from dataset import ResizeAndNormalize_new
    except ImportError as e:
        print(f"❌ 无法导入dataset模块: {e}")
        sys.exit(1)

    train_transform = ResizeAndNormalize_new(size=(args.input_h, args.input_w))
    val_transform = ResizeAndNormalize_new(size=(args.input_h, args.input_w))
    test_transform = ResizeAndNormalize_new(size=(args.input_h, args.input_w))

    train_dataset, val_dataset, test_domain_datasets = build_fundus_datasets(
        args=args,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
    )

    print(f"\n📊 Dataset Summary:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    total_test_samples = sum(len(ds) for ds in test_domain_datasets.values())
    print(f"   Total test samples: {total_test_samples}")
    for domain_id, ds in test_domain_datasets.items():
        print(f"   Domain{domain_id} test samples: {len(ds)}")
    print(f"   Number of classes: {args.num_classes}")
    print(f"   Input size: {args.input_h}x{args.input_w}")

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

    print("\n🔍 Checking data format...")
    for images, masks, _ in train_loader:
        print(f"   Image shape: {images.shape}")
        print(f"   Mask shape: {masks.shape}")
        print(f"   Mask unique values: {torch.unique(masks)}")
        break

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
            epoch=epoch,
            vis_dir=train_vis_dir,
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
    best_ckpt = torch.load(best_model_path, map_location=device)
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
            epoch=args.epochs + 1,
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

        print("-" * 80)
        print(f"📍 Domain{domain_id} Results:")
        print(f"📊 Test Loss:     {test_loss:.4f}")
        print(f"🎯 Disc Dice:     {test_disc_dice:.4f}")
        print(f"🎯 Cup Dice:      {test_cup_dice:.4f}")
        print(f"🎯 Mean Dice:     {test_mean_dice:.4f}")
        print(f"🎯 Disc IoU:      {test_disc_iou:.4f}")
        print(f"🎯 Cup IoU:       {test_cup_iou:.4f}")
        print(f"🎯 Mean IoU:      {test_mean_iou:.4f}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("✅ Summary of Per-Domain Test Results:")
    print("=" * 80)
    for domain_id, result in all_domain_results.items():
        print(
            f"Domain{domain_id} | "
            f"Loss={result['loss']:.4f}  "
            f"DiscDice={result['disc_dice']:.4f}  "
            f"CupDice={result['cup_dice']:.4f}  "
            f"MeanDice={result['mean_dice']:.4f}  "
            f"DiscIoU={result['disc_iou']:.4f}  "
            f"CupIoU={result['cup_iou']:.4f}  "
            f"MeanIoU={result['mean_iou']:.4f}"
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
        "best_epoch_loss": int(best_val_loss_epoch),
        "best_epoch_disc_dice": int(best_val_disc_dice_epoch),
        "best_epoch_cup_dice": int(best_val_cup_dice_epoch),
        "best_epoch_mean_dice": int(best_val_mean_dice_epoch),
        "best_epoch_disc_iou": int(best_val_disc_iou_epoch),
        "best_epoch_cup_iou": int(best_val_cup_iou_epoch),
        "best_epoch_mean_iou": int(best_val_mean_iou_epoch),
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
