# train_segdinov3_cnn_train_val_test.py
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

sys.path.append('./dino')

try:
    import loralib as lora
except ImportError:
    print("请安装loralib：pip install loralib")
    sys.exit(1)


# ==================== CNN解码器定义 ====================
class SegDecoderCNN(nn.Module):
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
    def __init__(self, image_encoder, mask_decoder, img_size=224, patch_size=16, dino_size="b"):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.dino_dim = 768 if dino_size == "b" else 384
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


# ==================== 辅助函数 ====================
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

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, num_classes=1, ce_class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes

        if num_classes == 1:
            self.ce_loss = nn.BCEWithLogitsLoss()
        else:
            if ce_class_weights is not None:
                self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)
            else:
                self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        pred:
            二分类: [B, 1, H, W] logits
            多分类: [B, C, H, W] logits
        target:
            二分类: [B, 1, H, W] 或 [B, H, W]
            多分类: one-hot [B, C, H, W] 或 index [B, H, W]
        """
        if self.num_classes == 1:
            target = target.float()
            if target.ndim == 3:
                target = target.unsqueeze(1)

            ce_loss = self.ce_loss(pred, target)

            prob = torch.sigmoid(pred)
            inter = (prob * target).sum(dim=(2, 3))
            union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            dice = (2.0 * inter + 1e-6) / (union + 1e-6)
            dice_loss = 1.0 - dice.mean()

        else:
            if target.ndim == 4:
                target_indices = torch.argmax(target, dim=1).long()
                target_onehot = target.float()
            else:
                target_indices = target.long()
                target_onehot = F.one_hot(target_indices, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

            ce_loss = self.ce_loss(pred, target_indices)

            probs = F.softmax(pred, dim=1)
            dice_scores = []
            for c in range(1, self.num_classes):
                pred_class = probs[:, c, :, :]
                target_class = target_onehot[:, c, :, :]
                inter = (pred_class * target_class).sum(dim=(1, 2))
                union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
                dice = (2.0 * inter + 1e-6) / (union + 1e-6)
                dice_scores.append(dice)

            dice_scores = torch.stack(dice_scores)
            dice_loss = 1.0 - dice_scores.mean()

        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return total_loss, dice_loss, ce_loss

def train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes=1, dice_thr=0.5, vis_dir=None, epoch=0):
    model.train()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_ce_loss = 0.0

    dice_scores = []
    iou_scores = []

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
            dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            iou = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            dice_scores.append(dice)
            iou_scores.append(iou)

        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals(epoch, inputs, logits, targets, out_dir=vis_dir, max_save=8, thr=dice_thr)
            first_batch_logged = True

    avg_loss = total_loss / max(1, len(train_loader))
    avg_dice_loss = total_dice_loss / max(1, len(train_loader))
    avg_ce_loss = total_ce_loss / max(1, len(train_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0

    print(
        f"[Train Epoch {epoch}] "
        f"total={avg_loss:.4f}  dice_loss={avg_dice_loss:.4f}  ce_loss={avg_ce_loss:.4f}  "
        f"dice={avg_dice:.4f}  iou={avg_iou:.4f}"
    )

    return avg_loss, avg_dice, avg_iou


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, num_classes=1, dice_thr=0.5, vis_dir=None, split_name="Val"):
    model.eval()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_ce_loss = 0.0

    dice_scores = []
    iou_scores = []

    idx_global = 0

    for inputs, targets, _ in data_loader:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        logits = model(inputs)
        total_batch_loss, dice_loss, ce_loss = criterion(logits, targets)

        total_loss += total_batch_loss.item()
        total_dice_loss += dice_loss.item()
        total_ce_loss += ce_loss.item()

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
                    fname_prefix=split_name.lower(),
                )
                idx_global += 1

    avg_loss = total_loss / max(1, len(data_loader))
    avg_dice_loss = total_dice_loss / max(1, len(data_loader))
    avg_ce_loss = total_ce_loss / max(1, len(data_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0

    print(
        f"[{split_name}] "
        f"total={avg_loss:.4f}  dice_loss={avg_dice_loss:.4f}  ce_loss={avg_ce_loss:.4f}  "
        f"dice={avg_dice:.4f}  iou={avg_iou:.4f}"
    )

    return avg_loss, avg_dice, avg_iou


# ==================== 数据集构建 ====================
def build_npc_datasets(args, train_transform, val_transform, test_transform):
    from dataset import FolderDataset_NPC

    train_domains = [int(domain_id) for domain_id in args.trainid.split(',')]
    test_domains = [int(domain_id) for domain_id in args.testid.split(',')]

    print(f"✅ Training domains: {train_domains}")
    print(f"✅ Testing domains: {test_domains}")

    train_datasets = []
    val_datasets = []
    test_domain_datasets = {}

    # train: trainid_train
    for domain_id in train_domains:
        path = os.path.join(args.data_dir, f"Domain{domain_id}_train/")
        if os.path.exists(path):
            dataset = FolderDataset_NPC(
                root=path,
                split="train",
                img_dir_name=args.img_dir_name,
                label_dir_name=args.label_dir_name,
                mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
                transform=train_transform,
                use_pkl=True,
                pkl_dir=path,
            )
            train_datasets.append(dataset)
            print(f"✅ Loaded train dataset from Domain{domain_id}_train: {len(dataset)} samples")
        else:
            print(f"⚠️ Warning: train path not found: {path}")

    # val: trainid_test
    for domain_id in train_domains:
        path = os.path.join(args.data_dir, f"Domain{domain_id}_test/")
        if os.path.exists(path):
            dataset = FolderDataset_NPC(
                root=path,
                split="test",
                img_dir_name=args.img_dir_name,
                label_dir_name=args.label_dir_name,
                mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
                transform=val_transform,
                use_pkl=True,
                pkl_dir=path,
            )
            val_datasets.append(dataset)
            print(f"✅ Loaded val dataset from Domain{domain_id}_test: {len(dataset)} samples")
        else:
            print(f"⚠️ Warning: val path not found: {path}")

    # test: 每个 testid 单独构建
    for domain_id in test_domains:
        domain_subsets = []

        for split_name, split_arg in [("train", "train"), ("test", "test")]:
            path = os.path.join(args.data_dir, f"Domain{domain_id}_{split_name}/")
            if os.path.exists(path):
                dataset = FolderDataset_NPC(
                    root=path,
                    split=split_arg,
                    img_dir_name=args.img_dir_name,
                    label_dir_name=args.label_dir_name,
                    mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
                    transform=test_transform,
                    use_pkl=True,
                    pkl_dir=path,
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--in_ch", type=int, default=1)
    parser.add_argument("--repo_dir", type=str, default="./dino")
    parser.add_argument("--dino_ckpt", type=str, required=True, help="Path to the pretrained DINO checkpoint (.pth).")
    parser.add_argument("--dino_size", type=str, default="b", choices=["b", "s"], help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--last_layer_idx", type=int, default=-1)
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--cuda_id", type=int, default=1)
    parser.add_argument("--trainid", type=str, default="1,2,3", help="Comma-separated list of domain IDs for training domains")
    parser.add_argument("--testid", type=str, default="4", help="Comma-separated list of domain IDs for testing domains")
    parser.add_argument("--task_name", type=str, default="NPC_D_CNN")
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

    save_root = f"./runs/segdinov3_cnn_{args.dino_size}_{args.input_h}_{args.task_name}_train{args.trainid}_test{args.testid}"
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
        from vision_transformer import vit_base, vit_small
    except ImportError:
        print("❌ 无法导入DINO模块，请确保代码在 ./dino 下")
        sys.exit(1)

    if args.dino_size == "b":
        backbone = vit_base(patch_size=16, num_classes=0, img_size=[args.input_h, args.input_w])
    else:
        backbone = vit_small(patch_size=16, num_classes=0, img_size=[args.input_h, args.input_w])

    state_dict = torch.load(args.dino_ckpt, map_location="cpu")
    if "teacher" in state_dict:
        state_dict = state_dict["teacher"]
    msg = backbone.load_state_dict(state_dict, strict=False)
    print(f"✅ 加载DINO权重: {msg}")
    backbone.eval()

    if use_lora:
        for param in backbone.parameters():
            param.requires_grad = False
        replace_linear_with_lora(backbone, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        print("✅ LoRA已应用到DINO backbone")

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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params_count:,}")
    print(f"✅ Trainable ratio: {trainable_params_count / total_params:.2%}")

    # ==================== 组合损失 ====================
    if args.num_classes == 1:
        criterion = DiceCELoss(
            dice_weight=0.5,
            ce_weight=0.5,
            num_classes=1,
        )
    else:
        ce_weights = torch.tensor([0.2, 1.0, 1.0], device=device)
        criterion = DiceCELoss(
            dice_weight=0.5,
            ce_weight=0.5,
            num_classes=args.num_classes,
            ce_class_weights=ce_weights,
        )

    try:
        from dataset import ResizeAndNormalize_NPC
    except ImportError as e:
        print(f"❌ 无法导入dataset模块: {e}")
        sys.exit(1)

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
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loaders = {}
    for domain_id, test_dataset in test_domain_datasets.items():
        test_loaders[domain_id] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

    print("\n🚀 开始训练")
    print("=" * 60)

    best_val_dice = -1.0
    best_val_dice_epoch = -1
    best_val_iou = -1.0
    best_val_iou_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=args.num_classes,
            dice_thr=0.5,
            vis_dir=train_vis_dir,
            epoch=epoch,
        )

        val_loss, val_dice, val_iou = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=args.num_classes,
            dice_thr=0.5,
            vis_dir=val_vis_dir,
            split_name="Val",
        )

        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
                "best_val_dice": best_val_dice,
                "best_val_iou": best_val_iou,
            },
            latest_model_path,
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
                },
                best_model_path,
            )
            print(f"💾 [Save] New best ckpt: {best_model_path}")
        elif val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_iou_epoch = epoch

    print("=" * 60)
    print(f"🏆 [Summary] Best Val Dice = {best_val_dice:.4f} @ epoch {best_val_dice_epoch}")
    print(f"🏆 [Summary] Best Val IoU  = {best_val_iou:.4f} @ epoch {best_val_iou_epoch}")
    print("=" * 60)

    print("\n🧪 Loading best model for final testing...")
    best_ckpt = torch.load(best_model_path, map_location=device)
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

        test_loss, test_dice, test_iou = evaluate(
            model,
            test_loader,
            criterion,
            device,
            num_classes=args.num_classes,
            dice_thr=0.5,
            vis_dir=domain_vis_dir,
            split_name=f"Test_Domain{domain_id}",
        )

        all_domain_results[domain_id] = {
            "loss": test_loss,
            "dice": test_dice,
            "iou": test_iou,
        }

        print("-" * 60)
        print(f"📍 Domain{domain_id} Results:")
        print(f"[Final Test][Domain{domain_id}] Loss = {test_loss:.4f}")
        print(f"[Final Test][Domain{domain_id}] Dice = {test_dice:.4f}")
        print(f"[Final Test][Domain{domain_id}] IoU  = {test_iou:.4f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("✅ Summary of Per-Domain Test Results:")
    print("=" * 60)
    for domain_id, result in all_domain_results.items():
        print(
            f"Domain{domain_id} | "
            f"Loss={result['loss']:.4f}  "
            f"Dice={result['dice']:.4f}  "
            f"IoU={result['iou']:.4f}"
        )
    print("=" * 60)

    results = {
        "dino_size": args.dino_size,
        "best_val_dice": float(best_val_dice),
        "best_val_iou": float(best_val_iou),
        "best_epoch": int(best_val_dice_epoch),
        "train_id": args.trainid,
        "test_id": args.testid,
        "input_size": f"{args.input_h}x{args.input_w}",
        "per_domain_test_results": {
            str(domain_id): {
                "loss": float(result["loss"]),
                "dice": float(result["dice"]),
                "iou": float(result["iou"]),
            }
            for domain_id, result in all_domain_results.items()
        }
    }
    results_path = os.path.join(save_root, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
