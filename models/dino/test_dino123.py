# train_dinov1_working.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import random

# ========== 工具函数 ==========
def tensor_to_rgb(img_t: torch.Tensor) -> np.ndarray:
    """Tensor转RGB图像"""
    img = img_t.detach().cpu().float()
    img = img.clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    """Mask tensor转灰度图"""
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

def iou_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    """计算IoU"""
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    iou = (inter + eps) / union
    return iou.view(-1)

def dice_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    """计算Dice系数"""
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / union
    return dice

# ========== DINOv1加载函数 ==========
def load_dinov1_backbone_simple(args):
    """
    简单的DINOv1加载函数
    """
    print("=" * 60)
    print("加载DINOv1 Backbone")
    print(f"权重文件: {args.dino_ckpt}")
    print(f"模型大小: {args.dino_size}")
    print("=" * 60)
    
    # 检查权重文件
    if not os.path.exists(args.dino_ckpt):
        raise FileNotFoundError(f"权重文件不存在: {args.dino_ckpt}")
    
    # 加载权重文件
    print(f"加载权重文件...")
    checkpoint = torch.load(args.dino_ckpt, map_location='cpu')
    print(f"检查点类型: {type(checkpoint)}")
    
    # 提取state_dict
    if isinstance(checkpoint, dict):
        if 'teacher' in checkpoint:
            print("✓ 检测到DINOv1格式 (teacher/student)")
            state_dict = checkpoint['teacher']
        elif 'state_dict' in checkpoint:
            print("✓ 检测到state_dict格式")
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            print("✓ 检测到model格式")
            state_dict = checkpoint['model']
        else:
            print("✓ 直接使用checkpoint作为state_dict")
            state_dict = checkpoint
    else:
        print("✓ checkpoint直接是state_dict")
        state_dict = checkpoint
    
    print(f"state_dict键数量: {len(state_dict)}")
    
    # 检查是否是真正的DINOv1
    keys = list(state_dict.keys())
    has_mask_token = any('mask_token' in k for k in keys)
    has_ls_gamma = any('ls1.gamma' in k or 'ls2.gamma' in k for k in keys)
    
    if has_mask_token or has_ls_gamma:
        print("⚠️ 警告：权重文件包含DINOv2/v3特有的参数")
        print("可能不是纯DINOv1权重，但我们会尝试加载")
    
    # 从torch.hub加载DINOv1模型
    print(f"从torch.hub加载DINOv1模型...")
    
    if args.dino_size == "s":
        print("加载DINOv1 ViT-Small/16...")
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        embed_dim = 384
    elif args.dino_size == "b":
        print("加载DINOv1 ViT-Base/16...")
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        embed_dim = 768
    else:
        raise ValueError(f"不支持的dino_size: {args.dino_size}")
    
    print("✓ 模型架构加载成功")
    
    # 打印模型信息
    print(f"模型embed_dim: {embed_dim}")
    
    # 检查patch_size
    if hasattr(backbone, 'patch_embed'):
        patch_size = backbone.patch_embed.proj.kernel_size[0]
        print(f"模型patch_size: {patch_size}x{patch_size}")
    
    # 加载权重
    print("加载预训练权重...")
    try:
        # 首先尝试严格模式
        backbone.load_state_dict(state_dict, strict=True)
        print("✓ 权重加载成功（严格模式）")
    except Exception as e:
        print(f"严格模式加载失败: {e}")
        print("尝试非严格模式...")
        
        # 非严格模式
        missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ 缺失 {len(missing_keys)} 个键:")
            for key in list(missing_keys)[:5]:
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... 还有 {len(missing_keys)-5} 个")
        
        if unexpected_keys:
            print(f"⚠️ 有 {len(unexpected_keys)} 个意外键:")
            for key in list(unexpected_keys)[:5]:
                print(f"  - {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... 还有 {len(unexpected_keys)-5} 个")
    
    # 冻结backbone
    print("冻结backbone参数...")
    for param in backbone.parameters():
        param.requires_grad = False
    
    print("=" * 60)
    print("DINOv1加载完成 ✓")
    print("=" * 60)
    
    return backbone, embed_dim

# ========== 简单的DPT模型 ==========
class SimpleSegmentationModel(nn.Module):
    """简单的分割模型，适配DINOv1"""
    def __init__(self, backbone, embed_dim=384, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
    def forward(self, x):
        # DINOv1前向传播
        with torch.no_grad():
            # 获取特征
            features = self.backbone.get_intermediate_layers(x, n=1)
            if isinstance(features, list):
                features = features[0]  # 取第一层
            
        # features形状: [B, N, C]
        B, N, C = features.shape
        
        # 假设输入是256x256，patch_size=16，那么特征图是16x16=256
        # 但DINOv1会有一个额外的cls_token，所以N=257
        H = W = int((N - 1) ** 0.5)  # 减去cls_token
        
        # 分离cls_token和patch tokens
        cls_token = features[:, 0, :]  # [B, C]
        patch_tokens = features[:, 1:, :]  # [B, H*W, C]
        
        # reshape为空间特征图
        patch_features = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 解码
        output = self.decoder(patch_features)
        
        # 上采样到输入尺寸
        output = nn.functional.interpolate(
            output, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        
        return output

# ========== 训练函数 ==========
def train_one_epoch(model, train_loader, optimizer, device, epoch=0):
    model.train()
    total_loss = 0.0
    dice_scores = []
    
    criterion = nn.BCEWithLogitsLoss()
    pbar = tqdm(train_loader, desc=f"[Train Epoch {epoch}]")
    
    for batch_idx, (inputs, targets, _) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算Dice
        with torch.no_grad():
            dice = dice_binary_torch(logits, targets, thresh=0.5).mean().item()
            dice_scores.append(dice)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    
    print(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    return avg_loss, avg_dice

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    
    criterion = nn.BCEWithLogitsLoss()
    pbar = tqdm(val_loader, desc="[Eval]")
    
    for batch_idx, (inputs, targets, _) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        
        # 计算指标
        dice = dice_binary_torch(logits, targets, thresh=0.5).mean().item()
        iou = iou_binary_torch(logits, targets, thresh=0.5).mean().item()
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}", iou=f"{iou:.4f}")
    
    avg_loss = total_loss / len(val_loader)
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    avg_iou = np.mean(iou_scores) if iou_scores else 0.0
    
    print(f"[Eval] Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
    return avg_loss, avg_dice, avg_iou

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="DINOv1分割训练")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--trainid", type=str, default="1,2,3", help="训练域ID")
    parser.add_argument("--img_dir_name", type=str, default="image", help="图像目录名")
    parser.add_argument("--label_dir_name", type=str, default="mask", help="标签目录名")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--input_h", type=int, default=256, help="输入高度")
    parser.add_argument("--input_w", type=int, default=256, help="输入宽度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # DINO参数
    parser.add_argument("--dino_ckpt", type=str, required=True, 
                       help="DINOv1权重路径，如: dino_deitsmall16_pretrain.pth")
    parser.add_argument("--dino_size", type=str, default="s", choices=["s", "b"],
                       help="DINO模型大小: s=small, b=base")
    
    # 其他参数
    parser.add_argument("--cuda_id", type=int, default=0, help="CUDA设备ID")
    parser.add_argument("--task_name", type=str, default="Fundus_D", help="任务名称")

    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    
    args = parser.parse_args()
    
    # 设置设备
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_id)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建保存目录
    save_root = f"./runs/dinov1_{args.dino_size}_{args.input_h}_{args.task_name}_{args.trainid}"
    os.makedirs(save_root, exist_ok=True)
    ckpt_dir = os.path.join(save_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"保存目录: {save_root}")
    
    # ========== 关键部分：加载DINOv1 ==========
    backbone, embed_dim = load_dinov1_backbone_simple(args)
    
    # 创建模型
    from dpt import DPT
    model = DPT(backbone=backbone,  nclass=1)
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # # 优化器（只优化解码器）
    # optimizer = torch.optim.AdamW(
    #     model.decoder.parameters(),
    #     lr=args.lr,
    #     weight_decay=1e-4
    # )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    
    # 数据加载
    print("\n" + "=" * 60)
    print("加载数据集")
    print("=" * 60)
    
    # 这里使用你的数据加载逻辑
    # 由于我不知道你的具体数据格式，这里用简单的示例
    # 你需要替换为你的实际数据加载代码
    
    try:
        from dataset import ResizeAndNormalize,FolderDataset
    

        data_root = args.data_dir
        domain_ids = [int(domain_id) for domain_id in args.trainid.split(',')]
        train_paths = [os.path.join(data_root, f"Domain{domain_id}/train/ROIs/") for domain_id in domain_ids]
        val_paths = [os.path.join(data_root, f"Domain{domain_id}/test/ROIs/") for domain_id in domain_ids]
        # root = os.path.join(args.data_dir, args.dataset)

        train_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
        val_transform   = ResizeAndNormalize(size=(args.input_h, args.input_w))

        train_datasets=[FolderDataset(
            root=path,
            split="train",
            img_dir_name=args.img_dir_name,
            label_dir_name=args.label_dir_name,
            mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
            transform=train_transform,
        )for path in train_paths]
        val_datasets=[FolderDataset(
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
        print("Number of training samples: ", len(train_dataset))
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
            
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("创建虚拟数据用于测试...")
        
        # 创建虚拟数据用于测试
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=100, img_size=(3, 256, 256)):
                self.num_samples = num_samples
                self.img_size = img_size
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                img = torch.randn(self.img_size)
                mask = torch.randn(1, self.img_size[1], self.img_size[2])
                meta = {"id": f"sample_{idx}"}
                return img, mask, meta
        
        train_dataset = DummyDataset(num_samples=100)
        val_dataset = DummyDataset(num_samples=20)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False
        )
        
        print("使用虚拟数据，仅用于测试模型是否可以运行")
    
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    best_val_dice = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # 验证
        if val_loader:
            val_loss, val_dice, val_iou = evaluate(model, val_loader, device)
        else:
            val_dice = 0.0
            val_iou = 0.0
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_dice': val_dice if val_loader else 0.0,
            'val_iou': val_iou if val_loader else 0.0,
        }
        
        # 保存最新
        torch.save(checkpoint, os.path.join(ckpt_dir, 'latest.pth'))
        
        # 保存最佳
        if val_loader and val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save(checkpoint, os.path.join(ckpt_dir, f'best_epoch{epoch}_dice{val_dice:.4f}.pth'))
            print(f"✓ 新的最佳模型保存: epoch {epoch}, Dice: {val_dice:.4f}")
    
    print("\n" + "=" * 60)
    print("训练完成")
    print(f"最佳验证Dice: {best_val_dice:.4f} @ epoch {best_epoch}")
    print(f"模型保存到: {save_root}")
    print("=" * 60)

if __name__ == "__main__":
    main()