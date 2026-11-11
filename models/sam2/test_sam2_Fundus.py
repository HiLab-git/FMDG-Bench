# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""
import pdb
import math
import loralib as lora
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

join = os.path.join

import torch.nn.functional as F
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
import sys
sys.path.append("./sam2")
import argparse
import random
from datetime import datetime
import shutil
import glob

from naso_dataset import nasopharyngealDataset
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from skimage import io
from seg_head.SamFeatSeg import SegDecoderCNN
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from Datasets.wjx_dataloader import PolypDataset, RetinaDataset_prototype

torch.cuda.empty_cache()
# set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
# 在导入后立即禁用所有自动混合精度
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False

# 设置默认数据类型
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
# Apply LoRA to image_encoder and mask_decoder
def replace_linear_with_lora(module, lora_r=4, lora_alpha=4):
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

def create_color_mask(mask):
    color_map = {1: [0, 255, 0], 2: [0, 0, 255]}
    color_mask = torch.zeros((mask.shape[1], mask.shape[2], 3), dtype=torch.uint8)
    
    for class_id in range(1, mask.shape[0]):
        class_mask = mask[class_id, :, :] > 0.5
        color = torch.tensor(color_map[class_id], dtype=torch.uint8)
        color_mask[class_mask] = color
    
    return color_mask

def multi_class_dice(pred, target, num_classes):
    dice_scores = []
    for c in range(1, num_classes):
        pred_class = pred[:, c, :, :]
        target_class = target[:, c, :, :]
        intersection = (pred_class * target_class).sum(dim=[1, 2])
        union = pred_class.sum(dim=[1, 2]) + target_class.sum(dim=[1, 2])
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
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

#%%
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--tr_npy_path", type=str, default="./data/Fundus/Domain1/train/ROIs/", help="path to training npy files")
parser.add_argument("-j", "--ts_npy_path", type=str, default="./data/Fundus/Domain1/test/ROIs/", help="path to test npy files")
parser.add_argument("--data_root", type=str, default="./data/Fundus/", help="contain 4 domain")
parser.add_argument("--trainid", type=str, default="1,2,3", help="Comma-separated list of domain IDs for training")
parser.add_argument("--testid", type=str, default="4", help="Comma-separated list of domain IDs for testing")
parser.add_argument("--task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--iternum", type=int, default=20000)
parser.add_argument("--checkpoint", type=str, default="./sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
parser.add_argument("--sam2_checkpoint", type=str, default="./sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="path to SAM2 checkpoint")
parser.add_argument("--load_pretrain", type=bool, default=True, help="load pretrain model")
parser.add_argument("--pretrain_model_path", type=str, default="")
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--warmup_epochs", type=int, default=5, help="number of warmup epochs")
parser.add_argument("--min_lr", type=float, default=1e-6, help="minimum learning rate")
parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping value")
parser.add_argument("--use_wandb", type=bool, default=False, help="use wandb to monitor training")
parser.add_argument("--use_amp", action="store_true", default=False, help="use amp")
parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.use_wandb:
    import wandb
    wandb.login()
    wandb.init(
        project=args.task_name,
        config={"lr": args.lr, "batch_size": args.batch_size, "data_path": args.tr_npy_path, "model_type": args.model_type},
    )

run_id = datetime.now().strftime("%Y%m%d-%H%M")
if "na" in args.task_name:
    model_save_path = join(args.work_dir, "naso_auto", args.task_name+f"-domain{args.trainid}_{args.iternum}", run_id)
else:
    model_save_path = join(args.work_dir, "fundus_auto", args.task_name+f"-domain{args.trainid}_{args.iternum}", run_id)
device = torch.device(args.device)

writer = SummaryWriter(log_dir=f"./eca72c87-0efd-4c31-86af-e7a3bc8ac1fd/wjx_sam/result/log/{args.task_name}/domain{args.trainid}_{args.iternum}/{run_id}")

#%%
class SamFeatSeg(nn.Module):
    def __init__(self, image_encoder, mask_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder

    def forward(self, image):
        original_size = image.shape[2:]
        
        # ImageEncoder返回一个字典，我们需要提取vision_features
        encoder_output = self.image_encoder(image)
        
        # 从字典中提取vision_features
        image_embedding = encoder_output["vision_features"]
        
        # 如果vision_features是(B, H*W, C)格式，重塑为(B, C, H, W)
        if len(image_embedding.shape) == 3:
            b, n, c = image_embedding.shape
            h = w = int(n**0.5)
            image_embedding = image_embedding.permute(0, 2, 1).reshape(b, c, h, w)
        
        low_res_masks = self.mask_decoder(image_embedding)
        ori_res_masks = F.interpolate(low_res_masks, size=original_size, mode="bilinear", align_corners=False)
        ori_res_masks = F.softmax(ori_res_masks, dim=1)
        return ori_res_masks

def setup_optimizer_scheduler(model, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    warmup_epochs = args.warmup_epochs
    total_epochs = args.num_epochs
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(args.min_lr / args.lr, cosine_decay)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

def setup_dataloaders(train_domains, test_domains, data_root, task_name, batch_size, num_workers):
    if "na" in task_name:
        train_paths = [os.path.join(data_root, f"Domain{domain_id}/train/") for domain_id in train_domains]
        val_paths = [os.path.join(data_root, f"Domain{domain_id}/test/") for domain_id in train_domains]
        test_paths = [os.path.join(data_root, f"Domain{domain_id}/train/") for domain_id in test_domains]
        
        train_datasets = [nasopharyngealDataset(path) for path in train_paths]
        val_datasets = [nasopharyngealDataset(path) for path in val_paths]
        test_datasets = [nasopharyngealDataset(path) for path in test_paths]
    elif "poly" in task_name:
        train_paths = [os.path.join(data_root, f"Domain{domain_id}/split_dataset/train/") for domain_id in train_domains]
        val_paths = [os.path.join(data_root, f"Domain{domain_id}/split_dataset/test/") for domain_id in train_domains]
        test_paths = [os.path.join(data_root, f"Domain{domain_id}/split_dataset/test/") for domain_id in test_domains]
        
        train_datasets = [PolypDataset(path) for path in train_paths]
        val_datasets = [PolypDataset(path) for path in val_paths]
        test_datasets = [PolypDataset(path) for path in test_paths]
    else:
        train_paths = [os.path.join(data_root, f"Domain{domain_id}/train/ROIs/") for domain_id in train_domains]
        val_paths = [os.path.join(data_root, f"Domain{domain_id}/test/ROIs/") for domain_id in train_domains]
        test_paths = [os.path.join(data_root, f"Domain{domain_id}/test/ROIs/") for domain_id in test_domains]
        
        train_datasets = [RetinaDataset_prototype(path) for path in train_paths]
        val_datasets = [RetinaDataset_prototype(path) for path in val_paths]
        test_datasets = [RetinaDataset_prototype(path) for path in test_paths]

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    
    generator = torch.Generator(device='cpu')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, generator=generator)
    
    return train_loader, val_loader, test_loader
def setup_model_and_optimizer(args, device):
    """设置模型和优化器 - 使用正确的SAM2库结构"""
    seg_decoder = SegDecoderCNN(num_classes=3)
    
    print(f"=== Loading SAM2 Model ===")
    print(f"Checkpoint path: {args.sam2_checkpoint}")
    
    if not os.path.exists(args.sam2_checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found at: {args.sam2_checkpoint}")
    
    # 先加载检查点
    checkpoint = torch.load(args.sam2_checkpoint, map_location="cpu")
    
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Invalid checkpoint format. Expected dict with 'model' key.")
    
    model_state_dict = checkpoint["model"]
    
    # 构建完整的SAM2图像编码器
    print("\nBuilding complete SAM2 image encoder...")
    
    # 导入必要的模块
    from sam2.modeling.backbones.image_encoder import ImageEncoder
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.backbones.image_encoder import FpnNeck
    
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
    
    # 根据Hiera源代码，我们需要传递正确的参数
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
    # 关键修改：使用trunk.channel_list的相同顺序
    # 根据错误信息，trunk.channel_list是[896, 448, 224, 112]
    # 所以neck.backbone_channel_list应该也是[896, 448, 224, 112]
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
            new_key = k[14:]
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
    
    # 将图像编码器移动到设备
    image_encoder = image_encoder.to(device)
    
    # 冻结图像编码器参数
    print("\n=== Freezing Image Encoder ===")
    for param in image_encoder.parameters():
        param.requires_grad = False
    
    frozen_params = sum(p.numel() for p in image_encoder.parameters())
    trainable_params = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)
    print(f"Total parameters in image encoder: {frozen_params:,}")
    print(f"Trainable parameters in image encoder: {trainable_params:,}")
    
    # 应用LoRA
    if "lora" in args.task_name:
        print("\n=== Applying LoRA ===")
        replace_linear_with_lora(image_encoder)
        print("Applied LoRA to image encoder")
        
        lora_params = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)
        print(f"Trainable parameters after LoRA: {lora_params:,}")
    
    # 创建模型
    print("\n=== Creating Complete Model ===")
    medsam_model = SamFeatSeg(image_encoder=image_encoder, mask_decoder=seg_decoder)
    medsam_model = medsam_model.to(device)
    
    total_params = sum(p.numel() for p in medsam_model.parameters())
    trainable_params = sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable model parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    return medsam_model
def train_epoch(model, dataloader, optimizer, device, iter_num, log_interval, writer, args, scaler=None):
    model.train()
    epoch_loss = 0
    epoch_dice = torch.zeros(2)
    epoch_iou = torch.zeros(2)
    
    for step, (image, gt2D, _) in enumerate(dataloader):
        optimizer.zero_grad()
        image, gt2D = image.to(device), gt2D.to(device)
        
        if args.use_amp and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                medsam_pred = model(image)
                loss = 1 - multi_class_dice(medsam_pred, gt2D, 3).mean()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            medsam_pred = model(image)
            multi_dice = multi_class_dice(medsam_pred, gt2D, 3)
            loss = 1 - (1*multi_dice[0] + 2*multi_dice[1]).mean()
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=args.grad_clip)
            
            optimizer.step()
        
        epoch_loss += loss.item()
        pred = medsam_pred > 0.5
        
        dice_score = multi_class_dice(pred, gt2D, 3)
        epoch_dice[0] += dice_score[0, :].mean().item()
        epoch_dice[1] += dice_score[1, :].mean().item()
        
        iou_score = multi_class_iou(pred, gt2D, 3)
        epoch_iou[0] += iou_score[0, :].mean().item()
        epoch_iou[1] += iou_score[1, :].mean().item()
        
        iter_num += 1
        
        if iter_num % log_interval == 0:
            writer.add_scalar('Train/Loss', loss.item(), iter_num)
            writer.add_scalar('Train/Dice_0', dice_score[0, :].mean().item(), iter_num)
            writer.add_scalar('Train/Dice_1', dice_score[1, :].mean().item(), iter_num)
            writer.add_scalar('Train/IoU_0', iou_score[0, :].mean().item(), iter_num)
            writer.add_scalar('Train/IoU_1', iou_score[1, :].mean().item(), iter_num)
            
            disc_dice = dice_score[0, :].mean().item()
            cup_dice = dice_score[1, :].mean().item()
            disc_iou = iou_score[0, :].mean().item()
            cup_iou = iou_score[1, :].mean().item()
            mean_dice = (disc_dice + cup_dice) / 2
            mean_iou = (disc_iou + cup_iou) / 2
            
            print(f"[Iter {iter_num:5d}] Loss: {loss.item():.4f} | "
                  f"Disc Dice: {disc_dice:.4f} | Cup Dice: {cup_dice:.4f} | Mean Dice: {mean_dice:.4f} | "
                  f"Disc IoU: {disc_iou:.4f} | Cup IoU: {cup_iou:.4f} | Mean IoU: {mean_iou:.4f}")
    
    epoch_loss /= len(dataloader)
    epoch_dice /= len(dataloader)
    epoch_iou /= len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_iou, iter_num

def validate(model, dataloader, device, writer, epoch, max_steps=100):
    model.eval()
    val_loss = 0
    epoch_val_dice = torch.zeros(2)
    epoch_val_iou = torch.zeros(2)
    
    with torch.no_grad():
        for val_step, (image, gt2D, _) in enumerate(dataloader):
            if val_step >= max_steps:
                break
                
            image, gt2D = image.to(device), gt2D.to(device)
            medsam_pred = model(image)
            loss = 1 - multi_class_dice(medsam_pred, gt2D, 3).mean()
            
            val_loss += loss.item()
            
            val_dice = multi_class_dice(medsam_pred, gt2D, 3)
            epoch_val_dice[0] += val_dice[0, :].mean().item()
            epoch_val_dice[1] += val_dice[1, :].mean().item()
            
            val_iou = multi_class_iou(medsam_pred, gt2D, 3)
            epoch_val_iou[0] += val_iou[0, :].mean().item()
            epoch_val_iou[1] += val_iou[1, :].mean().item()
    
    val_loss /= min(len(dataloader), max_steps)
    epoch_val_dice /= min(len(dataloader), max_steps)
    epoch_val_iou /= min(len(dataloader), max_steps)
    
    val_dice_mean = epoch_val_dice.mean().item()
    val_iou_mean = epoch_val_iou.mean().item()
    
    print(f"[Validation Epoch {epoch+1}] Loss: {val_loss:.4f}")
    print(f"  Disc Dice: {epoch_val_dice[0].item():.4f}, Cup Dice: {epoch_val_dice[1].item():.4f}, Mean Dice: {val_dice_mean:.4f}")
    print(f"  Disc IoU: {epoch_val_iou[0].item():.4f}, Cup IoU: {epoch_val_iou[1].item():.4f}, Mean IoU: {val_iou_mean:.4f}")
    
    writer.add_scalar('Validation/Loss', val_loss, epoch+1)
    writer.add_scalar('Validation/Dice_0', epoch_val_dice[0], epoch+1)
    writer.add_scalar('Validation/Dice_1', epoch_val_dice[1], epoch+1)
    writer.add_scalar('Validation/Dice_mean', val_dice_mean, epoch+1)
    writer.add_scalar('Validation/IoU_0', epoch_val_iou[0], epoch+1)
    writer.add_scalar('Validation/IoU_1', epoch_val_iou[1], epoch+1)
    writer.add_scalar('Validation/IoU_mean', val_iou_mean, epoch+1)
    
    return val_loss, epoch_val_dice, epoch_val_iou, val_dice_mean, val_iou_mean

def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, join(model_save_path, run_id + "_" + os.path.basename(__file__)))
    
    if "test" in args.task_name:
        phase = "test"
    else:
        phase = "train"
    
    device = torch.device(args.device)
    
    train_domains = [int(id) for id in args.trainid.split(',')]
    test_domains = [int(id) for id in args.testid.split(',')]
    
    print(f"Training domains: {train_domains}")
    print(f"Testing domains: {test_domains}")
    
    train_loader, val_loader, test_loader = setup_dataloaders(
        train_domains, test_domains, args.data_root, args.task_name, 
        args.batch_size, args.num_workers
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    if phase == "train":
        medsam_model = setup_model_and_optimizer(args, device)
        medsam_model.train()
        
        total_params = sum(p.numel() for p in medsam_model.parameters())
        trainable_params = sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params:.2%}")
        
        optimizer, scheduler = setup_optimizer_scheduler(medsam_model, args)
        
        print(f"\n=== Learning Rate Strategy ===")
        print(f"Initial learning rate: {args.lr}")
        print(f"Warmup epochs: {args.warmup_epochs}")
        print(f"Minimum learning rate: {args.min_lr}")
        print(f"Total epochs: {args.num_epochs}")
        print(f"Gradient clipping: {args.grad_clip}")
        
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        
        num_epochs = args.num_epochs
        iter_num = 0
        log_interval = 100
        
        best_val_disc_dice = 0.0
        best_val_cup_dice = 0.0
        best_val_disc_iou = 0.0
        best_val_cup_iou = 0.0
        best_val_mean_dice = 0.0
        
        best_val_disc_dice_epoch = 0
        best_val_cup_dice_epoch = 0
        best_val_disc_iou_epoch = 0
        best_val_cup_iou_epoch = 0
        best_val_mean_dice_epoch = 0
        
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
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
            
            epoch_loss, epoch_dice, epoch_iou, iter_num = train_epoch(
                medsam_model, train_loader, optimizer, device, 
                iter_num, log_interval, writer, args, scaler
            )
            
            scheduler.step()
            
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch+1)
            writer.add_scalar('Epoch/Loss', epoch_loss, epoch+1)
            writer.add_scalar('Epoch/Dice_0', epoch_dice[0].item(), epoch+1)
            writer.add_scalar('Epoch/Dice_1', epoch_dice[1].item(), epoch+1)
            writer.add_scalar('Epoch/IoU_0', epoch_iou[0].item(), epoch+1)
            writer.add_scalar('Epoch/IoU_1', epoch_iou[1].item(), epoch+1)
            
            epoch_mean_dice = epoch_dice.mean().item()
            epoch_mean_iou = epoch_iou.mean().item()
            
            print(f"[Train Epoch {epoch+1}] Loss: {epoch_loss:.4f}")
            print(f"  Disc Dice: {epoch_dice[0].item():.4f}, Cup Dice: {epoch_dice[1].item():.4f}, Mean Dice: {epoch_mean_dice:.4f}")
            print(f"  Disc IoU: {epoch_iou[0].item():.4f}, Cup IoU: {epoch_iou[1].item():.4f}, Mean IoU: {epoch_mean_iou:.4f}")
            
            print("Validating...")
            val_loss, val_dice, val_iou, val_dice_mean, val_iou_mean = validate(
                medsam_model, val_loader, device, writer, epoch+1
            )
            
            if val_dice[0] > best_val_disc_dice:
                best_val_disc_dice = val_dice[0]
                best_val_disc_dice_epoch = epoch + 1
                
            if val_dice[1] > best_val_cup_dice:
                best_val_cup_dice = val_dice[1]
                best_val_cup_dice_epoch = epoch + 1
                
            if val_iou[0] > best_val_disc_iou:
                best_val_disc_iou = val_iou[0]
                best_val_disc_iou_epoch = epoch + 1
                
            if val_iou[1] > best_val_cup_iou:
                best_val_cup_iou = val_iou[1]
                best_val_cup_iou_epoch = epoch + 1
                
            if val_dice_mean > best_val_mean_dice:
                best_val_mean_dice = val_dice_mean
                best_val_mean_dice_epoch = epoch + 1                
                torch.save(medsam_model.state_dict(), join(model_save_path, "_best_model.pth"))
                print(f"Saved new best model with Mean Dice: {val_dice_mean:.4f}")
            
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        
        print(f"\n=== Training completed ===")
        print(f"Best validation results:")
        print(f"  Disc Dice: {best_val_disc_dice:.4f} at epoch {best_val_disc_dice_epoch}")
        print(f"  Cup Dice: {best_val_cup_dice:.4f} at epoch {best_val_cup_dice_epoch}")
        print(f"  Disc IoU: {best_val_disc_iou:.4f} at epoch {best_val_disc_iou_epoch}")
        print(f"  Cup IoU: {best_val_cup_iou:.4f} at epoch {best_val_cup_iou_epoch}")
        print(f"  Mean Dice: {best_val_mean_dice:.4f} at epoch {best_val_mean_dice_epoch}")
    
    else:
        medsam_model = setup_model_and_optimizer(args, device)
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                medsam_model.load_state_dict(checkpoint["model"])
            else:
                medsam_model.load_state_dict(checkpoint)
            medsam_model.eval()
            print(f"Loaded model from: {args.checkpoint}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
        print("\n=== Testing ===")
        medsam_model.eval()
        
        with torch.no_grad():
            total_dice0 = 0
            total_dice1 = 0
            total_iou0 = 0
            total_iou1 = 0
            num_batches = len(test_loader)
            
            for step, (image, gt2D, _) in enumerate(test_loader):
                image, gt2D = image.to(device), gt2D.to(device)
                
                medsam_pred = model(image)
                
                pred = torch.zeros_like(medsam_pred, dtype=torch.long)
                pred = pred.scatter(1, torch.argmax(medsam_pred, dim=1, keepdim=True), 1)
                
                new_pred = pred.clone()
                new_pred[:, 1] = (pred[:, 1] + pred[:, 2]).clamp(0, 1)
                new_gt2D = gt2D.clone()
                new_gt2D[:, 1] = (gt2D[:, 1] + gt2D[:, 2]).clamp(0, 1)
                
                test_dice = multi_class_dice(new_pred, new_gt2D, 3)
                total_dice0 += test_dice[0, :].mean().item()
                total_dice1 += test_dice[1, :].mean().item()
                
                test_iou = multi_class_iou(new_pred, new_gt2D, 3)
                total_iou0 += test_iou[0, :].mean().item()
                total_iou1 += test_iou[1, :].mean().item()
                
                if step % 10 == 0:
                    image1 = F.interpolate(image[0].unsqueeze(0), size=(800, 800), mode='bilinear', align_corners=False)
                    visualize_on_tensorboard(image1[0], gt2D[0], pred[0], step, writer, f"Test_Step_{step}")
            
            avg_dice0 = total_dice0 / num_batches
            avg_dice1 = total_dice1 / num_batches
            avg_iou0 = total_iou0 / num_batches
            avg_iou1 = total_iou1 / num_batches
            
            print(f"\n=== Test Results ===")
            print(f"Disc Dice: {avg_dice0:.4f}")
            print(f"Cup Dice: {avg_dice1:.4f}")
            print(f"Disc IoU: {avg_iou0:.4f}")
            print(f"Cup IoU: {avg_iou1:.4f}")

if __name__ == "__main__":
    main()