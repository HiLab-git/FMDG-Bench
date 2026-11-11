import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from seg_head.SamFeatSeg import SegDecoderCNN, DPTDecoderForSAM
import loralib as lora
from Datasets.new_dataloader import PolypDatasetInfer, RetinaDatasetInfer, PolypDataset, RetinaDataset_prototype
from naso_dataset_16 import nasopharyngealDataset_single, nasopharyngealDataset
import re
import glob
import sys

# ---------- SAM2 相关导入 ----------
# 请根据你的实际环境调整 sam2 的路径
sys.path.append("./sam2")
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine

DATASET_ROOTS = {
    "fundus": "./data/Fundus",
    "na": "./data/NPC",
    "poly": "./data/PolypDB",
}


def is_medsam_task(task_name: str) -> bool:
    name = task_name.lower()
    return ("medsam" in name) or ("sam-med" in name) or ("sam_med" in name)


def is_sam2_task(task_name: str) -> bool:
    name = task_name.lower()
    return "sam2" in name


def get_sam_checkpoint(method_name, args):
    if is_sam2_task(method_name):
        print("👉 Using SAM2 checkpoint")
        return args.sam2_ckpt
    elif is_medsam_task(method_name):
        print("👉 Using MedSAM checkpoint")
        return args.medsam_ckpt
    else:
        print("👉 Using original SAM checkpoint")
        return args.sam_ckpt


def is_lora_method(name: str) -> bool:
    return "lora" in name.lower()


def get_data_root(task_name):
    name = task_name.lower()
    if "na" in name:
        return DATASET_ROOTS["na"]
    elif "poly" in name:
        return DATASET_ROOTS["poly"]
    else:
        return DATASET_ROOTS["fundus"]


def parse_log_for_model(log_file):
    model_paths = []
    with open(log_file, "r") as f:
        for line in f:
            if "_best_model.pth" in line:
                match = re.search(r'(/.*?_best_model\.pth)', line)
                if match:
                    model_paths.append(match.group(1))
    if len(model_paths) == 0:
        raise ValueError(f"No best model found in {log_file}")
    model_path = model_paths[-1]
    match = re.search(r'(?:fundus_auto|naso_auto|poly_auto)/(.*?)/\d{8}', model_path)
    if not match:
        raise ValueError(f"Cannot parse method name from {model_path}")
    method_name = match.group(1)
    return model_path, method_name


def parse_method(method_name):
    name = method_name.lower()
    if "fundus" in name or "fu" in name:
        task = "fundus"
    elif "poly" in name:
        task = "poly"
    elif "na" in name:
        task = "na"
    else:
        raise ValueError(f"Unknown task in {method_name}")

    if "cnn" in name:
        decoder = "cnn"
    elif "dpt" or "dppt" in name:
        decoder = "dpt"
    else:
        raise ValueError(f"Unknown decoder in {method_name}")

    match = re.search(r'domain([\d,]+)', name)
    train_domains = [int(x) for x in match.group(1).split(",")]
    all_domains = [1, 2, 3, 4]
    test_domains = [d for d in all_domains if d not in train_domains]
    return task, decoder, train_domains, test_domains


def replace_linear_with_lora(module, lora_r=4, lora_alpha=4):
    if lora is None:
        raise ModuleNotFoundError("当前 task_name 启用了 LoRA，但环境中未安装 loralib。请先安装 loralib。")
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


def compute_dice(pred, gt, num_classes):
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        gt_c = (gt == c).astype(np.uint8)
        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        if union == 0:
            dice = 1.0
        else:
            dice = 2.0 * intersection / union
        dices.append(dice)
    return dices


def is_na_task(name): return "na" in name.lower()
def is_poly_task(name): return "poly" in name.lower()
def is_fundus_task(name): return not is_na_task(name) and not is_poly_task(name)
def get_num_classes(name):
    return 2 if (is_na_task(name) or is_poly_task(name)) else 3


# ---------- SAM2 编码器构建函数 ----------
def setup_sam2_image_encoder(ckpt_path, device):
    """手动搭建 SAM2 图像编码器并加载预训练权重"""
    print(f"Loading SAM2 image encoder from {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "model" not in checkpoint:
        raise ValueError("Invalid SAM2 checkpoint format")
    model_state_dict = checkpoint["model"]

    # 推断 embed_dim
    patch_embed_key = "image_encoder.trunk.patch_embed.proj.weight"
    if patch_embed_key in model_state_dict:
        embed_dim = model_state_dict[patch_embed_key].shape[0]
    else:
        embed_dim = 112  # fallback

    # 构建 Hiera backbone（参数需与训练时一致）
    trunk = Hiera(
        embed_dim=embed_dim,
        num_heads=1,
        drop_path_rate=0.0,
        q_pool=3,
        q_stride=(2, 2),
        stages=(2, 3, 16, 3),
        dim_mul=2.0,
        head_mul=2.0,
        window_pos_embed_bkg_spatial_size=(14, 14),
        window_spec=(8, 4, 14, 7),
        global_att_blocks=(12, 16, 20),
        weights_path=None,
        return_interm_layers=True,
    )

    position_encoding = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=trunk.channel_list,
        kernel_size=1, stride=1, padding=0,
        fpn_interp_model="bilinear",
        fuse_type="sum",
    )

    image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=0)

    # 提取 image_encoder 权重
    image_encoder_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith("image_encoder."):
            new_key = k[14:]   # 去除前缀
            image_encoder_state_dict[new_key] = v

    missing, unexpected = image_encoder.load_state_dict(image_encoder_state_dict, strict=False)
    print(f"Loaded image encoder weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return image_encoder


# ---------- 模型定义（兼容 SAM 和 SAM2） ----------
class SamFeatSeg(torch.nn.Module):
    def __init__(self, image_encoder, mask_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.is_sam2 = False  # 由外部设置

    def forward(self, image):
        if self.is_sam2:
            # SAM2 返回 dict，提取 vision_features 并可能 reshape
            encoder_output = self.image_encoder(image)
            image_embedding = encoder_output["vision_features"]
            if len(image_embedding.shape) == 3:   # (B, N, C) -> (B, C, H, W)
                b, n, c = image_embedding.shape
                h = w = int(n ** 0.5)
                image_embedding = image_embedding.permute(0, 2, 1).reshape(b, c, h, w)
        else:
            image_embedding = self.image_encoder(image)

        out = self.mask_decoder(image_embedding)
        return F.interpolate(out, size=image.shape[2:], mode="bilinear", align_corners=False)


def build_model(model_path, task, decoder_type, method_name, args, device):
    num_classes = get_num_classes(task)

    # 解码器
    if decoder_type == "cnn":
        seg_decoder = SegDecoderCNN(num_classes=num_classes)
    else:
        seg_decoder = DPTDecoderForSAM(
            num_classes=num_classes,
            embed_dim=256,
            features=128,
            out_channels=[256, 512, 1024, 2048],
            use_bn=True,
        )

    use_sam2 = is_sam2_task(method_name)

    if use_sam2:
        # ---- SAM2 编码器 ----
        sam2_ckpt = get_sam_checkpoint(method_name, args)
        image_encoder = setup_sam2_image_encoder(sam2_ckpt, device)
        # 冻结整个编码器
        for p in image_encoder.parameters():
            p.requires_grad = False
    else:
        # ---- SAM / MedSAM 编码器 ----
        from segment_anything import sam_model_registry
        sam_ckpt_path = get_sam_checkpoint(method_name, args)
        sam = sam_model_registry[args.model_type](checkpoint=sam_ckpt_path)
        sam.prompt_decoder = torch.nn.Identity()
        sam.mask_decoder = torch.nn.Identity()
        image_encoder = sam.image_encoder
        for p in image_encoder.parameters():
            p.requires_grad = False

    # LoRA 注入（在冻结之后）
    if is_lora_method(method_name):
        print("👉 Injecting LoRA modules into image encoder")
        replace_linear_with_lora(image_encoder)

    model = SamFeatSeg(image_encoder, seg_decoder).to(device)
    model.is_sam2 = use_sam2

    # 加载训练好的权重（必须包含 LoRA 参数）
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------- 数据集 ----------
def build_dataset(data_root, domains, task_name, mode):
    datasets = []
    for d in domains:
        if is_na_task(task_name):
            if mode == "val_vis":
                paths = [os.path.join(data_root, f"Domain{d}/test/")]
            else:
                paths = [
                    os.path.join(data_root, f"Domain{d}/train/"),
                    os.path.join(data_root, f"Domain{d}/test/")
                ]
            datasets += [nasopharyngealDataset(p) for p in paths]
        elif is_poly_task(task_name):
            if mode == "val_vis":
                paths = [os.path.join(data_root, f"Domain{d}/split_dataset/test/")]
            else:
                paths = [
                    os.path.join(data_root, f"Domain{d}/split_dataset/train/"),
                    os.path.join(data_root, f"Domain{d}/split_dataset/test/")
                ]
            datasets += [PolypDataset(data_root=p, is_train=False) for p in paths]
        else:  # fundus
            if mode == "val_vis":
                paths = [os.path.join(data_root, f"Domain{d}/test/ROIs/")]
            else:
                paths = [
                    os.path.join(data_root, f"Domain{d}/train/ROIs/"),
                    os.path.join(data_root, f"Domain{d}/test/ROIs/")
                ]
            datasets += [RetinaDataset_prototype(data_root=p, is_train=False) for p in paths]
    return ConcatDataset(datasets)


def save_mask(pred, save_path, is_fundus):
    pred = pred.astype(np.uint8)
    if not is_fundus:
        cv2.imwrite(save_path, pred * 255)
        return
    color = np.zeros((*pred.shape, 3), dtype=np.uint8)
    color[pred == 1] = [0, 255, 0]
    color[pred == 2] = [0, 0, 255]
    cv2.imwrite(save_path, color)


def split_16_patches(img):
    H, W = img.shape[:2]
    h, w = H // 4, W // 4
    patches = []
    for i in range(4):
        for j in range(4):
            patch = img[i*h:(i+1)*h, j*w:(j+1)*w]
            patches.append(patch)
    return patches


def inference(args, domains):
    device = torch.device(args.device)
    model = build_model(args.model_path, args.task_name, args.decoder, args.method_name, args, device)
    is_fundus = is_fundus_task(args.task_name)

    for d in domains:
        print(f"\n🚀 Inference Domain{d}")
        domain_dice = []
        global_idx = 0
        dataset = build_dataset(args.data_root, [d], args.task_name, args.mode)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        save_root = os.path.join(args.save_dir, args.method_name, args.mode)
        if args.mode == "test_vis":
            save_root = os.path.join(save_root, f"Domain{d}")
        os.makedirs(save_root, exist_ok=True)

        for batch in loader:
            image = batch[0].to(device)
            gt = batch[1]

            with torch.no_grad():
                out = model(image)
                pred = torch.argmax(F.softmax(out, dim=1), dim=1)

            pred_np = pred.squeeze().cpu().numpy()
            if gt is not None:
                gt_np = gt.squeeze().numpy()
                if gt_np.ndim == 3:
                    gt_np = np.argmax(gt_np, axis=0)

            num_classes = get_num_classes(args.task_name)
            dice_list = compute_dice(pred_np, gt_np, num_classes)
            if num_classes == 3:
                dice_od = dice_list[1]
                dice_oc = dice_list[2]
                print(f"   🟢 OD Dice: {dice_od:.4f} | 🔴 OC Dice: {dice_oc:.4f}")
                domain_dice.append([dice_od, dice_oc])
            else:
                dice_mean = np.mean(dice_list)
                domain_dice.append(dice_mean)

            img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)

            pred_patches = split_16_patches(pred_np)
            gt_patches = split_16_patches(gt_np)
            img_patches = split_16_patches(img_np)

            base_name = f"{args.mode}_domain{d}"
            for i in range(16):
                name = f"{base_name}_{global_idx:05d}"
                save_mask(pred_patches[i],
                          os.path.join(save_root, name + "_pred.png"),
                          is_fundus)
                if gt is not None:
                    save_mask(gt_patches[i],
                              os.path.join(save_root, name + "_gt.png"),
                              is_fundus)
                cv2.imwrite(os.path.join(save_root, name + "_img.png"),
                            img_patches[i])
                global_idx += 1

        # 打印统计
        if num_classes == 3:
            domain_dice_np = np.array(domain_dice)
            mean_od = domain_dice_np[:, 0].mean()
            mean_oc = domain_dice_np[:, 1].mean()
            mean_all = (mean_od + mean_oc) / 2
            print(f"✅ Domain{d} Mean OD Dice: {mean_od:.4f}")
            print(f"✅ Domain{d} Mean OC Dice: {mean_oc:.4f}")
            print(f"✅ Domain{d} Mean Dice (OD+OC): {mean_all:.4f}")
        else:
            mean_dice = np.mean(domain_dice)
            print(f"✅ Domain{d} Mean Dice: {mean_dice:.4f}")


def auto_run(args):
    log_files = glob.glob(os.path.join(args.log_dir, "*.log"))
    for log_file in log_files:
        print(f"\n=== Processing {log_file} ===")
        model_path, method_name = parse_log_for_model(log_file)
        task, decoder, train_domains, test_domains = parse_method(method_name)
        print(f"Model: {model_path}")
        print(f"Method: {method_name}")
        print(f"Decoder: {decoder}")
        print(f"Task: {task}")
        print(f"Train domains: {train_domains}")
        print(f"Test domains: {test_domains}")

        args.model_path = model_path
        args.task_name = task
        args.decoder = decoder
        args.data_root = get_data_root(task)
        args.method_name = method_name

        if not os.path.exists(args.data_root):
            raise ValueError(f"Dataset path not found: {args.data_root}")

        args.mode = "val_vis"
        inference(args, train_domains)

        args.mode = "test_vis"
        inference(args, test_domains)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--domains", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, required=False)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--medsam_ckpt", type=str, default="./wjx_sam/medsam_vit_b.pth")
    parser.add_argument("--sam_ckpt", type=str, default="./wjx_sam/sam_vit_b_01ec64.pth")
    # 新增 SAM2 checkpoint 参数
    parser.add_argument("--sam2_ckpt", type=str,
                        default="./sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--save_dir", type=str, default="./runs")
    parser.add_argument("--device", type=str, default="cuda:2")
    args = parser.parse_args()

    if args.log_dir is not None:
        auto_run(args)
    else:
        if args.model_path is None or args.domains is None or args.task_name is None:
            raise ValueError("手动模式必须提供 model_path + domains + task_name")
        args.method_name = "manual_run"
        args.mode = "test_vis"
        inference(args, [int(x) for x in args.domains.split(",")])