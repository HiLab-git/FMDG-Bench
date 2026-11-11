import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import types
import re
import glob
import sys

# 数据集导入（按需，缺失时占位）
try:
    from seg_head.SamFeatSeg import SegDecoderCNN, DPTDecoderForSAM
except ImportError:
    SegDecoderCNN = None
    DPTDecoderForSAM = None

try:
    from Datasets.new_dataloader import PolypDataset, RetinaDataset_prototype
except ImportError:
    PolypDataset = None
    RetinaDataset_prototype = None

try:
    from naso_dataset_16 import nasopharyngealDataset
except ImportError:
    nasopharyngealDataset = None

# LoRA 按需导入
try:
    import loralib as lora
except ModuleNotFoundError:
    lora = None

# SAM3 路径
sys.path.append("./sam3")
from sam3.model.vitdet import ViT

DATASET_ROOTS = {
    "fundus": "./data/Fundus/",
    "na": "./data/MS_NPC_guran/preprocessed/",
    "poly": "./data/PolypDB/PolypDB/PolypDB_modality_wise/",
}

# ----------------------- 辅助函数 -----------------------
def is_sam3_task(task_name: str) -> bool:
    return "sam3" in task_name.lower()

def is_lora_method(name: str) -> bool:
    return "lora" in name.lower()

def is_medsam_task(task_name: str) -> bool:
    name = task_name.lower()
    return ("medsam" in name) or ("sam-med" in name) or ("sam_med" in name)

def get_sam_checkpoint(method_name, args):
    if is_sam3_task(method_name):
        print("👉 Using SAM3 checkpoint")
        return args.sam3_ckpt
    elif is_medsam_task(method_name):
        print("👉 Using MedSAM checkpoint")
        return args.medsam_ckpt
    else:
        print("👉 Using original SAM checkpoint")
        return args.sam_ckpt

def get_data_root(task_name):
    name = task_name.lower()
    if "na" in name: return DATASET_ROOTS["na"]
    elif "poly" in name: return DATASET_ROOTS["poly"]
    else: return DATASET_ROOTS["fundus"]

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
    if "fundus" in name or "fu-" in name: task = "fundus"
    elif "poly" in name: task = "poly"
    elif "na" in name: task = "na"
    else: raise ValueError(f"Unknown task in {method_name}")

    if "cnn" in name: decoder = "cnn"
    elif "dpt" in name: decoder = "dpt"
    else: raise ValueError(f"Unknown decoder in {method_name}")

    match = re.search(r'domain([\d,]+)', name)
    train_domains = [int(x) for x in match.group(1).split(",")]
    all_domains = [1,2,3,4]
    test_domains = [d for d in all_domains if d not in train_domains]
    return task, decoder, train_domains, test_domains

def replace_linear_with_lora(module, lora_r=4, lora_alpha=4):
    if lora is None:
        raise ModuleNotFoundError("当前 task_name 启用了 LoRA，但环境中未安装 loralib。")
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

def disable_sam3_fused_mlp_for_lora(module):
    """将 SAM3 中的 fused MLP 替换为普通 forward，避免 LoRA 梯度错误"""
    patched_count = 0
    for name, child in module.named_modules():
        if hasattr(child, "fc1") and hasattr(child, "fc2") and hasattr(child, "act"):
            fc1 = getattr(child, "fc1")
            fc2 = getattr(child, "fc2")
            act = getattr(child, "act")
            if isinstance(fc1, nn.Module) and isinstance(fc2, nn.Module):
                def new_forward(self, x):
                    x = self.fc1(x)
                    x = self.act(x)
                    x = self.fc2(x)
                    return x
                child.forward = types.MethodType(new_forward, child)
                patched_count += 1
    print(f"✅ 已为 LoRA 模式关闭 SAM3 fused MLP，patched modules: {patched_count}")

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

# ----------------------- SAM3 编码器加载（复用训练脚本逻辑） -----------------------
def _unwrap_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model", "state_dict", "model_state_dict", "network", "net"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint

def _build_best_matched_state_dict(model, raw_state_dict):
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    prefix_candidates = [
        "", "trunk.", "image_encoder.", "image_encoder.trunk.", "backbone.", "backbone.trunk.",
        "backbone.vision_backbone.", "backbone.vision_backbone.trunk.",
        "detector.", "detector.backbone.", "detector.backbone.vision_backbone.",
        "detector.backbone.vision_backbone.trunk.",
        "module.", "module.trunk.", "module.image_encoder.", "module.backbone.",
        "module.backbone.vision_backbone.", "module.backbone.vision_backbone.trunk.",
        "module.detector.", "module.detector.backbone.", "module.detector.backbone.vision_backbone.",
        "module.detector.backbone.vision_backbone.trunk.",
    ]
    best_loaded = {}
    best_prefix = None
    for prefix in prefix_candidates:
        current_loaded = {}
        for ckpt_key, value in raw_state_dict.items():
            new_key = ckpt_key
            if prefix and ckpt_key.startswith(prefix):
                new_key = ckpt_key[len(prefix):]
            if new_key in model_keys and model_state[new_key].shape == value.shape:
                target_dtype = model_state[new_key].dtype
                if value.dtype != target_dtype:
                    value = value.to(dtype=target_dtype)
                current_loaded[new_key] = value
        if len(current_loaded) > len(best_loaded):
            best_loaded = current_loaded
            best_prefix = prefix
    return best_loaded, best_prefix

def setup_sam3_image_encoder(ckpt_path, device, method_name, lora_r=4, lora_alpha=4):
    print("=== 加载 SAM3 图像编码器 ===")
    image_encoder = ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        in_chans=3,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        qkv_bias=True,
        drop_path_rate=0.1,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    raw_state_dict = _unwrap_checkpoint_state_dict(checkpoint)
    if not isinstance(raw_state_dict, dict):
        raise ValueError(f"SAM3 checkpoint 格式不对，得到的是 {type(raw_state_dict)}")

    matched_state_dict, best_prefix = _build_best_matched_state_dict(image_encoder, raw_state_dict)
    if len(matched_state_dict) == 0:
        sample_keys = list(raw_state_dict.keys())[:30]
        raise RuntimeError(f"SAM3 权重一个都没匹配上。\n样本keys: {sample_keys}")

    load_msg = image_encoder.load_state_dict(matched_state_dict, strict=False)
    print(f"SAM3 权重加载完成 (prefix={repr(best_prefix)}), missing={len(load_msg.missing_keys)}, unexpected={len(load_msg.unexpected_keys)}")

    # 冻结所有参数
    for param in image_encoder.parameters():
        param.requires_grad = False

    # 如果需要 LoRA
    if is_lora_method(method_name):
        disable_sam3_fused_mlp_for_lora(image_encoder)
        replace_linear_with_lora(image_encoder, lora_r=lora_r, lora_alpha=lora_alpha)
        # 仅 LoRA 参数可训
        for name, param in image_encoder.named_parameters():
            param.requires_grad = "lora_" in name
        lora_trainable = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)
        print(f"✅ LoRA 注入完成，可训练参数: {lora_trainable:,}")
    else:
        print("✅ 不使用 LoRA")

    return image_encoder

# ----------------------- 模型定义 -----------------------
class SamFeatSeg(torch.nn.Module):
    def __init__(self, image_encoder, mask_decoder, method_name):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.adapter = nn.Conv2d(1024, 256, kernel_size=1)
        self.is_sam3 = True          # 强制为 SAM3
        self.use_lora = is_lora_method(method_name)

    def forward(self, image):
        original_size = image.shape[2:]
        target_size = 1008
        if image.shape[-2:] != (target_size, target_size):
            image = F.interpolate(image, (target_size, target_size), mode="bilinear", align_corners=False)

        # 推断时无梯度
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16): 
                feat = self.image_encoder(image)

        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.dim() == 3:
            B, L, C = feat.shape
            H = W = int(L ** 0.5)
            feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
        elif feat.dim() != 4:
            raise ValueError(f"Unexpected feat shape: {feat.shape}")

        feat = self.adapter(feat)
        out = self.mask_decoder(feat)
        return F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)


def build_model(model_path, task, decoder_type, method_name, args, device):
    num_classes = get_num_classes(task)

    # 解码器
    if decoder_type == "cnn":
        if SegDecoderCNN is None:
            raise ImportError("SegDecoderCNN 未导入")
        seg_decoder = SegDecoderCNN(num_classes=num_classes)
    else:
        if DPTDecoderForSAM is None:
            raise ImportError("DPTDecoderForSAM 未导入")
        seg_decoder = DPTDecoderForSAM(
            num_classes=num_classes,
            embed_dim=256,
            features=128,
            out_channels=[256, 512, 1024, 2048],
            use_bn=True,
        )

    use_sam3 = is_sam3_task(method_name)
    if use_sam3:
        # 加载 SAM3 编码器（自动处理冻结和 LoRA）
        sam3_ckpt = get_sam_checkpoint(method_name, args)
        image_encoder = setup_sam3_image_encoder(sam3_ckpt, device, method_name)
    else:
        # SAM / MedSAM 分支（此处省略，您可按需添加）
        raise NotImplementedError("当前仅支持 SAM3，如需其他编码器请参考之前的推断脚本")

    model = SamFeatSeg(image_encoder, seg_decoder, method_name).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ----------------------- 数据集 -----------------------
def build_dataset(data_root, domains, task_name, mode):
    datasets = []
    for d in domains:
        if is_na_task(task_name):
            paths = [os.path.join(data_root, f"Domain{d}/test/")] if mode == "val_vis" else [
                os.path.join(data_root, f"Domain{d}/train/"),
                os.path.join(data_root, f"Domain{d}/test/")
            ]
            datasets += [nasopharyngealDataset(p) for p in paths]
        elif is_poly_task(task_name):
            paths = [os.path.join(data_root, f"Domain{d}/split_dataset/test/")] if mode == "val_vis" else [
                os.path.join(data_root, f"Domain{d}/split_dataset/train/"),
                os.path.join(data_root, f"Domain{d}/split_dataset/test/")
            ]
            datasets += [PolypDataset(data_root=p, is_train=False) for p in paths]
        else:  # fundus
            paths = [os.path.join(data_root, f"Domain{d}/test/ROIs/")] if mode == "val_vis" else [
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
            patches.append(img[i*h:(i+1)*h, j*w:(j+1)*w])
    return patches

# ----------------------- 推理 -----------------------
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
                domain_dice.append(np.mean(dice_list))

            img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)

            pred_patches = split_16_patches(pred_np)
            gt_patches = split_16_patches(gt_np)
            img_patches = split_16_patches(img_np)

            base_name = f"{args.mode}_domain{d}"
            for i in range(16):
                name = f"{base_name}_{global_idx:05d}"
                save_mask(pred_patches[i], os.path.join(save_root, name + "_pred.png"), is_fundus)
                if gt is not None:
                    save_mask(gt_patches[i], os.path.join(save_root, name + "_gt.png"), is_fundus)
                cv2.imwrite(os.path.join(save_root, name + "_img.png"), img_patches[i])
                global_idx += 1

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
    parser.add_argument("--medsam_ckpt", type=str, default="/path/to/medsam_vit_b.pth")
    parser.add_argument("--sam_ckpt", type=str, default="/path/to/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam3_ckpt", type=str, default="./sam3/sam3/moxing/sam3.pt",
                        help="Path to SAM3 checkpoint (.pt)")
    parser.add_argument("--save_dir", type=str, default="./runs")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.log_dir is not None:
        auto_run(args)
    else:
        if args.model_path is None or args.domains is None or args.task_name is None:
            raise ValueError("手动模式必须提供 model_path + domains + task_name")
        args.method_name = "manual_run"
        args.mode = "test_vis"
        inference(args, [int(x) for x in args.domains.split(",")])