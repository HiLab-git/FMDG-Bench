# train_clip_cnn_final.py
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import random
import json

# 添加LoRA相关导入
try:
    import loralib as lora
except ImportError:
    print("请安装loralib：pip install loralib")
    sys.exit(1)


# ==================== CNN解码器定义 ====================
class SegDecoderCNN(nn.Module):
    """与第一个程序完全相同的CNN解码器"""
    def __init__(self,
                 num_classes=1,
                 embed_dim=256,
                 num_depth=2,
                 top_channel=64,
                 ):
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
                    nn.ConvTranspose2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 2,
                                       stride=2)
                )

            self.blocks.append(block)

        self.final = nn.Sequential(
            nn.Conv2d(top_channel, top_channel, 3, padding=1),
            nn.Conv2d(top_channel, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_block(x)
        for blk in self.blocks:
            x = blk(x)
        return self.final(x)


class CLIPFeatSeg(nn.Module):
    """CLIP + CNN解码器的组合模型"""
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        img_size=224,
        patch_size=16,
        clip_model_type="ViT-B/16"
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder

        if "ViT-B" in clip_model_type:
            self.clip_dim = 768
        elif "ViT-L" in clip_model_type:
            self.clip_dim = 1024
        elif "RN" in clip_model_type:
            self.clip_dim = 512
        else:
            self.clip_dim = 512

        self.num_patches = (img_size // patch_size) ** 2
        self.feature_projection = nn.Sequential(
            nn.Conv2d(self.clip_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.upsample_to_decoder = nn.Upsample(
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        )

    def extract_patch_tokens(self, x):
        self.image_encoder.eval()

        if hasattr(self.image_encoder, '__call__'):
            try:
                output = self.image_encoder(x)
                if isinstance(output, torch.Tensor):
                    last_hidden_state = output
                elif hasattr(output, 'last_hidden_state'):
                    last_hidden_state = output.last_hidden_state
                else:
                    raise AttributeError("无法从模型输出中提取特征")
                patch_tokens = last_hidden_state[:, 1:, :]
                return patch_tokens
            except Exception as e:
                print(f"直接调用模型失败: {e}")

        try:
            x = self.image_encoder.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([
                self.image_encoder.class_embedding.to(x.dtype) +
                torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
            ], dim=1)
            x = x + self.image_encoder.positional_embedding.to(x.dtype)
            x = self.image_encoder.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.image_encoder.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.image_encoder.ln_post(x)
            patch_tokens = x[:, 1:, :]
            return patch_tokens
        except AttributeError as e:
            print(f"标准CLIP方式失败: {e}")
            raise RuntimeError(f"无法从CLIP视觉编码器提取特征。编码器类型: {type(self.image_encoder)}")

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
            out = nn.functional.interpolate(
                out,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )
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
                bias=child.bias is not None
            )
            lora_module.weight = child.weight
            if child.bias is not None:
                lora_module.bias = child.bias
            setattr(module, name, lora_module)
        else:
            replace_linear_with_lora(child, lora_r=lora_r, lora_alpha=lora_alpha)


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

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, eps=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        """
        logits: [B, 1, H, W]
        target: [B, 1, H, W]
        """
        target = target.float().clamp(0, 1)

        bce_loss = self.bce(logits, target)

        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_score = (2.0 * inter + self.eps) / (union + self.eps)
        dice_loss = 1.0 - dice_score.mean()

        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total_loss, dice_loss, bce_loss

def train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes=1, dice_thr=0.5, vis_dir=None, epoch=0):
    model.train()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    dice_scores = []
    iou_scores = []
    first_batch_logged = False

    for step, (inputs, targets, _) in enumerate(train_loader):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()
        logits = model(inputs)

        if num_classes == 1:
            total_batch_loss, dice_loss, bce_loss = criterion(logits, targets)
        else:
            raise NotImplementedError("当前这个Polyp脚本是二分类版，组合损失按二分类实现。")

        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_dice_loss += dice_loss.item()
        total_bce_loss += bce_loss.item()

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
    avg_bce_loss = total_bce_loss / max(1, len(train_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0

    print(
        f"[Train Epoch {epoch}] "
        f"loss={avg_loss:.4f}  bce={avg_bce_loss:.4f}  diceloss={avg_dice_loss:.4f}  "
        f"dice={avg_dice:.4f}  iou={avg_iou:.4f}"
    )
    return avg_loss, avg_dice_loss, avg_bce_loss, avg_dice, avg_iou

@torch.no_grad()
def evaluate(model, data_loader, criterion, device, num_classes=1, dice_thr=0.5, vis_dir=None, split_name="Val"):
    model.eval()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    dice_scores = []
    iou_scores = []
    idx_global = 0

    for inputs, targets, _ in data_loader:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        logits = model(inputs)

        if num_classes == 1:
            total_batch_loss, dice_loss, bce_loss = criterion(logits, targets)
        else:
            raise NotImplementedError("当前这个Polyp脚本是二分类版，组合损失按二分类实现。")

        total_loss += total_batch_loss.item()
        total_dice_loss += dice_loss.item()
        total_bce_loss += bce_loss.item()

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
                    fname_prefix=split_name.lower()
                )
                idx_global += 1

    avg_loss = total_loss / max(1, len(data_loader))
    avg_dice_loss = total_dice_loss / max(1, len(data_loader))
    avg_bce_loss = total_bce_loss / max(1, len(data_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0

    print(
        f"[{split_name}] "
        f"loss={avg_loss:.4f}  bce={avg_bce_loss:.4f}  diceloss={avg_dice_loss:.4f}  "
        f"dice={avg_dice:.4f}  iou={avg_iou:.4f}"
    )
    return avg_loss, avg_dice_loss, avg_bce_loss, avg_dice, avg_iou



def load_clip_model_simple(model_type, model_name, device, local_model_dir="./models"):
    """极简版本：从本地加载模型"""
    if model_type == 'medclip':
        print(f"Loading MedCLIP: {model_name}")
        if model_name == 'ViT-B/32':
            local_path = os.path.join(local_model_dir, "pubmed_clip_vit_b32")
        else:
            local_path = os.path.join(local_model_dir, "pubmed_clip")

        print(f"Model path: {local_path}")
        if not os.path.exists(local_path):
            print(f"ERROR: Model directory not found: {local_path}")
            print("Falling back to standard CLIP...")
            import clip
            model, preprocess = clip.load("ViT-B/32", device=device)
            model = model.float()
            return model.visual, preprocess

        print("✓ All model files found!")

        try:
            from transformers import CLIPVisionModel, CLIPProcessor, CLIPVisionConfig
            from transformers import CLIPImageProcessor
            import json

            print("Attempting to load model manually...")
            config_path = os.path.join(local_path, "config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            if "vision_config" in config_dict:
                vision_config_dict = config_dict["vision_config"]
            else:
                vision_config_dict = config_dict

            vision_config = CLIPVisionConfig.from_dict(vision_config_dict)
            vision_model = CLIPVisionModel(vision_config)
            weights_path = os.path.join(local_path, "pytorch_model.bin")

            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                print("✓ Loaded weights with weights_only=True")
            except Exception as e:
                print(f"Weights_only loading failed: {e}")
                print("Warning: Using standard torch.load")
                state_dict = torch.load(weights_path, map_location="cpu")

            print(f"Total weights in model: {len(state_dict.keys())}")
            print("First 5 keys:", list(state_dict.keys())[:5])

            vision_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("vision_model."):
                    vision_state_dict[key] = value

            print(f"Extracted {len(vision_state_dict)} visual weights")
            if len(vision_state_dict) == 0:
                print("Warning: No vision-specific weights found. Trying alternative prefix...")
                for key, value in state_dict.items():
                    if key.startswith("visual."):
                        new_key = "vision_model." + key.replace("visual.", "")
                        vision_state_dict[new_key] = value
                    elif key.startswith("vision_encoder."):
                        new_key = "vision_model." + key.replace("vision_encoder.", "")
                        vision_state_dict[new_key] = value

            if len(vision_state_dict) == 0:
                print("No vision weights found, using all weights...")
                vision_state_dict = state_dict

            missing_keys, unexpected_keys = vision_model.load_state_dict(vision_state_dict, strict=False)
            if missing_keys:
                print(f"⚠ Missing keys count: {len(missing_keys)}")
                if len(missing_keys) > 0:
                    print(f"First 5 missing keys: {missing_keys[:5]}")
                    for missing_key in missing_keys[:10]:
                        alt_key = missing_key.replace("vision_model.", "")
                        if alt_key in state_dict:
                            print(f"  Found alternative key for {missing_key}: {alt_key}")
                            vision_state_dict[missing_key] = state_dict[alt_key]
                    vision_model.load_state_dict(vision_state_dict, strict=False)

            if unexpected_keys:
                print(f"⚠ Unexpected keys count: {len(unexpected_keys)}")
                if len(unexpected_keys) > 0:
                    print(f"First 5 unexpected keys: {unexpected_keys[:5]}")

            print("Loading processor offline...")
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"

            try:
                processor = CLIPProcessor.from_pretrained(local_path, local_files_only=True)
                print("✓ CLIPProcessor loaded from local (offline)")
            except Exception as e:
                print(f"Failed to load processor from local: {e}")
                try:
                    image_processor = CLIPImageProcessor.from_pretrained(local_path, local_files_only=True)

                    class SimpleCLIPProcessor:
                        def __init__(self, image_processor):
                            self.image_processor = image_processor
                        def __call__(self, images, return_tensors="pt", **kwargs):
                            return self.image_processor(images, return_tensors=return_tensors, **kwargs)
                    processor = SimpleCLIPProcessor(image_processor)
                    print("✓ Created simple processor from local image processor")
                except Exception as e2:
                    print(f"Failed to create processor: {e2}")
                    from transformers.image_processing_utils import BatchFeature
                    import torchvision.transforms as transforms
                    from PIL import Image

                    class CustomImageProcessor:
                        def __init__(self):
                            self.mean = [0.48145466, 0.4578275, 0.40821073]
                            self.std = [0.26862954, 0.26130258, 0.27577711]
                            self.size = 224
                            self.transform = transforms.Compose([
                                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.CenterCrop(self.size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=self.mean, std=self.std)
                            ])
                        def __call__(self, images, return_tensors="pt", **kwargs):
                            if not isinstance(images, list):
                                images = [images]
                            processed_images = []
                            for img in images:
                                if isinstance(img, str):
                                    img = Image.open(img).convert("RGB")
                                elif isinstance(img, Image.Image):
                                    pass
                                elif isinstance(img, np.ndarray):
                                    img = Image.fromarray(img)
                                else:
                                    raise TypeError(f"Unsupported image type: {type(img)}")
                                processed = self.transform(img)
                                processed_images.append(processed)
                            pixel_values = torch.stack(processed_images)
                            return BatchFeature(data={"pixel_values": pixel_values})
                    processor = CustomImageProcessor()
                    print("✓ Created custom image processor")

            vision_model = vision_model.to(device)
            vision_model = vision_model.float()
            vision_model.eval()

            print("Testing forward pass...")
            with torch.no_grad():
                test_input = torch.randn(1, 3, 224, 224).to(device)
                try:
                    output = vision_model(test_input, output_hidden_states=True)
                    print(f"✓ Forward pass successful")
                    print(f"Output shape: {output.last_hidden_state.shape}")
                except Exception as e:
                    print(f"Forward pass test failed: {e}")

            class SimpleVisionWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def eval(self):
                    self.model.eval()
                    return self
                def train(self, mode=True):
                    self.model.train(mode)
                    return self
                def parameters(self):
                    return self.model.parameters()
                def named_parameters(self):
                    return self.model.named_parameters()
                def to(self, device):
                    self.model = self.model.to(device)
                    return self
                def __call__(self, x):
                    return self.model(x, output_hidden_states=True).last_hidden_state

            backbone = SimpleVisionWrapper(vision_model)
            backbone = backbone.to(device)
            backbone.eval()
            for param in backbone.model.parameters():
                param.requires_grad = False

            print("✓ PubMedCLIP loaded successfully!")
            print(f"  Backbone 类型: {type(backbone)}")
            print(f"  Backbone 参数数量: {len(list(backbone.parameters()))}")
            return backbone, processor

        except Exception as e:
            print(f"❌ Failed to load PubMedCLIP: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to standard CLIP...")
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
            import clip
            model, preprocess = clip.load("ViT-B/32", device=device)
            model = model.float()
            return model.visual, None
    else:
        print(f"{model_type} not implemented, using standard CLIP...")
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.float()
        return model.visual, None


def build_polyp_datasets(args, train_transform, val_transform, test_transform):
    from dataset import FolderDataset_PolypDB

    train_domains = [int(domain_id) for domain_id in args.trainid.split(',')]
    test_domains = [int(domain_id) for domain_id in args.testid.split(',')]

    print(f"✅ Training domains: {train_domains}")
    print(f"✅ Testing domains: {test_domains}")

    train_datasets = []
    val_datasets = []
    test_domain_datasets = {}

    # train: trainid/train
    for domain_id in train_domains:
        path = os.path.join(args.data_dir, f"Domain{domain_id}/split_dataset/train/")
        if os.path.exists(path):
            dataset = FolderDataset_PolypDB(
                root=path,
                split="train",
                img_dir_name=args.img_dir_name,
                label_dir_name=args.label_dir_name,
                mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
                transform=train_transform,
            )
            train_datasets.append(dataset)
            print(f"✅ Loaded train dataset from Domain{domain_id}/split_dataset/train: {len(dataset)} samples")
        else:
            print(f"⚠️ Warning: train path not found: {path}")

    # val: trainid/test
    for domain_id in train_domains:
        path = os.path.join(args.data_dir, f"Domain{domain_id}/split_dataset/test/")
        if os.path.exists(path):
            dataset = FolderDataset_PolypDB(
                root=path,
                split="test",
                img_dir_name=args.img_dir_name,
                label_dir_name=args.label_dir_name,
                mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
                transform=val_transform,
            )
            val_datasets.append(dataset)
            print(f"✅ Loaded val dataset from Domain{domain_id}/split_dataset/test: {len(dataset)} samples")
        else:
            print(f"⚠️ Warning: val path not found: {path}")

    # test: 每个 testid 单独构建
    for domain_id in test_domains:
        domain_subsets = []

        for split_name in ["train", "test"]:
            path = os.path.join(args.data_dir, f"Domain{domain_id}/split_dataset/{split_name}/")
            if os.path.exists(path):
                dataset = FolderDataset_PolypDB(
                    root=path,
                    split=split_name,
                    img_dir_name=args.img_dir_name,
                    label_dir_name=args.label_dir_name,
                    mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
                    transform=test_transform,
                )
                domain_subsets.append(dataset)
                print(f"✅ Loaded test dataset from Domain{domain_id}/split_dataset/{split_name}: {len(dataset)} samples")
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--model_type", type=str, default="clip",
                       choices=["clip", "medclip", "chexclip", "fundusclip"],
                       help="Type of CLIP model to use")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                       choices=["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
                               "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"])
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--img_dir_name", type=str, default="images")
    parser.add_argument("--label_dir_name", type=str, default="masks")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--trainid", type=str, default="1,2,3")
    parser.add_argument("--testid", type=str, default="4")
    parser.add_argument("--task_name", type=str, default="PolypDB_D_CNN")
    parser.add_argument("--model_cache_dir", type=str, default="./models",
                       help="本地模型缓存目录")
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
            raise ValueError(f"Invalid CUDA ID {args.cuda_id}! Only {cuda_count} GPU(s) available")
        torch.cuda.set_device(args.cuda_id)
        device = f"cuda:{args.cuda_id}"
        print(f"Using CUDA device ID: {args.cuda_id} (device: {device})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_root = f"./runs/{args.model_type}_cnn_{args.clip_model.replace('/', '_')}_{args.input_h}_{args.task_name}_train{args.trainid}_test{args.testid}"
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

    print(f"\n{'='*60}")
    print(f"Loading {args.model_type} model: {args.clip_model}")
    print(f"{'='*60}")

    try:
        backbone, preprocess = load_clip_model_simple(
            args.model_type,
            args.clip_model,
            device,
            local_model_dir=args.model_cache_dir
        )
    except Exception as e:
        print(f"Failed to load {args.model_type} model: {e}")
        print("Falling back to standard CLIP...")
        import clip
        model_tmp, preprocess = clip.load(args.clip_model, device=device)
        model_tmp = model_tmp.float()
        backbone = model_tmp.visual
        args.model_type = "clip"

    if not isinstance(backbone, nn.Module):
        print(f"⚠️ Warning: backbone is not nn.Module, type: {type(backbone)}")
        class BackboneWrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            def forward(self, x):
                return self.backbone(x)
        backbone = BackboneWrapper(backbone)
        print("✅ 已将backbone包装为nn.Module")

    backbone = backbone.to(device)
    backbone.eval()

    if use_lora:
        visual_model = backbone
        if hasattr(backbone, 'model'):
            visual_model = backbone.model
        replace_linear_with_lora(visual_model, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        print("✅ LoRA已应用到CLIP视觉编码器")

    seg_decoder = SegDecoderCNN(
        num_classes=args.num_classes,
        embed_dim=256,
        num_depth=2,
        top_channel=64
    )

    model = CLIPFeatSeg(
        image_encoder=backbone,
        mask_decoder=seg_decoder,
        img_size=args.input_h,
        patch_size=16,
        clip_model_type=args.clip_model
    )
    model = model.to(device)

    print("\n🔍 模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params_count:,}")
    print(f"  可训练比例: {trainable_params_count/total_params:.2%}")

    if use_lora:
        trainable_params = []
        trainable_params.extend(model.mask_decoder.parameters())
        trainable_params.extend(model.feature_projection.parameters())
        trainable_params.extend(model.upsample_to_decoder.parameters())
        for _, param in backbone.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print(f"\n✅ LoRA训练模式:")
        print(f"   优化参数数量: {len(trainable_params)}组")
    else:
        for param in backbone.parameters():
            param.requires_grad = False
        trainable_params = list(model.mask_decoder.parameters()) + \
                          list(model.feature_projection.parameters()) + \
                          list(model.upsample_to_decoder.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print(f"\n✅ 标准训练模式（冻结backbone）:")
        print(f"   解码器和适配器参数数量: {len(trainable_params)}组")

    criterion = DiceBCELoss(
        dice_weight=0.5,
        bce_weight=0.5,
    )

    from dataset import ResizeAndNormalize

    train_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    val_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    test_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))

    train_dataset, val_dataset, test_domain_datasets = build_polyp_datasets(
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
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    test_loaders = {}
    for domain_id, test_dataset in test_domain_datasets.items():
        test_loaders[domain_id] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )

    best_val_dice = -1.0
    best_val_dice_epoch = -1
    best_val_iou = -1.0
    best_val_iou_epoch = -1

    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice_loss, train_bce_loss, train_dice, train_iou = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=args.num_classes,
            dice_thr=0.5,
            vis_dir=train_vis_dir,
            epoch=epoch
        )

        val_loss, val_dice_loss, val_bce_loss, val_dice, val_iou = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=args.num_classes,
            dice_thr=0.5,
            vis_dir=val_vis_dir,
            split_name="Val"
        )

        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "best_val_dice": best_val_dice,
                "best_val_iou": best_val_iou,
                "train_loss": train_loss,
                "train_dice_loss": train_dice_loss,
                "train_bce_loss": train_bce_loss,
                "val_loss": val_loss,
                "val_dice_loss": val_dice_loss,
                "val_bce_loss": val_bce_loss
            },
            latest_model_path
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
                    "args": vars(args),
                    "best_val_dice": best_val_dice,
                    "best_val_iou": best_val_iou,
                    "train_loss": train_loss,
                    "train_dice_loss": train_dice_loss,
                    "train_bce_loss": train_bce_loss,
                    "val_loss": val_loss,
                    "val_dice_loss": val_dice_loss,
                    "val_bce_loss": val_bce_loss
                },
                best_model_path
            )
            print(f"[Save] New best ckpt: {best_model_path}")
        elif val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_iou_epoch = epoch

    print("=" * 60)
    print(f"[Summary] Best Val Dice = {best_val_dice:.4f} @ epoch {best_val_dice_epoch}")
    print(f"[Summary] Best Val IoU = {best_val_iou:.4f} @ epoch {best_val_iou_epoch}")
    print("=" * 60)

    print("\n🧪 Loading best model for final testing...")
    best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
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

        test_loss, test_dice_loss, test_bce_loss, test_dice, test_iou = evaluate(
            model,
            test_loader,
            criterion,
            device,
            num_classes=args.num_classes,
            dice_thr=0.5,
            vis_dir=domain_vis_dir,
            split_name=f"Test_Domain{domain_id}"
        )

        all_domain_results[domain_id] = {
            "loss": test_loss,
            "dice_loss": test_dice_loss,
            "bce_loss": test_bce_loss,
            "dice": test_dice,
            "iou": test_iou,
        }

        print("-" * 60)
        print(f"📍 Domain{domain_id} Results:")
        print(f"[Final Test][Domain{domain_id}] Loss     = {test_loss:.4f}")
        print(f"[Final Test][Domain{domain_id}] BCE      = {test_bce_loss:.4f}")
        print(f"[Final Test][Domain{domain_id}] DiceLoss = {test_dice_loss:.4f}")
        print(f"[Final Test][Domain{domain_id}] Dice     = {test_dice:.4f}")
        print(f"[Final Test][Domain{domain_id}] IoU      = {test_iou:.4f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("✅ Summary of Per-Domain Test Results:")
    print("=" * 60)
    for domain_id, result in all_domain_results.items():
        print(
            f"Domain{domain_id} | "
            f"Loss={result['loss']:.4f}  "
            f"BCE={result['bce_loss']:.4f}  "
            f"DiceLoss={result['dice_loss']:.4f}  "
            f"Dice={result['dice']:.4f}  "
            f"IoU={result['iou']:.4f}"
        )
    print("=" * 60)

    results = {
        "model_type": args.model_type,
        "clip_model": args.clip_model,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "best_epoch": best_val_dice_epoch,
        "train_id": args.trainid,
        "test_id": args.testid,
        "input_size": f"{args.input_h}x{args.input_w}",
        "final_test_loss": float(test_loss),
        "final_test_bce": float(test_bce_loss),
        "final_test_dice_loss": float(test_dice_loss),
        "final_test_dice": float(test_dice),
        "final_test_iou": float(test_iou),
    }

    results_path = os.path.join(save_root, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
