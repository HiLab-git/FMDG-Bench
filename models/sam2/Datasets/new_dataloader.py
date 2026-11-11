import pdb
import loralib as lora
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用 'Agg' 后端以适应无 GUI 环境
import matplotlib.pyplot as plt
import os
import cv2
join = os.path.join
import torch.nn.functional as F
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai

import torch.nn.functional as F
import argparse
import random
from datetime import datetime
from skimage.transform import resize
import shutil
import glob
from PIL import Image
import numpy as np
from monai.metrics import compute_dice
from medpy.metric.binary import dc
from torch.utils.tensorboard import SummaryWriter
from skimage import io
import math
import json

# 全局随机种子（保证可复现）
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.empty_cache()

class PolypDataset(Dataset):
    def __init__(
        self,
        data_root,
        target_size=(1024, 1024),
        stats_file="stats.txt",
        return_prompt=False,
        is_train=True,
        
    ):
        self.random = random
        self.data_root = data_root
        self.target_size = target_size
        self.single_size = (256, 256)
        self.grid_size = 4
        self.num_patches = self.grid_size * self.grid_size
        self.stats_file = stats_file
        self.return_prompt = return_prompt
        self.is_train = is_train

        

        self.gt_path_files = sorted(
            glob.glob(os.path.join(data_root, "masks/**/*.png"), recursive=True)
        )
        self.img_path_files = sorted(
            glob.glob(os.path.join(data_root, "images/**/*.jpg"), recursive=True)
        )

        assert len(self.gt_path_files) == len(self.img_path_files), \
            "图像和掩码数量不匹配"

        self.total_samples = len(self.img_path_files)
        self.num_groups = math.ceil(self.total_samples / self.num_patches)

        self.stats = []

        mode_str = "训练" if is_train else "测试"
        print(
            f"PolypDataset [{mode_str}] - "
            f"数据量: {self.total_samples}, "
            f"组数: {self.num_groups}, "
            f"每组: {self.num_patches}"
        )

    def _resize_single(self, img, mask):
        img = transform.resize(
            img, self.single_size,
            order=1, mode='reflect',
            anti_aliasing=True,
            preserve_range=True
        )
        mask = transform.resize(
            mask, self.single_size,
            order=0, mode='reflect',
            anti_aliasing=False,
            preserve_range=True
        )
        return img, mask

    def _concat_16_to_1024(self, img_list, mask_list):
        img_rows = []
        mask_rows = []

        for i in range(self.grid_size):
            img_rows.append(
                np.hstack(img_list[i*self.grid_size:(i+1)*self.grid_size])
            )
            mask_rows.append(
                np.hstack(mask_list[i*self.grid_size:(i+1)*self.grid_size])
            )

        return np.vstack(img_rows), np.vstack(mask_rows)

    def _process_item(self, img_path, mask_path):
        img = io.imread(img_path)
        mask = io.imread(mask_path, as_gray=True)

        img, mask = self._resize_single(img, mask)

        if img.max() > 1:
            img = img / 255.0

        mask = (mask >= 0.5).astype(np.uint8)
        return img, mask

    def _get_group_16_pairs(self, index):
        start = index * self.num_patches
        end = min(start + self.num_patches, self.total_samples)

        indices = list(range(start, end))

        # ✅ 测试时也随机补齐
        if len(indices) < self.num_patches:
            pad_num = self.num_patches - len(indices)
            extra_indices = [
                self.random.randint(0, self.total_samples - 1)
                for _ in range(pad_num)
            ]
            indices.extend(extra_indices)

        return (
            [self.img_path_files[i] for i in indices],
            [self.gt_path_files[i] for i in indices]
        )

    def augment_image(self, img, mask):
        """✅ 仅在训练阶段生效"""
        if not self.is_train:
            return img, mask

        if self.random.random() < 0.5:
            img, mask = np.fliplr(img), np.fliplr(mask)

        if self.random.random() < 0.5:
            img, mask = np.flipud(img), np.flipud(mask)

        if self.random.random() < 0.5:
            angle = self.random.uniform(-15, 15)
            img = transform.rotate(img, angle, mode='reflect', preserve_range=True)
            mask = transform.rotate(mask, angle, order=0, mode='reflect', preserve_range=True)

        if self.random.random() < 0.5:
            brightness = self.random.uniform(0.8, 1.2)
            contrast = self.random.uniform(0.8, 1.2)
            img = img * brightness
            mean = img.mean()
            img = (img - mean) * contrast + mean
            img = np.clip(img, 0, 1)

        if self.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, img.shape)
            img = np.clip(img + noise, 0, 1)

        return img, mask

    def __len__(self):
        return self.num_groups

    def __getitem__(self, index):
        batch_img_files, batch_gt_files = self._get_group_16_pairs(index)

        img_list, mask_list = [], []

        for img_path, gt_path in zip(batch_img_files, batch_gt_files):
            img, mask = self._process_item(img_path, gt_path)
            img_list.append(img)
            mask_list.append(mask)

        img, mask = self._concat_16_to_1024(img_list, mask_list)

        # ✅ 测试时不做增强
        img, mask = self.augment_image(img, mask)

        mask = (mask >= 0.5).astype(np.uint8)

        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).float()

        if self.is_train:
            self.stats.append({
                "idx": index,
                "img_mean": img_tensor.mean().item(),
                "img_std": img_tensor.std().item(),
                "mask_mean": mask_tensor.mean().item(),
                "mask_std": mask_tensor.std().item(),
            })

        return img_tensor, mask_tensor, f"batch_{index}", 1


import os
import glob
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io, transform


class RetinaDataset_prototype(Dataset):
    def __init__(
        self,
        data_root,
        bbox_shift=20,
        target_size=(1024, 1024),
        return_prompt=False,
        prompt_point_foreground_mode="disc_or_cup",
        num_points=2,
        is_train=True,  # 区分训练/测试模式
    ):
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "mask")
        self.img_path = os.path.join(data_root, "image")
        
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.png"), recursive=True)
        )
        self.img_path_files = sorted(
            glob.glob(os.path.join(self.img_path, "**/*.png"), recursive=True)
        )
        
        assert len(self.gt_path_files) == len(self.img_path_files), \
            "图像和掩码数量不匹配"

        self.bbox_shift = bbox_shift
        self.target_size = target_size
        self.single_size = (256, 256)
        self.grid_size = 4
        self.num_patches = self.grid_size * self.grid_size
        
        self.return_prompt = return_prompt
        self.prompt_point_foreground_mode = prompt_point_foreground_mode
        self.num_points = num_points
        
        self.is_train = is_train
        
        # 直接使用 Python 内置的 random
        self.random = random
            
        self.total_samples = len(self.img_path_files)
        self.num_groups = math.ceil(self.total_samples / self.num_patches)

        mode_str = "训练" if is_train else "测试"
        print(f"RetinaDataset [{mode_str}] - "
              f"原始数据量: {self.total_samples} | "
              f"共 {self.num_groups} 组 | "
              f"每组 {self.num_patches} 张 (不足时随机补齐)")

    def _resize_single(self, img, mask):
        """将单张图像和掩码统一 resize 到 256x256"""
        img_256 = transform.resize(
            img, self.single_size, order=1, mode='reflect',
            anti_aliasing=True, preserve_range=True
        )
        mask_256 = transform.resize(
            mask, self.single_size, order=0, mode='reflect',
            anti_aliasing=False, preserve_range=True
        )
        return img_256, mask_256

    def _concat_16_to_1024(self, img_list, mask_list):
        """将 4x4 的 patch 拼接成 1024x1024 的大图"""
        img_rows = []
        mask_rows = []
        for i in range(self.grid_size):
            row_imgs = img_list[i*self.grid_size : (i+1)*self.grid_size]
            row_masks = mask_list[i*self.grid_size : (i+1)*self.grid_size]
            img_rows.append(np.hstack(row_imgs))
            mask_rows.append(np.hstack(row_masks))
            
        img_1024 = np.vstack(img_rows)
        mask_1024 = np.vstack(mask_rows)
        return img_1024, mask_1024

    def _get_group_16_pairs(self, index):
        """获取一组 16 张图像的索引，不足 16 张时随机补齐"""
        start = index * self.num_patches
        end = min(start + self.num_patches, self.total_samples)
        
        indices = list(range(start, end))
        
        # ✅ 核心修改：无论训练还是测试，不足时均采用随机补齐
        if len(indices) < self.num_patches:
            pad_num = self.num_patches - len(indices)
            extra_indices = [self.random.randint(0, self.total_samples - 1) 
                             for _ in range(pad_num)]
            indices.extend(extra_indices)

        return [self.img_path_files[i] for i in indices], \
               [self.gt_path_files[i] for i in indices]

    def augment_image(self, img, mask):
        """⚠️ 仅训练阶段生效的增强函数"""
        if not self.is_train:
            return img, mask
            
        if self.random.random() < 0.5:
            img, mask = np.fliplr(img), np.fliplr(mask)
        if self.random.random() < 0.5:
            img, mask = np.flipud(img), np.flipud(mask)
        if self.random.random() < 0.5:
            angle = self.random.uniform(-15, 15)
            img = transform.rotate(img, angle, mode='reflect', preserve_range=True)
            mask = transform.rotate(mask, angle, order=0, mode='reflect', preserve_range=True)
        if self.random.random() < 0.5:
            factor_brightness = random.uniform(0.8, 1.2)
            factor_contrast = random.uniform(0.8, 1.2)
            img = img * factor_brightness
            mean = img.mean()
            img = (img - mean) * factor_contrast + mean
            img = np.clip(img, 0, 1)
        if self.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, img.shape)
            img = np.clip(img + noise, 0, 1)
        return img, mask

    def __len__(self):
        return self.num_groups

    def __getitem__(self, index):
        batch_img_files, batch_gt_files = self._get_group_16_pairs(index)
        
        img_256_list = []
        mask_256_list = []
        
        # 记录本 batch 内每张图片的原始尺寸，测试时反推回原图必备
        original_sizes = [] 
        file_names = []

        for img_path, gt_path in zip(batch_img_files, batch_gt_files):
            img = io.imread(img_path) / 255.0
            mask = io.imread(gt_path, as_gray=True)
            
            # 记录原始信息
            original_sizes.append(mask.shape[:2]) 
            file_names.append(os.path.basename(img_path))
            
            img_256, mask_256 = self._resize_single(img, mask)
            
            # 将 0~255 的灰度图根据阈值转换为 0, 1, 2 的三分类掩码
            converted_mask = np.zeros_like(mask_256, dtype=np.uint8)
            converted_mask[mask_256 > 200/255] = 0
            converted_mask[(mask_256 >= 100/255) & (mask_256 <= 200/255)] = 1
            converted_mask[mask_256 < 100/255] = 2
            
            mask_256_list.append(converted_mask)
            img_256_list.append(img_256)

        img_1024, mask_1024 = self._concat_16_to_1024(img_256_list, mask_256_list)
        
        # 测试阶段不会进行数据增强
        img_1024, mask_1024 = self.augment_image(img_1024, mask_1024)

        # 测试阶段：使用单通道长整型张量 (0, 1, 2)
        mask_tensor = torch.tensor(mask_1024, dtype=torch.long)
        img_tensor = torch.tensor(img_1024.transpose(2, 0, 1), dtype=torch.float32)

        # 返回测试必需的上下文信息：文件名列表和原始尺寸
        return img_tensor, mask_tensor, file_names, original_sizes
    
class PolypDatasetInfer(Dataset):
    def __init__(self, data_root, target_size=(1024, 1024)):
        self.img_files = sorted(glob.glob(os.path.join(data_root, "images/**/*.jpg"), recursive=True))
        self.gt_files = sorted(glob.glob(os.path.join(data_root, "masks/**/*.png"), recursive=True))

        assert len(self.img_files) == len(self.gt_files)

        self.target_size = target_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        gt_path = self.gt_files[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(gt_path, 0)

        # resize到1024
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, self.target_size, interpolation=cv2.INTER_NEAREST)

        img = img / 255.0
        gt = (gt > 127).astype(np.uint8)

        img = torch.tensor(img.transpose(2,0,1)).float()
        gt = torch.tensor(gt).long()

        name = os.path.basename(img_path).replace(".jpg", "")

        return img, gt, name
    
class RetinaDatasetInfer(Dataset):
    def __init__(self, data_root, target_size=(1024, 1024)):
        self.img_files = sorted(glob.glob(os.path.join(data_root, "image/**/*.png"), recursive=True))
        self.gt_files = sorted(glob.glob(os.path.join(data_root, "mask/**/*.png"), recursive=True))

        assert len(self.img_files) == len(self.gt_files)

        self.target_size = target_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        gt_path = self.gt_files[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        mask = cv2.imread(gt_path, 0)

        # resize
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 👉 和你训练完全一致的label逻辑
        converted = np.zeros_like(mask, dtype=np.uint8)
        converted[mask > 200] = 0
        converted[(mask >= 100) & (mask <= 200)] = 1  # OD
        converted[mask < 100] = 2                     # OC

        img = torch.tensor(img.transpose(2,0,1)).float()
        gt = torch.tensor(converted).long()

        name = os.path.basename(img_path).replace(".png", "")

        return img, gt, name