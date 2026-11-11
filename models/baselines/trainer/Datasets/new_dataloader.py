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

from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai


import argparse
import random
from datetime import datetime
from skimage.transform import resize
import shutil
import glob
from PIL import Image
import numpy as np
from monai.metrics import compute_dice

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
        single_size=(256, 256),
        stats_file="stats.txt",
        return_prompt=False,
        is_train=True,

    ):
        self.random = random
        self.data_root = data_root
        self.single_size = single_size
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
        self.stats = []

        mode_str = "训练" if is_train else "测试"
        print(
            f"PolypDataset [{mode_str}] - "
            f"数据量: {self.total_samples}, "
            f"单张尺寸: {self.single_size}"
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

    def _process_item(self, img_path, mask_path):
        img = io.imread(img_path)
        mask = io.imread(mask_path, as_gray=True)

        img, mask = self._resize_single(img, mask)

        if img.max() > 1:
            img = img / 255.0

        mask = (mask >= 0.5).astype(np.uint8)
        return img, mask

    def augment_image(self, img, mask):
        """仅在训练阶段生效"""
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
        return self.total_samples

    def __getitem__(self, index):
        img_path = self.img_path_files[index]
        gt_path = self.gt_path_files[index]

        img, mask = self._process_item(img_path, gt_path)

        # 数据增强 (训练时)
        img, mask = self.augment_image(img, mask)

        mask = (mask >= 0.5).astype(np.uint8)

        # .copy() 防止 flip 等操作产生的负 stride
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        mask_tensor = torch.from_numpy(mask.copy()).float()

        if self.is_train:
            self.stats.append({
                "idx": index,
                "img_mean": img_tensor.mean().item(),
                "img_std": img_tensor.std().item(),
                "mask_mean": mask_tensor.mean().item(),
                "mask_std": mask_tensor.std().item(),
            })

        file_name = os.path.basename(img_path).replace(".jpg", "")
        return img_tensor, mask_tensor, file_name, 1


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
        single_size=(256, 256),
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
        self.single_size = single_size

        self.return_prompt = return_prompt
        self.prompt_point_foreground_mode = prompt_point_foreground_mode
        self.num_points = num_points

        self.is_train = is_train

        # 直接使用 Python 内置的 random
        self.random = random

        self.total_samples = len(self.img_path_files)

        mode_str = "训练" if is_train else "测试"
        print(f"RetinaDataset [{mode_str}] - "
              f"数据量: {self.total_samples} | "
              f"单张尺寸: {self.single_size}")

    def _resize_single(self, img, mask):
        """将单张图像和掩码统一 resize"""
        img_256 = transform.resize(
            img, self.single_size, order=1, mode='reflect',
            anti_aliasing=True, preserve_range=True
        )
        mask_256 = transform.resize(
            mask, self.single_size, order=0, mode='reflect',
            anti_aliasing=False, preserve_range=True
        )
        return img_256, mask_256

    def augment_image(self, img, mask):
        """仅训练阶段生效的增强函数"""
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
        return self.total_samples

    def __getitem__(self, index):
        img_path = self.img_path_files[index]
        gt_path = self.gt_path_files[index]

        img = io.imread(img_path) / 255.0
        mask = io.imread(gt_path, as_gray=True)

        # 记录原始信息
        original_size = mask.shape[:2]
        file_name = os.path.basename(img_path)

        img_256, mask_256 = self._resize_single(img, mask)

        # 将 0~255 的灰度图根据阈值转换为 0, 1, 2 的三分类掩码
        converted_mask = np.zeros_like(mask_256, dtype=np.uint8)
        converted_mask[mask_256 > 200/255] = 0
        converted_mask[(mask_256 >= 100/255) & (mask_256 <= 200/255)] = 1
        converted_mask[mask_256 < 100/255] = 2

        # 数据增强
        img_256, converted_mask = self.augment_image(img_256, converted_mask)

        # .copy() 防止 flip 等操作产生的负 stride
        mask_tensor = torch.tensor(converted_mask.copy(), dtype=torch.long)
        img_tensor = torch.tensor(img_256.transpose(2, 0, 1).copy(), dtype=torch.float32)

        return img_tensor, mask_tensor, file_name, original_size
    
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