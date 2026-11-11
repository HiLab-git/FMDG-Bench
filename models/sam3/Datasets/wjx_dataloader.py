import pdb

# Import loralib for LoRA
import loralib as lora
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
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
import shutil
import glob

from naso_dataset import nasopharyngealDataset

import os
import glob  # 确保添加这一行
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import torch.nn as nn
from skimage.transform import resize
import torch.nn.functional as F

from monai.metrics import compute_dice

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
from skimage import io
import argparse
from seg_head.SamFeatSeg import SegDecoderCNN



torch.cuda.empty_cache()
# set seeds
torch.manual_seed(2025)
torch.cuda.empty_cache()


join = os.path.join

import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from os.path import join
from skimage import io
from torch.utils.data import Dataset
from skimage.transform import resize
import pickle
import hashlib


import json

class PolypDataset(Dataset):
    """不使用缓存，直接从磁盘读取数据的内窥镜数据集类，统计每个样本的统计特征并记录在文件中"""
    def __init__(self, data_root, target_size=(1024, 1024), stats_file="stats.txt"):
        import random
        self.random = random
        self.data_root = data_root
        self.target_size = target_size  # 最终拼接后的尺寸
        self.single_size = (256, 256)   # 单张缩放后的尺寸
        self.grid_size = 4              # 4×4网格拼接16张
        self.num_patches = self.grid_size * self.grid_size  # 16张
        self.stats_file = stats_file    # 统计特征的文件
        
        # 获取所有图像和掩码路径
        self.gt_path_files = sorted(glob.glob(join(data_root, "masks/**/*.png"), recursive=True))
        self.img_path_files = sorted(glob.glob(join(data_root, "images/**/*.jpg"), recursive=True))
        assert len(self.gt_path_files) == len(self.img_path_files), "图像和掩码数量不匹配"

        # 确保数据集数量是16的倍数（不足则重复最后几张）
        self.padded_len = ((len(self.gt_path_files) + self.num_patches - 1) // self.num_patches) * self.num_patches
        self.padded_img_files = self.img_path_files + self.img_path_files[:self.padded_len - len(self.img_path_files)]
        self.padded_gt_files = self.gt_path_files + self.gt_path_files[:self.padded_len - len(self.gt_path_files)]

        # 打开文件准备写入
        self.stats = []
    
    def _resize_single(self, img, mask):
        """单张图像/掩码缩放到256×256"""
        # 缩放图像到256×256（抗锯齿）
        img_256 = resize(
            img, 
            self.single_size,
            order=1,          # 1 = 双线性插值（保证图像平滑）
            mode='reflect',
            anti_aliasing=True,
            preserve_range=True
        )
        # 缩放掩码到256×256（最近邻插值，避免类别混淆）
        mask_256 = resize(
            mask, 
            self.single_size,
            order=0,          # 0 = 最近邻插值（保证掩码类别不模糊）
            mode='reflect',   # 填充模式（符合skimage要求）
            anti_aliasing=False,
            preserve_range=True
        )
        return img_256, mask_256

    def _concat_16_to_1024(self, img_list, mask_list):
        """将16张256×256的图像/掩码拼接成1024×1024"""
        # 图像拼接：4×4网格
        img_rows = []
        mask_rows = []
        for i in range(self.grid_size):
            # 拼接一行（4张）
            row_imgs = img_list[i*self.grid_size : (i+1)*self.grid_size]
            row_masks = mask_list[i*self.grid_size : (i+1)*self.grid_size]
            
            img_row = np.hstack(row_imgs)  # 水平拼接
            mask_row = np.hstack(row_masks)
            
            img_rows.append(img_row)
            mask_rows.append(mask_row)
        
        # 垂直拼接所有行
        img_1024 = np.vstack(img_rows)  # (1024, 1024, 3)
        mask_1024 = np.vstack(mask_rows)  # (1024, 1024)
        return img_1024, mask_1024

    def _process_item(self, img_path, mask_path):
        """处理单张图像（读取+缩放256×256）"""
        # 读取原始图像
        img = io.imread(img_path)
        # 读取原始掩码
        mask = io.imread(mask_path, as_gray=True)  # 灰度方式读取会自动缩放到0-1
        
        # 第一步：缩放到256×256
        img_256, mask_256 = self._resize_single(img, mask)

        # 归一化图像
        if img_256.max() > 1.0:
            img_256 = img_256 / 255.0
        
        # 掩码二值化
        threshold = 0.5
        binary_mask = np.zeros_like(mask_256)
        binary_mask[mask_256 >= threshold] = 1

        return img_256, binary_mask

    def __len__(self):
        # 每个样本是16张拼接后的大图，因此长度是填充后长度/16
        return self.padded_len // self.num_patches

    def __getitem__(self, index):
        # 1. 获取当前样本对应的16张图像索引
        start_idx = index * self.num_patches
        end_idx = start_idx + self.num_patches
        batch_img_files = self.padded_img_files[start_idx:end_idx]
        batch_gt_files = self.padded_gt_files[start_idx:end_idx]

        # 2. 逐个处理16张图像：读取→缩放256×256
        img_256_list = []
        mask_256_list = []
        for img_path, gt_path in zip(batch_img_files, batch_gt_files):
            img_256, mask_256 = self._process_item(img_path, gt_path)
            img_256_list.append(img_256)
            mask_256_list.append(mask_256)

        # 3. 拼接16张256×256成1024×1024
        img_1024, mask_1024 = self._concat_16_to_1024(img_256_list, mask_256_list)

        # 4. 图像增强（仅对拼接后的大图做增强，保持逻辑一致）
        img_1024 = self.augment_image(img_1024)

        # 5. 转换为tensor并调整维度（C, H, W）
        img_tensor = torch.tensor(img_1024.transpose(2, 0, 1)).float()
        mask_tensor = torch.tensor(mask_1024).long()

        # 6. 统计特征（仅统计拼接后大图）
        mean = img_tensor.mean().item()
        std = img_tensor.std().item()
        mask_mean = mask_tensor.float().mean().item()
        mask_std = mask_tensor.float().std().item()
        self.stats.append({
            "img_batch_idx": index,
            "img_mean": mean,
            "img_std": std,
            "mask_mean": mask_mean,
            "mask_std": mask_std
        })

        return img_tensor, mask_tensor, f"batch_{index}", 1

    def augment_image(self, img):
        # 亮度与对比度调整（概率为 0.5）
        if self.random.random() < 0.5:
            factor_brightness = random.uniform(0.8, 1.2)
            factor_contrast = random.uniform(0.8, 1.2)
            img = img * factor_brightness
            mean = img.mean()
            img = (img - mean) * factor_contrast + mean
            img = np.clip(img, 0, 1)

        # 添加高斯噪声（概率为 0.5）
        if self.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, img.shape)
            img = np.clip(img + noise, 0, 1)

        return img

    def save_stats(self):
        # 保存统计特征到文件
        with open(self.stats_file, 'w') as f:
            for stat in self.stats:
                f.write(json.dumps(stat) + "\n")

    # 辅助方法（与原版相同）
    def _get_bbox_from_mask(self, mask):
        return [1, 1, 1, 1]

    def _generate_point_prompts(self, mask):
        return 1, -1


class RetinaDataset_prototype(Dataset):
    """"眼底数据集的处理：先缩256×256，再拼接16张成1024×1024"""

    def __init__(self, data_root, bbox_shift=20,target_size=(1024, 1024)):
        self.data_root = data_root
        self.gt_path = join(data_root, "mask")
        self.img_path = join(data_root, "image")
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, "**/*.png"), recursive=True))
        self.img_path_files = sorted(glob.glob(join(self.img_path, "**/*.png"), recursive=True))
        assert len(self.gt_path_files) == len(self.img_path_files), "Mismatch in images and masks count."
        self.bbox_shift = bbox_shift
        self.target_size = target_size  # 最终拼接后的尺寸
        self.single_size = (256, 256)   # 单张缩放后的尺寸
        self.grid_size = 4              # 4×4网格拼接16张
        self.num_patches = self.grid_size * self.grid_size  # 16张

        # 确保数据集数量是16的倍数（不足则重复最后几张）
        self.padded_len = ((len(self.gt_path_files) + self.num_patches - 1) // self.num_patches) * self.num_patches
        self.padded_img_files = self.img_path_files + self.img_path_files[:self.padded_len - len(self.img_path_files)]
        self.padded_gt_files = self.gt_path_files + self.gt_path_files[:self.padded_len - len(self.gt_path_files)]

        print(f"Number of images (original): {len(self.gt_path_files)}")
        print(f"Number of images (padded): {self.padded_len}")
        print(f"Number of 1024×1024 samples: {self.__len__()}")

    def __len__(self):
        # 每个样本是16张拼接后的大图，因此长度是填充后长度/16
        return self.padded_len // self.num_patches

    def _resize_single(self, img, mask):
        """单张图像/掩码缩放到256×256"""
        # 缩放图像到256×256（抗锯齿）
        img_256 = resize(
            img, 
            self.single_size,
            order=1,          # 1 = 双线性插值（保证图像平滑）
            mode='reflect',
            anti_aliasing=True,
            preserve_range=True
        )
        # 缩放掩码到256×256（最近邻插值，避免类别混淆）
        mask_256 = resize(
            mask, 
            self.single_size,
            order=0,          # 0 = 最近邻插值（保证掩码类别不模糊）
            mode='reflect',   # 填充模式（符合skimage要求）
            anti_aliasing=False,
            preserve_range=True
        )
        return img_256, mask_256

    def _concat_16_to_1024(self, img_list, mask_list):
        """将16张256×256的图像/掩码拼接成1024×1024"""
        # 图像拼接：4×4网格
        img_rows = []
        mask_rows = []
        for i in range(self.grid_size):
            # 拼接一行（4张）
            row_imgs = img_list[i*self.grid_size : (i+1)*self.grid_size]
            row_masks = mask_list[i*self.grid_size : (i+1)*self.grid_size]
            
            img_row = np.hstack(row_imgs)  # 水平拼接
            mask_row = np.hstack(row_masks)
            
            img_rows.append(img_row)
            mask_rows.append(mask_row)
        
        # 垂直拼接所有行
        img_1024 = np.vstack(img_rows)  # (1024, 1024, 3)
        mask_1024 = np.vstack(mask_rows)  # (1024, 1024)
        return img_1024, mask_1024

    def __getitem__(self, index):
        # 1. 获取当前样本对应的16张图像索引
        start_idx = index * self.num_patches
        end_idx = start_idx + self.num_patches
        batch_img_files = self.padded_img_files[start_idx:end_idx]
        batch_gt_files = self.padded_gt_files[start_idx:end_idx]

        # 2. 逐个处理16张图像：读取→缩放256×256→掩码分类
        img_256_list = []
        mask_256_list = []
        for img_path, gt_path in zip(batch_img_files, batch_gt_files):
            # 读取原始图像并归一化
            img = io.imread(img_path) / 255.0
            # 读取原始掩码
            mask = io.imread(gt_path, as_gray=True)

            # 第一步：缩放到256×256
            img_256, mask_256 = self._resize_single(img, mask)

            # 掩码分类（和原版逻辑一致，仅针对256×256掩码）
            converted_mask = np.zeros_like(mask_256, dtype=np.uint8)
            converted_mask[mask_256 > 200/255] = 0          # 背景
            converted_mask[(mask_256 >= 100/255) & (mask_256 <= 200/255)] = 1  # 视盘
            converted_mask[mask_256 < 100/255] = 2          # 视杯

            mask_256_list.append(converted_mask)
            img_256_list.append(img_256)

        # 3. 拼接16张256×256成1024×1024
        img_1024, mask_1024 = self._concat_16_to_1024(img_256_list, mask_256_list)

        # 4. 掩码转换为3通道（和原版逻辑一致）
        new_mask = np.zeros((3, mask_1024.shape[0], mask_1024.shape[1]), dtype=np.uint8)
        new_mask[0,(mask_1024==0)]=1  # 背景
        new_mask[1,(mask_1024==1)]=1  # 视盘
        new_mask[2,(mask_1024==2)]=1  # 视杯

        # 5. 转换为tensor并调整维度
        img_tensor = torch.tensor(img_1024.transpose(2, 0, 1)).float()  # (C, H, W)
        mask_tensor = torch.tensor(new_mask).long()

        # 6. 保持原版的bbox和点提示逻辑（适配1024×1024尺寸）
        bboxes = np.array([200,200,800,800])  # 可根据需要调整为1024尺寸的bbox
        pos_indices = np.column_stack(np.where(mask_tensor.squeeze(0) > 0))
        neg_indices = np.column_stack(np.where(mask_tensor.squeeze(0) == 0))
        
        pos_point = pos_indices[np.random.choice(pos_indices.shape[0], 1)][0] if len(pos_indices) > 0 else np.array([512, 512])
        neg_point = neg_indices[np.random.choice(neg_indices.shape[0], 1)][0] if len(neg_indices) > 0 else np.array([100, 100])
        
        points_np = np.array([pos_point, neg_point])  
        point_coords = torch.from_numpy(points_np).float()
        point_labels = torch.tensor([1, 0]).long()

        return img_tensor, mask_tensor, 1

# class PolypDataset_old(Dataset):
#     """使用磁盘缓存存储预处理数据的内窥镜数据集类"""
#     def __init__(self, data_root, target_size=(1024, 1024), cache_dir='./dataset_cache'):
#         self.data_root = data_root
#         self.target_size = target_size
#         self.cache_dir = cache_dir  # 缓存目录路径
        
#         # 如果缓存目录不存在则创建
#         os.makedirs(self.cache_dir, exist_ok=True)
        
#         # 获取所有图像和掩码路径
#         self.gt_path_files = sorted(glob.glob(join(data_root, "masks/**/*.png"), recursive=True))
#         self.img_path_files = sorted(glob.glob(join(data_root, "images/**/*.jpg"), recursive=True))
#         assert len(self.gt_path_files) == len(self.img_path_files), "图像和掩码数量不匹配"
        
#         # 预处理并缓存所有数据
#         self.cached_items = []  # 存储缓存键而非数据本身
#         for img_path, mask_path in tqdm(zip(self.img_path_files, self.gt_path_files), total=len(self.img_path_files)):
#             # 为每个数据项生成唯一缓存键
#             cache_key = self._generate_cache_key(img_path, mask_path)
#             cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
#             # 检查缓存是否存在
#             if os.path.exists(cache_path):
#                 # 从缓存加载
#                 with open(cache_path, 'rb') as f:
#                     cached_data = pickle.load(f)
#             else:
#                 # 处理并保存到缓存
#                 img, mask = self._process_item(img_path, mask_path)
#                 cached_data = {
#                     "img": img,
#                     "mask": mask,
#                     "name": os.path.basename(mask_path)
#                 }
#                 with open(cache_path, 'wb') as f:
#                     pickle.dump(cached_data, f)
            
#             self.cached_items.append(cache_key)
    
#     def _generate_cache_key(self, img_path, mask_path):
#         """基于文件路径和参数生成唯一缓存键"""
#         key_str = f"{img_path}_{mask_path}_{self.target_size}"
#         return hashlib.md5(key_str.encode()).hexdigest()
    
#     def _process_item(self, img_path, mask_path):
#         # 图像处理
#         img = io.imread(img_path)
#         img = resize(img, self.target_size, mode='reflect', anti_aliasing=True)
#         # 判断是否需要归一化（假设 img 是 NumPy 数组）
#         if img.max() > 1.0:
#             img = img / 255.0
#         img = torch.tensor(img.transpose(2, 0, 1)).float()  # (C, H, W)

#         # 掩码处理
#         mask = io.imread(mask_path, as_gray=True)#灰度方式读取会自动缩放到0-1
#         # 假设mask是读取的灰度图像（numpy数组）
#         threshold = 0.5  # 阈值，可根据实际情况调整
#         binary_mask = np.zeros_like(mask)
#         binary_mask[mask >= threshold] = 1  # 大于等于阈值的设为1，其余为0
#         new_mask = resize(binary_mask, self.target_size, mode='reflect', anti_aliasing=False)
#         mask = torch.tensor(new_mask).long()
#         return img, mask

#     def __len__(self):
#         return len(self.cached_items)

#     def __getitem__(self, index):
#         # 从缓存加载数据项
#         cache_key = self.cached_items[index]
#         cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
#         with open(cache_path, 'rb') as f:
#             data = pickle.load(f)
#         #填充最后一个是为了和nano数据集保持一致，因为用的同一个训练代码
#         img, mask, name = data["img"], data["mask"], data["name"]

#         # 2. 图像增强：亮度/对比度 + 高斯噪声（假设 img 是 float 格式，范围在 [0, 1]）
#         img = self.augment_image(img)
#         return img, mask, name,1
#         # return data["img"], data["mask"], data["name"],1


#     def augment_image(self, img):
#         # 亮度与对比度调整（概率为 0.5）
#         if random.random() < 0.5:
#             factor_brightness = random.uniform(0.8, 1.2)
#             factor_contrast = random.uniform(0.8, 1.2)
#             img = img * factor_brightness
#             mean = img.mean()
#             img = (img - mean) * factor_contrast + mean
#             img = np.clip(img, 0, 1)

#         # 添加高斯噪声（概率为 0.5）
#         if random.random() < 0.5:
#             noise = np.random.normal(0, 0.01, img.shape)
#             img = np.clip(img + noise, 0, 1)

#     # 辅助方法（与原版相同）
#     def _get_bbox_from_mask(self, mask):
#         return [1, 1, 1, 1]

#     def _generate_point_prompts(self, mask):
#         return 1, -1
   

# class RetinaDataset(Dataset):
#     """使用磁盘缓存存储预处理数据的视网膜数据集类"""
#     def __init__(self, data_root, target_size=(800, 800), cache_dir='./dataset_cache'):
#         self.data_root = data_root
#         self.target_size = target_size
#         self.cache_dir = cache_dir  # 缓存目录路径
        
#         # 如果缓存目录不存在则创建
#         os.makedirs(self.cache_dir, exist_ok=True)
        
#         # 获取所有图像和掩码路径
#         self.gt_path_files = sorted(glob.glob(join(data_root, "mask/**/*.png"), recursive=True))
#         self.img_path_files = sorted(glob.glob(join(data_root, "image/**/*.png"), recursive=True))
#         assert len(self.gt_path_files) == len(self.img_path_files), "图像和掩码数量不匹配"
        
#         # 预处理并缓存所有数据
#         self.cached_items = []  # 存储缓存键而非数据本身
#         for img_path, mask_path in tqdm(zip(self.img_path_files, self.gt_path_files), total=len(self.img_path_files)):
#             # 为每个数据项生成唯一缓存键
#             cache_key = self._generate_cache_key(img_path, mask_path)
#             cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
#             # 检查缓存是否存在
#             if os.path.exists(cache_path):
#                 # 从缓存加载
#                 with open(cache_path, 'rb') as f:
#                     cached_data = pickle.load(f)
#             else:
#                 # 处理并保存到缓存
#                 img, mask = self._process_item(img_path, mask_path)
#                 cached_data = {
#                     "img": img,
#                     "mask": mask,
#                     "name": os.path.basename(mask_path)
#                 }
#                 with open(cache_path, 'wb') as f:
#                     pickle.dump(cached_data, f)
            
#             self.cached_items.append(cache_key)
    
#     def _generate_cache_key(self, img_path, mask_path):
#         """基于文件路径和参数生成唯一缓存键"""
#         key_str = f"{img_path}_{mask_path}_{self.target_size}"
#         return hashlib.md5(key_str.encode()).hexdigest()
    
#     def _process_item(self, img_path, mask_path):
#         # 图像处理
#         img = io.imread(img_path)
#         img = resize(img, self.target_size, mode='reflect', anti_aliasing=True) / 255.0
        

#         # —— 数据增强开始 ——

#         # 1. 亮度/对比度调整（随机增减）
#         if random.random() < 0.5:
#             factor_brightness = random.uniform(0.8, 1.2)
#             factor_contrast = random.uniform(0.8, 1.2)
#             img = img * factor_brightness
#             mean = img.mean()
#             img = (img - mean) * factor_contrast + mean
#             img = np.clip(img, 0, 1)

#         # 2. 添加高斯噪声
#         if random.random() < 0.5:
#             noise = np.random.normal(0, 0.01, img.shape)
#             img = np.clip(img + noise, 0, 1)

#         # —— 数据增强结束 ——

        
#         img = torch.tensor(img.transpose(2, 0, 1)).float()  # (C, H, W)

#         # 掩码处理
#         mask = io.imread(mask_path, as_gray=True)
#         converted_mask = np.zeros_like(mask, dtype=np.uint8)
#         converted_mask[mask > 200/255] = 0
#         converted_mask[(mask >= 100/255) & (mask <= 200/255)] = 1
#         converted_mask[mask < 100/255] = 2
#         new_mask = np.zeros((3, mask.shape[0], mask.shape[1]), dtype=np.uint8)
#         new_mask[0,(converted_mask==0)]=1
#         new_mask[1,(converted_mask==1)]=1
#         new_mask[2,(converted_mask==2)]=1 
#         mask = torch.tensor(new_mask).long()

#         return img, mask

#     def __len__(self):
#         return len(self.cached_items)

#     def __getitem__(self, index):
#         # 从缓存加载数据项
#         cache_key = self.cached_items[index]
#         cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
#         with open(cache_path, 'rb') as f:
#             data = pickle.load(f)
        
#         return data["img"], data["mask"], data["name"]

#     # 辅助方法（与原版相同）
#     def _get_bbox_from_mask(self, mask):
#         return [1, 1, 1, 1]

#     def _generate_point_prompts(self, mask):
#         return 1, -1
    
# class RetinaDataset_flash(Dataset):
#     """数据集预先加载到内存中，提高训练速度"""
#     def __init__(self, data_root, target_size=(1024, 1024)):
#         self.data_root = data_root
#         self.target_size = target_size
#         self.gt_path_files = sorted(glob.glob(join(data_root, "mask/**/*.png"), recursive=True))
#         self.img_path_files = sorted(glob.glob(join(data_root, "image/**/*.png"), recursive=True))
#         assert len(self.gt_path_files) == len(self.img_path_files), "Mismatch in images and masks count."
        
#         # 预加载所有数据到内存
#         self.cached_data = []
#         for img_path, mask_path in tqdm(zip(self.img_path_files, self.gt_path_files), total=len(self.img_path_files)):
#             img, mask = self._process_item(img_path, mask_path)
#             self.cached_data.append({
#                 "img": img,
#                 "mask": mask,
#                 "name": os.path.basename(mask_path)
#             })

#     def _process_item(self, img_path, mask_path):
#         # 图像处理
#         img = io.imread(img_path) / 255.0
#         img = resize(img, self.target_size, mode='reflect', anti_aliasing=True)
        
#         img = torch.tensor(img.transpose(2, 0, 1)).float()  # (C, H, W)

#         # 掩码处理
#         mask = io.imread(mask_path, as_gray=True)
#         converted_mask = np.zeros_like(mask, dtype=np.uint8)
#         converted_mask[mask > 200/255] = 0
#         converted_mask[(mask >= 100/255) & (mask <= 200/255)] = 1
#         converted_mask[mask < 100/255] = 2
#         new_mask = np.zeros((3, mask.shape[0], mask.shape[1]), dtype=np.uint8)
#         new_mask[0,(converted_mask==0)]=1
#         new_mask[1,(converted_mask==1)]=1
#         new_mask[2,(converted_mask==2)]=1 
#         mask = torch.tensor(new_mask).long()

#         # # 动态生成边界框和点提示
#         # bbox = self._get_bbox_from_mask(mask.numpy())
#         # pos_point, neg_point = self._generate_point_prompts(mask.numpy())
#         # point_coords = torch.tensor([pos_point, neg_point]).float()
#         # point_labels = torch.tensor([1, 0]).long()

#         return img, mask

#     def __len__(self):
#         return len(self.cached_data)

#     def __getitem__(self, index):
#         data = self.cached_data[index]
#         return data["img"], data["mask"], None,data["name"],None,None

#     # 辅助函数：动态生成边界框
#     def _get_bbox_from_mask(self, mask):
#         # from skimage.measure import regionprops
#         # regions = regionprops(mask)
#         # if len(regions) == 0:
#         #     return [0, 0, mask.shape[1], mask.shape[0]]  # 默认全图
#         # y1, x1, y2, x2 = regions[0].bbox
#         # return [x1, y1, x2, y2]
#         return [1, 1, 1, 1]

#     # 辅助函数：生成点提示
#     def _generate_point_prompts(self, mask):
#         # pos_indices = np.column_stack(np.where(mask > 0))
#         # if len(pos_indices) == 0:
#         #     pos_point = np.array([mask.shape[0]//2, mask.shape[1]//2])
#         # else:
#         #     pos_point = pos_indices[np.random.choice(pos_indices.shape[0], 1)][0]
#         # neg_indices = np.column_stack(np.where(mask == 0))
#         # neg_point = neg_indices[np.random.choice(neg_indices.shape[0], 1)][0]
#         # return pos_point, neg_point
#         return 1, -1
    
