import os
import json
import math
import glob
import pickle
import random
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize


class nasopharyngealDataset(Dataset):
    """
    鼻咽数据集：
    - 每个样本 = 1 张 256×256 切片
    - 返回三通道图像 + 单通道二值mask
    """
    def __init__(
        self,
        preprocessed_dir,
        single_size=(256, 256),
        stats_file="stats.txt",
    ):
        self.preprocessed_dir = preprocessed_dir
        self.single_size = single_size
        self.stats_file = stats_file
        self.random = random

        # 收集所有 pkl 文件
        self.pkl_files = sorted(
            [
                os.path.join(preprocessed_dir, f)
                for f in os.listdir(preprocessed_dir)
                if f.endswith(".pkl")
            ]
        )

        # 构建全量切片索引
        self.slice_records = []
        for pkl_file in self.pkl_files:
            with open(pkl_file, "rb") as f:
                vol_data = pickle.load(f)
            for slice_idx in range(len(vol_data)):
                self.slice_records.append(
                    {"pkl_file": pkl_file, "slice_idx": slice_idx}
                )

        self.total_samples = len(self.slice_records)
        self.stats = []

        print(
            f"nasopharyngealDataset - 切片数: {self.total_samples} | "
            f"单张尺寸: {self.single_size}"
        )

    @lru_cache(maxsize=100)
    def _load_volume(self, pkl_file):
        with open(pkl_file, "rb") as f:
            return pickle.load(f)

    def _resize_single(self, image, mask):
        resized_image = resize(
            image,
            self.single_size,
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
        resized_mask = resize(
            mask,
            self.single_size,
            order=0,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        )
        return resized_image, resized_mask

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        record = self.slice_records[index]

        vol_data = self._load_volume(record["pkl_file"])
        slice_data = vol_data[record["slice_idx"]]

        image = slice_data["image"].copy()
        mask = slice_data["mask"].copy()

        # 统一成二值mask
        mask = np.where(mask > 128 / 255, 1.0, 0.0).astype(np.float32)

        img_256, mask_256 = self._resize_single(image, mask)

        # Z-Score 归一化 (nnU-Net 标准预处理, 在 repeat 到3通道之前)
        img_256 = (img_256 - img_256.mean()) / max(img_256.std(), 1e-8)

        # 再做一次二值化，保证mask干净
        mask_256 = (mask_256 >= 0.5).astype(np.float32)

        # 记录统计信息
        self.stats.append(
            {
                "idx": index,
                "image_mean": float(np.mean(img_256)),
                "image_std": float(np.std(img_256)),
                "mask_mean": float(np.mean(mask_256)),
                "mask_std": float(np.std(mask_256)),
                "pkl_file": os.path.basename(record["pkl_file"]),
                "slice_idx": record["slice_idx"],
            }
        )

        # tensor 化: .copy() 防止负stride
        image_tensor = torch.from_numpy(img_256.copy()).unsqueeze(0).float()   # [1,256,256]
        mask_tensor = torch.from_numpy(mask_256.copy()).unsqueeze(0).float()   # [1,256,256]

        # 灰度转3通道
        image_tensor = image_tensor.repeat(3, 1, 1)  # [3,256,256]

        domain = os.path.basename(os.path.dirname(record["pkl_file"]))
        sample_name = os.path.basename(record["pkl_file"]).replace(".pkl", "") + f"_slice{record['slice_idx']:04d}"

        return image_tensor, mask_tensor, domain, sample_name

    def save_stats(self):
        with open(self.stats_file, "w") as f:
            for stat in self.stats:
                f.write(json.dumps(stat) + "\n")



class nasopharyngealDataset_single(Dataset):
    """
    推理版：
    - 每个样本 = 1 张 slice
    - resize 到 1024×1024
    - 返回 3 通道 image + 1 通道 mask
    """

    def __init__(
        self,
        preprocessed_dir,
        target_size=(1024, 1024),
    ):
        self.preprocessed_dir = preprocessed_dir
        self.target_size = target_size

        # 收集 pkl
        self.pkl_files = sorted(
            [
                os.path.join(preprocessed_dir, f)
                for f in os.listdir(preprocessed_dir)
                if f.endswith(".pkl")
            ]
        )

        # 建 slice 索引
        self.slice_records = []
        for pkl_file in self.pkl_files:
            with open(pkl_file, "rb") as f:
                vol_data = pickle.load(f)

            for slice_idx in range(len(vol_data)):
                self.slice_records.append(
                    {"pkl_file": pkl_file, "slice_idx": slice_idx}
                )

        print(f"[Single] naso dataset slices: {len(self.slice_records)}")

    @lru_cache(maxsize=50)
    def _load_volume(self, pkl_file):
        with open(pkl_file, "rb") as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.slice_records)

    def __getitem__(self, index):
        record = self.slice_records[index]

        vol = self._load_volume(record["pkl_file"])
        slice_data = vol[record["slice_idx"]]

        image = slice_data["image"].copy()
        mask = slice_data["mask"].copy()

        # 二值化
        mask = (mask > 128 / 255).astype(np.float32)

        # resize 到 1024
        image = resize(
            image,
            self.target_size,
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )

        mask = resize(
            mask,
            self.target_size,
            order=0,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        )

        mask = (mask >= 0.5).astype(np.float32)

        # tensor
        image = torch.from_numpy(image).unsqueeze(0).float()  # [1,H,W]
        image = image.repeat(3, 1, 1)                         # → [3,H,W]

        mask = torch.from_numpy(mask).unsqueeze(0).float()    # [1,H,W]

        # 命名（用于保存）
        base = os.path.basename(record["pkl_file"]).replace(".pkl", "")
        sample_name = f"{base}_slice{record['slice_idx']:04d}"

        return image, mask, sample_name