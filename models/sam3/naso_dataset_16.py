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
    - 按顺序每16张切片组成一个样本
    - 每张先缩放到256×256
    - 4×4拼接成1024×1024
    - 最后一组不足16张时，随机补齐
    - 返回三通道图像 + 单通道二值mask
    """
    def __init__(
        self,
        preprocessed_dir,
        target_size=(1024, 1024),
        single_size=(256, 256),
        stats_file="stats.txt",
    ):
        self.preprocessed_dir = preprocessed_dir
        self.target_size = target_size
        self.single_size = single_size
        self.grid_size = 4
        self.num_patches = self.grid_size * self.grid_size  # 16
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
        self.num_groups = math.ceil(self.total_samples / self.num_patches)
        self.stats = []

        print(
            f"nasopharyngealDataset - 原始切片数: {self.total_samples} | "
            f"共 {self.num_groups} 组 | 每组16张顺序拼接，最后不足16张时随机补齐"
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

    def _concat_16_to_1024(self, image_list, mask_list):
        img_rows = []
        mask_rows = []
        for i in range(self.grid_size):
            row_imgs = image_list[i * self.grid_size : (i + 1) * self.grid_size]
            row_masks = mask_list[i * self.grid_size : (i + 1) * self.grid_size]
            img_rows.append(np.hstack(row_imgs))
            mask_rows.append(np.hstack(row_masks))

        img_1024 = np.vstack(img_rows)
        mask_1024 = np.vstack(mask_rows)
        return img_1024, mask_1024

    def _get_group_16_records(self, index):
        """
        按顺序取第 index 组的16张切片。
        最后一组不足16张时，保留最后这些，再随机补齐。
        """
        start = index * self.num_patches
        end = min(start + self.num_patches, self.total_samples)

        indices = list(range(start, end))

        if len(indices) < self.num_patches:
            pad_num = self.num_patches - len(indices)
            extra_indices = [
                self.random.randint(0, self.total_samples - 1)
                for _ in range(pad_num)
            ]
            indices.extend(extra_indices)

        batch_records = [self.slice_records[i] for i in indices]
        return batch_records

    def __len__(self):
        return self.num_groups

    def __getitem__(self, index):
        batch_records = self._get_group_16_records(index)

        img_256_list = []
        mask_256_list = []
        batch_info = []

        for record in batch_records:
            pkl_file = record["pkl_file"]
            slice_idx = record["slice_idx"]

            vol_data = self._load_volume(pkl_file)
            slice_data = vol_data[slice_idx]

            image = slice_data["image"].copy()
            mask = slice_data["mask"].copy()

            # 统一成二值mask
            mask = np.where(mask > 128 / 255, 1.0, 0.0).astype(np.float32)

            img_256, mask_256 = self._resize_single(image, mask)

            img_256_list.append(img_256)
            mask_256_list.append(mask_256)
            batch_info.append(
                {
                    "pkl_file": pkl_file,
                    "slice_idx": slice_idx,
                }
            )

        img_1024, mask_1024 = self._concat_16_to_1024(img_256_list, mask_256_list)

        # 避免负步长/视图问题
        img_1024 = img_1024.copy()
        mask_1024 = mask_1024.copy()

        # 再做一次二值化，保证mask干净
        mask_1024 = (mask_1024 >= 0.5).astype(np.float32)

        # 记录统计信息
        self.stats.append(
            {
                "batch_idx": index,
                "image_mean": float(np.mean(img_1024)),
                "image_std": float(np.std(img_1024)),
                "mask_mean": float(np.mean(mask_1024)),
                "mask_std": float(np.std(mask_1024)),
                "batch_files": [os.path.basename(x["pkl_file"]) for x in batch_info],
                "batch_slices": [x["slice_idx"] for x in batch_info],
            }
        )

        # tensor 化
        image_tensor = torch.from_numpy(img_1024).unsqueeze(0).float()   # [1,1024,1024]
        mask_tensor = torch.from_numpy(mask_1024).unsqueeze(0).float()   # [1,1024,1024]

        # 灰度转3通道
        image_tensor = image_tensor.repeat(3, 1, 1)  # [3,1024,1024]

        first_record = batch_info[0]
        domain = os.path.basename(os.path.dirname(first_record["pkl_file"]))
        sample_name = os.path.basename(first_record["pkl_file"]).replace(".pkl", "") + f"_batch{index}"

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