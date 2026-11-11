# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
目前缺乏数据增强
"""




import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from functools import lru_cache
import matplotlib.pyplot as plt

from skimage.transform import resize



import torch
import os
import pickle
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms as T
import json

class nasopharyngealDataset(Dataset):
    def __init__(self, 
                 preprocessed_dir, 
                 target_size=(1024, 1024), 
                 transform=None, 
                 max_cache_size=100, 
                 stats_file="stats.txt"):
        self.preprocessed_dir = preprocessed_dir
        self.target_size = target_size
        self.single_size = (256, 256)  # 单张缩放后的尺寸
        self.grid_size = 4              # 4×4网格拼接16张
        self.num_patches = self.grid_size * self.grid_size  # 16张
        self.transform = transform or T.Compose([])
        self.max_cache_size = max_cache_size
        self.stats_file = stats_file  # 保存统计特征的文件
        
        # 获取所有预处理文件
        self.pkl_files = sorted([
            os.path.join(preprocessed_dir, f) 
            for f in os.listdir(preprocessed_dir) 
            if f.endswith('.pkl')
        ])
        
        # 构建切片索引
        self.slice_records = []
        for pkl_file in self.pkl_files:
            with open(pkl_file, 'rb') as f:
                vol_data = pickle.load(f)
                for slice_idx in range(len(vol_data)):
                    self.slice_records.append({
                        'pkl_file': pkl_file,
                        'slice_idx': slice_idx
                    })

        # 确保切片数量是16的倍数（不足则重复最后几张）
        self.padded_len = ((len(self.slice_records) + self.num_patches - 1) // self.num_patches) * self.num_patches
        self.padded_slice_records = self.slice_records + self.slice_records[:self.padded_len - len(self.slice_records)]

        # 创建CPU随机数生成器
        self.cpu_generator = torch.Generator(device='cpu')
        self.cpu_generator.manual_seed(torch.seed())

        # 统计特征列表
        self.stats = []

    @lru_cache(maxsize=100)
    def _load_volume(self, pkl_file):
        """带缓存的加载volume数据"""
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)

    def _resize_single(self, image, mask):
        """将单张图像/掩码缩放到256×256（修复mode参数错误）"""
        # 图像缩放：双线性插值（order=1），填充模式用reflect
        resized_image = resize(image, 
                              self.single_size,
                              order=1,        # 1=双线性插值（图像用）
                              mode='reflect', # 填充模式（只能是这几个值）
                              anti_aliasing=True,
                              preserve_range=True)
        # 掩码缩放：最近邻插值（order=0），填充模式用reflect（关键修复）
        resized_mask = resize(mask, 
                              self.single_size,
                              order=0,        # 0=最近邻插值（掩码用）
                              mode='reflect', # 填充模式，不能写nearest！
                              anti_aliasing=False,
                              preserve_range=True)
        return resized_image, resized_mask

    def _concat_16_to_1024(self, image_list, mask_list):
        """将16张256×256的图像/掩码拼接成1024×1024"""
        # 图像拼接：4×4网格
        img_rows = []
        mask_rows = []
        for i in range(self.grid_size):
            # 拼接一行（4张）
            row_imgs = image_list[i*self.grid_size : (i+1)*self.grid_size]
            row_masks = mask_list[i*self.grid_size : (i+1)*self.grid_size]
            
            # 水平拼接
            img_row = np.hstack(row_imgs)  # (256, 1024)
            mask_row = np.hstack(row_masks)
            
            img_rows.append(img_row)
            mask_rows.append(mask_row)
        
        # 垂直拼接所有行
        img_1024 = np.vstack(img_rows)  # (1024, 1024)
        mask_1024 = np.vstack(mask_rows)  # (1024, 1024)
        return img_1024, mask_1024

    def __len__(self):
        # 每个样本是16张拼接后的大图，长度为填充后数量/16
        return self.padded_len // self.num_patches

    def __getitem__(self, idx):
        # 1. 获取当前样本对应的16张切片索引
        start_idx = idx * self.num_patches
        end_idx = start_idx + self.num_patches
        batch_records = self.padded_slice_records[start_idx:end_idx]

        # 2. 逐个处理16张切片：加载→缩放256×256
        img_256_list = []
        mask_256_list = []
        batch_info = []  # 记录批次内的文件信息
        for record in batch_records:
            pkl_file = record['pkl_file']
            slice_idx = record['slice_idx']
            
            # 加载数据
            vol_data = self._load_volume(pkl_file)
            slice_data = vol_data[slice_idx]
            
            # 获取图像和mask
            image = slice_data['image'].copy()
            mask = slice_data['mask'].copy()
            
            # 统一mask数值范围到 [0,1]（和旧版本逻辑对齐）
            mask = np.where(mask > 128/255, 1.0, 0.0).astype(np.float32)
            # =====================================

            # 缩放到256×256（修复后）
            img_256, mask_256 = self._resize_single(image, mask)
            
            img_256_list.append(img_256)
            mask_256_list.append(mask_256)
            batch_info.append({
                'pkl_file': pkl_file,
                'slice_idx': slice_idx
            })

        # 3. 拼接16张256×256成1024×1024
        img_1024, mask_1024 = self._concat_16_to_1024(img_256_list, mask_256_list)

        # 4. 计算拼接后大图的统计特征
        image_mean = np.mean(img_1024)
        image_std = np.std(img_1024)
        mask_mean = np.mean(mask_1024)
        mask_std = np.std(mask_1024)

        # 记录统计特征
        self.stats.append({
            'batch_idx': idx,
            'image_mean': image_mean,
            'image_std': image_std,
            'mask_mean': mask_mean,
            'mask_std': mask_std,
            'batch_files': [os.path.basename(info['pkl_file']) for info in batch_info],
            'batch_slices': [info['slice_idx'] for info in batch_info]
        })

        # 5. 转换为tensor并调整维度
        image_tensor = torch.from_numpy(img_1024).unsqueeze(0).float()  # (1, 1024, 1024)
        mask_tensor = torch.from_numpy(mask_1024).unsqueeze(0).float()    # (1, 1024, 1024)

        # 转换为三通道
        image_tensor = image_tensor.repeat(3, 1, 1)  # (3, 1024, 1024)

        # 6. 生成批次的域信息和volume ID（取第一个切片的信息）
        first_record = batch_info[0]
        domain = os.path.basename(os.path.dirname(first_record['pkl_file']))
        volume_id = os.path.basename(first_record['pkl_file']).replace('.pkl', '') + f"_batch{idx}"

        return image_tensor, mask_tensor, domain, volume_id

    def save_stats(self):
        """保存统计特征到文件"""
        with open(self.stats_file, 'w') as f:
            for stat in self.stats:
                f.write(json.dumps(stat) + "\n")



class nasopharyngealDataset_old(Dataset):
    def __init__(self, 
                 preprocessed_dir, 
                 target_size=(1024, 1024), 
                 transform=None, 
                 max_cache_size=100):
        self.preprocessed_dir = preprocessed_dir
        self.target_size = target_size
        self.transform = transform or T.Compose([])
        self.max_cache_size = max_cache_size
        
        # 获取所有预处理文件
        self.pkl_files = sorted([
            os.path.join(preprocessed_dir, f) 
            for f in os.listdir(preprocessed_dir) 
            if f.endswith('.pkl')
        ])
        
        # 构建切片索引
        self.slice_records = []
        for pkl_file in self.pkl_files:
            with open(pkl_file, 'rb') as f:
                vol_data = pickle.load(f)
                for slice_idx in range(len(vol_data)):
                    self.slice_records.append({
                        'pkl_file': pkl_file,
                        'slice_idx': slice_idx
                    })

        # 创建CPU随机数生成器
        self.cpu_generator = torch.Generator(device='cpu')
        self.cpu_generator.manual_seed(torch.seed())

    @lru_cache(maxsize=100)
    def _load_volume(self, pkl_file):
        """带缓存的加载volume数据"""
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.slice_records)

    def __getitem__(self, idx):
        # 获取切片信息
        record = self.slice_records[idx]
        pkl_file = record['pkl_file']
        slice_idx = record['slice_idx']
        
        # 加载数据
        vol_data = self._load_volume(pkl_file)
        slice_data = vol_data[slice_idx]
        
        # 获取图像和mask
        image = slice_data['image'].copy()
        mask = slice_data['mask'].copy()

        # 调整尺寸（双线性插值用于图像，最近邻用于mask）
        resized_image = resize(image, 
                              self.target_size,
                              order=1,  # 双线性插值
                              mode='reflect',
                              anti_aliasing=True,
                              preserve_range=True)
        

        # 转换为tensor
        image_tensor = torch.from_numpy(resized_image).unsqueeze(0).float()  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()    # (1, H, W)
        # max_value = torch.max(image_tensor)
        # print(f"最大值: {max_value.item()}")  # 使用.item()获取Python标量值

        # 转换为三通道
        image_tensor = image_tensor.repeat(3, 1, 1)  # (3, H, W)
        
        # return {
        #     'image': image_tensor,
        #     'mask': mask_tensor,
        #     'domain': os.path.basename(os.path.dirname(pkl_file)),
        #     'volume_id': os.path.basename(pkl_file).replace('.pkl', '')
        # }#修改这里，
        return  image_tensor,mask_tensor, os.path.basename(os.path.dirname(pkl_file)),os.path.basename(pkl_file).replace('.pkl', '')
        #修改这里，

def visualize_sample(sample):
    try:
        plt.figure(figsize=(12, 6))
        
        # 获取并检查图像数据
        img = sample['image'].cpu().numpy()
        if img.shape[0] != 3:
            raise ValueError(f"图像通道顺序异常，应为(3, H, W)，实际为{img.shape}")
        
        # 转换为(H, W, 3)并取第一个通道
        img_gray = img.transpose(1, 2, 0)[..., 0]
        
        # 获取并检查掩码数据
        mask = sample['mask'].cpu().numpy().squeeze()
        if mask.ndim != 2:
            raise ValueError(f"掩码维度异常，预期2D，实际为{mask.shape}")
        
        # 显示图像
        plt.subplot(1, 2, 1)
        plt.imshow(img_gray, cmap='gray')
        plt.title(f"Image Shape: {img_gray.shape}")
        
        # 显示叠加效果
        plt.subplot(1, 2, 2)
        plt.imshow(img_gray, cmap='gray')
        plt.imshow(mask, cmap='viridis', alpha=0.3)
        plt.title(f"Mask Shape: {mask.shape}")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"可视化失败: {str(e)}")
        print("调试信息:")
        print(f"Domain: {sample.get('domain', '未知')}")
        print(f"Volume ID: {sample.get('volume_id', '未知')}")
        print(f"图像尺寸: {img.shape if 'image' in locals() else '未获取'}")
        print(f"掩码尺寸: {mask.shape if 'mask' in locals() else '未获取'}")
        if 'img' in locals():
            print(f"图像数值范围: {np.min(img)} - {np.max(img)}")
        if 'mask' in locals():
            print(f"掩码数值范围: {np.unique(mask)}")


if __name__ == "__main__":
    # 初始化数据集
    dataset = nasopharyngealDataset(
        preprocessed_dir="./ubuntu/Desktop/MS_NPC_guran/preprocessed/Domain1_train/",
    )
    
    # 打印数据集信息
    print(f"数据集包含切片数量: {len(dataset)}")
    # print(f"第一个样本的域: {dataset[0]['domain']}")
    # print(f"第一个样本的volume ID: {dataset[0]['volume_id']}")
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=torch.Generator(device='cpu')  # 使用CPU生成器
    )
    
    # 打印dataloader信息
    print(f"\nDataLoader包含批次数量: {len(dataloader)}")
    # print(f"每个批次形状: {next(iter(dataloader))['image'].shape}")
    
    # 可视化第一个样本
    img,mask,_,_ = dataset[0]
    # 转换为numpy数组
    img_np = img.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    mask_np = mask.numpy().squeeze()        # (1, H, W) -> (H, W)

    # 可视化
    plt.figure(figsize=(12, 6))

    # 显示图像（取第一个通道）
    plt.subplot(1, 2, 1)
    plt.imshow(img_np[..., 0], cmap='gray')
    plt.title('MRI Image')

    # 显示mask叠加
    plt.subplot(1, 2, 2)
    plt.imshow(img_np[..., 0], cmap='gray')
    plt.imshow(mask_np, cmap='viridis', alpha=0.3)
    plt.title('Mask Overlay')

    plt.tight_layout()
    plt.show()
    print("\n可视化第一个样本:")
    # visualize_sample(sample)




# if __name__ == "__main__":
#    # %% sanity test of dataset class
#     tr_dataset = nasopharyngealDataset("./eca72c87-0efd-4c31-86af-e7a3bc8ac1fd/pywjx/domain1/")
#     generator = torch.Generator(device='cpu')
#     tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True, pin_memory=True,
#                             generator=generator)
#     for step, (image, gt, bboxes, names_temp,_,_) in enumerate(tr_dataloader):
#         print(image.shape, gt.shape, bboxes.shape)
#         # show the example

#         # 提取第一个样本
#         image = image[3]  # [3, 1024, 1024]
#         mask = gt[3]  # [1, 1024, 1024]
#         image = image.permute(1, 2, 0).cpu().numpy()  # [1024, 1024, 3]
#         mask = mask.squeeze(0).cpu().numpy()         # [1024, 1024]
#         # 创建图像和 mask 的可视化
#         plt.figure(figsize=(12, 6))

#         # 显示原始图像
#         plt.subplot(1, 2, 1)
#         plt.imshow(image)
#         plt.title("Original Image")
#         plt.axis("off")

#         # 显示 Mask
#         plt.subplot(1, 2, 2)
#         plt.imshow(mask, cmap='gray')
#         plt.title("Mask")
#         plt.axis("off")

#         # 创建一个叠加效果
#         overlay = image.copy()
#         overlay[mask > 0] = [1, 0, 0]  # 在 mask 区域添加红色覆盖

#         # 可视化叠加效果
#         plt.figure(figsize=(6, 6))
#         plt.imshow(overlay)
#         plt.title("Image with Mask Overlay")
#         plt.axis("off")
#         plt.show()

#         # 显示结果
#         # plt.tight_layout()
#         # plt.show()

#         break
