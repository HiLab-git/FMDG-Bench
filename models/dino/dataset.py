import os
from typing import List, Tuple, Optional
import cv2
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class ResizeAndNormalize:
    def __init__(self, size=(256, 256), mean=IMAGENET_MEAN, std=IMAGENET_STD, thr=0.5):
        self.size = size  # (H, W)
        self.mean = mean
        self.std = std
        self.thr = thr

    def __call__(self, img_bgr: np.ndarray, mask_hwc: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = TF.resize(
            img_pil, self.size,
            interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        img_t = TF.to_tensor(img_resized)                   # Convert image to tensor
        img_t = TF.normalize(img_t, self.mean, self.std)    # Normalize with ImageNet mean and std
        H, W = self.size
        mask_ts = []
        for c in range(mask_hwc.shape[2]):
            m = Image.fromarray(mask_hwc[..., c])
            m = TF.resize(m, (H, W), interpolation=InterpolationMode.NEAREST)
            mt = TF.to_tensor(m)[0]        # Take the single channel
            mask_ts.append(mt)
        mask_t = torch.stack(mask_ts, dim=0).float()   # Stack into (C,H,W)
        mask_t = (mask_t > self.thr).float()           # Binarize mask by threshold
        return img_t, mask_t

class ResizeAndNormalize_NPC:
    def __init__(self, size=(256, 256), mean=IMAGENET_MEAN, std=IMAGENET_STD, thr=0.5):
        self.size = size  # (H, W)
        self.mean = mean
        self.std = std
        self.thr = thr

    def __call__(self, img_bgr: np.ndarray, mask_hwc: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = TF.resize(
            img_pil, self.size,
            interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        img_t = TF.to_tensor(img_resized)                   # Convert image to tensor
        img_t = TF.normalize(img_t, self.mean, self.std)    # Normalize with ImageNet mean and std
        H, W = self.size
        mask_ts = []
        for c in range(mask_hwc.shape[2]):
            m = Image.fromarray(mask_hwc[..., c])
            m = TF.resize(m, (H, W), interpolation=InterpolationMode.NEAREST)
            m_np = np.array(m, dtype=np.float32)  # 转为numpy数组
            mt = torch.from_numpy(m_np)  # 直接转为tensor，不做归一化
            mask_ts.append(mt)
        mask_t = torch.stack(mask_ts, dim=0).float()   # Stack into (C,H,W)
        mask_t = (mask_t > self.thr).float()           # Binarize mask by threshold
        return img_t, mask_t

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
def _list_files(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    out.sort()
    return out

def _swap_dir_keep_name(path: str, src_dir_name: str, dst_dir_name: str, dst_ext: Optional[str]) -> str:
    parts = path.replace("\\", "/").split("/")
    try:
        i = parts.index(src_dir_name)
    except ValueError:
        raise RuntimeError(f"Directory name '{src_dir_name}' not found in path: {path}")
    parts[i] = dst_dir_name
    dst_path = "/".join(parts)
    if dst_ext is not None:
        base, _ = os.path.splitext(dst_path)
        dst_path = base + dst_ext
    return dst_path

class FolderDataset(data.Dataset):
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dir_name: str = "image",
        label_dir_name: str = "mask",
        img_exts: Tuple[str, ...] = SUPPORTED_EXTS,
        mask_ext: Optional[str] = None,  
        transform: Optional[ResizeAndNormalize] = None,
        strict_pair: bool = True,       
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir_name = img_dir_name
        self.label_dir_name = label_dir_name
        self.img_dir = os.path.join(root, img_dir_name)
        self.label_dir = os.path.join(root, label_dir_name)
        self.mask_ext = mask_ext
        self.transform = transform
        self.strict_pair = strict_pair

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.img_paths = _list_files(self.img_dir, img_exts)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir} (supported extensions: {img_exts})")

        self.pairs: List[Tuple[str, str]] = []
        for ip in self.img_paths:
            mp = _swap_dir_keep_name(ip, self.img_dir_name, self.label_dir_name, self.mask_ext)
            if not os.path.isfile(mp) and self.mask_ext is None:
                base, _ = os.path.splitext(mp)
                found = False
                for ext in SUPPORTED_EXTS:
                    cand = base + ext
                    if os.path.isfile(cand):
                        mp = cand
                        found = True
                        break
                if not found and self.strict_pair:
                    raise FileNotFoundError(f"Label file not found: {base}.(any extension with same name)")
                elif not found:
                    continue
            elif not os.path.isfile(mp) and self.strict_pair:
                raise FileNotFoundError(f"Label file not found: {mp}")
            elif not os.path.isfile(mp):
                continue

            self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid (img, label) pairs in {self.split} set!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3) BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
        if mask is None:
            raise RuntimeError(f"Failed to read label: {mask_path}")
        # (H,W) -> (H,W,1)
        mask_hwc = mask[..., None]

        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask_hwc)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            m = torch.from_numpy(mask_hwc).permute(2, 0, 1).float() / 255.0
            mask_t = (m > 0.5).float()

        if mask_t.max().item() < 1.0:
            mask_t = (mask_t > 0).float()

        meta = {
            "img_path": img_path,
            "mask_path": mask_path,
            "id": os.path.splitext(os.path.basename(img_path))[0],
        }
        return img_t, mask_t, meta

class FolderDataset_PolypDB(data.Dataset):
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dir_name: str = "images",
        label_dir_name: str = "masks",
        img_exts: Tuple[str, ...] = SUPPORTED_EXTS,
        mask_ext: Optional[str] = None,  
        transform: Optional[ResizeAndNormalize] = None,
        strict_pair: bool = True,       
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir_name = img_dir_name
        self.label_dir_name = label_dir_name
        self.img_dir = os.path.join(root, img_dir_name)
        self.label_dir = os.path.join(root, label_dir_name)
        self.mask_ext = mask_ext
        self.transform = transform
        self.strict_pair = strict_pair

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.img_paths = _list_files(self.img_dir, img_exts)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir} (supported extensions: {img_exts})")

        self.pairs: List[Tuple[str, str]] = []
        for ip in self.img_paths:
            mp = _swap_dir_keep_name(ip, self.img_dir_name, self.label_dir_name, self.mask_ext)
            if not os.path.isfile(mp) and self.mask_ext is None:
                base, _ = os.path.splitext(mp)
                found = False
                for ext in SUPPORTED_EXTS:
                    cand = base + ext
                    if os.path.isfile(cand):
                        mp = cand
                        found = True
                        break
                if not found and self.strict_pair:
                    raise FileNotFoundError(f"Label file not found: {base}.(any extension with same name)")
                elif not found:
                    continue
            elif not os.path.isfile(mp) and self.strict_pair:
                raise FileNotFoundError(f"Label file not found: {mp}")
            elif not os.path.isfile(mp):
                continue

            self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid (img, label) pairs in {self.split} set!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3) BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
        if mask is None:
            raise RuntimeError(f"Failed to read label: {mask_path}")
        # (H,W) -> (H,W,1)
        mask_hwc = mask[..., None]

        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask_hwc)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            m = torch.from_numpy(mask_hwc).permute(2, 0, 1).float() / 255.0
            mask_t = (m > 0.5).float()

        if mask_t.max().item() < 1.0:
            mask_t = (mask_t > 0).float()

        meta = {
            "img_path": img_path,
            "mask_path": mask_path,
            "id": os.path.splitext(os.path.basename(img_path))[0],
        }
        return img_t, mask_t, meta


import pickle
import numpy as np
import cv2
import torch
import os
from typing import Tuple, List, Optional
from torch.utils import data
from functools import lru_cache

class FolderDataset_NPC(data.Dataset):
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dir_name: str = "image",
        label_dir_name: str = "mask",
        img_exts: Tuple[str, ...] = SUPPORTED_EXTS,
        mask_ext: Optional[str] = None,  
        transform: Optional[ResizeAndNormalize] = None,
        strict_pair: bool = True,
        use_pkl: bool = False,  # 新增：是否使用pkl文件
        pkl_dir: Optional[str] = None,  # 新增：pkl文件目录
        max_cache_size: int = 100,  # 新增：缓存大小
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir_name = img_dir_name
        self.label_dir_name = label_dir_name
        self.mask_ext = mask_ext
        self.transform = transform
        self.strict_pair = strict_pair
        self.use_pkl = use_pkl
        self.max_cache_size = max_cache_size
        
        if self.use_pkl:
            # 使用pkl文件模式
            if pkl_dir is None:
                # 默认使用root作为pkl目录
                pkl_dir = root
            self.pkl_dir = pkl_dir
            
            # 获取所有pkl文件
            self.pkl_files = sorted([
                os.path.join(self.pkl_dir, f) 
                for f in os.listdir(self.pkl_dir) 
                if f.endswith('.pkl')
            ])
            
            if len(self.pkl_files) == 0:
                raise RuntimeError(f"No pkl files found in {self.pkl_dir}")
            
            # 构建切片索引（类似nasopharyngealDataset的做法）
            self.slice_records = []
            for pkl_file in self.pkl_files:
                # 预加载一次来获取切片数量
                with open(pkl_file, 'rb') as f:
                    vol_data = pickle.load(f)
                    
                # 根据pkl文件的结构进行处理
                if isinstance(vol_data, dict):
                    # 如果是字典格式，假设包含images和masks
                    if 'images' in vol_data and 'masks' in vol_data:
                        num_slices = len(vol_data['images'])
                        for slice_idx in range(num_slices):
                            self.slice_records.append({
                                'pkl_file': pkl_file,
                                'slice_idx': slice_idx,
                                'data_type': 'dict'  # 标记数据类型
                            })
                    else:
                        # 如果是其他字典格式，假设每个键值对是一个样本
                        num_slices = len(vol_data)
                        for slice_idx in range(num_slices):
                            self.slice_records.append({
                                'pkl_file': pkl_file,
                                'slice_idx': slice_idx,
                                'data_type': 'dict_items'
                            })
                elif isinstance(vol_data, list):
                    # 如果是列表格式，假设每个元素是一个切片
                    for slice_idx in range(len(vol_data)):
                        self.slice_records.append({
                            'pkl_file': pkl_file,
                            'slice_idx': slice_idx,
                            'data_type': 'list'
                        })
                elif isinstance(vol_data, np.ndarray):
                    # 如果是numpy数组，假设第一维是切片维度
                    for slice_idx in range(vol_data.shape[0]):
                        self.slice_records.append({
                            'pkl_file': pkl_file,
                            'slice_idx': slice_idx,
                            'data_type': 'numpy_volume'
                        })
                else:
                    raise ValueError(f"Unsupported pkl file format in {pkl_file}")
            
            print(f"Loaded {len(self.slice_records)} slices from {len(self.pkl_files)} pkl files")
            
        else:
            # 原始的文件系统模式
            self.img_dir = os.path.join(root, img_dir_name)
            self.label_dir = os.path.join(root, label_dir_name)

            if not os.path.isdir(self.img_dir):
                raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
            if not os.path.isdir(self.label_dir):
                raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

            self.img_paths = _list_files(self.img_dir, img_exts)
            if len(self.img_paths) == 0:
                raise RuntimeError(f"No images found in {self.img_dir} (supported extensions: {img_exts})")

            self.pairs: List[Tuple[str, str]] = []
            for ip in self.img_paths:
                mp = _swap_dir_keep_name(ip, self.img_dir_name, self.label_dir_name, self.mask_ext)
                if not os.path.isfile(mp) and self.mask_ext is None:
                    base, _ = os.path.splitext(mp)
                    found = False
                    for ext in SUPPORTED_EXTS:
                        cand = base + ext
                        if os.path.isfile(cand):
                            mp = cand
                            found = True
                            break
                    if not found and self.strict_pair:
                        raise FileNotFoundError(f"Label file not found: {base}.(any extension with same name)")
                    elif not found:
                        continue
                elif not os.path.isfile(mp) and self.strict_pair:
                    raise FileNotFoundError(f"Label file not found: {mp}")
                elif not os.path.isfile(mp):
                    continue

                self.pairs.append((ip, mp))

            if len(self.pairs) == 0:
                raise RuntimeError(f"No valid (img, label) pairs in {self.split} set!")

    @lru_cache(maxsize=100)
    def _load_pkl_data(self, pkl_file):
        """带缓存的加载pkl数据（类似nasopharyngealDataset的做法）"""
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        if self.use_pkl:
            return len(self.slice_records)
        else:
            return len(self.pairs)

    def _extract_slice_data(self, vol_data, slice_idx, data_type):
        """从pkl数据中提取单个切片的数据"""
        if data_type == 'dict':
            # 字典格式，包含images和masks
            image = vol_data['images'][slice_idx]
            mask = vol_data['masks'][slice_idx]
        elif data_type == 'dict_items':
            # 字典格式，键值对是样本
            keys = list(vol_data.keys())
            key = keys[slice_idx]
            sample = vol_data[key]
            if isinstance(sample, dict) and 'image' in sample and 'mask' in sample:
                image = sample['image']
                mask = sample['mask']
            else:
                # 假设样本本身就是图像/掩码对
                image = sample[0] if isinstance(sample, (list, tuple)) else sample
                mask = sample[1] if isinstance(sample, (list, tuple)) else None
        elif data_type == 'list':
            # 列表格式
            slice_data = vol_data[slice_idx]
            if isinstance(slice_data, dict) and 'image' in slice_data and 'mask' in slice_data:
                # 类似nasopharyngealDataset的格式
                image = slice_data['image']
                mask = slice_data['mask']
            elif isinstance(slice_data, (list, tuple)) and len(slice_data) >= 2:
                # 假设是(image, mask)元组
                image = slice_data[0]
                mask = slice_data[1]
            else:
                raise ValueError(f"Unsupported list item format at index {slice_idx}")
        elif data_type == 'numpy_volume':
            # numpy数组格式
            image = vol_data[slice_idx]
            mask = None  # 需要根据实际情况调整
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return image, mask

    def __getitem__(self, idx: int):
        if self.use_pkl:
            # pkl文件模式
            record = self.slice_records[idx]
            pkl_file = record['pkl_file']
            slice_idx = record['slice_idx']
            data_type = record['data_type']
            
            # 加载数据（使用缓存）
            vol_data = self._load_pkl_data(pkl_file)
            
            # 提取切片数据
            image, mask = self._extract_slice_data(vol_data, slice_idx, data_type)
            
            # 确保数据是numpy数组
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # 处理不同的数据格式
            if image.ndim == 3 and image.shape[-1] == 1:
                # 单通道扩展为三通道
                image = np.repeat(image, 3, axis=-1)
            elif image.ndim == 2:
                # 灰度图扩展为三通道
                image = np.stack([image] * 3, axis=-1)
            
            # 确保数据类型正确
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            if mask is not None:
                if mask.ndim == 3:
                    if mask.shape[-1] == 1:
                        mask = mask.squeeze(-1)
                    else:
                        # 如果是三通道mask，转换为灰度
                        mask = np.mean(mask, axis=-1).astype(np.uint8)
                
                if mask.dtype != np.uint8:
                    if mask.max() <= 1.0:
                        mask = (mask * 255).astype(np.uint8)
                    else:
                        mask = mask.astype(np.uint8)
            else:
                # 如果没有mask，创建全零mask（仅用于测试）
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # 准备文件名信息
            base_name = os.path.splitext(os.path.basename(pkl_file))[0]
            img_path = f"{base_name}_slice{slice_idx}.pkl"
            mask_path = f"{base_name}_slice{slice_idx}_mask.pkl"
            
        else:
            # 原始的文件系统模式
            img_path, mask_path = self.pairs[idx]

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3) BGR uint8
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
            if mask is None:
                raise RuntimeError(f"Failed to read label: {mask_path}")
            
            image = img
            mask = mask

        # (H,W) -> (H,W,1)
        mask_hwc = mask[..., None] if mask is not None else None

        if self.transform is not None:
            if mask_hwc is not None:
                img_t, mask_t = self.transform(image, mask_hwc)
            else:
                img_t = self.transform(image)
                mask_t = torch.zeros(img_t.shape[1:], dtype=torch.float32)
        else:
            if image.shape[-1] == 3 and len(image.shape) == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image
            else:
                img_rgb = image
                
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            
            if mask_hwc is not None:
                m = torch.from_numpy(mask_hwc).permute(2, 0, 1).float() / 255.0
                mask_t = (m > 0.5).float()
            else:
                mask_t = torch.zeros(img_t.shape[1:], dtype=torch.float32)

        if mask_t.max().item() < 1.0:
            mask_t = (mask_t > 0).float()

        meta = {
            "img_path": img_path,
            "mask_path": mask_path if 'mask_path' in locals() else img_path.replace('.pkl', '_mask.pkl'),
            "id": os.path.splitext(os.path.basename(img_path))[0],
            "from_pkl": self.use_pkl,
            "slice_idx": slice_idx if self.use_pkl else 0,
            "pkl_file": pkl_file if self.use_pkl else None,
        }
        return img_t, mask_t, meta

class FolderDataset_two(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dir_name: str = "image",
        label_dir_name: str = "mask",
        img_exts: Tuple[str, ...] = SUPPORTED_EXTS,
        mask_ext: Optional[str] = None,
        transform: Optional[object] = None,
        strict_pair: bool = True,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir_name = img_dir_name
        self.label_dir_name = label_dir_name
        self.img_dir = os.path.join(root, img_dir_name)
        self.label_dir = os.path.join(root, label_dir_name)
        self.mask_ext = mask_ext
        self.transform = transform
        self.strict_pair = strict_pair

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.img_paths = _list_files(self.img_dir, img_exts)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir} (supported extensions: {img_exts})")

        self.pairs: List[Tuple[str, str]] = []
        for ip in self.img_paths:
            mp = _swap_dir_keep_name(ip, self.img_dir_name, self.label_dir_name, self.mask_ext)
            if not os.path.isfile(mp) and self.mask_ext is None:
                base, _ = os.path.splitext(mp)
                found = False
                for ext in SUPPORTED_EXTS:
                    cand = base + ext
                    if os.path.isfile(cand):
                        mp = cand
                        found = True
                        break
                if not found and self.strict_pair:
                    raise FileNotFoundError(f"Label file not found: {base}.(any extension with same name)")
                elif not found:
                    continue
            elif not os.path.isfile(mp) and self.strict_pair:
                raise FileNotFoundError(f"Label file not found: {mp}")
            elif not os.path.isfile(mp):
                continue

            self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid (img, label) pairs in {self.split} set!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        # 读取图像为BGR格式 (H, W, 3) uint8
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # 读取mask为灰度图 (H, W) uint8
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read label: {mask_path}")

        # 将灰度mask转换为3个类别的one-hot编码
        converted_mask = np.zeros_like(mask, dtype=np.uint8)
        converted_mask[mask > 200] = 0       # 背景
        converted_mask[(mask >= 100) & (mask <= 200)] = 1  # 视盘
        converted_mask[mask < 100] = 2       # 视杯
        
        # 转换为one-hot编码 (H, W, 3) 【修改点1：调整维度顺序为HWC，适配transform】
        new_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
        new_mask[converted_mask == 0, 0] = 1  # 背景通道
        new_mask[converted_mask == 1, 1] = 1  # 视盘通道
        new_mask[converted_mask == 2, 2] = 1  # 视杯通道

        if self.transform is not None:
            # 传递HWC格式的mask给transform 【修改点2：维度顺序匹配】
            img_t, mask_t = self.transform(img, new_mask)
        else:
            # 如果没有transform，手动处理
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(new_mask).permute(2, 0, 1).float()

        meta = {
            "img_path": img_path,
            "mask_path": mask_path,
            "id": os.path.splitext(os.path.basename(img_path))[0],
        }
        
        return img_t, mask_t, meta

class ResizeAndNormalize_two:
    def __init__(self, size=(256, 256), mean=IMAGENET_MEAN, std=IMAGENET_STD, thr=0.5):
        self.size = size  # (H, W)
        self.mean = mean
        self.std = std
        self.thr = thr

    def __call__(self, img_bgr: np.ndarray, mask_hwc: np.ndarray):
        # 处理图像
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = TF.resize(
            img_pil, self.size,
            interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        img_t = TF.to_tensor(img_resized)                  
        img_t = TF.normalize(img_t, self.mean, self.std)    

        # 处理mask 【修改点3：简化mask处理逻辑，避免维度错误】
        H, W = self.size
        # 将HWC格式的mask转换为PIL Image（按类别通道拆分后resize）
        mask_ts = []
        for c in range(3):  # 固定3个类别：背景、视盘、视杯
            # 提取单通道mask (H,W)
            single_channel = mask_hwc[..., c].astype(np.uint8)
            m = Image.fromarray(single_channel)
            # Resize到目标尺寸（最近邻插值，避免类别值失真）
            m_resized = TF.resize(m, (H, W), interpolation=InterpolationMode.NEAREST)
            # 转换为tensor并取单通道
            mt = TF.to_tensor(m_resized)[0]
            # 二值化（确保是0/1）
            mt = (mt > self.thr).float()
            mask_ts.append(mt)
        
        # 堆叠为 (3, H, W) 的tensor
        mask_t = torch.stack(mask_ts, dim=0).float()
        
        return img_t, mask_t

import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from typing import Optional, Tuple, List
import torch.utils.data as data
class ResizeAndNormalize_new:
    def __init__(self, size=(256, 256), mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.size = size  # (H, W)
        self.mean = mean
        self.std = std

    def __call__(self, img_bgr: np.ndarray, mask_hwc: np.ndarray):
        # 调整图像大小
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = TF.resize(
            img_pil, self.size,
            interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        img_t = TF.to_tensor(img_resized)                   # Convert image to tensor
        img_t = TF.normalize(img_t, self.mean, self.std)    # Normalize with ImageNet mean and std
        
        # 调整mask大小
        H, W = self.size
        if mask_hwc.ndim == 2:
            mask_hwc = mask_hwc[..., None]  # 添加通道维度
        
        # 对于多类别分割，我们需要将灰度图转换为one-hot编码
        # 假设mask的格式与RetinaDataset_prototype中相同
        mask_np = mask_hwc.squeeze()  # [H, W] 或 [H, W, 1] -> [H, W]
        
        # 根据灰度值将mask转换为3个类别（0: 背景, 1: 视盘, 2: 视杯）
        # 阈值与RetinaDataset_prototype保持一致
        converted_mask = np.zeros_like(mask_np, dtype=np.uint8)
        
        # 将白色（>200）的像素设置为0，表示背景
        converted_mask[mask_np > 200/255] = 0
        
        # 对灰色值范围 [100/255, 200/255] 的像素赋值为1（表示视盘）
        converted_mask[(mask_np >= 100/255) & (mask_np <= 200/255)] = 1
        
        # 对灰色值范围 [<100/255] 的像素赋值为2（表示视杯）
        converted_mask[mask_np < 100/255] = 2
        
        # 转换为one-hot编码（3个通道）
        onehot_mask = np.zeros((3, H, W), dtype=np.float32)
        for c in range(3):
            # 调整每个通道的大小
            channel_mask = (converted_mask == c).astype(np.float32)
            channel_pil = Image.fromarray(channel_mask)
            channel_resized = TF.resize(channel_pil, (H, W), 
                                       interpolation=InterpolationMode.NEAREST)
            channel_t = TF.to_tensor(channel_resized)[0]  # 取单通道
            onehot_mask[c] = channel_t
        
        return img_t, torch.from_numpy(onehot_mask).float()

class FolderDataset_new(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dir_name: str = "image",
        label_dir_name: str = "mask",
        img_exts: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'),
        mask_ext: Optional[str] = None,  
        transform: Optional[ResizeAndNormalize_new] = None,
        strict_pair: bool = True,
        num_classes: int = 3  # 3个类别：背景、视盘、视杯
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir_name = img_dir_name
        self.label_dir_name = label_dir_name
        self.img_dir = os.path.join(root, img_dir_name)
        self.label_dir = os.path.join(root, label_dir_name)
        self.mask_ext = mask_ext
        self.transform = transform
        self.strict_pair = strict_pair
        self.num_classes = num_classes

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        # 获取图像和掩码文件列表
        self.img_paths = []
        for ext in img_exts:
            self.img_paths.extend(glob.glob(os.path.join(self.img_dir, f"**/*{ext}"), recursive=True))
        self.img_paths = sorted(self.img_paths)
        
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir} (supported extensions: {img_exts})")

        self.pairs: List[Tuple[str, str]] = []
        for ip in self.img_paths:
            # 构建对应的掩码路径
            base_name = os.path.basename(ip)
            name_without_ext = os.path.splitext(base_name)[0]
            
            if self.mask_ext:
                mp = os.path.join(self.label_dir, f"{name_without_ext}{self.mask_ext}")
            else:
                # 尝试所有支持的扩展名
                mp = None
                for ext in img_exts:
                    cand = os.path.join(self.label_dir, f"{name_without_ext}{ext}")
                    if os.path.isfile(cand):
                        mp = cand
                        break
            
            if mp is None or not os.path.isfile(mp):
                if self.strict_pair:
                    raise FileNotFoundError(f"Label file not found for {ip}")
                else:
                    continue
                    
            self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid (img, label) pairs in {self.split} set!")
            
        print(f"✅ Found {len(self.pairs)} image-mask pairs in {root}/{split}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3) BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # 读取掩码（灰度图）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
        if mask is None:
            raise RuntimeError(f"Failed to read label: {mask_path}")
        
        # 将掩码归一化到[0, 1]
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        
        # 转换为[H, W, 1]格式
        mask_hwc = mask[..., None]

        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask_hwc)
        else:
            # 如果没有transform，直接转换为tensor
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            
            # 根据灰度值将mask转换为3个类别的one-hot编码
            converted_mask = np.zeros_like(mask, dtype=np.uint8)
            converted_mask[mask > 200/255] = 0
            converted_mask[(mask >= 100/255) & (mask <= 200/255)] = 1
            converted_mask[mask < 100/255] = 2
            
            # 转换为one-hot编码
            h, w = converted_mask.shape
            mask_t = torch.zeros(self.num_classes, h, w, dtype=torch.float32)
            for c in range(self.num_classes):
                mask_t[c] = torch.from_numpy((converted_mask == c).astype(np.float32))

        meta = {
            "img_path": img_path,
            "mask_path": mask_path,
            "id": os.path.splitext(os.path.basename(img_path))[0],
        }
        return img_t, mask_t, meta

