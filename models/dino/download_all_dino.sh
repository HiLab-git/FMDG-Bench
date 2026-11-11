#!/bin/bash

# 配置
SAVE_DIR="./web_pth"
mkdir -p "$SAVE_DIR"

echo "Downloading DINO models to $SAVE_DIR..."

# DINOv1
echo "=== Downloading DINOv1 ==="
wget -P "$SAVE_DIR" https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
wget -P "$SAVE_DIR" https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
wget -P "$SAVE_DIR" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth

# DINOv2
echo "=== Downloading DINOv2 ==="
wget -P "$SAVE_DIR" https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
wget -P "$SAVE_DIR" https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
wget -P "$SAVE_DIR" https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth

# DINOv3
echo "=== Downloading DINOv3 ==="
wget -P "$SAVE_DIR" https://dl.fbaipublicfiles.com/dinov3/dinov3_vits14_pretrain.pth
wget -P "$SAVE_DIR" https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb14_pretrain.pth
wget -P "$SAVE_DIR" https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl14_pretrain.pth

echo "Download completed!"
echo "Models saved in: $SAVE_DIR"
ls -lh "$SAVE_DIR"