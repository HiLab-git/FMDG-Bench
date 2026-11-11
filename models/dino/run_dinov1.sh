# python train_dinov1_NPC_D.py --data_dir ./web_pth/dino_deitsmall16_pretrain.pth --trainid 1,2,3 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./dino3/SegDINO/web_pth/dinov2_vits14_pretrain.pth  --img_dir_name image --label_dir_name mask  --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0
# python train_dinov1_NPC_D.py \
#     --dino_ckpt ./dino_deitsmall16_pretrain.pth \
#     --dino_size s \
#     --data_dir ./dummy_data --batch_size 2 --epochs 2

# python train_dinov1_D_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 2>&1 | tee ./logs/train_F_dino1_234_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_F_dino1_134_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_F_dino1_124_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_F_dino1_123_$(date +%Y%m%d_%H%M%S).log

# python train_dinov1_D_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_N_dino1_234_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_N_dino1_134_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_N_dino1_124_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_N_dino1_123_$(date +%Y%m%d_%H%M%S).log

# python train_dinov1_D_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_P_dino1_234_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_P_dino1_134_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_P_dino1_124_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 0 2>&1 | tee ./logs/train_P_dino1_123_$(date +%Y%m%d_%H%M%S).log

#DPT+lora
# python train_dinov1_D_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name Fundus_D_lora 2>&1 | tee ./logs/train_F_dino1_Dlora_234_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name NPC_D_lora 2>&1 | tee ./logs/train_N_dino1_Dlora_234_$(date +%Y%m%d_%H%M%S).log
# python train_dinov1_D_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name PolypDB_D_lora 2>&1 | tee ./logs/train_P_dino1_Dlora_234_$(date +%Y%m%d_%H%M%S).log

#CNN
# python dinov1_cnn_lora_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name Fundus_C_lora #2>&1 | tee ./logs/train_F_dino1_cnn_234_$(date +%Y%m%d_%H%M%S).log
# python dinov1_cnn_lora_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 # 2>&1 | tee ./logs/train_N_dino1_234_$(date +%Y%m%d_%H%M%S).log
python dinov1_cnn_lora_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 1 # 2>&1 | tee ./logs/train_P_dino1_234_$(date +%Y%m%d_%H%M%S).log
