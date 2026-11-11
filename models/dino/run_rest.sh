#########################
#####dinov1

##DPT
python train_dinov1_D_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name Fundus_D 2>&1 | tee ./log/train_F_dino1_D_1234_$(date +%Y%m%d_%H%M%S).log
python train_dinov1_D_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name NPC_D 2>&1 | tee ./log/train_N_dino1_D_1234_$(date +%Y%m%d_%H%M%S).log
python train_dinov1_D_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name PolypDB_D 2>&1 | tee ./log/train_P_dino1_D_1234_$(date +%Y%m%d_%H%M%S).log

##DPT+lora
python train_dinov1_D_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name Fundus_D_lora 2>&1 | tee ./log/train_F_dino1_Dlora_1234_$(date +%Y%m%d_%H%M%S).log
python train_dinov1_D_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name NPC_D_lora 2>&1 | tee ./log/train_N_dino1_Dlora_1234_$(date +%Y%m%d_%H%M%S).log
python train_dinov1_D_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name PolypDB_D_lora 2>&1 | tee ./log/train_P_dino1_Dlora_1234_$(date +%Y%m%d_%H%M%S).log

##CNN
python dinov1_cnn_lora_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name Fundus_cnn 2>&1 | tee ./log/F_dino1_cnn_1234_$(date +%Y%m%d_%H%M%S).log
python dinov1_cnn_lora_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name NPC_cnn 2>&1 | tee ./log/N_dino1_cnn_1234_$(date +%Y%m%d_%H%M%S).log
python dinov1_cnn_lora_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name PolypDB_cnn 2>&1 | tee ./log/P_dino1_cnn_1234_$(date +%Y%m%d_%H%M%S).log

##CNN+lora
python dinov1_cnn_lora_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name Fundus_cnn_lora 2>&1 | tee ./log/F_dino1_cnnlora_1234_$(date +%Y%m%d_%H%M%S).log
python dinov1_cnn_lora_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name NPC_cnn_lora 2>&1 | tee ./log/N_dinov1_cnnlora_1234_$(date +%Y%m%d_%H%M%S).log
python dinov1_cnn_lora_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 224 --input_w 224 --dino_size s --dino_ckpt ./web_pth/dino_deitsmall16_pretrain.pth --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name PolypDB_cnn_lora 2>&1 | tee ./log/P_dinov1_cnnlora_1234_$(date +%Y%m%d_%H%M%S).log


#########################
######dinov2

###DPT
python train_dinov2_D_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name Fundus_D 2>&1 | tee ./log/F_dino2_D_1234_$(date +%Y%m%d_%H%M%S).log
python train_dinov2_D_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name NPC_D 2>&1 | tee ./log/N_dino2_D_1234_$(date +%Y%m%d_%H%M%S).log
python train_dinov2_D_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name PolypDB_D 2>&1 | tee ./log/P_dino2_D_1234_$(date +%Y%m%d_%H%M%S).log
##DPT+lora
python train_dinov2_D_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name Fundus_D_lora 2>&1 | tee ./log/F_dino2_Dlora_1234_$(date +%Y%m%d_%H%M%S).log
python train_dinov2_D_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name NPC_D_lora 2>&1 | tee ./log/N_dino2_Dlora_1234_$(date +%Y%m%d_%H%M%S).log
python train_dinov2_D_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name PolypDB_D_lora 2>&1 | tee ./log/P_dino2_Dlora_1234_$(date +%Y%m%d_%H%M%S).log

##cnn
python dinov2_cnn_lora_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name Fundus_cnn  2>&1 | tee ./log/F_dino2_cnn_1234_$(date +%Y%m%d_%H%M%S).log
python dinov2_cnn_lora_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name NPC_cnn 2>&1 | tee ./log/N_dino2_cnn_1234_$(date +%Y%m%d_%H%M%S).log
python dinov2_cnn_lora_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name PolypDB_cnn 2>&1 | tee ./log/P_dino2_cnn_234_$(date +%Y%m%d_%H%M%S).log

##cnn+lora
python dinov2_cnn_lora_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name Fundus_cnn_lora  2>&1 | tee ./log/F_dino2_cnnlora_1234_$(date +%Y%m%d_%H%M%S).log
python dinov2_cnn_lora_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name NPC_cnn_lora 2>&1 | tee ./log/N_dino2_cnnlora_1234_$(date +%Y%m%d_%H%M%S).log
python dinov2_cnn_lora_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 518 --input_w 518 --dino_size s --dino_ckpt ./web_pth/dinov2_vits14_pretrain.pth --epochs 50 --batch_size 4 --lr 1e-4 --cuda_id 2 --task_name PolypDB_cnn_lora 2>&1 | tee ./log/P_dino2_cnnlora_1234_$(date +%Y%m%d_%H%M%S).log

#########################
######dinov3

###DPT
python train_segdinov3_Fundus_D.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name image --label_dir_name mask --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name Fundus_D 2>&1 | tee ./log/F_dinov3_D_1234_$(date +%Y%m%d_%H%M%S).log
python train_segdinov3_NPC_D.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name image --label_dir_name mask --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name NPC_D 2>&1 | tee ./log/N_dinov3_D_1234_$(date +%Y%m%d_%H%M%S).log
python train_segdinov3_PolypDB_D.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name images --label_dir_name masks --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name PolypDB_D 2>&1 | tee ./log/P_dinov3_D_1234_$(date +%Y%m%d_%H%M%S).log

###DPT+lora
python train_segdinov3_Fundus_D.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name image --label_dir_name mask --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name Fundus_D_lora 2>&1 | tee ./log/F_dinov3_Dlora_1234_$(date +%Y%m%d_%H%M%S).log
python train_segdinov3_NPC_D.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name image --label_dir_name mask --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name NPC_D_lora 2>&1 | tee ./log/N_dinov3_Dlora_1234_$(date +%Y%m%d_%H%M%S).log
python train_segdinov3_PolypDB_D.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name images --label_dir_name masks --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name PolypDB_D_lora 2>&1 | tee ./log/P_dinov3_Dlora_1234_$(date +%Y%m%d_%H%M%S).log

###cnn
python dinov3_cnn_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name image --label_dir_name mask --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name Fundus_cnn 2>&1 | tee ./log/F_dinov3_cnn_1234_$(date +%Y%m%d_%H%M%S).log
python dinov3_cnn_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name image --label_dir_name mask --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name NPC_cnn 2>&1 | tee ./log/N_dinov3_cnn_1234_$(date +%Y%m%d_%H%M%S).log
python dinov3_cnn_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name images --label_dir_name masks --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name PolypDB_cnn 2>&1 | tee ./log/P_dinov3_cnn_1234_$(date +%Y%m%d_%H%M%S).log

###cnn+lora
python dinov3_cnn_Fundus.py --data_dir ${FMDG_DATA_ROOT}/Fundus --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name image --label_dir_name mask --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name Fundus_cnn_lora 2>&1 | tee ./log/F_dinov3_cnnlora_1234_$(date +%Y%m%d_%H%M%S).log
python dinov3_cnn_NPC.py --data_dir ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name image --label_dir_name mask --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name NPC_cnn_lora 2>&1 | tee ./log/N_dinov3_cnnlora_1234_$(date +%Y%m%d_%H%M%S).log
python dinov3_cnn_PolypDB.py --data_dir ${FMDG_DATA_ROOT}/PolypDB --trainid 1,2,3,4 --input_h 256 --input_w 256 --dino_size s --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --repo_dir ./dinov3 --img_dir_name images --label_dir_name masks --mask_ext '.png' --epochs 50 --batch_size 16 --lr 1e-4 --cuda_id 2 --task_name PolypDB_cnn_lora 2>&1 | tee ./log/P_dinov3_cnnlora_234_$(date +%Y%m%d_%H%M%S).log
