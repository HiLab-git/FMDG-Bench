# python cnn_sam3_Fundus.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR}  --task_name sam3_cnn_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_cnn_Fundus_1_$(date +%Y%m%d_%H%M%S).log


# ##cnn;sam
#  python cnn_sam3_PolypDB.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/PolypDB/PolypDB_modality_wise/ --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_cnn_poly-domaim1,2,3,4_3000000/ --task_name sam3_cnn_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_cnn_Poly_1_$(date +%Y%m%d_%H%M%S).log

#  python cnn_sam3_PolypDB.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/PolypDB/PolypDB_modality_wise/ --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_cnn_poly-domaim1,2,3,4_3000000/ --task_name sam3_cnn_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_cnn_Poly_2_$(date +%Y%m%d_%H%M%S).log

#  python cnn_sam3_Fundus.py --data_root ${FMDG_DATA_ROOT}/Fundus/ --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_cnn_fu-domaim1,2,3,4_3000000/ --task_name sam3_cnn_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam3_cnn_Fundus_1_$(date +%Y%m%d_%H%M%S).log

#  python cnn_sam3_Fundus.py --data_root ${FMDG_DATA_ROOT}/Fundus/ --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_cnn_fu-domaim1,2,3,4_3000000/ --task_name sam3_cnn_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_cnn_Fundus_2_$(date +%Y%m%d_%H%M%S).log

#  python cnn_sam3_NPC.py --data_root ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed/ --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR}  --checkpoint ${FMDG_WORK_DIR}naso_auto/sam_cnn_na-domaim1,2,3,4_3000000/ --task_name sam3_cnn_na --num_epochs 50 --iternum 3000000 --device cuda:0  2>&1 | tee ./logs/sam_cnn_NPC_1_$(date +%Y%m%d_%H%M%S).log

#  python cnn_sam3_NPC.py --data_root ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed/ --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR}  --checkpoint ${FMDG_WORK_DIR}naso_auto/sam_cnn_na-domaim1,2,3,4_3000000/ --task_name sam3_cnn_na --num_epochs 50 --iternum 3000000 --device cuda:0  2>&1 | tee ./logs/sam_cnn_NPC_2_$(date +%Y%m%d_%H%M%S).log



#baseline
python TransUNet.py --data_root ${FMDG_DATA_ROOT}/Fundus/ --batch_size 1 --trainid 1,3 --testid 2,4  --task_name UNet-Baseline --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/TransUNet_Fundus_1_$(date +%Y%m%d_%H%M%S).log

python TransUNet.py --data_root ${FMDG_DATA_ROOT}/Fundus/ --batch_size 1 --trainid 2,4 --testid 1,3   --task_name UNet-Baseline --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/TransUNet_Fundus_2_$(date +%Y%m%d_%H%M%S).log

python TransUNet.py --data_root ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed/ --batch_size 1 --trainid 1,4 --testid 2,3   --task_name UNet-Baseline-na --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/TransUNet_NPC_1_$(date +%Y%m%d_%H%M%S).log

python TransUNet.py --data_root ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed/ --batch_size 1 --trainid 2,3 --testid 1,4   --task_name UNet-Baseline-na --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/TransUNet_NPC_2_$(date +%Y%m%d_%H%M%S).log

python TransUNet.py --data_root ${FMDG_DATA_ROOT}/PolypDB/PolypDB/PolypDB_modality_wise/ --batch_size 1 --trainid 1,2 --testid 3,4   --task_name UNet-Baseline-poly --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/TransUNet_POLYPDB_1_$(date +%Y%m%d_%H%M%S).log

python TransUNet.py --data_root ${FMDG_DATA_ROOT}/PolypDB/PolypDB/PolypDB_modality_wise/ --batch_size 1 --trainid 3,4 --testid 1,2   --task_name UNet-Baseline-poly --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/TransUNet_POLYPDB_2_$(date +%Y%m%d_%H%M%S).log


#使用DPT解码器sam

# python DPT_sam3_Fundus.py --data_root ${FMDG_DATA_ROOT}/Fundus/ --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPT_fu-domaim1,2,3,4_300000/ --task_name sam3_DPT_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_DPT_Fundus_1_$(date +%Y%m%d_%H%M%S).log

# python DPT_sam3_Fundus.py --data_root ${FMDG_DATA_ROOT}/Fundus/ --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPT_fu-domaim1,2,3,4_300000/ --task_name sam3_DPT_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_DPT_Fundus_2_$(date +%Y%m%d_%H%M%S).log

python DPT_sam3_NPC.py --data_root ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed/ --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPT_na-domaim1,2,3,4_300000/ --task_name sam3_DPT_na --num_epochs 50 --iternum 3000000 --device cuda:0 --lr 0.0001 2>&1 | tee ./logs/sam_DPT_NPC_1_$(date +%Y%m%d_%H%M%S).log

python DPT_sam3_NPC.py --data_root ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed/ --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPT_na-domaim1,2,3,4_300000/ --task_name sam3_DPT_na --num_epochs 50 --iternum 3000000 --device cuda:0 --lr 0.0001 2>&1 | tee ./logs/sam_DPT_NPC_2_$(date +%Y%m%d_%H%M%S).log

python DPT_sam3_PolypDB.py --data_root ${FMDG_DATA_ROOT}/PolypDB/PolypDB/PolypDB_modality_wise --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPT_poly-domaim1,2,3,4_300000/ --task_name sam3_DPT_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_DPT_Poly_1_$(date +%Y%m%d_%H%M%S).log

python DPT_sam3_PolypDB.py --data_root ${FMDG_DATA_ROOT}/PolypDB/PolypDB/PolypDB_modality_wise --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPT_poly-domaim1,2,3,4_300000/ --task_name sam3_DPT_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_DPT_Poly_2_$(date +%Y%m%d_%H%M%S).log


#使用DPT+lora;模型sam

python DPT_sam3_Fundus.py --data_root ${FMDG_DATA_ROOT}/Fundus/ --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPTlora_fu-domaim1,2,3,4_300000/ --task_name sam3_DPTlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_DPTlora_Fundus_1_$(date +%Y%m%d_%H%M%S).log

python DPT_sam3_Fundus.py --data_root ${FMDG_DATA_ROOT}/Fundus/ --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPTlora_fu-domaim1,2,3,4_300000/ --task_name sam3_DPTlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_DPTlora_Fundus_2_$(date +%Y%m%d_%H%M%S).log

python DPT_sam3_NPC.py --data_root ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed/ --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPTlora_na-domaim1,2,3,4_300000/ --task_name sam3_DPTlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_DPTlora_NPC_1_$(date +%Y%m%d_%H%M%S).log

python DPT_sam3_NPC.py --data_root ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed/ --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPTlora_na-domaim1,2,3,4_300000/ --task_name sam3_DPTlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_DPTlora_NPC_2_$(date +%Y%m%d_%H%M%S).log

python DPT_sam3_PolypDB.py --data_root ${FMDG_DATA_ROOT}/PolypDB/PolypDB/PolypDB_modality_wise --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPTlora_poly-domaim1,2,3,4_300000/ --task_name sam3_DPTlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_DPTlora_Poly_1_$(date +%Y%m%d_%H%M%S).log

python DPT_sam3_PolypDB.py --data_root ${FMDG_DATA_ROOT}/PolypDB/PolypDB/PolypDB_modality_wise --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_DPTlora_poly-domaim1,2,3,4_300000/ --task_name sam3_DPTlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_DPTlora_Poly_2_$(date +%Y%m%d_%H%M%S).log

# ##cnn+lora;sam

 python cnn_sam3_Fundus.py --data_root ${FMDG_DATA_ROOT}/Fundus/ --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_cnnlora_fu-domaim1,2,3,4_3000000/ --task_name sam3_cnnlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_cnnlora_Fundus_1_$(date +%Y%m%d_%H%M%S).log

 python cnn_sam3_Fundus.py --data_root ${FMDG_DATA_ROOT}/Fundus/ --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_cnnlora_fu-domaim1,2,3,4_3000000/ --task_name sam3_cnnlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_cnnlora_Fundus_2_$(date +%Y%m%d_%H%M%S).log

 python cnn_sam3_NPC.py --data_root ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed/ --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR}  --checkpoint ${FMDG_WORK_DIR}naso_auto/sam_cnnlora_na-domaim1,2,3,4_3000000/ --task_name sam3_cnnlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_cnnlora_NPC_1_$(date +%Y%m%d_%H%M%S).log

 python cnn_sam3_NPC.py --data_root ${FMDG_DATA_ROOT}/MS_NPC_guran/preprocessed/ --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR}  --checkpoint ${FMDG_WORK_DIR}naso_auto/sam_cnnlora_na-domaim1,2,3,4_3000000/ --task_name sam3_cnnlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_cnnlora_NPC_2_$(date +%Y%m%d_%H%M%S).log

 python cnn_sam3_PolypDB.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/PolypDB/PolypDB_modality_wise/ --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_cnnlora_poly-domaim1,2,3,4_3000000/ --task_name sam3_cnnlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_cnnlora_Poly_1_$(date +%Y%m%d_%H%M%S).log

 python cnn_sam3_PolypDB.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/PolypDB/PolypDB_modality_wise/ --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR} --checkpoint ${FMDG_WORK_DIR}fundus_auto/sam_cnnlora_poly-domaim1,2,3,4_3000000/ --task_name sam3_cnnlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logs/sam_cnnlora_Poly_2_$(date +%Y%m%d_%H%M%S).log