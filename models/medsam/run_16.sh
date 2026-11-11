#使用DPT解码器sam

python sam_DPT_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPT_fu-domaim1,2,3,4_300000/ --task_name sam_DPT_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_DPT_Fundus_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPT_fu-domaim1,2,3,4_300000/ --task_name sam_DPT_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_DPT_Fundus_2_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPT_na-domaim1,2,3,4_300000/ --task_name sam_DPT_na --num_epochs 50 --iternum 3000000 --device cuda:0 --lr 0.0001 2>&1 | tee ./logsss/sam_DPT_NPC_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPT_na-domaim1,2,3,4_300000/ --task_name sam_DPT_na --num_epochs 50 --iternum 3000000 --device cuda:0 --lr 0.0001 2>&1 | tee ./logsss/sam_DPT_NPC_2_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_poly_16.py --data_root ${FMDG_DATA_ROOT}/PolypDB --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPT_poly-domaim1,2,3,4_300000/ --task_name sam_DPT_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_DPT_Poly_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_poly_16.py --data_root ${FMDG_DATA_ROOT}/PolypDB --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPT_poly-domaim1,2,3,4_300000/ --task_name sam_DPT_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_DPT_Poly_2_$(date +%Y%m%d_%H%M%S).log


#使用DPT+lora;模型sam

python sam_DPT_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPTlora_fu-domaim1,2,3,4_300000/ --task_name sam_DPTlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_DPTlora_Fundus_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPTlora_fu-domaim1,2,3,4_300000/ --task_name sam_DPTlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_DPTlora_Fundus_2_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPTlora_na-domaim1,2,3,4_300000/ --task_name sam_DPTlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_DPTlora_NPC_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPTlora_na-domaim1,2,3,4_300000/ --task_name sam_DPTlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_DPTlora_NPC_2_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_poly_16.py --data_root ${FMDG_DATA_ROOT}/PolypDB --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPTlora_poly-domaim1,2,3,4_300000/ --task_name sam_DPTlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_DPTlora_Poly_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_poly_16.py --data_root ${FMDG_DATA_ROOT}/PolypDB --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_DPTlora_poly-domaim1,2,3,4_300000/ --task_name sam_DPTlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_DPTlora_Poly_2_$(date +%Y%m%d_%H%M%S).log


#使用DPT解码器medsam

python sam_DPT_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPT_fu-domaim1,2,3,4_300000/ --task_name medsam_DPT_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPT_Fundus_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPT_fu-domaim1,2,3,4_300000/ --task_name medsam_DPT_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPT_Fundus_2_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPT_na-domaim1,2,3,4_300000/ --task_name medsam_DPT_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPT_NPC_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPT_na-domaim1,2,3,4_300000/ --task_name medsam_DPT_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPT_NPC_2_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_poly_16.py --data_root ${FMDG_DATA_ROOT}/PolypDB --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPT_poly-domaim1,2,3,4_300000/ --task_name medsam_DPT_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPT_Poly_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_poly_16.py --data_root ${FMDG_DATA_ROOT}/PolypDB --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPT_poly-domaim1,2,3,4_300000/ --task_name medsam_DPT_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPT_Poly_2_$(date +%Y%m%d_%H%M%S).log


# #使用DPT+lora;模型medsam

python sam_DPT_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPTlora_fu-domaim1,2,3,4_300000/ --task_name medsam_DPTlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPTlora_Fundus_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPTlora_fu-domaim1,2,3,4_300000/ --task_name medsam_DPTlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPTlora_Fundus_2_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPTlora_na-domaim1,2,3,4_300000/ --task_name medsam_DPTlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPTlora_NPC_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPTlora_na-domaim1,2,3,4_300000/ --task_name medsam_DPTlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPTlora_NPC_2_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_poly_16.py --data_root ${FMDG_DATA_ROOT}/PolypDB --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPTlora_poly-domaim1,2,3,4_300000/ --task_name medsam_DPTlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPTlora_Poly_1_$(date +%Y%m%d_%H%M%S).log

python sam_DPT_poly_16.py --data_root ${FMDG_DATA_ROOT}/PolypDB --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_DPTlora_poly-domaim1,2,3,4_300000/ --task_name medsam_DPTlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_DPTlora_Poly_2_$(date +%Y%m%d_%H%M%S).log



#cnn;sam

 python cnn_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_cnn_fu-domaim1,2,3,4_3000000/ --task_name sam_cnn_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_cnn_Fundus_1_$(date +%Y%m%d_%H%M%S).log

 python cnn_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_cnn_fu-domaim1,2,3,4_3000000/ --task_name sam_cnn_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_cnn_Fundus_2_$(date +%Y%m%d_%H%M%S).log

 python cnn_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR}/  --checkpoint ${FMDG_WORK_DIR}/naso_auto/sam_cnn_na-domaim1,2,3,4_3000000/ --task_name sam_cnn_na --num_epochs 50 --iternum 3000000 --device cuda:0  2>&1 | tee ./logsss/sam_cnn_NPC_1_$(date +%Y%m%d_%H%M%S).log

 python cnn_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR}/  --checkpoint ${FMDG_WORK_DIR}/naso_auto/sam_cnn_na-domaim1,2,3,4_3000000/ --task_name sam_cnn_na --num_epochs 50 --iternum 3000000 --device cuda:0  2>&1 | tee ./logsss/sam_cnn_NPC_2_$(date +%Y%m%d_%H%M%S).log

 python cnn_poly_16.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/ --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_cnn_poly-domaim1,2,3,4_3000000/ --task_name sam_cnn_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_cnn_Poly_1_$(date +%Y%m%d_%H%M%S).log

 python cnn_poly_16.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/ --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_cnn_poly-domaim1,2,3,4_3000000/ --task_name sam_cnn_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_cnn_Poly_2_$(date +%Y%m%d_%H%M%S).log

##baseline
python UNet.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 1,3 --testid 2,4   --task_name UNet-Baseline --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/UNET_Fundus_1_$(date +%Y%m%d_%H%M%S).log

python UNet.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 2,4 --testid 1,3   --task_name UNet-Baseline --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/UNET_Fundus_2_$(date +%Y%m%d_%H%M%S).log

python UNet.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 1,4 --testid 2,3   --task_name UNet-Baseline-na --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/UNET_NPC_1_$(date +%Y%m%d_%H%M%S).log

python UNet.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 2,3 --testid 1,4   --task_name UNet-Baseline-na --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/UNET_NPC_2_$(date +%Y%m%d_%H%M%S).log

python UNet.py --data_root ${FMDG_DATA_ROOT}/PolypDB --batch_size 1 --trainid 1,2 --testid 3,4   --task_name UNet-Baseline-poly --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/UNET_POLYPDB_1_$(date +%Y%m%d_%H%M%S).log

python UNet.py --data_root ${FMDG_DATA_ROOT}/PolypDB --batch_size 1 --trainid 3,4 --testid 1,2   --task_name UNet-Baseline-poly --num_epochs 100 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/UNET_POLYPDB_2_$(date +%Y%m%d_%H%M%S).log


# ##cnn+lora;sam

 python cnn_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_cnnlora_fu-domaim1,2,3,4_3000000/ --task_name sam_cnnlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_cnnlora_Fundus_1_$(date +%Y%m%d_%H%M%S).log

 python cnn_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_cnnlora_fu-domaim1,2,3,4_3000000/ --task_name sam_cnnlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_cnnlora_Fundus_2_$(date +%Y%m%d_%H%M%S).log

 python cnn_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR}/  --checkpoint ${FMDG_WORK_DIR}/naso_auto/sam_cnnlora_na-domaim1,2,3,4_3000000/ --task_name sam_cnnlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_cnnlora_NPC_1_$(date +%Y%m%d_%H%M%S).log

 python cnn_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR}/  --checkpoint ${FMDG_WORK_DIR}/naso_auto/sam_cnnlora_na-domaim1,2,3,4_3000000/ --task_name sam_cnnlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_cnnlora_NPC_2_$(date +%Y%m%d_%H%M%S).log

 python cnn_poly_16.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/ --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_cnnlora_poly-domaim1,2,3,4_3000000/ --task_name sam_cnnlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_cnnlora_Poly_1_$(date +%Y%m%d_%H%M%S).log

 python cnn_poly_16.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/ --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/sam_cnnlora_poly-domaim1,2,3,4_3000000/ --task_name sam_cnnlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/sam_cnnlora_Poly_2_$(date +%Y%m%d_%H%M%S).log



# ##cnn;medsam

python cnn_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_cnn_fu-domaim1,2,3,4_3000000/ --task_name medsam_cnn_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnn_Fundus_1_$(date +%Y%m%d_%H%M%S).log

python cnn_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_cnn_fu-domaim1,2,3,4_3000000/ --task_name medsam_cnn_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnn_Fundus_2_$(date +%Y%m%d_%H%M%S).log

python cnn_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR}/  --checkpoint ${FMDG_WORK_DIR}/naso_auto/medsam_cnn_na-domaim1,2,3,4_3000000/ --task_name medsam_cnn_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnn_NPC_1_$(date +%Y%m%d_%H%M%S).log

python cnn_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR}/  --checkpoint ${FMDG_WORK_DIR}/naso_auto/medsam_cnn_na-domaim1,2,3,4_3000000/ --task_name medsam_cnn_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnn_NPC_2_$(date +%Y%m%d_%H%M%S).log

python cnn_poly_16.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/ --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_cnn_poly-domaim1,2,3,4_3000000/ --task_name medsam_cnn_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnn_Poly_1_$(date +%Y%m%d_%H%M%S).log

python cnn_poly_16.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/ --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_cnn_poly-domaim1,2,3,4_3000000/ --task_name medsam_cnn_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnn_Poly_2_$(date +%Y%m%d_%H%M%S).log


# ##cnn+lora;medsam

python cnn_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 1,3 --testid 2,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_cnnlora_fu-domaim1,2,3,4_3000000/ --task_name medsam_cnnlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnnlora_Fundus_1_$(date +%Y%m%d_%H%M%S).log

python cnn_fundus_16.py --data_root ${FMDG_DATA_ROOT}/Fundus --batch_size 1 --trainid 2,4 --testid 1,3 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_cnnlora_fu-domaim1,2,3,4_3000000/ --task_name medsam_cnnlora_fu --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnnlora_Fundus_2_$(date +%Y%m%d_%H%M%S).log

python cnn_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 1,4 --testid 2,3 --work_dir ${FMDG_WORK_DIR}/  --checkpoint ${FMDG_WORK_DIR}/naso_auto/medsam_cnnlora_na-domaim1,2,3,4_3000000/ --task_name medsam_cnnlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnnlora_NPC_1_$(date +%Y%m%d_%H%M%S).log

python cnn_npc_16.py --data_root ${FMDG_DATA_ROOT}/NPC --batch_size 1 --trainid 2,3 --testid 1,4 --work_dir ${FMDG_WORK_DIR}/  --checkpoint ${FMDG_WORK_DIR}/naso_auto/medsam_cnnlora_na-domaim1,2,3,4_3000000/ --task_name medsam_cnnlora_na --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnnlora_NPC_2_$(date +%Y%m%d_%H%M%S).log

python cnn_poly_16.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/ --batch_size 1 --trainid 1,2 --testid 3,4 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_cnnlora_poly-domaim1,2,3,4_3000000/ --task_name medsam_cnnlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnnlora_Poly_1_$(date +%Y%m%d_%H%M%S).log

python cnn_poly_16.py  --data_root ${FMDG_DATA_ROOT}/PolypDB/ --batch_size 1 --trainid 3,4 --testid 1,2 --work_dir ${FMDG_WORK_DIR}/ --checkpoint ${FMDG_WORK_DIR}/fundus_auto/medsam_cnnlora_poly-domaim1,2,3,4_3000000/ --task_name medsam_cnnlora_poly --num_epochs 50 --iternum 3000000 --device cuda:0 2>&1 | tee ./logsss/medsam_cnnlora_Poly_2_$(date +%Y%m%d_%H%M%S).log
