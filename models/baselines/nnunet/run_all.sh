#!/bin/bash
# ============================================================
# 多模型并行实验启动脚本 (3x 1080Ti)
#
# 策略: 每个 GPU 串行跑自己分配到的实验, 4 个 GPU 并行
#       这样每张卡同时只跑 1 个实验, 不会 OOM
#
# 用法:
#   bash run_all.sh                      # 运行所有模型
#   bash run_all.sh unetpp               # 只运行 UNet++
#   bash run_all.sh attention_unet       # 只运行 Attention UNet
#   bash run_all.sh nnunet               # 只运行 nnU-Net
#
# 注意: 建议在 tmux/screen 中运行, 防止 SSH 断开导致进程终止
# ============================================================

MODEL=${1:-"all"}

LOG_DIR="./logsu"
mkdir -p "$LOG_DIR"

# ============================================================
# 数据集路径 (修改为服务器实际路径)
# ============================================================
DATA_FUNDUS="${FMDG_DATA_ROOT}/Fundus"
DATA_NPC="${FMDG_DATA_ROOT}/NPC"
DATA_POLY="${FMDG_DATA_ROOT}/PolypDB"

# ============================================================
# 实验网格定义
# ============================================================
# 格式: "dataset_name|data_root|trainid|testid|task_suffix|fold_name"
#
# Fundus:        2 folds (train 1,3->test 2,4 / train 2,4->test 1,3)
# NPC (na):      2 folds (train 1,4->test 2,3 / train 2,3->test 1,4)
# PolypDB (poly):2 folds (train 1,2->test 3,4 / train 3,4->test 1,2)

declare -a EXPERIMENTS=(
    # Fundus
    "Fundus|${DATA_FUNDUS}|1,3|2,4||fold1"
    "Fundus|${DATA_FUNDUS}|2,4|1,3||fold2"
    # NPC
    "NPC|${DATA_NPC}|1,4|2,3|-na|fold1"
    "NPC|${DATA_NPC}|2,3|1,4|-na|fold2"
    # PolypDB
    "PolypDB|${DATA_POLY}|1,2|3,4|-poly|fold1"
    "PolypDB|${DATA_POLY}|3,4|1,2|-poly|fold2"
)

# ============================================================
# 要运行的模型列表
# ============================================================
if [ "$MODEL" == "all" ]; then
    MODELS=("unetpp" "attention_unet" "nnunet" "transunet")
else
    MODELS=("$MODEL")
fi

# ---- 分离 TransUNet (多卡) 和其他模型 (单卡) ----
TRANSUNET_MODELS=()
OTHER_MODELS=()
for m in "${MODELS[@]}"; do
    if [ "$m" == "transunet" ]; then
        TRANSUNET_MODELS+=("$m")
    else
        OTHER_MODELS+=("$m")
    fi
done

TOTAL_GPU=3       # 3x 1080Ti

# ============================================================
# TransUNet 实验: 串行, DataParallel 使用全部 3 卡
# ============================================================
if [ ${#TRANSUNET_MODELS[@]} -gt 0 ]; then
    echo "============================================"
    echo "TransUNet: Sequential on all 3 GPUs (DataParallel)"
    echo "Experiments: ${#EXPERIMENTS[@]}"
    echo "============================================"
    echo ""

    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name data_root trainid testid suffix fold <<< "$exp"
        task_name="transunet-Baseline${suffix}"
        log_file="${LOG_DIR}/transunet_${name}_${fold}_$(date +%Y%m%d_%H%M%S).log"

        echo "============================================"
        echo "[TransUNet] Starting: ${name} | ${fold} | train=${trainid} test=${testid}"
        echo "  Log: ${log_file}"
        echo "============================================"
        CUDA_VISIBLE_DEVICES=1,2 python train.py \
            --model_type transunet \
            --data_root "${data_root}" \
            --batch_size 16 \
            --trainid "${trainid}" \
            --testid "${testid}" \
            --task_name "${task_name}" \
            --num_epochs 100 \
            --iternum 3000000 \
            --device cuda:0 \
            --multi_gpu 0,1,2 \
            --use_amp \
            2>&1 | tee "${log_file}"
        echo "[TransUNet] Completed: ${name} | ${fold}"
    done

    echo ""
    echo "TransUNet experiments completed."
    echo ""
fi

# ============================================================
# 其他模型: per-GPU 并行
# ============================================================
if [ ${#OTHER_MODELS[@]} -gt 0 ]; then

# ============================================================
# 为每个 GPU 生成独立的串行运行脚本
# ============================================================
GPU_SCRIPTS=()
for gpu in $(seq 0 $((TOTAL_GPU - 1))); do
    GPU_SCRIPT="${LOG_DIR}/_gpu_${gpu}_run.sh"
    echo "#!/bin/bash" > "$GPU_SCRIPT"
    echo "set -e" >> "$GPU_SCRIPT"
    chmod +x "$GPU_SCRIPT"
    GPU_SCRIPTS[$gpu]="$GPU_SCRIPT"
done

exp_idx=0
for model in "${OTHER_MODELS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name data_root trainid testid suffix fold <<< "$exp"
        gpu=$((exp_idx % TOTAL_GPU))
        task_name="${model}-Baseline${suffix}"
        # log 文件名里的时间戳用运行时展开
        log_file="${LOG_DIR}/${model}_${name}_${fold}_\$(date +%Y%m%d_%H%M%S).log"

        cat >> "${GPU_SCRIPTS[$gpu]}" << 'INNER_EOF'
echo "============================================"
echo "[GPU GPU_ID] Starting: MODEL | NAME | FOLD | train=TRAINID test=TESTID"
echo "  Log: LOG_FILE"
echo "============================================"
CUDA_VISIBLE_DEVICES=GPU_ID python train.py \
    --model_type MODEL \
    --data_root DATA_ROOT \
    --batch_size 16 \
    --trainid TRAINID \
    --testid TESTID \
    --task_name TASK_NAME \
    --num_epochs 100 \
    --iternum 3000000 \
    --device cuda:0 \
    2>&1 | tee LOG_FILE
echo "[GPU GPU_ID] Completed: MODEL | NAME | FOLD"
INNER_EOF

        # 用 sed 替换占位符 (避免 heredoc 变量提前展开)
        # TASK_NAME 必须在 NAME 之前, 否则 NAME 会先替换掉 TASK_NAME 里的 "NAME"
        sed -i "s/GPU_ID/${gpu}/g; s/MODEL/${model}/g; s|DATA_ROOT|${data_root}|g; s/TASK_NAME/${task_name}/g; s/NAME/${name}/g; s/TRAINID/${trainid}/g; s/TESTID/${testid}/g; s/FOLD/${fold}/g; s|LOG_FILE|${log_file}|g" "${GPU_SCRIPTS[$gpu]}"

        exp_idx=$((exp_idx + 1))
    done
done

# 给每个 GPU 脚本加完成标记
for gpu in $(seq 0 $((TOTAL_GPU - 1))); do
    echo "echo '[GPU ${gpu}] ALL DONE'" >> "${GPU_SCRIPTS[$gpu]}"
done

# ============================================================
# 统计信息
# ============================================================
echo "============================================"
echo "Model(s): ${OTHER_MODELS[*]}"
echo "Total experiments: $((${#OTHER_MODELS[@]} * ${#EXPERIMENTS[@]}))"
echo "GPUs available: ${TOTAL_GPU}"
echo "Strategy: per-GPU sequential, 3 GPUs parallel"
echo "============================================"
echo ""

# 显示每个 GPU 的任务分配
for gpu in $(seq 0 $((TOTAL_GPU - 1))); do
    exp_count=$(grep -c "Starting:" "${GPU_SCRIPTS[$gpu]}" 2>/dev/null || echo 0)
    echo "GPU ${gpu}: ${exp_count} experiments → ${GPU_SCRIPTS[$gpu]}"
done
echo ""

# ============================================================
# 启动: 每个 GPU 一个后台进程, 内部串行执行
# ============================================================
for gpu in $(seq 0 $((TOTAL_GPU - 1))); do
    bash "${GPU_SCRIPTS[$gpu]}" &
    echo "[Launched] GPU ${gpu} (PID $!)"
done

echo ""
echo "All GPU queues launched. Waiting for completion..."
echo "Run 'watch nvidia-smi' to monitor GPU usage."
wait
echo "All experiments completed."
fi  # end if OTHER_MODELS
