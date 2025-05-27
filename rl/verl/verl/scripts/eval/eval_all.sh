#!/bin/bash
#set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

INPUT="$1"

# 第二个参数：每节点使用的 GPU 数量（默认为 8）
NGPUS_PER_NODE="${2:-8}"  # 如果未提供，则默认值为8

MODEL_PATH=$MODEL_LOAD

echo "CHECKPOINT_LOAD=$CHECKPOINT_LOAD"
echo "CHECKPOINT_SAVE=$CHECKPOINT_SAVE"
echo "MODEL_PATH=$MODEL_PATH"

OUTPUT_DIR="${CHECKPOINT_SAVE}/eval"


# 获取待评测的 step 列表
if [[ "$INPUT" == "--all" ]]; then
    echo "🚀 Running inference on all checkpoints..."
    step_dirs=($(ls -d ${CHECKPOINT_LOAD}/global_step_* 2>/dev/null | sort -t '_' -k3 -n))
    STEP_ARRAY=()
    for dir in "${step_dirs[@]}"; do
        step=$(basename "$dir" | awk -F'_' '{print $3}')
        STEP_ARRAY+=("$step")
    done
elif [[ "$INPUT" =~ ^[0-9]+-[0-9]+$ ]]; then
    echo "📈 Running range: $INPUT"
    IFS='-' read -r start end <<< "$INPUT"
    STEP_ARRAY=()
    for ((i=start; i<=end; i++)); do
        STEP_ARRAY+=("$i")
    done
else
    echo "🔢 Running selected steps: $INPUT"
    IFS=',' read -ra STEP_ARRAY <<< "$INPUT"
fi

# 遍历 STEP_ARRAY
for step in "${STEP_ARRAY[@]}"; do
    ckpt_path="${CHECKPOINT_LOAD}/global_step_${step}"

    if [ -d "$ckpt_path" ]; then
        output_path="${OUTPUT_DIR}/global_step_${step}"
        echo "当前模型路径: $ckpt_path"
        echo "输出路径: $output_path"
        # data.batch_size 和 tensor_model_parallel_size 必须和边训练边评测时保持一致，否则评测结果会不一样

        python3 -m verl.trainer.main_eval_xrh \
            trainer.nnodes=1 \
            trainer.n_gpus_per_node=$NGPUS_PER_NODE \
            data.val_files=$val_data \
            data.output_path=$output_path \
            data.batch_size=128 \
            model.path=$MODEL_PATH \
            model.ckpt_path=$ckpt_path \
            rollout.temperature=0.0 \
            rollout.prompt_length=1024 \
            rollout.response_length=3072 \
            rollout.gpu_memory_utilization=0.7 \
            rollout.tensor_model_parallel_size=2
    else
        echo "[⚠️ 警告] global_step_${step} 不存在，跳过"
    fi
done
