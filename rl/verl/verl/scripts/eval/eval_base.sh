set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# 第1个参数：每节点使用的 GPU 数量（默认为 8）
NGPUS_PER_NODE="${1:-8}"  # 如果未提供，则默认值为8


MODEL_PATH=$MODEL_LOAD


echo "MODEL_LOAD=$MODEL_LOAD"
echo "CHECKPOINT_SAVE=$CHECKPOINT_SAVE"

OUTPUT_DIR="${CHECKPOINT_SAVE}/eval"  # Add default output directory



# 从 MODEL_PATH 中提取模型名称和 global step
# 示例 MODEL_PATH 格式：.../rl/<model_name>/<global_step>
global_step=$(basename "$MODEL_PATH")
model_name=$(basename "$(dirname "$MODEL_PATH")")


# 拼接得到 output_path
output_path="${OUTPUT_DIR}/${model_name}/${global_step}/"



echo "输出路径: $output_path"

#rm -rf $output_path

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${val_data}"
echo "Output Directory: ${output_path}"

# data.batch_size 和 tensor_model_parallel_size 必须和边训练边评测时保持一致，否则评测结果会不一样

python3 -m verl.trainer.main_eval_xrh \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    data.val_files=$val_data \
    data.output_path=$output_path \
    data.batch_size=128 \
    model.path=$MODEL_PATH \
    rollout.temperature=0.0 \
    rollout.prompt_length=1024 \
    rollout.response_length=3072 \
    rollout.gpu_memory_utilization=0.7 \
    rollout.tensor_model_parallel_size=2

