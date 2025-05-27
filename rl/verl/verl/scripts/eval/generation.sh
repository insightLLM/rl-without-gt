set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# 第1个参数：每节点使用的 GPU 数量（默认为 8）
NGPUS_PER_NODE="${1:-8}"  # 如果未提供，则默认值为8

# 采样的参数个数
n_samples=$2

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
echo "Output Directory: ${output_path}"


# Possible values: aime, amc, math, minerva, olympiad_bench

DATATYPES=(
'aime2024'
'math500'
'amc2023'
)



# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=$NGPUS_PER_NODE \
        data.path= {user-defined-data-path}/${DATA_TYPE}.parquet \
        data.output_path=${output_path}/${DATA_TYPE}.parquet \
        data.n_samples=$n_samples \
        data.batch_size=1024 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.prompt_length=1024 \
        rollout.response_length=3072 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.7 \
        rollout.tensor_model_parallel_size=2
done






