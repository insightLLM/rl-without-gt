#!/bin/bash

set -x


if [ "$#" -lt 7 ]; then
  echo "Usage: $0 <nnodes> <save_freq> <test_freq> <total_training_steps> <global_step_id> <logger_resume_id> <val_before_train: true|false>"
  exit 1
fi


nnodes=$1
save_freq=$2
test_freq=$3
total_training_steps=$4
global_step_id=$5
logger_resume_id=$6
val_before_train=$7



echo "Number of nodes: ${nnodes}, save checkpoint freq: ${save_freq}"


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORK_DIR=`dirname $(dirname $SCRIPT_DIR)`
cd $WORK_DIR
echo "workspace_dir=$PWD"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# 仅当 MASTER_IP 环境变量为空时，执行脚本获取 IP
if [ -z "$MASTER_IP" ]; then
    MASTER_IP=$(python3 ip_utils/get_domain_ip.py $MASTER_ADDR)
fi

curr_ip=$(python $WORK_DIR/ip_utils/get_host_ip.py)
echo $MASTER_IP $curr_ip



MODEL_PATH=$MODEL_LOAD

project_name="rl"
experiment_name="grpo-math-7b_math-bs_128-3k-zs_qwen-r_a_g_f"

echo "checkpoint_dir=$CHECKPOINT_SAVE"

resume_from_path="${CHECKPOINT_LOAD}/global_step_${global_step_id}"

echo "resume_from_path=$resume_from_path"

# cluster
train_data="{user-defined-data-path}/math_train.parquet"

#train_data="{user-defined-data-path}/deepscaler_17k.parquet"

val_data='[ {user-defined-data-path}/aime2024.parquet, {user-defined-data-path}/amc2023.parquet, {user-defined-data-path}/math500.parquet]'


output_path="${CHECKPOINT_SAVE}/outputs/eval/${experiment_name}/"

timestamp=$(date +%Y%m%d_%H%M%S)

log_path="${CHECKPOINT_SAVE}/log_${experiment_name}_${timestamp}.out"


MASTER_ADDR=$MASTER_IP

if [ "$MASTER_IP" = "$curr_ip" ]; then
    export VLLM_ATTENTION_BACKEND=XFORMERS

    echo "[主节点]清理残留的 ray, 避免端口被占用"
    ray stop --force
    pkill -f "ray"

    echo "开始启动 ray 主节点.................."

    ray start --head \
        --port=$MASTER_PORT \
        --min-worker-port=20000 \
        --max-worker-port=21000 \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

    # 等待节点连接（硬编码预期节点数，可根据需要改为变量）

    TIMEOUT=600  # 10分钟超时
    EXPECTED_NODES=$nnodes
    echo "等待至少 ${EXPECTED_NODES} 个节点连接..."

    # 阶段1：等待集群响应
    echo "等待Ray集群响应..."
    if ! timeout $TIMEOUT bash -c "until ray status --address=$MASTER_ADDR:$MASTER_PORT >/dev/null 2>&1; do sleep 5; done"; then
        echo "❌ 错误：Ray集群未在 ${TIMEOUT}秒 内响应"
        ray stop --force
        exit 1
    fi

    # 阶段2：等待节点数达标（修复版）
    echo "等待节点连接（预期: $EXPECTED_NODES，超时: ${TIMEOUT}秒）..."
    if ! timeout $TIMEOUT bash -c "\
        while :; do \
            alive_nodes=\$(ray status 2>/dev/null | awk '/Active:/{f=1;c=0} /Pending:/{f=0} f&&/node_/{c++} END{print c+0}'); \
            echo \"当前节点数: \$alive_nodes/$EXPECTED_NODES\"; \
            [ \$alive_nodes -ge $EXPECTED_NODES ] && break; \
            sleep 5; \
        done"; then
        echo "❌ 超时：未在 ${TIMEOUT}秒 内达到预期节点数"
        ray status
        ray stop --force
        exit 1
    fi

    sleep 20s

    echo " [集群就绪] 提交Ray任务"

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{"working_dir": "'$WORK_DIR'"}'   \
        -- python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        reward_model.reward_manager=xrh \
        data.train_files=$train_data \
        data.val_files=$val_data \
        data.output_path=$output_path \
        data.train_batch_size=128 \
        data.max_prompt_length=1024 \
        data.max_response_length=3072 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=$project_name \
        trainer.experiment_name=$experiment_name \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$nnodes \
        trainer.save_freq=$save_freq \
        trainer.default_local_dir=$CHECKPOINT_SAVE \
        trainer.test_freq=$test_freq \
        trainer.total_training_steps=$total_training_steps \
        trainer.resume_mode=$resume_from_path \
        trainer.val_before_train=$val_before_train \
        trainer.logger_resume_id=$logger_resume_id \
        trainer.checkpoint_condition.enable=True \
        trainer.checkpoint_condition.MATH500_sota=0.75 \
        trainer.checkpoint_condition.AIME2024_sota=0.3 \
        trainer.checkpoint_condition.AMC2023_sota=0.6 \
        trainer.checkpoint_condition.MATH500=0.7 \
        trainer.checkpoint_condition.AIME2024=0.2 \
        trainer.checkpoint_condition.AMC2023=0.5 2>&1 | tee $log_path

    echo 'job done, now shutdown ray cluster'
    ray stop --force
else
    export VLLM_ATTENTION_BACKEND=XFORMERS

    echo "[从节点]清理残留的 ray, 避免端口被占用"
    ray stop --force
    pkill -f "ray"

    echo "开始启动 ray 从节点.................."
    sleep 20s

    echo "等待 ray 主节点启动 ${MASTER_IP}:6379..."

    until ray status --address="${MASTER_IP}:${MASTER_PORT}" &> /dev/null; do
        sleep 5
    done

    echo "ray 主节点已启动，启动 ray 从节点"

    ray start --address $MASTER_IP:$MASTER_PORT --block
fi


