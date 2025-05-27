set -x
master_ip=$(python3 ip_utils/get_domain_ip.py $MASTER_ADDR)
curr_ip=$(python ip_utils/get_host_ip.py)
echo $master_ip $curr_ip

# Train over a single node, 8 A100-80GB GPUs.
set -x
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORK_DIR=`dirname $(dirname $SCRIPT_DIR)`

cd $WORK_DIR
echo "workspace_dir=$PWD"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
master_ip=$(python3 ip_utils/get_domain_ip.py $MASTER_ADDR)
curr_ip=$(python $WORK_DIR/ip_utils/get_host_ip.py)
echo $master_ip $curr_ip

if [ "$master_ip" = "$curr_ip" ]; then
    export VLLM_ATTENTION_BACKEND=XFORMERS
    echo "run ray!!!!!!!!!!!!!!!!!"
    ray start --head --num-gpus 8
    sleep 50s
    echo "start job!!!!!!!!!!!!!!!!!"
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/global_data/pretrain/zhouhuozhi/data/math/train_verl.parquet \
    data.val_files=/global_data/pretrain/zhouhuozhi/data/math/test_verl.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=/global_data/pretrain/zhouhuozhi/models/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_one_node' \
    trainer.experiment_name='grpo_one_node_test' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.default_local_dir=/checkpoint_save \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    echo 'job done, now shutdown ray cluster'
    ray stop --force
else
    export VLLM_ATTENTION_BACKEND=XFORMERS
    sleep 20s
    ray start --address $master_ip:6379 --block 
fi
