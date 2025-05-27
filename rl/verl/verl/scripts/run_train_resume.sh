#!/bin/bash
set -x

DIR=/opt
cd $DIR


#git clone https://github.com/insightLLM/rl-without-gt.git

# 指定项目名称
project_name="rl-without-gt"

# 安装 math-evaluation
cd $DIR/$project_name/rl/verl/lib/src/eval/MARIO_EVAL
pip install -e .

# 安装 lib
cd $DIR/$project_name/rl/verl/lib
pip install -e .


pip install antlr4-python3-runtime==4.9.3

# 更新 torch 适配 cu124
pip uninstall -y torch torchvision torchaudio
pip install torch==2.4.0+cu124  torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


cd $DIR/$project_name/rl/verl/verl

pip install wandb aiohttp pylatexenc cyac argparse


# 需要安装 ray[default]==2.10 否则无法提交任务
pip uninstall ray -y
pip install ray[default]==2.10


# 安装 verl
pip install -e .

# 训练数据和测试数据的路径

export train_data="/opt/rl-without-gt/rl/verl/lib/data/qwen_template/math_train.parquet"
#export train_data="/opt/without-GT-RL/rl/verl/lib/data/qwen_template/deepscaler_17k.parquet"

export val_data='[/opt/rl-without-gt/rl/verl/lib/data/qwen_template/aime2024.parquet,/opt/rl-without-gt/rl/verl/lib/data/qwen_template/amc2023.parquet,/opt/rl-without-gt/rl/verl/lib/data/qwen_template/math500.parquet]'


# 本地运行需要加上环境变量，如果在云服务上运行，由云服务注入
export MASTER_IP=$(hostname -I | awk '{print $1}') # 如果只有一个节点
export MASTER_PORT=6379

export MODEL_LOAD="/global_data/pretrain/rihui/model_zoo/Qwen2.5-Math-7B"
export CHECKPOINT_LOAD="/opt/checkpoints"
export CHECKPOINT_SAVE="/opt/checkpoints"

# <nnodes> <save_freq> <test_freq> <total_training_steps> <global_step_id>  <logger_resume_id> <val_before_train: true|false>
bash ./scripts/train/run_grpo_math_7b_resume.sh 1 20 5 60 50 oex5ovi7 false


