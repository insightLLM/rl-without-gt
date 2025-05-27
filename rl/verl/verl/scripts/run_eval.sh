#!/bin/bash
set -x

DIR=/opt
cd $DIR


#git clone https://github.com/insightLLM/rl-without-gt.git

# 指定项目名称
project_name="rl-without-gt"

# 安装 lib
cd $DIR/$project_name/rl/verl/lib
pip install -e .


# 不要装 ray==2.10 ，否则会报错(GPU占用冲突)
pip uninstall ray -y
pip install ray

pip install cyac argparse

# 更新 torch 适配 cu124
pip uninstall -y torch torchvision torchaudio
pip install torch==2.4.0+cu124  torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


# 安装 verl
cd $DIR/$project_name/rl/verl/verl

pip install -e .


# 测试数据的路径
export val_data='[/opt/rl-without-gt/rl/verl/lib/data/qwen_template/aime2024.parquet,/opt/rl-without-gt/rl/verl/lib/data/qwen_template/amc2023.parquet,/opt/rl-without-gt/rl/verl/lib/data/qwen_template/math500.parquet]'


# 本地运行需要配置如下环境变量，如果在云服务上运行，由云服务注入
export MODEL_LOAD="/global_data/pretrain/rihui/model_zoo/Qwen2.5-Math-7B"
export CHECKPOINT_LOAD="/opt/checkpoints"
export CHECKPOINT_SAVE="/opt/checkpoints"

# 参数为huggingface 格式的模型
bash ./scripts/eval/eval_base.sh 8

# 训练生成的模型，指定某些 step
#bash ./scripts/eval/eval_all.sh 90,100 8


# 训练生成的模型，评测所有已有的 step
#bash ./scripts/eval/eval_all.sh --all

# 训练生成的模型，评测 step 区间（闭区间）
#bash ./scripts/eval/eval_all.sh 80-100


