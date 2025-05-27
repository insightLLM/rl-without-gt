#!/bin/bash
set -x

DIR=/opt
cd $DIR


#git clone https://github.com/insightLLM/rl-without-gt.git

# 安装 lib
cd $DIR/$project_name/rl/verl/lib
pip install -e .


pip uninstall ray -y
pip install ray

pip install cyac argparse

# 更新 torch 适配 cu124
pip uninstall -y torch torchvision torchaudio
pip install torch==2.4.0+cu124  torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


# 安装 verl
cd $DIR/$project_name/rl/verl/verl

pip install -e .


# 本地运行需要加上环境变量，如果在云服务上运行，由云服务注入
export MODEL_LOAD=""
export CHECKPOINT_SAVE=""



# huggingface 模型
bash ./scripts/eval/generation.sh 8 64




