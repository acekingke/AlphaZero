#!/bin/bash
# 创建并激活conda环境

# 环境名称
ENV_NAME="alphazero_env"

# 创建conda环境，使用Python 3.9
conda create -y -n $ENV_NAME python=3.9

# 激活环境
echo "正在激活环境 $ENV_NAME..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 安装依赖
echo "正在安装依赖..."
conda install -y pytorch torchvision torchaudio -c pytorch
conda install -y numpy matplotlib tqdm pillow

# 如果需要CUDA支持，可以使用下面的命令代替上面的PyTorch安装
# conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

echo "环境设置完成！使用以下命令激活环境："
echo "conda activate $ENV_NAME"