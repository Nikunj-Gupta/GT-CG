#!/bin/bash
set -euo pipefail

# Clean working PyMARL/PyTorch environment
# conda create -n gtcg python=3.10 -y
# conda activate gtcg

python -m pip install --upgrade pip

# Install PyTorch (stable, no Intel oneDNN issues)
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install SMACv2
pip install git+https://github.com/oxwhirl/smacv2.git

# Install SMAC
pip install git+https://github.com/oxwhirl/smac.git

# Install PettingZoo MARL benchmark (with MAgent extras for smoke tests)
pip install "pettingzoo"

# Install PyTorch Geometric (matching cu121 build)
pip install --force-reinstall pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install --force-reinstall torch_geometric

# Install other dependencies
pip install pymongo setproctitle sacred pyyaml tensorboard_logger matplotlib wandb 
