#!/bin/bash
# install_mamba_ssm.sh — Install mamba_ssm CUDA kernels on Neptune
# Run AFTER current training completes to avoid GPU conflicts during compilation
#
# Prerequisites verified:
#   - PyTorch 2.5.1+cu121
#   - CUDA 12.6 system (nvcc available)
#   - GCC 13.3
#   - Python 3.11
#
# Usage: ssh nick@neptune "bash /path/to/install_mamba_ssm.sh"

set -e

CONDA_ENV="/home/nick/miniconda3/envs/py311-train"
PIP="$CONDA_ENV/bin/pip"
PYTHON="$CONDA_ENV/bin/python"

echo "============================================"
echo "Installing mamba_ssm CUDA kernels"
echo "============================================"

# Set CUDA_HOME for compilation
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

echo "CUDA_HOME=$CUDA_HOME"
echo "Python=$($PYTHON --version)"
echo "PyTorch=$($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA=$($PYTHON -c 'import torch; print(torch.version.cuda)')"
echo ""

# Step 1: Install causal-conv1d (required dependency)
echo "[1/3] Installing causal-conv1d..."
$PIP install causal-conv1d>=1.4.0 2>&1 | tail -5

# Step 2: Install mamba-ssm
echo ""
echo "[2/3] Installing mamba-ssm..."
$PIP install mamba-ssm 2>&1 | tail -5

# Step 3: Verify installation
echo ""
echo "[3/3] Verifying installation..."
$PYTHON -c "
import mamba_ssm
print(f'mamba_ssm version: {mamba_ssm.__version__}')
from mamba_ssm import Mamba
import torch
# Quick smoke test
model = Mamba(d_model=64, d_state=32, d_conv=4, expand=2).cuda()
x = torch.randn(2, 100, 64).cuda()
y = model(x)
print(f'Input: {x.shape} -> Output: {y.shape}')
print('✓ mamba_ssm CUDA kernels working!')
"

echo ""
echo "============================================"
echo "✓ Installation complete!"
echo "============================================"
