#!/usr/bin/env bash
# Setup script for local conda environment
set -euo pipefail

ENV_NAME="vlm-self-reflection-swift"

echo "========================================="
echo "Setting up conda environment: ${ENV_NAME}"
echo "========================================="

# Activate the conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "ERROR: Failed to activate environment ${ENV_NAME}"
    exit 1
fi

echo "Environment activated: $CONDA_DEFAULT_ENV"

# Install PyTorch with CUDA support (adjust CUDA version as needed)
echo -e "\nInstalling PyTorch with CUDA 12.1 support..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
echo -e "\nInstalling project requirements..."
pip install -r requirements.txt

# Verify installations
echo -e "\n========================================="
echo "Verifying installations..."
echo "========================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" || echo "CUDA not available"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" || echo "No GPUs detected"

# Check ms-swift installation
python -c "import swift; print(f'ms-swift installed: OK')" || echo "WARNING: ms-swift not installed"

# Check transformers version
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"

echo -e "\n========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To use this environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test locally (requires GPU):"
echo "  bash scripts/run_sft_ddp.sh"
echo ""
