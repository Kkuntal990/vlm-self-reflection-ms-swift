#!/usr/bin/env bash
# Local environment configuration for development/testing
# Use this instead of env.sh when running on your local machine

# Local cache and output directories (not using PVCs)
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"
export TORCH_HOME="${HOME}/.cache/torch"
export PIP_CACHE_DIR="${HOME}/.cache/pip"

# Local output directory (create in project)
export OUTPUT_DIR="${PWD}/outputs"

# Create directories if they don't exist
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${TORCH_HOME}"

# DDP and NCCL settings for single-node multi-GPU training
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Distributed training configuration
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29501}

# Optional: HuggingFace token (if you have one)
# export HF_TOKEN="your_token_here"

echo "Local environment configured:"
echo "  HF_HOME: ${HF_HOME}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
