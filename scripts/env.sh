#!/usr/bin/env bash
# Environment configuration for ms-swift training
# This script sets up cache and output directories optimized for dual-PVC setup

# Cache PVC (rook-ceph-block) - mounted at /cache
# Optimized for many small files (HuggingFace models, datasets, pip cache)
export HF_HOME=/cache/hf
export TRANSFORMERS_CACHE=/cache/hf/transformers
export HF_DATASETS_CACHE=/cache/hf/datasets
export TORCH_HOME=/cache/torch
export PIP_CACHE_DIR=/cache/pip

# Outputs PVC (rook-cephfs) - mounted at /outputs
# Optimized for large files (checkpoints, model weights)
export OUTPUT_DIR=/outputs

# DDP and NCCL settings for single-node multi-GPU training
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Distributed training configuration
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29501}

# Optional: Uncomment for additional debugging
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
