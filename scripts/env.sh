#!/usr/bin/env bash

# Cache PVC (rook-ceph-block) - mounted at /cache
export HF_HOME=/cache/hf
export HF_HUB_CACHE=/cache/hf/hub
export TRANSFORMERS_CACHE=/cache/hf/transformers
export HF_DATASETS_CACHE=/cache/hf/datasets
export TORCH_HOME=/cache/torch
export PIP_CACHE_DIR=/cache/pip

# Outputs PVC (rook-cephfs) - mounted at /outputs
export OUTPUT_DIR=/outputs

# Make cached runs deterministic (enable once model is fully cached)
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# DDP / NCCL
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29501}

# Debug toggles (enable when needed)
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
