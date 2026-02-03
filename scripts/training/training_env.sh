#!/usr/bin/env bash
#
# Training Environment Configuration
#
# This file sets all environment variables required for full fine-tuning.
# Source this file before running the training script, or set variables
# via K8s job spec to override these defaults.
#
# Usage:
#   source scripts/training/training_env.sh
#   scripts/training/run_full_sft_qwen3vl_fire_8gpu.sh
#

# ============================================
# Cache and Storage Configuration
# ============================================
# Cache PVC (rook-ceph-block) - mounted at /cache
export HF_HOME="${HF_HOME:-/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/cache/hf/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/cache/hf/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/cache/hf/datasets}"
export TORCH_HOME="${TORCH_HOME:-/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/cache/pip}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/cache/modelscope}"

# Outputs PVC (rook-cephfs) - mounted at /outputs
export OUTPUT_DIR="${OUTPUT_DIR:-/outputs}"

# HuggingFace settings
export HF_HUB_DISABLE_TELEMETRY=1
# export TRANSFORMERS_OFFLINE=1  # Enable once model is fully cached
# export HF_DATASETS_OFFLINE=1

# ============================================
# Model Configuration
# ============================================
export MODEL_ID="${MODEL_ID:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"

# ============================================
# Dataset Configuration
# ============================================
export DATASET_PATH="${DATASET_PATH:-/outputs/fire_preprocessed_v3/fire_feedback_train.jsonl /outputs/fire_preprocessed_v3/fire_messages_train.jsonl}"
export VAL_DATASET_PATH="${VAL_DATASET_PATH:-}"

# ============================================
# Sequence and Vision Configuration
# ============================================
export MAX_LEN="${MAX_LEN:-9856}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2048}"

# ============================================
# Training Hyperparameters
# ============================================
export BATCH="${BATCH:-2}"              # Per-GPU batch size
export GRAD_ACC="${GRAD_ACC:-16}"       # Gradient accumulation steps
export EPOCHS="${EPOCHS:-2}"            # Number of training epochs
export LR="${LR:-3e-6}"                 # Learning rate (lower for full fine-tuning)
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export LOSS_SCALE="${LOSS_SCALE:-default}"

# ============================================
# DDP / Distributed Configuration
# ============================================
export NPROC="${NPROC:-4}"              # Number of GPUs
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"  # 30 min timeout

# Debug toggles (enable when needed)
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ============================================
# Output and Checkpoint Configuration
# ============================================
export RUN_NAME="${RUN_NAME:-llava-ov-feedback-training-data2}"
export RESUME_PATH="${RESUME_PATH:-}"   # Set to checkpoint path to resume

# ============================================
# Logging and Checkpointing
# ============================================
export LOGGING_STEPS="${LOGGING_STEPS:-5}"
export EVAL_STEPS="${EVAL_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# ============================================
# Python and CUDA Settings
# ============================================
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Optional: Set dtype (auto-detected if not set)
# export DTYPE="bfloat16"

# ============================================
# Summary
# ============================================
echo "Training Environment Configured:"
echo "  Model: ${MODEL_ID}"
echo "  Dataset: ${DATASET_PATH}"
echo "  GPUs: ${NPROC}"
echo "  Batch: ${BATCH} x ${GRAD_ACC} (grad acc) x ${NPROC} (GPUs) = $((BATCH * GRAD_ACC * NPROC)) effective"
echo "  Max Length: ${MAX_LEN}"
echo "  Learning Rate: ${LR}"
echo "  Epochs: ${EPOCHS}"
echo "  Output: ${OUTPUT_DIR}/${RUN_NAME}"
