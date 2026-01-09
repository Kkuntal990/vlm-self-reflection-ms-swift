#!/usr/bin/env bash
#
# Full Fine-Tuning Script for Qwen2.5-VL-7B on FIRE Dataset
#
# This script performs FULL parameter fine-tuning (not LoRA) on the
# FIRE behavior cloning dataset using 8 A100 GPUs.
#
# Hardware: 8 × A100 80GB (single node DDP)
# Model: Qwen/Qwen2.5-VL-7B-Instruct with full parameter updates
#
set -euo pipefail

# Source environment configuration
source /workspace/scripts/env.sh

# ============================================
# Model Configuration
# ============================================
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"

# ============================================
# Dataset Configuration
# ============================================
DATASET_PATH="${DATASET_PATH:-/outputs/fire_bc/fire_sharegpt_train.jsonl}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-/outputs/fire_bc/fire_sharegpt_test.jsonl}"

# ============================================
# Sequence and Vision Configuration
# ============================================
MAX_LEN="${MAX_LEN:-3584}"  # Optimized for FIRE dataset (samples ~2600 tokens)
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2048}"
export IMAGE_MAX_TOKEN_NUM

# Memory optimization for large models
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ============================================
# Training Hyperparameters
# ============================================
# Batch size per GPU (with MAX_LEN=3584, we can fit batch=4)
BATCH="${BATCH:-4}"
# Gradient accumulation to maintain effective batch size
GRAD_ACC="${GRAD_ACC:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOSS_SCALE="${LOSS_SCALE:-last_round}"
# ============================================
# DDP Configuration
# ============================================
NPROC="${NPROC:-8}"

# ============================================
# Output Configuration
# ============================================
RUN_NAME="${RUN_NAME:-qwen2_5vl-7b-instruct-fire-full-sft}"
OUTPUT_PATH="${OUTPUT_DIR}/${RUN_NAME}"

# ============================================
# Logging and Checkpointing
# ============================================
LOGGING_STEPS="${LOGGING_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# ============================================
# Auto-detect dtype
# ============================================
DTYPE="${DTYPE:-auto}"
if [ "$DTYPE" = "auto" ]; then
    echo "Auto-detecting optimal dtype..."
    DTYPE=$(python -c "import torch; print('bfloat16' if torch.cuda.is_bf16_supported() else 'float16')")
    echo "Selected dtype: $DTYPE"
fi

# ============================================
# Pre-flight Checks
# ============================================
echo "Checking GPU availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'Found {torch.cuda.device_count()} GPU(s)')"

# Verify dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Training dataset not found: $DATASET_PATH"
    echo "Please ensure the dataset file exists."
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_PATH}"

# ============================================
# Training Configuration Summary
# ============================================
echo "========================================="
echo "FIRE Full Fine-Tuning"
echo "Full Parameter Update (Not LoRA)"
echo "========================================="
echo ""
echo "Model Configuration:"
echo "  Model: ${MODEL_ID}"
echo "  Train Type: FULL"
echo "  Dtype: ${DTYPE}"
echo ""
echo "Dataset Configuration:"
echo "  Train: ${DATASET_PATH}"
echo "  Val: ${VAL_DATASET_PATH}"
echo ""
echo "Sequence Configuration:"
echo "  Max Length: ${MAX_LEN}"
echo "  Image Max Tokens: ${IMAGE_MAX_TOKEN_NUM}"
echo ""
echo "Training Configuration:"
echo "  GPUs: ${NPROC}"
echo "  Batch Size per GPU: ${BATCH}"
echo "  Gradient Accumulation: ${GRAD_ACC}"
echo "  Effective Batch Size: $((BATCH * GRAD_ACC * NPROC))"
echo "  Epochs: ${EPOCHS}"
echo "  Learning Rate: ${LR}"
echo "  Warmup Ratio: ${WARMUP_RATIO}"
echo "  Max Grad Norm: ${MAX_GRAD_NORM}"
echo ""
echo "Output: ${OUTPUT_PATH}"
echo "========================================="

# ============================================
# Launch Distributed Training
# ============================================
# ms-swift handles multi-GPU training internally via NPROC_PER_NODE

# Auto-generate CUDA_VISIBLE_DEVICES based on NPROC
CUDA_DEVICES=$(seq -s, 0 $((NPROC - 1)))

NPROC_PER_NODE="${NPROC}" \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
swift sft \
    --model "${MODEL_ID}" \
    --train_type full \
    --dataset "${DATASET_PATH}" \
    --split_dataset_ratio 0.1 \
    --max_length "${MAX_LEN}" \
    --loss_scale "${LOSS_SCALE}" \
    --per_device_train_batch_size "${BATCH}" \
    --per_device_eval_batch_size "${BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --torch_dtype "${DTYPE}" \
    --output_dir "${OUTPUT_PATH}" \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --use_hf true \
    --gradient_checkpointing false \
    --freeze_vit true \
    --freeze_aligner true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 16 \
    --report_to tensorboard \
    --save_only_model true \
    --deepspeed zero2 \
    --packing true \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 4 \
    --attn_impl flash_attn \
    --load_from_cache_file true


echo ""
echo "========================================="
echo "Training completed successfully!"
echo "========================================="
echo "Model saved to: ${OUTPUT_PATH}"
echo ""
echo "To use the model for inference:"
echo "  swift infer --model_dir ${OUTPUT_PATH}"
echo "========================================="
