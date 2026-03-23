#!/usr/bin/env bash
#
# LoRA SFT Script for Qwen2.5-VL-7B
#
# Parameter-efficient fine-tuning using LoRA on the LLM component only.
# Vision encoder and aligner are frozen.
#
# Hardware: 4 × A100 80GB (single node DDP)
# Model: Qwen/Qwen2.5-VL-7B-Instruct with LoRA adapters
#
set -euo pipefail

# Source environment configuration
source /workspace/scripts/training/env.sh

# ============================================
# Model Configuration
# ============================================
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"

# ============================================
# Dataset Configuration
# ============================================
DATASET_PATH="${DATASET_PATH:-/outputs/mixed_training_v1/mixed_training_v1.jsonl}"

# ============================================
# Sequence and Vision Configuration
# ============================================
MAX_LEN="${MAX_LEN:-3124}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2048}"
export IMAGE_MAX_TOKEN_NUM

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ============================================
# LoRA Configuration
# ============================================
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"

# ============================================
# Training Hyperparameters
# ============================================
BATCH="${BATCH:-8}"
GRAD_ACC="${GRAD_ACC:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOSS_SCALE="${LOSS_SCALE:-default}"
DEEPSPEED="${DEEPSPEED:-zero2}"
PACKING="${PACKING:-false}"

# ============================================
# DDP Configuration
# ============================================
NPROC="${NPROC:-4}"

# ============================================
# Output Configuration
# ============================================
RUN_NAME="${RUN_NAME:-qwen25vl-7b-lora-sft}"
OUTPUT_PATH="${OUTPUT_DIR}/${RUN_NAME}"

# Checkpoint resumption
RESUME_PATH="${RESUME_PATH:-}"

# ============================================
# Logging and Checkpointing
# ============================================
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-1000000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# ============================================
# Auto-detect dtype
# ============================================
DTYPE="${DTYPE:-auto}"
if [ "$DTYPE" = "auto" ]; then
    DTYPE=$(python -c "import torch; print('bfloat16' if torch.cuda.is_bf16_supported() else 'float16')")
fi

# ============================================
# Pre-flight Checks
# ============================================
echo "Checking GPU availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'Found {torch.cuda.device_count()} GPU(s)')"

for dataset in ${DATASET_PATH}; do
    dataset_file="${dataset%%#*}"
    if [ ! -f "${dataset_file}" ]; then
        echo "ERROR: Training dataset not found: ${dataset_file}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_PATH}"

# ============================================
# Training Configuration Summary
# ============================================
echo "========================================="
echo "LoRA SFT for Qwen2.5-VL-7B"
echo "========================================="
echo ""
echo "Model: ${MODEL_ID}"
echo "Train Type: LoRA (rank=${LORA_RANK}, alpha=${LORA_ALPHA})"
echo "Target Modules: ${TARGET_MODULES}"
echo "Dtype: ${DTYPE}"
echo ""
echo "Dataset: ${DATASET_PATH}"
echo ""
echo "Sequence: max_len=${MAX_LEN}, image_max_tokens=${IMAGE_MAX_TOKEN_NUM}"
echo ""
echo "Training:"
echo "  GPUs: ${NPROC}"
echo "  Batch/GPU: ${BATCH}"
echo "  Grad Acc: ${GRAD_ACC}"
echo "  Effective Batch: $((BATCH * GRAD_ACC * NPROC))"
echo "  Epochs: ${EPOCHS}"
echo "  LR: ${LR}"
echo "  Warmup: ${WARMUP_RATIO}"
echo ""
echo "Frozen: ViT=yes, Aligner=yes, LLM=LoRA"
echo "Output: ${OUTPUT_PATH}"
echo "========================================="

# ============================================
# Launch Training
# ============================================
CUDA_DEVICES=$(seq -s, 0 $((NPROC - 1)))

RESUME_ARG=""
if [ -n "${RESUME_PATH}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME_PATH}"
    echo "Resuming from: ${RESUME_PATH}"
fi

NPROC_PER_NODE="${NPROC}" \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
swift sft \
    --model "${MODEL_ID}" \
    --train_type lora \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --target_modules "${TARGET_MODULES}" \
    --freeze_vit true \
    --freeze_aligner true \
    --dataset ${DATASET_PATH} \
    --split_dataset_ratio 0 \
    --max_length "${MAX_LEN}" \
    --loss_scale "${LOSS_SCALE}" \
    --per_device_train_batch_size "${BATCH}" \
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
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed "${DEEPSPEED}" \
    --packing "${PACKING}" \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 4 \
    --attn_impl flash_attn \
    --load_from_cache_file true \
    --dataset_shuffle true \
    ${RESUME_ARG}

echo ""
echo "========================================="
echo "LoRA Training completed!"
echo "========================================="
echo "Adapter saved to: ${OUTPUT_PATH}"
