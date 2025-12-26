#!/usr/bin/env bash
#
# Behavior Cloning Training Script for Qwen3-VL-32B on FIRE Dataset
#
# This script implements offline imitation learning via behavior cloning
# over reflection trajectories, following the RePer paradigm:
#
#   State  = (image, question, previous attempts, previous feedback)
#   Action = next student response
#   Loss   = cross-entropy on expert action given state
#
# Even though we use standard SFT tooling, this is NOT "just SFT" - it is
# trajectory-level behavior cloning where each timestep is a training example.
#
# Hardware: 4 × A100 80GB (single node DDP)
# Model: Qwen/Qwen3-VL-32B-Instruct with LoRA
#
set -euo pipefail

# Source environment configuration
source /workspace/scripts/env.sh

# ============================================
# Model Configuration
# ============================================
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-32B-Instruct}"

# ============================================
# Dataset Configuration (Behavior Cloning JSONL)
# ============================================
DATASET_PATH="${DATASET_PATH:-/outputs/fire_bc/fire_bc_train.jsonl}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-/outputs/fire_bc/fire_bc_test.jsonl}"

# ============================================
# Sequence and Vision Configuration
# ============================================
MAX_LEN="${MAX_LEN:-8192}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2048}"
export IMAGE_MAX_TOKEN_NUM

# Memory optimization for large models
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ============================================
# Training Hyperparameters
# ============================================
# Batch size reduced for 32B model (memory constraint)
BATCH="${BATCH:-1}"
# Gradient accumulation increased to maintain effective batch size
GRAD_ACC="${GRAD_ACC:-16}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

# ============================================
# LoRA Configuration
# ============================================
# Higher rank for complex behavior cloning task
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"

# ============================================
# DDP Configuration
# ============================================
NPROC="${NPROC:-4}"
PORT="${MASTER_PORT:-29501}"

# ============================================
# Output Configuration
# ============================================
RUN_NAME="${RUN_NAME:-qwen3vl-32b-fire-bc}"
OUTPUT_PATH="${OUTPUT_DIR}/${RUN_NAME}"

# ============================================
# Logging and Checkpointing
# ============================================
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

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
    echo "Run prepare_fire_full_state_jsonl.py first to create the behavior cloning dataset."
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_PATH}"

# ============================================
# Training Configuration Summary
# ============================================
echo "========================================="
echo "FIRE Behavior Cloning Training"
echo "Offline Imitation Learning via SFT"
echo "========================================="
echo ""
echo "Model Configuration:"
echo "  Model: ${MODEL_ID}"
echo "  Train Type: LoRA"
echo "  LoRA Rank: ${LORA_RANK} | Alpha: ${LORA_ALPHA}"
echo "  Target Modules: ${TARGET_MODULES}"
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
echo ""
echo "Output: ${OUTPUT_PATH}"
echo "========================================="

# ============================================
# Launch Distributed Training
# ============================================
torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${PORT}" \
    $(which swift) sft \
    --model "${MODEL_ID}" \
    --train_type lora \
    --dataset "${DATASET_PATH}" \
    --val_dataset "${VAL_DATASET_PATH}" \
    --max_length "${MAX_LEN}" \
    --per_device_train_batch_size "${BATCH}" \
    --per_device_eval_batch_size "${BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --target_modules "${TARGET_MODULES}" \
    --torch_dtype "${DTYPE}" \
    --output_dir "${OUTPUT_PATH}" \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --use_hf true \
    --gradient_checkpointing true \
    --freeze_vit true \
    --freeze_aligner true \
    --attn_impl flash_attn \
    --dataloader_num_workers 4 \
    --report_to tensorboard

echo ""
echo "========================================="
echo "Training completed successfully!"
echo "========================================="
echo "Checkpoints saved to: ${OUTPUT_PATH}"
echo ""
echo "To merge LoRA weights for inference:"
echo "  swift merge --model ${MODEL_ID} --adapter ${OUTPUT_PATH}"
echo "========================================="
