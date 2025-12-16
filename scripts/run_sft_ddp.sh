#!/usr/bin/env bash
set -euo pipefail

# Source environment configuration
source /workspace/scripts/env.sh

# Training configuration with sensible defaults
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
DATASET_ID="${DATASET_ID:-AI-ModelScope/alpaca-gpt4-data-en#500}"
MAX_LEN="${MAX_LEN:-2048}"
BATCH="${BATCH:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
PORT="${MASTER_PORT:-29501}"
NPROC="${NPROC:-2}"

# Auto-detect optimal dtype (bf16 if supported, else fp16)
DTYPE="${DTYPE:-auto}"
if [ "$DTYPE" = "auto" ]; then
    echo "Auto-detecting optimal dtype..."
    DTYPE=$(python -c "import torch; print('bfloat16' if torch.cuda.is_bf16_supported() else 'float16')")
    echo "Selected dtype: $DTYPE"
fi

# Verify GPU availability
echo "Checking GPU availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'Found {torch.cuda.device_count()} GPU(s)')"

# Create output directory
mkdir -p "${OUTPUT_DIR}/qwen3-8b-sft"

echo "========================================="
echo "Starting 2-GPU DDP Training"
echo "========================================="
echo "Model: ${MODEL_ID}"
echo "Dataset: ${DATASET_ID}"
echo "Max Length: ${MAX_LEN}"
echo "Batch Size per GPU: ${BATCH}"
echo "Gradient Accumulation: ${GRAD_ACC}"
echo "Effective Batch Size: $((BATCH * GRAD_ACC * NPROC))"
echo "Epochs: ${EPOCHS}"
echo "Learning Rate: ${LR}"
echo "Dtype: ${DTYPE}"
echo "LoRA Rank: ${LORA_RANK}"
echo "Output Dir: ${OUTPUT_DIR}/qwen3-8b-sft"
echo "========================================="

# Launch distributed training with torchrun
torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${PORT}" \
    $(which swift) sft \
    --model_id_or_path "${MODEL_ID}" \
    --train_type lora \
    --dataset "${DATASET_ID}" \
    --max_length "${MAX_LEN}" \
    --per_device_train_batch_size "${BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --torch_dtype "${DTYPE}" \
    --output_dir "${OUTPUT_DIR}/qwen3-8b-sft" \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 2 \
    --use_hf true \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --gradient_checkpointing true \
    --dataloader_num_workers 4

echo "========================================="
echo "Training completed successfully!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/qwen3-8b-sft"
echo "========================================="
