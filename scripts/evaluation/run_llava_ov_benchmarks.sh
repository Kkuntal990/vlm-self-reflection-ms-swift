#!/bin/bash
# run_llava_ov_benchmarks.sh - Evaluation script for LLaVA-OneVision models via lmms-eval
#
# Uses the built-in llava_hf model type in lmms-eval, which supports
# LlavaOnevisionForConditionalGeneration from HuggingFace transformers natively.
# No custom wrappers or LLaVA-NeXT package needed.
#
# Usage:
#   bash scripts/evaluation/run_llava_ov_benchmarks.sh \
#     --model-path /outputs/checkpoint-2735 \
#     --output-dir /outputs/benchmark_results/llava_ov \
#     --num-gpus 4
#
#   # Quick sanity check (10 samples, 1 benchmark)
#   bash scripts/evaluation/run_llava_ov_benchmarks.sh \
#     --model-path /outputs/checkpoint-2735 \
#     --output-dir /outputs/benchmark_results/test \
#     --num-gpus 1 \
#     --limit 10 \
#     --benchmarks mme

set -euo pipefail

# ========================================
# Default Configuration
# ========================================
MODEL_PATH=""
OUTPUT_DIR="/outputs/benchmark_results"
NUM_GPUS=1
BENCHMARKS="all"
LIMIT=0

# All supported benchmarks
ALL_BENCHMARKS="mmbench_en_dev,mme,seedbench,mmmu_val,mmvet,ai2d,ocrbench,mathvista_testmini"

# ========================================
# Parse Arguments
# ========================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            echo "LLaVA-OneVision Benchmark Evaluation (lmms-eval built-in llava_hf)"
            echo ""
            echo "Usage: bash run_llava_ov_benchmarks.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-path       Path to model checkpoint or HF model ID (required)"
            echo "  --output-dir       Output directory (default: /outputs/benchmark_results)"
            echo "  --num-gpus         Number of GPUs (default: 1)"
            echo "  --benchmarks       Comma-separated list or 'all' (default: all)"
            echo "  --limit            Limit samples per benchmark, 0 = all (default: 0)"
            echo ""
            echo "Available benchmarks:"
            echo "  mmbench_en_dev, mme, seedbench, mmmu_val, mmvet, ai2d, ocrbench, mathvista_testmini"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: --model-path is required"
    exit 1
fi

# ========================================
# Environment Setup
# ========================================
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Login to HuggingFace if token is available
if [ -n "${HF_TOKEN:-}" ]; then
    echo "Logging in to HuggingFace Hub..."
    huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || \
        python -c "import huggingface_hub; huggingface_hub.login(token='${HF_TOKEN}')" 2>/dev/null || \
        echo "WARNING: HF login failed, some datasets may not be accessible"
fi

# ========================================
# Print Configuration
# ========================================
echo "========================================="
echo "LLaVA-OneVision Benchmark Evaluation"
echo "========================================="
echo "Model Path:  ${MODEL_PATH}"
echo "Output Dir:  ${OUTPUT_DIR}"
echo "Num GPUs:    ${NUM_GPUS}"
echo "Benchmarks:  ${BENCHMARKS}"
echo "Limit:       ${LIMIT} (0 = all)"
echo ""

# Verify model path
if [ -d "${MODEL_PATH}" ]; then
    echo "Local model checkpoint found:"
    ls -lh "${MODEL_PATH}/config.json" 2>/dev/null || echo "  (no config.json)"
else
    echo "MODEL_PATH is not a local directory: ${MODEL_PATH}"
    echo "Treating as HuggingFace model ID..."
fi
echo ""

# Verify GPUs
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "  No GPUs detected"
echo ""

# ========================================
# Install lmms-eval (editable mode for task YAMLs)
# ========================================
if ! python -c "import lmms_eval" 2>/dev/null; then
    echo "Installing lmms-eval..."
    LMMS_EVAL_DIR="/tmp/lmms-eval"
    if [ ! -d "${LMMS_EVAL_DIR}" ]; then
        git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git "${LMMS_EVAL_DIR}"
    fi
    pip install -e "${LMMS_EVAL_DIR}"
fi

# ========================================
# Resolve Benchmarks
# ========================================
if [ "${BENCHMARKS}" = "all" ]; then
    RESOLVED_BENCHMARKS="${ALL_BENCHMARKS}"
else
    RESOLVED_BENCHMARKS="${BENCHMARKS}"
fi

echo "Resolved benchmarks: ${RESOLVED_BENCHMARKS}"
echo ""

# ========================================
# Create Output Directory
# ========================================
mkdir -p "${OUTPUT_DIR}"

# ========================================
# Build and Run lmms-eval Command
# ========================================
# Use built-in llava_hf model type which supports LlavaOnevision natively.
# For multi-GPU: accelerate data parallelism (no device_map, each process gets own GPU)
# For single GPU: device_map=auto
if [ "${NUM_GPUS}" -gt 1 ]; then
    MODEL_ARGS="pretrained=${MODEL_PATH},attn_implementation=flash_attention_2"
    CMD="accelerate launch --num_processes ${NUM_GPUS} --main_process_port 29500 -m lmms_eval"
else
    MODEL_ARGS="pretrained=${MODEL_PATH},device_map=auto,attn_implementation=flash_attention_2"
    CMD="python -m lmms_eval"
fi

CMD="${CMD} --model llava_hf"
CMD="${CMD} --model_args ${MODEL_ARGS}"
CMD="${CMD} --tasks ${RESOLVED_BENCHMARKS}"
CMD="${CMD} --batch_size 1"
CMD="${CMD} --output_path ${OUTPUT_DIR}"
CMD="${CMD} --log_samples"

if [ "${LIMIT}" -gt 0 ]; then
    CMD="${CMD} --limit ${LIMIT}"
fi

echo "========================================="
echo "Running lmms-eval with built-in llava_hf model"
echo "========================================="
echo "Command:"
echo "  ${CMD}"
echo ""

eval "${CMD}"

echo ""
echo "========================================="
echo "Evaluation Complete"
echo "========================================="
echo "Results saved to: ${OUTPUT_DIR}"
ls -lhR "${OUTPUT_DIR}/" 2>/dev/null || true
