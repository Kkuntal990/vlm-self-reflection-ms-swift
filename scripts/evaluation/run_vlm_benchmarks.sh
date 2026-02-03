#!/bin/bash
# run_vlm_benchmarks.sh - Unified evaluation script for VLMEvalKit and lmms-eval
#
# Usage:
#   # Run with lmms-eval (default)
#   bash scripts/evaluation/run_vlm_benchmarks.sh \
#     --framework lmms-eval \
#     --model-path /outputs/checkpoint-3752 \
#     --output-dir /outputs/benchmark_results/lmms_eval \
#     --num-gpus 2
#
#   # Run with VLMEvalKit
#   bash scripts/evaluation/run_vlm_benchmarks.sh \
#     --framework vlmevalkit \
#     --model-path /outputs/checkpoint-3752 \
#     --output-dir /outputs/benchmark_results/vlmevalkit \
#     --num-gpus 2
#
#   # Quick sanity check (10 samples, 1 benchmark)
#   bash scripts/evaluation/run_vlm_benchmarks.sh \
#     --framework lmms-eval \
#     --model-path /outputs/checkpoint-3752 \
#     --output-dir /outputs/benchmark_results/test \
#     --num-gpus 1 \
#     --limit 10 \
#     --benchmarks mmbench

set -euo pipefail

# ========================================
# Default Configuration
# ========================================
FRAMEWORK="lmms-eval"
MODEL_PATH=""
MODEL_NAME="FireSFT-Qwen2-5-VL-7B"
MODEL_TYPE=""  # lmms-eval model type: qwen2_5_vl, llava_onevision, llava, etc.
OUTPUT_DIR="/outputs/benchmark_results"
NUM_GPUS=2
BENCHMARKS="all"
LIMIT=0
VLMEVALKIT_DIR="/tmp/VLMEvalKit"

# ========================================
# Benchmark Definitions
# ========================================

# VLMEvalKit benchmark names
VLMEVALKIT_ALL="MMBench_DEV_EN MME SEEDBench_IMG MMMU_DEV_VAL MMVet AI2D_TEST OCRBench MathVista_MINI"

# lmms-eval benchmark names
LMMS_EVAL_ALL="mmbench_en_dev,mme,seedbench,mmmu_val,mmvet,ai2d,ocrbench,mathvista_testmini"

# Mapping for individual benchmark selection
declare -A VLMEVALKIT_MAP=(
    ["mmbench"]="MMBench_DEV_EN"
    ["mme"]="MME"
    ["seedbench"]="SEEDBench_IMG"
    ["mmmu"]="MMMU_DEV_VAL"
    ["mmvet"]="MMVet"
    ["ai2d"]="AI2D_TEST"
    ["ocrbench"]="OCRBench"
    ["mathvista"]="MathVista_MINI"
)

declare -A LMMS_EVAL_MAP=(
    ["mmbench"]="mmbench_en_dev"
    ["mme"]="mme"
    ["seedbench"]="seedbench"
    ["mmmu"]="mmmu_val"
    ["mmvet"]="mmvet"
    ["ai2d"]="ai2d"
    ["ocrbench"]="ocrbench"
    ["mathvista"]="mathvista_testmini"
)

# ========================================
# Parse Arguments
# ========================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
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
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --vlmevalkit-dir)
            VLMEVALKIT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --framework        vlmevalkit or lmms-eval (default: lmms-eval)"
            echo "  --model-path       Path to model checkpoint (required)"
            echo "  --model-name       Model name for VLMEvalKit registry (default: FireSFT-Qwen2-5-VL-7B)"
            echo "  --model-type       lmms-eval model type (default: auto-detect from model path)"
            echo "                     Options: qwen2_5_vl, llava_onevision, llava, etc."
            echo "  --output-dir       Output directory (default: /outputs/benchmark_results)"
            echo "  --num-gpus         Number of GPUs (default: 2)"
            echo "  --benchmarks       Comma-separated list or 'all' (default: all)"
            echo "                     Options: mmbench,mme,seedbench,mmmu,mmvet,ai2d,ocrbench,mathvista"
            echo "  --limit            Limit samples per benchmark, 0=all (default: 0)"
            echo "  --vlmevalkit-dir   VLMEvalKit install directory (default: /tmp/VLMEvalKit)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
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

# Login to HuggingFace if token is available (required for gated datasets)
if [ -n "${HF_TOKEN:-}" ]; then
    echo "Logging in to HuggingFace Hub..."
    huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || \
        python -c "import huggingface_hub; huggingface_hub.login(token='${HF_TOKEN}')" 2>/dev/null || \
        echo "WARNING: HF login failed, some datasets may not be accessible"
fi

# ========================================
# Auto-detect Model Type
# ========================================
detect_model_type() {
    local model_path="$1"

    # Check for model type indicators in path or config
    if [[ "${model_path}" =~ [Ll]lava.*[Oo]ne[Vv]ision ]] || \
       [[ "${model_path}" =~ llava-ov ]] || \
       [[ "${model_path}" =~ llava_ov ]]; then
        echo "llava_onevision"
    elif [[ "${model_path}" =~ [Ll]lava ]]; then
        echo "llava"
    elif [[ "${model_path}" =~ [Qq]wen.*2.*5.*[Vv][Ll] ]] || \
         [[ "${model_path}" =~ [Qq]wen2.5-VL ]]; then
        echo "qwen2_5_vl"
    elif [[ "${model_path}" =~ [Qq]wen.*[Vv][Ll] ]]; then
        echo "qwen_vl"
    else
        # Default fallback - user should specify --model-type
        echo ""
    fi
}

# Auto-detect model type if not specified
if [ -z "$MODEL_TYPE" ]; then
    MODEL_TYPE=$(detect_model_type "$MODEL_PATH")
    if [ -z "$MODEL_TYPE" ]; then
        echo "WARNING: Could not auto-detect model type from path."
        echo "         Please specify --model-type (e.g., qwen2_5_vl, llava_onevision, llava)"
        echo "         Defaulting to 'qwen2_5_vl'"
        MODEL_TYPE="qwen2_5_vl"
    else
        echo "Auto-detected model type: ${MODEL_TYPE}"
    fi
fi

echo "========================================="
echo "VLM Benchmark Evaluation"
echo "========================================="
echo "Framework:   ${FRAMEWORK}"
echo "Model Path:  ${MODEL_PATH}"
echo "Model Type:  ${MODEL_TYPE}"
echo "Output Dir:  ${OUTPUT_DIR}"
echo "Num GPUs:    ${NUM_GPUS}"
echo "Benchmarks:  ${BENCHMARKS}"
echo "Limit:       ${LIMIT} (0 = all)"
echo ""

# Verify model path (supports both local dirs and HuggingFace model IDs)
if [ -d "${MODEL_PATH}" ]; then
    echo "Local model checkpoint found:"
    ls -lh "${MODEL_PATH}/config.json" 2>/dev/null || echo "  (no config.json)"
else
    echo "MODEL_PATH is not a local directory: ${MODEL_PATH}"
    echo "Treating as HuggingFace model ID (will download at runtime)..."
fi
echo ""

# Verify GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ========================================
# Resolve Benchmarks
# ========================================
resolve_benchmarks() {
    local framework="$1"
    local selection="$2"

    if [ "$selection" = "all" ]; then
        if [ "$framework" = "vlmevalkit" ]; then
            echo "$VLMEVALKIT_ALL"
        else
            echo "$LMMS_EVAL_ALL"
        fi
        return
    fi

    local result=""
    IFS=',' read -ra BENCH_ARRAY <<< "$selection"
    for bench in "${BENCH_ARRAY[@]}"; do
        bench=$(echo "$bench" | xargs)  # trim whitespace
        if [ "$framework" = "vlmevalkit" ]; then
            local mapped="${VLMEVALKIT_MAP[$bench]:-}"
            if [ -z "$mapped" ]; then
                echo "ERROR: Unknown benchmark '$bench'" >&2
                exit 1
            fi
            result="${result} ${mapped}"
        else
            local mapped="${LMMS_EVAL_MAP[$bench]:-}"
            if [ -z "$mapped" ]; then
                echo "ERROR: Unknown benchmark '$bench'" >&2
                exit 1
            fi
            if [ -n "$result" ]; then
                result="${result},${mapped}"
            else
                result="${mapped}"
            fi
        fi
    done
    echo "$result"
}

RESOLVED_BENCHMARKS=$(resolve_benchmarks "$FRAMEWORK" "$BENCHMARKS")
echo "Resolved benchmarks: ${RESOLVED_BENCHMARKS}"
echo ""

# ========================================
# Run VLMEvalKit
# ========================================
run_vlmevalkit() {
    echo "========================================="
    echo "Setting up VLMEvalKit"
    echo "========================================="

    # Clone and install if not present
    if [ ! -d "${VLMEVALKIT_DIR}" ]; then
        echo "Cloning VLMEvalKit..."
        git clone https://github.com/open-compass/VLMEvalKit.git "${VLMEVALKIT_DIR}"
    fi

    if ! python -c "import vlmeval" 2>/dev/null; then
        echo "Installing VLMEvalKit..."
        pip install -e "${VLMEVALKIT_DIR}"
    fi

    # Register custom model
    echo "Registering custom model '${MODEL_NAME}'..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "${SCRIPT_DIR}/register_vlmevalkit_model.py" \
        --model-path "${MODEL_PATH}" \
        --model-name "${MODEL_NAME}" \
        --vlmevalkit-dir "${VLMEVALKIT_DIR}"

    echo ""
    echo "========================================="
    echo "Running VLMEvalKit Evaluation"
    echo "========================================="

    local cmd="torchrun --nproc-per-node=${NUM_GPUS} ${VLMEVALKIT_DIR}/run.py"
    cmd="${cmd} --data ${RESOLVED_BENCHMARKS}"
    cmd="${cmd} --model ${MODEL_NAME}"
    cmd="${cmd} --work-dir ${OUTPUT_DIR}"
    cmd="${cmd} --verbose"

    echo "Command: ${cmd}"
    echo ""
    eval "${cmd}"
}

# ========================================
# Run lmms-eval
# ========================================
run_lmms_eval() {
    echo "========================================="
    echo "Setting up lmms-eval"
    echo "========================================="

    if ! python -c "import lmms_eval" 2>/dev/null; then
        echo "Installing lmms-eval..."
        # Must clone and install in editable mode so task YAML configs are included.
        # pip install from git URL does NOT include the YAML task definitions.
        local lmms_eval_dir="/tmp/lmms-eval"
        if [ ! -d "${lmms_eval_dir}" ]; then
            git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git "${lmms_eval_dir}"
        fi
        pip install -e "${lmms_eval_dir}"
    fi

    echo ""
    echo "========================================="
    echo "Running lmms-eval Evaluation"
    echo "========================================="

    local model_args="pretrained=${MODEL_PATH},device_map=auto,attn_implementation=flash_attention_2"

    # LLaVA-OneVision requires model_name to avoid a broken get_model_name_from_path call.
    # model_name determines the conversation template; "llava_qwen" is standard for Qwen-based LLaVA-OV.
    if [ "${MODEL_TYPE}" = "llava_onevision" ]; then
        model_args="${model_args},model_name=llava_qwen"
    fi

    local cmd=""
    if [ "${NUM_GPUS}" -gt 1 ]; then
        cmd="accelerate launch --num_processes ${NUM_GPUS} --main_process_port 29500 -m lmms_eval"
    else
        cmd="python -m lmms_eval"
    fi

    cmd="${cmd} --model ${MODEL_TYPE}"
    cmd="${cmd} --model_args ${model_args}"
    cmd="${cmd} --tasks ${RESOLVED_BENCHMARKS}"
    cmd="${cmd} --batch_size 1"
    cmd="${cmd} --output_path ${OUTPUT_DIR}"
    cmd="${cmd} --log_samples"

    if [ "${LIMIT}" -gt 0 ]; then
        cmd="${cmd} --limit ${LIMIT}"
    fi

    echo "Command: ${cmd}"
    echo ""
    eval "${cmd}"
}

# ========================================
# Dispatch
# ========================================
case "$FRAMEWORK" in
    vlmevalkit)
        run_vlmevalkit
        ;;
    lmms-eval)
        run_lmms_eval
        ;;
    *)
        echo "ERROR: Unknown framework '${FRAMEWORK}'. Use 'vlmevalkit' or 'lmms-eval'."
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Evaluation Complete"
echo "========================================="
echo "Results saved to: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}/"
