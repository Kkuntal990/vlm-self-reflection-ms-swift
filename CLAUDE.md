# VLM Self-Reflection - Developer Guide

## Overview

Production Kubernetes platform for VLM fine-tuning with **ms-swift**, **PyTorch DDP**, and **LoRA**.

**Training Paradigms**: FIRE behavior cloning (8-GPU), Standard SFT (2-GPU), Full fine-tuning
**Models**: Qwen2.5-VL-7B, Qwen3-8B, Qwen3-VL-32B
**Tech Stack**: ms-swift, PyTorch DDP, Kubernetes, LoRA, HuggingFace

## Quick Commands

### Docker (Two-Tier Build)

```bash
# Base image (run ONCE - ~2GB, contains PyTorch)
./build_base.sh

# App image (run FREQUENTLY - ~50-100MB, contains code)
./build_main.sh

# Test locally
docker run --rm -it --gpus all ghcr.io/kkuntal990/ms-swift-qwen:latest bash
```

**Registry**: Defaults to GHCR (`ghcr.io/kkuntal990`). For Docker Hub: `export DOCKER_REGISTRY=docker.io && export DOCKER_USERNAME=kkokate990`

### Kubernetes - Setup

```bash
# Create storage and secrets
kubectl apply -f k8s/pvc-cache.yaml
kubectl apply -f k8s/pvc-outputs.yaml
kubectl create secret generic hf-token --from-literal=token="hf_..."

# Verify
kubectl get pvc
kubectl get secret hf-token
```

### Kubernetes - FIRE Training (Recommended)

```bash
# Phase 1: Preprocess (CPU-only, 4-6 hours)
kubectl apply -f k8s/job-preprocess-fire-cpu.yaml
kubectl logs -f job/fire-preprocess-cpu-job

# Phase 2: Train (8-GPU full fine-tuning)
kubectl apply -f k8s/job-full-sft-qwen3vl-fire-8gpu.yaml
kubectl logs -f job/qwen2-5vl-7b-fire-full-sft-8gpu-job
```

### Kubernetes - Interactive Development

```bash
# Launch Jupyter pod
kubectl apply -f k8s/jupyter-2gpu-test.yaml
kubectl exec -it vlm-jupyter -- bash

# Inside pod
pip install jupyter jupyterlab ipywidgets
jupyter lab --ip=0.0.0.0 --port=8888 --token='medvae2024' --allow-root --no-browser

# Port forward (local terminal)
kubectl port-forward vlm-jupyter 8888:8888
# Access: http://localhost:8888
```

### Monitoring & Debugging

```bash
# Get pod name
POD_NAME=$(kubectl get pods -l job-name=<job-name> -o jsonpath='{.items[0].metadata.name}')

# GPU verification
kubectl exec -it $POD_NAME -- nvidia-smi

# Logs
kubectl logs -f job/<job-name>
kubectl describe job <job-name>

# Access results
kubectl exec -it $POD_NAME -- ls -lh /outputs/
kubectl cp $POD_NAME:/outputs/ ./local-outputs/
```

## Key Files

### Scripts

- [scripts/run_full_sft_qwen3vl_fire_8gpu.sh](scripts/run_full_sft_qwen3vl_fire_8gpu.sh) - 8-GPU full fine-tuning
- [scripts/prepare_fire_sharegpt.py](scripts/prepare_fire_sharegpt.py) - FIRE dataset preprocessing
- [scripts/env.sh](scripts/env.sh) - Environment configuration

### Kubernetes Jobs

- [k8s/job-preprocess-fire-cpu.yaml](k8s/job-preprocess-fire-cpu.yaml) - CPU preprocessing
- [k8s/job-full-sft-qwen3vl-fire-8gpu.yaml](k8s/job-full-sft-qwen3vl-fire-8gpu.yaml) - 8-GPU training
- [k8s/jupyter-2gpu-test.yaml](k8s/jupyter-2gpu-test.yaml) - Interactive dev pod

### Container

- [Dockerfile.base](Dockerfile.base) - Base image (CUDA 12.1, PyTorch 2.2.0)
- [Dockerfile](Dockerfile) - App image (ms-swift + scripts)
- [build_base.sh](build_base.sh) - Build base image
- [build_main.sh](build_main.sh) - Build app image

## Configuration

### Key Environment Variables (8-GPU FIRE Training)

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model identifier |
| `DATASET_PATH` | `/outputs/fire_bc/fire_bc_train.jsonl` | Preprocessed dataset |
| `BATCH` | `2` | Per-GPU batch size |
| `GRAD_ACC` | `8` | Gradient accumulation |
| `NPROC` | `8` | Number of GPUs |
| `MAX_LEN` | `8192` | Max sequence length |
| `LR` | `1e-5` | Learning rate |

**Effective Batch Size** = `BATCH × GRAD_ACC × NPROC` = 2 × 8 × 8 = 128

## Local Testing

### Test Preprocessing (Without Cluster)

```bash
# FIRE dataset (10 samples, no images)
python scripts/prepare_fire_sharegpt.py \
  --output_dir ./test_fire_local/outputs \
  --image_dir ./test_fire_local/images \
  --max_samples 10 \
  --splits train \
  --skip-images
```

### Validate Kubernetes Manifests

```bash
kubectl apply --dry-run=server -f k8s/job-*.yaml
```

## Common Issues

### OOM (Out of Memory)

1. Reduce `MAX_LEN`: 8192 → 4096
2. Reduce `BATCH`: 2 → 1
3. Increase `GRAD_ACC` to compensate
4. Reduce `LORA_RANK` if using LoRA

### DDP Hangs

1. Set `NCCL_DEBUG=INFO`
2. Verify GPUs on same node: `kubectl get pods -o wide`
3. Check `MASTER_PORT` not in use

### Preprocessing Slow

- First run downloads images (~23GB) - takes 4-6 hours
- Subsequent runs are cached
- Use `--max_samples 100` for quick testing

## Pull Request Workflow

### Commit Message Format

```text
<type>: <short summary> (max 50 chars)

<detailed description>

- Bullet points for changes
```

**Types**: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

### PR Checklist

- [ ] Code follows PEP 8
- [ ] Local preprocessing tests pass
- [ ] Docker image builds successfully
- [ ] K8s manifests validated with `--dry-run`
- [ ] No secrets/credentials committed

## Resources

- [README.md](README.md) - Comprehensive documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick deployment guide
- [DOCKER_BUILD.md](DOCKER_BUILD.md) - Docker build system
- [GHCR_SETUP.md](GHCR_SETUP.md) - GitHub Container Registry setup
- [ms-swift](https://github.com/modelscope/swift) - Framework reference
- [FIRE dataset](https://huggingface.co/datasets/PengxiangLi/FIRE) - Behavior cloning data

## Environment

- **Conda environment**: `vlm-self-reflection-swift`
- **Storage**: `/cache` (models/datasets), `/outputs` (checkpoints)
- **Container registry**: `ghcr.io/kkuntal990` (GHCR default)
