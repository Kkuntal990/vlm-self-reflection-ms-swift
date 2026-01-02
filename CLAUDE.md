# VLM Self-Reflection - Developer Guide

## Overview

Production-ready Kubernetes platform for Vision-Language Model (VLM) fine-tuning with two training paradigms:

1. **Standard SFT**: 2-GPU DDP fine-tuning (Qwen3-8B on Alpaca GPT4)
2. **FIRE Behavior Cloning**: 4-GPU trajectory-level imitation learning (Qwen3-VL-32B on FIRE dataset)

**Tech Stack**: ms-swift, PyTorch DDP, Kubernetes, LoRA, HuggingFace Datasets

## Repository Structure

```text
vlm-self-reflection/
├── scripts/                    # Training & preprocessing
│   ├── run_sft_ddp.sh         # 2-GPU DDP training
│   ├── run_sft_qwen3vl_fire_4gpu.sh  # 4-GPU FIRE training
│   ├── prepare_fire_sharegpt.py      # FIRE preprocessing (690 lines)
│   └── env.sh                  # Environment setup
├── k8s/                        # Kubernetes manifests
│   ├── job-sft-qwen3-8b-2gpu.yaml    # Standard SFT job
│   ├── job-sft-qwen3vl-fire-4gpu.yaml # FIRE training job
│   ├── job-preprocess-fire-cpu.yaml   # CPU preprocessing job
│   ├── pvc-*.yaml              # Storage (cache + outputs)
│   └── jupyter-2gpu-test.yaml  # Interactive dev pod
├── notebooks/
│   └── dataset_analysis.ipynb  # Dataset exploration
└── Dockerfile                  # CUDA 12.1 + ms-swift
```

## Important Files

### Core Scripts

- [scripts/run_sft_ddp.sh](scripts/run_sft_ddp.sh) - 2-GPU DDP orchestration with torchrun
- [scripts/run_sft_qwen3vl_fire_4gpu.sh](scripts/run_sft_qwen3vl_fire_4gpu.sh) - 4-GPU FIRE behavior cloning
- [scripts/prepare_fire_sharegpt.py](scripts/prepare_fire_sharegpt.py) - Converts FIRE trajectories to ShareGPT format
- [scripts/env.sh](scripts/env.sh) - Environment configuration (cache/output paths, NCCL settings)

### Kubernetes Jobs

- [k8s/job-sft-qwen3-8b-2gpu.yaml](k8s/job-sft-qwen3-8b-2gpu.yaml) - Standard SFT (2×A100, 16Gi RAM)
- [k8s/job-sft-qwen3vl-fire-4gpu.yaml](k8s/job-sft-qwen3vl-fire-4gpu.yaml) - FIRE training (4×A100, 128Gi RAM)
- [k8s/job-preprocess-fire-cpu.yaml](k8s/job-preprocess-fire-cpu.yaml) - CPU-only preprocessing (no GPU needed)

### Storage

- [k8s/pvc-cache.yaml](k8s/pvc-cache.yaml) - 500Gi linstor-ucsc for models/datasets
- [k8s/pvc-outputs.yaml](k8s/pvc-outputs.yaml) - 500Gi rook-cephfs for checkpoints

### Container

- [Dockerfile](Dockerfile) - CUDA 12.1, PyTorch 2.2.0, ms-swift
- [requirements.txt](requirements.txt) - Python dependencies

###

- Conda environment name - vlm-self-reflection-swift

## Quick Commands

### Docker

```bash
# Build and push
docker build -t <username>/ms-swift-qwen:latest .
docker push <username>/ms-swift-qwen:latest

# Test locally
docker run --rm -it --gpus all <username>/ms-swift-qwen:latest bash
```

### Kubernetes - Setup

```bash
# Create storage
kubectl apply -f k8s/pvc-cache.yaml
kubectl apply -f k8s/pvc-outputs.yaml

# Create HF token secret (for gated models)
kubectl create secret generic hf-token --from-literal=token="hf_..."

# Verify
kubectl get pvc
kubectl get secret hf-token
```

### Kubernetes - Standard SFT (Qwen3-8B)

```bash
# Submit job (smoke test with 500 samples)
kubectl apply -f k8s/job-sft-qwen3-8b-2gpu.yaml

# Monitor
kubectl get jobs
kubectl logs -f job/qwen3-8b-sft-job

# Check status
kubectl get job qwen3-8b-sft-job
POD_NAME=$(kubectl get pods -l job-name=qwen3-8b-sft-job -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $POD_NAME -- nvidia-smi
kubectl exec -it $POD_NAME -- ls -lh /outputs/qwen3-8b-sft/

# Clean up
kubectl delete job qwen3-8b-sft-job
```

### Kubernetes - FIRE Behavior Cloning (2-phase workflow)

```bash
# Phase 1: Preprocess data (CPU-only, 4-6 hours for full dataset)
kubectl apply -f k8s/job-preprocess-fire-cpu.yaml
kubectl logs -f job/fire-preprocess-cpu-job

# Verify preprocessing completed
kubectl get job fire-preprocess-cpu-job
PREP_POD=$(kubectl get pods -l job-name=fire-preprocess-cpu-job -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $PREP_POD -- ls -lh /outputs/fire_bc/

# Phase 2: GPU training (requires Phase 1 to complete)
kubectl apply -f k8s/job-sft-qwen3vl-fire-4gpu.yaml
kubectl logs -f job/qwen3vl-32b-fire-bc-job
```

### Interactive Development

```bash
# Launch Jupyter pod
kubectl apply -f k8s/jupyter-2gpu-test.yaml
kubectl exec -it vlm-jupyter -- bash

# Inside pod: Install Jupyter
pip install jupyter jupyterlab ipywidgets
jupyter lab --ip=0.0.0.0 --port=8888 --token='medvae2024' --allow-root --no-browser

# Local terminal: Port forward
kubectl port-forward vlm-jupyter 8888:8888
# Access http://localhost:8888
```

### Monitoring & Debugging

```bash
# GPU verification
kubectl exec -it $POD_NAME -- nvidia-smi
kubectl exec -it $POD_NAME -- python -c "import torch; print(torch.cuda.device_count())"

# Resource monitoring
kubectl top pod $POD_NAME
kubectl exec -it $POD_NAME -- nvidia-smi dmon -s u

# Logs
kubectl logs -f job/qwen3-8b-sft-job
kubectl describe job qwen3-8b-sft-job
kubectl describe pod $POD_NAME

# Access results
kubectl exec -it $POD_NAME -- ls -lh /outputs/qwen3-8b-sft/
kubectl cp $POD_NAME:/outputs/qwen3-8b-sft ./local-checkpoints/
```

## Local Testing

### Test Preprocessing (Without Cluster)

```bash
# FIRE dataset (10 samples, no image download)
./scripts/test_fire_preprocessing.sh

# Or manually
python scripts/prepare_fire_sharegpt.py \
  --output_dir ./test_fire_local/outputs \
  --image_dir ./test_fire_local/images \
  --max_samples 10 \
  --splits train \
  --skip-images

# Volcano dataset
python scripts/prepare_volcano_sharegpt.py \
  --output_dir ./test_volcano_local/outputs \
  --image_dir ./test_volcano_local/images \
  --max_samples 10 \
  --split train \
  --skip-images
```

### Validate Kubernetes Manifests

```bash
kubectl apply --dry-run=client -f k8s/job-sft-qwen3-8b-2gpu.yaml
kubectl apply --dry-run=server -f k8s/job-sft-qwen3-8b-2gpu.yaml
```

## Configuration

### Key Environment Variables (in Job YAMLs)

**Standard SFT**:

```yaml
MODEL_ID: "Qwen/Qwen3-8B"
DATASET_ID: "tatsu-lab/alpaca#500"  # Use #N for N samples
MAX_LEN: "2048"
BATCH: "1"           # Per-GPU batch size
GRAD_ACC: "16"       # Gradient accumulation
EPOCHS: "1"
LR: "2e-4"
LORA_RANK: "8"
NPROC: "2"           # Number of GPUs
```

**FIRE Training**:

```yaml
MODEL_ID: "Qwen/Qwen3-VL-32B-Instruct"
DATASET_PATH: "/outputs/fire_bc"  # From preprocessing job
BATCH: "1"
GRAD_ACC: "16"
LR: "1e-4"
LORA_RANK: "32"
MAX_LEN: "8192"
IMAGE_MAX_TOKEN_NUM: "2048"
NPROC: "4"
```

**Effective Batch Size** = `BATCH × GRAD_ACC × NPROC`

- Standard SFT: 1 × 16 × 2 = 32
- FIRE: 1 × 16 × 4 = 64

## Testing Best Practices

### Before Cluster Deployment

1. **Test preprocessing locally**:
   - Use `--skip-images` and `--max_samples 10`
   - Verify output JSONL format and stats.json

2. **Validate Docker image**:

   ```bash
   docker run --rm -it --gpus all <image> bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Validate K8s manifests**:

   ```bash
   kubectl apply --dry-run=server -f k8s/job-*.yaml
   ```

4. **Run smoke test first**:
   - Use 500 samples or less
   - Verify: model download → DDP init → training loss decreases → checkpoint saves

### During Development

- Monitor GPU utilization: `kubectl exec -it $POD_NAME -- nvidia-smi dmon -s u`
- Check logs continuously: `kubectl logs -f job/<job-name>`
- Validate checkpoints: `kubectl exec -it $POD_NAME -- ls -lh /outputs/`

## Pull Request Workflow

### 1. Branch Strategy

```bash
# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/my-feature

# Make changes
git add .
git commit -m "feat: add new dataset support"
git push origin feature/my-feature
```

### 2. Commit Message Format

```text
<type>: <short summary> (max 50 chars)

<detailed description if needed>

- Bullet points for changes
- Reference issues: Fixes #123
```

**Types**: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

### 3. PR Checklist

- [ ] Code follows PEP 8 style
- [ ] Local preprocessing tests pass
- [ ] Docker image builds successfully
- [ ] K8s manifests validated with `--dry-run`
- [ ] Documentation updated (README.md if needed)
- [ ] No secrets/credentials committed
- [ ] Smoke test on cluster (if modifying training)

### 4. PR Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
- [ ] Local preprocessing test passed
- [ ] Docker image builds
- [ ] Smoke test on cluster (if applicable)

## Related Issues
Closes #123
```

## Coding Standards

### Python

- Follow PEP 8
- Add docstrings to functions/classes
- Use type hints
- Keep functions under 50 lines

### Shell Scripts

- Use `set -e` (exit on error)
- Quote variables: `"$VAR"`
- Add comments for complex logic

### Kubernetes

- Include resource requests/limits
- Add labels and annotations
- Document env vars in comments

## Common Issues

### OOM (Out of Memory)

1. Reduce `MAX_LEN`: 2048 → 1024
2. Reduce `BATCH`: 1 (already minimum)
3. Increase `GRAD_ACC` to compensate
4. Reduce `LORA_RANK`: 8 → 4

### DDP Hangs

1. Set `NCCL_DEBUG=INFO`
2. Verify both GPUs on same node: `kubectl get pods -o wide`
3. Check `MASTER_PORT` not in use

### Preprocessing Slow

- First run downloads COCO images (~23GB) - takes 4-6 hours
- Subsequent runs are cached and much faster
- Use `--max_samples 100` for quick testing

## Resources

- [README.md](README.md) - Comprehensive documentation
- [QUICKSTART.md](QUICKSTART.md) - Rapid deployment guide
- [ms-swift docs](https://github.com/modelscope/swift) - Framework reference
- [Qwen models](https://huggingface.co/Qwen) - Base models
- [FIRE dataset](https://huggingface.co/datasets/PengxiangLi/FIRE) - Behavior cloning data
