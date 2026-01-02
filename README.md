# ms-swift Kubernetes 2-GPU DDP Training

Reproducible Kubernetes Job for 2-GPU DDP (Distributed Data Parallel) fine-tuning with ms-swift on Qwen3-8B.

## Overview

This project provides a production-ready setup for fine-tuning Qwen3-8B using:
- **Framework**: ms-swift with LoRA for memory efficiency
- **Training**: 2-GPU single-node DDP with torchrun
- **Dataset**: HuggingFace Alpaca GPT4 (smoke test with 500 samples, scalable to full dataset)
- **Storage**: Dual-PVC strategy optimized for Rook-Ceph

## Project Structure

```
.
├── Dockerfile                          # CUDA 12.1 + ms-swift container
├── requirements.txt                    # Python dependencies
├── .dockerignore                       # Build optimization
├── README.md                           # Comprehensive documentation (this file)
├── QUICKSTART.md                       # Quick deployment guide
├── CLAUDE.md                           # Developer guide for Claude Code
├── scripts/
│   ├── env.sh                         # Environment configuration
│   ├── run_sft_ddp.sh                 # 2-GPU DDP training (Qwen3-8B)
│   ├── run_sft_single_gpu.sh          # Single-GPU training (dev/test)
│   ├── run_sft_qwen3vl_fire_4gpu.sh  # 4-GPU FIRE behavior cloning (Qwen3-VL-32B)
│   ├── prepare_fire_sharegpt.py       # FIRE dataset preprocessing (690 lines)
│   ├── prepare_volcano_sharegpt.py    # Volcano dataset preprocessing
│   └── test_fire_preprocessing.sh     # Local preprocessing test
├── k8s/
│   ├── pvc-cache.yaml                 # linstor-ucsc for cache (500Gi)
│   ├── pvc-outputs.yaml               # rook-cephfs for checkpoints (500Gi)
│   ├── job-sft-qwen3-8b-2gpu.yaml    # 2-GPU Qwen3-8B Job
│   ├── job-preprocess-fire-cpu.yaml   # CPU-only FIRE preprocessing Job
│   ├── job-sft-qwen3vl-fire-4gpu.yaml # 4-GPU FIRE behavior cloning Job
│   ├── jupyter-2gpu-test.yaml         # Interactive development pod
│   └── secret-hf-token.yaml.template  # HuggingFace token template
├── notebooks/
│   └── dataset_analysis.ipynb         # Dataset exploration and validation
└── test_fire_local/                   # Local FIRE preprocessing outputs
```

## Storage Strategy

### Dual-PVC Architecture

1. **Cache PVC** (`rook-ceph-block`, 200Gi)
   - Mount: `/cache`
   - Purpose: HuggingFace models, datasets, pip cache
   - Rationale: Block storage avoids CephFS "many small files" performance issues

2. **Outputs PVC** (`rook-cephfs`, 300Gi)
   - Mount: `/outputs`
   - Purpose: Checkpoints, model weights, dataset shards
   - Rationale: CephFS handles large files well, enables sharing across jobs

## Quick Start

### 1. Build and Push Container Image

```bash
# Build the Docker image
docker build -t <your-dockerhub-username>/ms-swift-qwen:latest .

# Push to Docker Hub
docker push <your-dockerhub-username>/ms-swift-qwen:latest
```

**Important**: Update the image name in `k8s/job-sft-qwen3-8b-2gpu.yaml` with your actual Docker Hub username.

### 2. Create Storage Resources

```bash
# Create both PVCs
kubectl apply -f k8s/pvc-cache.yaml
kubectl apply -f k8s/pvc-outputs.yaml

# Verify PVCs are bound
kubectl get pvc
```

### 3. Create HuggingFace Token Secret (Optional)

For public models (like Qwen3-8B), this is optional. For gated/private models:

```bash
# Option 1: From command line (recommended)
kubectl create secret generic hf-token --from-literal=token="hf_your_actual_token_here"

# Option 2: From template file
# Edit k8s/secret-hf-token.yaml.template with your token, then:
kubectl apply -f k8s/secret-hf-token.yaml.template
```

### 4. Run Training Job

#### Phase 1: Smoke Test (5-15 minutes)

```bash
# Deploy the job (configured for 500-sample smoke test by default)
kubectl apply -f k8s/job-sft-qwen3-8b-2gpu.yaml

# Monitor progress
kubectl get jobs
kubectl logs -f job/qwen3-8b-sft-job

# Check for successful completion
kubectl get job qwen3-8b-sft-job
```

**Validation checklist**:
- Model and dataset downloads succeed to `/cache`
- DDP initializes with 2 ranks
- Training loss decreases
- Checkpoint appears in `/outputs/qwen3-8b-sft`

#### Phase 2: Full Dataset Training

After smoke test succeeds, scale to full dataset:

```bash
# Edit k8s/job-sft-qwen3-8b-2gpu.yaml
# Change DATASET_ID from:
#   value: "AI-ModelScope/alpaca-gpt4-data-en#500"
# To:
#   value: "AI-ModelScope/alpaca-gpt4-data-en"

# Optional: Adjust other hyperparameters
#   - EPOCHS: increase from 1 to 3
#   - MAX_LEN: try 4096 if you have sufficient GPU memory
#   - GRAD_ACC: adjust for desired effective batch size

# Delete previous job and rerun
kubectl delete job qwen3-8b-sft-job
kubectl apply -f k8s/job-sft-qwen3-8b-2gpu.yaml

```

#### Run  jupyter job for debugging

1. use `kubectl apply -f k8s/jupyter-2gpu-test.yaml`
2. `kubectl exec -it vlm-jupyter --bash`
3. `pip install jupyter jupyterlab ipywidgets`
4. `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \                                                                              --ServerApp.token='medvae2024' --ServerApp.password=''`
5. Now in your local terminal - `kubectl port-forward vlm-jupyter 8888:8888`
6. Connect to localhost:8888 in your local browser




### 5. Access Results

```bash
# Get pod name
POD_NAME=$(kubectl get pods -l job-name=qwen3-8b-sft-job -o jsonpath='{.items[0].metadata.name}')

# Check outputs
kubectl exec -it $POD_NAME -- ls -lh /outputs/qwen3-8b-sft/

# Check cache
kubectl exec -it $POD_NAME -- ls -lh /cache/hf/

# Copy checkpoints locally (after job completes)
kubectl cp $POD_NAME:/outputs/qwen3-8b-sft ./local-checkpoints/
```

## Configuration Reference

### Environment Variables

All training parameters can be customized via environment variables in `k8s/job-sft-qwen3-8b-2gpu.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-8B` | HuggingFace model identifier |
| `DATASET_ID` | `AI-ModelScope/alpaca-gpt4-data-en#500` | Dataset (use `#N` for N samples) |
| `MAX_LEN` | `2048` | Maximum sequence length |
| `BATCH` | `1` | Batch size per GPU |
| `GRAD_ACC` | `16` | Gradient accumulation steps |
| `EPOCHS` | `1` | Number of training epochs |
| `LR` | `2e-4` | Learning rate |
| `LORA_RANK` | `8` | LoRA rank |
| `LORA_ALPHA` | `16` | LoRA alpha parameter |
| `NPROC` | `2` | Number of GPUs |
| `DTYPE` | `auto` | Data type (auto-detects bf16/fp16) |

**Effective batch size** = `BATCH × GRAD_ACC × NPROC` = 1 × 16 × 2 = 32 (default)

### Resource Requests/Limits

Default configuration in Job manifest:
- **GPUs**: 2 (NVIDIA)
- **Memory**: 32Gi request, 64Gi limit
- **CPU**: 8 cores request, 16 cores limit

Adjust based on your cluster's GPU type (A100, V100, etc.)

## Debugging Guide

### GPU Verification

```bash
# Check GPU visibility in pod
kubectl exec -it $POD_NAME -- nvidia-smi

# Verify PyTorch sees both GPUs
kubectl exec -it $POD_NAME -- python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Common Issues

#### DDP Hangs at Initialization

**Symptoms**: Training stuck after "Initializing distributed training"

**Solutions**:
1. Check NCCL debug logs:
   ```yaml
   - name: NCCL_DEBUG
     value: "INFO"  # Change from WARN
   ```
2. Verify `MASTER_PORT` is not in use
3. Ensure both GPUs are on the same node (check `kubectl get pods -o wide`)

#### Out of Memory (OOM)

**Symptoms**: CUDA OOM errors, pod killed

**Solutions** (in order of preference):
1. Reduce `MAX_LEN`: `2048` → `1024`
2. Reduce `BATCH`: `1` → use `GRAD_ACC` to compensate
3. Increase `GRAD_ACC` (maintains effective batch size)
4. Enable gradient checkpointing (already enabled by default)
5. Reduce `LORA_RANK`: `8` → `4`

#### Model/Dataset Download Fails

**Solutions**:
1. Check network connectivity: `kubectl exec -it $POD_NAME -- ping huggingface.co`
2. Verify HF token (if using gated models): `kubectl get secret hf-token`
3. Check cache PVC has sufficient space: `kubectl exec -it $POD_NAME -- df -h /cache`

#### Slow Training

**Diagnostics**:
```bash
# Check cache hits (subsequent runs should be faster)
kubectl exec -it $POD_NAME -- ls -lh /cache/hf/hub/

# Check storage performance
kubectl exec -it $POD_NAME -- dd if=/dev/zero of=/cache/test bs=1G count=1 oflag=direct
```

### Logs and Monitoring

```bash
# Follow logs in real-time
kubectl logs -f job/qwen3-8b-sft-job

# Get logs after completion
kubectl logs job/qwen3-8b-sft-job > training.log

# Check job status
kubectl describe job qwen3-8b-sft-job

# Check pod events
kubectl describe pod $POD_NAME
```

## Advanced Configuration

### Custom Dataset Format

To use your own dataset, ensure it follows ms-swift's expected format:

```json
[
  {"instruction": "...", "output": "..."},
  {"instruction": "...", "input": "...", "output": "..."}
]
```

Upload to HuggingFace and set `DATASET_ID` accordingly.

### Multi-Epoch Training

For full dataset training:

```yaml
- name: EPOCHS
  value: "3"
- name: DATASET_ID
  value: "AI-ModelScope/alpaca-gpt4-data-en"  # Full dataset
```

### Memory Optimization

If you have GPUs with limited VRAM:

```yaml
- name: MAX_LEN
  value: "1024"
- name: BATCH
  value: "1"
- name: GRAD_ACC
  value: "32"  # Increased to maintain effective batch size
- name: LORA_RANK
  value: "4"   # Reduced from 8
```

## FIRE Full-State Imitation Learning (Qwen3-VL-32B)

This trains Qwen3-VL-32B using **offline imitation learning (behavior cloning)** on full reflection trajectories from the [FIRE dataset](https://huggingface.co/datasets/PengxiangLi/FIRE), following the RePer paradigm.

### Learning Paradigm

This is NOT standard final-answer SFT. It is trajectory-level behavior cloning where:

- **State** = (image, question, all previous attempts, all previous feedback)
- **Action** = next student response
- **Loss** = cross-entropy on expert action given state

Each timestep in the trajectory becomes a separate training example, teaching the model to iteratively refine its answers based on feedback.

### Step 1: Dataset Preprocessing (CPU-only)

**Important:** FIRE dataset only contains image paths. The actual images must be downloaded from the [COCO dataset](https://cocodataset.org/). The preprocessing script automatically downloads needed images from HuggingFace's COCO dataset.

#### Local Testing (without images)

Test the preprocessing logic locally without downloading COCO images:

```bash
# Test with 10 samples (uses placeholder image paths)
./scripts/test_fire_preprocessing.sh

# Or run directly with --skip-images flag
python scripts/prepare_fire_full_state_jsonl.py \
  --output_dir ./test_outputs \
  --image_dir ./test_images \
  --max_samples 10 \
  --splits train \
  --skip-images
```

#### Cluster Preprocessing (with real images)

Convert FIRE trajectories to behavior cloning format using CPU-only nodes (no GPU needed):

```bash
# Submit preprocessing job (CPU-only, cost-effective)
kubectl apply -f k8s/job-preprocess-fire-cpu.yaml

# Monitor progress
kubectl logs -f job/fire-preprocess-cpu-job

# Check completion
kubectl get job fire-preprocess-cpu-job
```

**For quick testing with 100 samples:**

Edit `k8s/job-preprocess-fire-cpu.yaml` and change:

```yaml
- name: MAX_SAMPLES
  value: "100"  # Change from "0"
```

**Output files:**

- `/outputs/fire_bc/fire_bc_train.jsonl` - Training examples (~367K from 105K trajectories)
- `/outputs/fire_bc/fire_bc_test.jsonl` - Test examples (~38K from 11K trajectories)
- `/outputs/fire_bc/stats.json` - Processing statistics
- `/cache/fire_images/{split}/` - Extracted images (~23GB total)

**Preprocessing time estimates:**

- Local testing with `--skip-images`: seconds
- 100 samples (with images): ~30-60 minutes (COCO download + processing)
- Full dataset (116K samples): ~4-6 hours (first run with COCO download)

**Note:** The first run requires downloading COCO images from HuggingFace, which is time-consuming. The preprocessing script searches through ~118K COCO images to find matches for FIRE samples. Subsequent runs will be faster if COCO is cached.

### Step 2: Training (4×A100 80GB)

After preprocessing completes, submit the GPU training job:

```bash
# Submit training job (requires preprocessed data)
kubectl apply -f k8s/job-sft-qwen3vl-fire-4gpu.yaml

# Monitor progress
kubectl logs -f job/qwen3vl-32b-fire-bc-job

# Check status
kubectl get job qwen3vl-32b-fire-bc-job
```

**Note:** Training job will fail if preprocessing hasn't completed. Verify data exists:

```bash
kubectl exec -it <preprocessing-pod> -- ls -lh /outputs/fire_bc/
```

### Configuration

**Preprocessing job** (`k8s/job-preprocess-fire-cpu.yaml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SAMPLES` | `0` (all) | Samples to preprocess (0 = all) |
| `MAX_HISTORY_ROUNDS` | `6` | Max history rounds in state |

**Training job** (`k8s/job-sft-qwen3vl-fire-4gpu.yaml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH` | `1` | Per-GPU batch size (reduced for 32B) |
| `GRAD_ACC` | `16` | Gradient accumulation |
| `LR` | `1e-4` | Learning rate |
| `LORA_RANK` | `32` | LoRA rank |
| `MAX_LEN` | `8192` | Max sequence length |
| `IMAGE_MAX_TOKEN_NUM` | `2048` | Image token budget |

**Effective batch size** = 1 × 16 × 4 GPUs = 64

### Checkpoints

Saved to: `/outputs/qwen3vl-32b-fire-bc/`

To merge LoRA weights for inference:

```bash
swift merge --model Qwen/Qwen3-VL-32B-Instruct --adapter /outputs/qwen3vl-32b-fire-bc
```

---

## Performance Tips

1. **Cache Warmup**: First run downloads models/datasets to cache. Subsequent runs will be much faster.

2. **Batch Size Tuning**: Start with `BATCH=1`, increase if memory allows. Monitor GPU utilization:
   ```bash
   kubectl exec -it $POD_NAME -- nvidia-smi dmon -s u
   ```

3. **Sequence Length**: Use the longest `MAX_LEN` your GPUs can handle without OOM for best quality.

4. **LoRA Parameters**: Higher `LORA_RANK` (8, 16, 32) improves quality but uses more memory.

## Future Enhancements (Out of Scope)

These features are intentionally excluded from the initial implementation:

- Multi-node DDP training
- DeepSpeed ZeRO optimization
- Flash Attention 2 integration
- GRPO/MT-GRPO post-training
- Automatic hyperparameter tuning

## Troubleshooting Checklist

Before asking for help, verify:

- [ ] GPUs are visible: `nvidia-smi` shows 2 GPUs
- [ ] PyTorch detects GPUs: `torch.cuda.device_count() == 2`
- [ ] PVCs are bound: `kubectl get pvc` shows `Bound` status
- [ ] Image is accessible: Check image pull secrets if using private registry
- [ ] HF token is valid (if using gated models)
- [ ] Sufficient storage: Cache PVC has space for model (~16GB) + dataset (~500MB)
- [ ] NCCL can communicate: Check firewall rules for inter-GPU communication

## Local Development

### Testing Preprocessing Scripts

Test dataset preprocessing locally before running on cluster:

```bash
# Test FIRE preprocessing with 10 samples (no image download)
./scripts/test_fire_preprocessing.sh

# Or manually with custom parameters
python scripts/prepare_fire_sharegpt.py \
  --output_dir ./test_fire_local/outputs \
  --image_dir ./test_fire_local/images \
  --max_samples 10 \
  --splits train \
  --skip-images
```

### Jupyter Notebook Analysis

Explore datasets and validate preprocessing logic:

```bash
# Install Jupyter locally
pip install jupyter jupyterlab ipywidgets

# Run notebook
jupyter lab notebooks/dataset_analysis.ipynb
```

### Single-GPU Development

For local development or testing on single GPU:

```bash
# Edit scripts/run_sft_single_gpu.sh to customize parameters
# Then run directly (not recommended for production)
bash scripts/run_sft_single_gpu.sh
```

Or use the Jupyter pod for interactive testing:

```bash
# Deploy Jupyter pod with 2 GPUs
kubectl apply -f k8s/jupyter-2gpu-test.yaml

# Access the pod
kubectl exec -it vlm-jupyter -- bash

# Inside pod: Start Jupyter Lab
pip install jupyter jupyterlab ipywidgets
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --ServerApp.token='medvae2024' --ServerApp.password=''

# In local terminal: Port forward
kubectl port-forward vlm-jupyter 8888:8888

# Access http://localhost:8888 in browser
```

## Volcano Dataset (Alternative to FIRE)

This project also supports the Volcano dataset for VLM training:

### Preprocessing Volcano Dataset

```bash
# Local test (10 samples, no images)
python scripts/prepare_volcano_sharegpt.py \
  --output_dir ./test_volcano_local/outputs \
  --image_dir ./test_volcano_local/images \
  --max_samples 10 \
  --split train \
  --skip-images

# Full preprocessing on cluster (create similar job to fire-preprocess-cpu-job)
```

**Dataset**: kaist-ai/volcano-train from HuggingFace

**Format**: Same ShareGPT format as FIRE, compatible with ms-swift training

## Testing Best Practices

### Before Submitting Jobs

1. **Validate Kubernetes manifests**:
   ```bash
   kubectl apply --dry-run=client -f k8s/job-sft-qwen3-8b-2gpu.yaml
   kubectl apply --dry-run=server -f k8s/job-sft-qwen3-8b-2gpu.yaml
   ```

2. **Test preprocessing locally**:
   - Always run preprocessing tests with `--skip-images` and small sample size
   - Verify output JSONL format matches expected structure
   - Check statistics in generated stats.json

3. **Verify Docker image**:
   ```bash
   # Test container locally
   docker run --rm -it --gpus all <your-image> bash
   # Inside container
   python -c "import torch; print(torch.cuda.is_available())"
   nvidia-smi
   ```

4. **Run smoke test before full training**:
   - Use 500 samples or less
   - Verify model downloads, DDP initialization, and training loss decrease
   - Check checkpoint saves correctly

### During Development

1. **Monitor resource usage**:
   ```bash
   kubectl top pod $POD_NAME
   kubectl exec -it $POD_NAME -- nvidia-smi dmon -s u
   ```

2. **Check logs continuously**:
   ```bash
   kubectl logs -f job/qwen3-8b-sft-job
   ```

3. **Validate checkpoints**:
   ```bash
   kubectl exec -it $POD_NAME -- ls -lh /outputs/qwen3-8b-sft/
   ```

## Contributing & Pull Requests

### Code Quality Standards

1. **Python Code**:
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Use type hints where applicable
   - Keep functions focused and under 50 lines when possible

2. **Shell Scripts**:
   - Use `set -e` to exit on errors
   - Add comments for complex operations
   - Quote variables to prevent word splitting

3. **Kubernetes Manifests**:
   - Include resource requests and limits
   - Add meaningful labels and annotations
   - Document environment variables in comments

### Git Workflow

1. **Branch Naming**:
   - Feature: `feature/description`
   - Bug fix: `fix/description`
   - Documentation: `docs/description`

2. **Commit Messages**:

   ```text
   <type>: <short summary> (max 50 chars)

   <detailed description if needed>

   - Bullet points for changes
   - Reference issues: Fixes #123
   ```

   Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

3. **Before Creating PR**:
   ```bash
   # Ensure you're on latest main
   git checkout main
   git pull origin main

   # Create feature branch
   git checkout -b feature/my-new-feature

   # Make changes and commit
   git add .
   git commit -m "feat: add support for new dataset"

   # Push to remote
   git push origin feature/my-new-feature
   ```

4. **PR Checklist**:
   - [ ] Code follows project style guidelines
   - [ ] All tests pass (local preprocessing tests)
   - [ ] Documentation updated (README.md if needed)
   - [ ] Environment variables documented
   - [ ] Smoke test validated on cluster (if applicable)
   - [ ] No secrets or credentials committed
   - [ ] Docker image builds successfully
   - [ ] Kubernetes manifests validated with `--dry-run`

5. **PR Description Template**:

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

### Code Review Guidelines

**For Reviewers**:

- Check for hardcoded values that should be environment variables
- Verify resource requests/limits are appropriate
- Ensure error handling is present
- Look for potential security issues (exposed secrets, etc.)

**For Authors**:

- Respond to all comments
- Update PR based on feedback
- Re-request review after changes
- Squash commits before merge (if requested)

## License

This project setup is provided as-is for educational and research purposes.

## Contributing

For issues or improvements, please file an issue or submit a pull request.

## Acknowledgments

- [ms-swift](https://github.com/modelscope/swift) - Model fine-tuning framework
- [Qwen3](https://huggingface.co/Qwen) - Base model
- [Alpaca GPT4](https://huggingface.co/datasets/AI-ModelScope/alpaca-gpt4-data-en) - Training dataset
