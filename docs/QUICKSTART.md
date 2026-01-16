# Quick Start Guide

## What Was Created

A complete Kubernetes-based 2-GPU DDP training setup with:

```
✓ Dockerfile (CUDA 12.1 + ms-swift)
✓ requirements.txt (all Python dependencies)
✓ .dockerignore (optimized build)
✓ scripts/env.sh (environment configuration)
✓ scripts/run_sft_ddp.sh (main training script)
✓ k8s/pvc-cache.yaml (rook-ceph-block, 300Gi)
✓ k8s/pvc-outputs.yaml (rook-cephfs, 300Gi)
✓ k8s/job-sft-qwen3-8b-2gpu.yaml (2-GPU Job)
✓ k8s/secret-hf-token.yaml.template (HF token template)
✓ README.md (comprehensive documentation)
```

All YAML files validated ✓
All shell scripts validated ✓
Executable permissions set ✓

## Immediate Next Steps

### 1. Update Docker Image Reference

Edit `k8s/job-sft-qwen3-8b-2gpu.yaml` line 42:

```yaml
# Change this:
image: <your-dockerhub-username>/ms-swift-qwen:latest

# To your actual Docker Hub username:
image: yourname/ms-swift-qwen:latest
```

### 2. Build and Push Container

```bash
# Build the image
docker build -t yourname/ms-swift-qwen:latest .

# Push to Docker Hub (requires login)
docker login
docker push yourname/ms-swift-qwen:latest
```

### 3. Deploy to Kubernetes

```bash
# Create storage
kubectl apply -f k8s/pvc-cache.yaml
kubectl apply -f k8s/pvc-outputs.yaml

# Wait for PVCs to bind
kubectl get pvc -w

# (Optional) Create HF token secret
kubectl create secret generic hf-token --from-literal=token="hf_your_token"

# Launch smoke test (500 samples, ~5-15 minutes)
kubectl apply -f k8s/job-sft-qwen3-8b-2gpu.yaml

# Monitor progress
kubectl logs -f job/qwen3-8b-sft-job
```

### 4. After Smoke Test Succeeds

Edit `k8s/job-sft-qwen3-8b-2gpu.yaml`:

```yaml
# Change dataset from 500 samples to full dataset
- name: DATASET_ID
  value: "AI-ModelScope/alpaca-gpt4-data-en"  # Remove #500

# Optional: increase epochs
- name: EPOCHS
  value: "3"
```

Then redeploy:

```bash
kubectl delete job qwen3-8b-sft-job
kubectl apply -f k8s/job-sft-qwen3-8b-2gpu.yaml
```

## Key Configuration Points

### Storage Strategy
- **Cache PVC** (rook-ceph-block): HF models, datasets, pip cache
- **Outputs PVC** (rook-cephfs): Checkpoints, model weights

### Default Training Parameters
- Model: Qwen/Qwen3-8B
- Dataset: 500 samples (smoke test) → full Alpaca GPT4
- GPUs: 2 (single node DDP)
- Batch size per GPU: 1
- Gradient accumulation: 16
- Effective batch size: 32
- LoRA rank: 8
- Max length: 2048
- Auto-dtype: bf16 (if supported) or fp16

### Expected Timeline
- **Smoke test** (500 samples): 5-15 minutes
- **Full dataset** (~52K samples): 2-4 hours (varies by GPU)

## Troubleshooting Quick Reference

```bash
# Check GPU visibility
kubectl exec -it <pod-name> -- nvidia-smi

# Check PyTorch GPU detection
kubectl exec -it <pod-name> -- python -c "import torch; print(torch.cuda.device_count())"

# Check PVC status
kubectl get pvc

# Check job status
kubectl describe job qwen3-8b-sft-job

# Get pod name
POD=$(kubectl get pods -l job-name=qwen3-8b-sft-job -o jsonpath='{.items[0].metadata.name}')

# View logs
kubectl logs -f $POD

# Access outputs
kubectl exec -it $POD -- ls -lh /outputs/qwen3-8b-sft/
```

## Common Issues

### OOM (Out of Memory)
Reduce memory usage in `k8s/job-sft-qwen3-8b-2gpu.yaml`:
```yaml
- name: MAX_LEN
  value: "1024"  # Reduce from 2048
- name: LORA_RANK
  value: "4"     # Reduce from 8
```

### DDP Hangs
Enable debug logging:
```yaml
- name: NCCL_DEBUG
  value: "INFO"  # Change from WARN
```

### Image Pull Failed
Check that your image is public or add image pull secret:
```bash
kubectl create secret docker-registry regcred \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=yourname \
  --docker-password=yourpassword
```

Then add to Job spec:
```yaml
imagePullSecrets:
- name: regcred
```

## Files to Customize for Your Use Case

1. **k8s/job-sft-qwen3-8b-2gpu.yaml**
   - Line 42: Docker image name
   - Lines 66-90: Environment variables (dataset, hyperparameters)
   - Lines 92-99: Resource requests/limits (adjust for your GPU type)

2. **scripts/run_sft_ddp.sh**
   - Only if you need custom ms-swift arguments

3. **k8s/pvc-*.yaml**
   - Storage sizes based on your needs
   - Storage classes if different from rook-ceph-*

## Validation Commands

```bash
# Verify all files exist
ls -lh Dockerfile requirements.txt .dockerignore
ls -lh scripts/*.sh
ls -lh k8s/*.yaml

# Verify script permissions
ls -l scripts/*.sh | grep rwx

# Validate YAML syntax
kubectl apply --dry-run=client -f k8s/
```

## What's Next?

1. **Build the container** with your Docker Hub username
2. **Deploy PVCs** and wait for them to bind
3. **Run smoke test** (500 samples) to validate setup
4. **Scale to full dataset** after smoke test succeeds
5. **Monitor and iterate** based on your results

See README.md for comprehensive documentation and advanced configuration options.
