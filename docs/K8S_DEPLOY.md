# Deploying Jobs on Kubernetes

## Prerequisites

Ensure storage and secrets are set up (one-time):

```bash
kubectl apply -f k8s/pvc-cache.yaml
kubectl apply -f k8s/pvc-outputs.yaml
kubectl create secret generic hf-token --from-literal=token="hf_..."
```

Verify:

```bash
kubectl get pvc
kubectl get secret hf-token
```

## Available Jobs

### Training

| Job File | Model | GPUs | Description |
|----------|-------|------|-------------|
| `job-full-qwen-multi-turn.yaml` | Qwen2.5-VL-7B | 4x A100 | Multi-turn feedback training |
| `job-full-llava-single-turn.yaml` | LLaVA-1.5-7B | 2x A100 | Single-turn feedback training |
| `job-full-llava-multi-turn.yaml` | LLaVA-1.5-7B | 2x A100 | Multi-turn feedback training |
| `job-full-llava-multi-and-single-turn.yaml` | LLaVA-1.5-7B | 2x A100 | Combined multi+single turn |
| `job-full-llava-multi-and-single-turn-feedbackv3.yaml` | LLaVA-1.5-7B | 2x A100 | Feedback v3 data |

### Evaluation

| Job File | Framework | GPUs | Description |
|----------|-----------|------|-------------|
| `job-eval-lmms-eval.yaml` | lmms-eval | 1x RTX A6000 | Qwen benchmark eval |
| `job-eval-llava-ov-lmms.yaml` | lmms-eval | 1x RTX A6000 | LLaVA-OV benchmark eval |
| `job-eval-vlmevalkit.yaml` | VLMEvalKit | 1x RTX A6000 | VLMEvalKit benchmarks |
| `job-evaluate-self-refinement.yaml` | Custom | 1x RTX A6000 | Self-reflective inference |
| `job-evaluate-self-refinement-multigpu.yaml` | Custom | Multi-GPU | Multi-GPU self-reflective inference |

### Inference

| Job File | GPUs | Description |
|----------|------|-------------|
| `job-run-inference.yaml` | 1x RTX A6000 | General VLM inference pipeline |

## Workflow

### 1. Configure the Job

Edit the YAML to set your env vars. Key fields to update per job type:

**Training jobs** — edit `env` section:

```yaml
- name: MODEL_ID
  value: "Qwen/Qwen2.5-VL-7B-Instruct"    # Base model
- name: DATASET_PATH
  value: "/outputs/fire_preprocessed_v3/fire_feedback_train_v2.jsonl"
- name: NPROC
  value: "4"                                 # Must match GPU count in resources
- name: BATCH
  value: "4"
- name: GRAD_ACC
  value: "8"
- name: LR
  value: "3e-6"
- name: RUN_NAME
  value: "my-training-run"
- name: RESUME_PATH                          # Set to "" for fresh start
  value: ""
```

**Evaluation jobs** — edit `env` section:

```yaml
- name: MODEL_PATH
  value: "/outputs/<run-name>/<version>/checkpoint-XXXX"
- name: OUTPUT_DIR
  value: "/outputs/benchmark_results/<eval-name>"
- name: BENCHMARKS
  value: "ai2d,mme,mmmu_val,ocrbench,seedbench"  # or "all"
- name: LIMIT
  value: "0"                                        # 0 = full eval
```

**Inference/self-refinement jobs** — edit `env` section:

```yaml
- name: MODEL_PATH
  value: "/outputs/<run-name>/<version>/checkpoint-XXXX"
- name: DATASET_PATH
  value: "/outputs/fire_preprocessed_v3/fire_messages_test.jsonl"
- name: OUTPUT_PATH
  value: "/outputs/inference_v4/my_results.jsonl"
- name: NUM_TURNS
  value: "2"
```

### 2. Deploy

```bash
kubectl apply -f k8s/<job-file>.yaml
```

### 3. Monitor

```bash
# Watch logs
kubectl logs -f job/<job-name>

# Check job status
kubectl get jobs
kubectl describe job <job-name>

# Get pod name
kubectl get pods -l job-name=<job-name>

# GPU check inside pod
kubectl exec -it <pod-name> -- nvidia-smi
```

### 4. Get Results

```bash
# List outputs inside the pod
kubectl exec -it <pod-name> -- ls -lh /outputs/

# Copy results locally
kubectl cp <pod-name>:/outputs/<result-path> ./local-results/
```

### 5. Clean Up

```bash
kubectl delete job <job-name>
```

## Rerunning a Job

Jobs must have unique names. To rerun, delete the old job first:

```bash
kubectl delete job <job-name>
kubectl apply -f k8s/<job-file>.yaml
```

Or change `metadata.name` in the YAML to a new name.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM | Reduce `BATCH` or `MAX_LEN`, increase `GRAD_ACC` to compensate |
| DDP hangs | Set `NCCL_DEBUG=INFO`, check GPUs on same node with `kubectl get pods -o wide` |
| Pod pending | Check GPU availability: `kubectl describe pod <pod-name>` (look at Events) |
| Dataset not found | Verify path exists: `kubectl exec -it <pod-name> -- ls /outputs/` |
| Job keeps restarting | Check `backoffLimit` and logs from previous attempt: `kubectl logs <pod-name> --previous` |
