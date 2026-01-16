# Self-Refinement Evaluation Guide

This guide explains how to evaluate your fine-tuned VLM's self-refinement capability using reward model scoring.

## Overview

The evaluation pipeline measures whether model responses **improve across multiple refinement turns**. It uses [Skywork-VL-Reward-7B](https://huggingface.co/Skywork/Skywork-VL-Reward-7B) to score each response and tracks improvement metrics.

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Reward Delta** | `(final_score - initial_score) / \|initial_score\|` - Percentage improvement |
| **Monotonic Improvement Rate** | % of samples where scores strictly increase each turn |
| **Average Improvement per Turn** | Mean score gain per refinement iteration |
| **Turns to Plateau** | Number of turns until improvement becomes negligible |
| **Improvement Rate** | % of samples where final score > initial score |

### Evaluation Modes

1. **Ground Truth Mode**: Score existing responses from the FIRE dataset
   - Validates the reward model works correctly
   - Establishes baseline improvement metrics
   - Does not require your fine-tuned model

2. **Generated Mode**: Generate new responses with your fine-tuned model, then score
   - Tests your model's actual self-refinement capability
   - Uses ground truth feedback from dataset to prompt refinements
   - Requires your fine-tuned model checkpoint

---

## Quick Start

### Prerequisites

- Kubernetes cluster access with GPU nodes
- PVCs set up (`ms-swift-cache-pvc`, `ms-swift-outputs-pvc`)
- HuggingFace token secret (`hf-token`)
- Preprocessed FIRE test dataset at `/outputs/fire_preprocessed_v2/fire_sharegpt_test.jsonl`

### Option A: Interactive Testing (Jupyter Pod)

Best for initial testing and debugging.

```bash
# 1. Deploy Jupyter pod
kubectl apply -f k8s/jupyter-1gpu-test.yaml

# 2. Wait for pod to be ready
kubectl get pods -w

# 3. Port forward
kubectl port-forward vlm-jupyter-eval 8888:8888

# 4. Access Jupyter at http://localhost:8888 (token: selfimprove)
```

### Option B: Batch Evaluation (Job)

Best for full-scale evaluation.

```bash
# 1. Update MODEL_PATH in the job manifest
# Edit k8s/job-evaluate-self-refinement.yaml

# 2. Deploy evaluation job
kubectl apply -f k8s/job-evaluate-self-refinement.yaml

# 3. Monitor progress
kubectl logs -f job/self-refinement-evaluation-job
```

---

## Detailed Instructions

### 1. Interactive Testing with Jupyter

#### Step 1: Deploy the Jupyter Pod

```bash
kubectl apply -f k8s/jupyter-1gpu-test.yaml
```

#### Step 2: Access the Pod

```bash
# Port forward
kubectl port-forward vlm-jupyter-eval 8888:8888

# Or exec into the pod
kubectl exec -it vlm-jupyter-eval -- bash
```

#### Step 3: Verify Dependencies

The evaluation dependencies (`trl`, `safetensors`, `matplotlib`, `seaborn`) are included in the Docker image via `requirements.txt`.

If using an older image, install them manually:

```bash
# Only needed if using older Docker image
pip install trl safetensors matplotlib seaborn
```

#### Step 4: Verify Environment

```python
# Check GPU
!nvidia-smi

# Check dataset exists
!ls -lh /outputs/fire_preprocessed_v2/
!wc -l /outputs/fire_preprocessed_v2/fire_sharegpt_test.jsonl

# Check model checkpoint (update path as needed)
!ls -lh /outputs/qwen2_5vl-7b-fire-full-sft/ 2>/dev/null || echo "Update MODEL_PATH"
```

---

### 2. Ground Truth Evaluation

Scores the existing FIRE dataset responses to validate the reward model.

#### From Terminal

```bash
python /workspace/scripts/evaluate_self_refinement.py \
    --dataset_path /outputs/fire_preprocessed_v2/fire_sharegpt_test.jsonl \
    --output_dir /outputs/eval_ground_truth \
    --mode ground_truth \
    --reward_model_id Skywork/Skywork-VL-Reward-7B \
    --max_samples 10  # Start small, set to 0 for all
```

#### From Jupyter Notebook

```python
import sys
sys.path.insert(0, '/workspace/scripts')

from score_with_reward_model import SkyworkVLRewardScorer
import json

# Initialize reward model
scorer = SkyworkVLRewardScorer(
    model_id="Skywork/Skywork-VL-Reward-7B",
    device="cuda",
    use_flash_attn=True
)

# Load a sample
with open("/outputs/fire_preprocessed_v2/fire_sharegpt_test.jsonl") as f:
    sample = json.loads(f.readline())

# Score all turns
result = scorer.score_conversation_turns(sample, return_details=True)

# Print results
print(f"Number of turns: {result['metrics']['num_turns']}")
print(f"Initial score: {result['metrics']['initial_score']:.2f}")
print(f"Final score: {result['metrics']['final_score']:.2f}")
print(f"Reward delta: {result['metrics']['reward_delta']*100:.1f}%")
print(f"Monotonic: {result['metrics']['is_monotonic']}")

for turn in result['turns']:
    print(f"\nTurn {turn['turn_index']}: {turn['reward_score']:.2f}")
```

#### Expected Output

```
Sample test_000001:
  Turn 0: 12.5
  Turn 1: 15.8 (+3.3)
  Turn 2: 18.2 (+2.4)
  Monotonic: Yes
  Delta: +45.6%
```

---

### 3. Generated Mode Evaluation

Tests your fine-tuned model's self-refinement capability.

#### Prerequisites

- Your fine-tuned model checkpoint (e.g., `/outputs/qwen2_5vl-7b-fire-full-sft/checkpoint-final`)

#### From Terminal

```bash
python /workspace/scripts/evaluate_self_refinement.py \
    --dataset_path /outputs/fire_preprocessed_v2/fire_sharegpt_test.jsonl \
    --model_path /outputs/qwen2_5vl-7b-fire-full-sft/checkpoint-final \
    --output_dir /outputs/eval_generated \
    --mode generated \
    --reward_model_id Skywork/Skywork-VL-Reward-7B \
    --max_samples 10 \
    --max_turns 3 \
    --temperature 0.7 \
    --top_p 0.9
```

#### From Jupyter Notebook

```python
import sys
sys.path.insert(0, '/workspace/scripts')

from generate_refinements import VLMInferenceEngine, generate_refinement_dialogue
from score_with_reward_model import SkyworkVLRewardScorer
import json

# Load models
print("Loading fine-tuned model...")
engine = VLMInferenceEngine(
    model_path="/outputs/qwen2_5vl-7b-fire-full-sft/checkpoint-final",
    device="cuda"
)

print("Loading reward model...")
scorer = SkyworkVLRewardScorer(
    model_id="Skywork/Skywork-VL-Reward-7B",
    device="cuda"
)

# Load test sample
with open("/outputs/fire_preprocessed_v2/fire_sharegpt_test.jsonl") as f:
    sample = json.loads(f.readline())

# Generate refinement dialogue
gen_result = generate_refinement_dialogue(
    engine=engine,
    sample=sample,
    max_turns=3,
    use_gt_feedback=True,
    generation_config={"temperature": 0.7, "top_p": 0.9}
)

# Convert to scoring format
scoring_sample = {
    "conversation": [
        {"human": turn["human"], "assistant": turn["assistant"]}
        for turn in gen_result["generated_conversation"]
    ],
    "images": gen_result["images"]
}

# Score generated responses
result = scorer.score_conversation_turns(scoring_sample, return_details=True)

# Print comparison
print("\n" + "="*60)
print("GENERATED RESPONSE EVALUATION")
print("="*60)
print(f"Turns generated: {len(gen_result['generated_conversation'])}")
print(f"Initial score: {result['metrics']['initial_score']:.2f}")
print(f"Final score: {result['metrics']['final_score']:.2f}")
print(f"Reward delta: {result['metrics']['reward_delta']*100:.1f}%")
print(f"Monotonic improvement: {result['metrics']['is_monotonic']}")
```

---

### 4. Analyzing Results

After running evaluation, analyze the results:

#### From Terminal

```bash
python /workspace/scripts/analyze_refinement_metrics.py \
    --results_path /outputs/eval_generated/sample_results.jsonl \
    --output_dir /outputs/eval_generated/analysis \
    --n_extreme 10
```

#### From Jupyter Notebook

```python
import sys
sys.path.insert(0, '/workspace/scripts')

from analyze_refinement_metrics import (
    load_results,
    analyze_by_turn,
    analyze_improvement_distribution,
    find_extreme_samples,
    create_visualizations
)
from pathlib import Path

# Load results
results = load_results("/outputs/eval_generated/sample_results.jsonl")

# Per-turn analysis
turn_analyses = analyze_by_turn(results)
for ta in turn_analyses:
    print(f"Turn {ta.turn_index}: score={ta.mean_score:.2f}, improvement={ta.mean_improvement:.2f}")

# Improvement distribution
dist = analyze_improvement_distribution(results)
print(f"\nImproved: {dist['improved_pct']:.1f}%")
print(f"Degraded: {dist['degraded_pct']:.1f}%")

# Find best/worst samples
best, worst = find_extreme_samples(results, n=5)
print("\nBest samples:")
for s in best:
    print(f"  {s['sample_id']}: delta={s['reward_delta']*100:.1f}%")

# Create visualizations
output_dir = Path("/outputs/eval_generated/analysis/plots")
create_visualizations(results, turn_analyses, output_dir)
print(f"\nPlots saved to {output_dir}")
```

---

### 5. Comparing Evaluations

Compare ground truth vs generated, or different model checkpoints:

```bash
python /workspace/scripts/analyze_refinement_metrics.py \
    --results_path /outputs/eval_ground_truth/sample_results.jsonl \
    --compare_path /outputs/eval_generated/sample_results.jsonl \
    --output_dir /outputs/comparison_analysis
```

---

## Full-Scale Evaluation (Kubernetes Job)

For evaluating the complete test set:

### Step 1: Update Job Configuration

Edit `k8s/job-evaluate-self-refinement.yaml`:

```yaml
env:
  - name: MODEL_PATH
    value: "/outputs/your-checkpoint-path"  # UPDATE THIS
  - name: MAX_SAMPLES
    value: "0"  # 0 = evaluate all samples
  - name: EVAL_MODE
    value: "generated"  # or "ground_truth"
```

### Step 2: Deploy Job

```bash
kubectl apply -f k8s/job-evaluate-self-refinement.yaml
```

### Step 3: Monitor Progress

```bash
# Watch logs
kubectl logs -f job/self-refinement-evaluation-job

# Check status
kubectl get jobs
kubectl describe job self-refinement-evaluation-job
```

### Step 4: Retrieve Results

```bash
# Get pod name
POD=$(kubectl get pods -l job-name=self-refinement-evaluation-job -o jsonpath='{.items[0].metadata.name}')

# Copy results locally
kubectl cp $POD:/outputs/evaluation_results ./local_eval_results

# View report
cat ./local_eval_results/evaluation_report.txt
```

---

## Output Files

After evaluation, you'll find these files in the output directory:

| File | Description |
|------|-------------|
| `sample_results.jsonl` | Per-sample results with all turn scores |
| `aggregate_metrics.json` | Summary statistics |
| `evaluation_report.txt` | Human-readable report |
| `analysis/detailed_analysis.json` | Turn-by-turn statistics |
| `analysis/detailed_analysis_report.txt` | Detailed analysis report |
| `analysis/plots/*.png` | Visualization charts |

---

## Interpreting Results

### Good Self-Refinement

```
Reward Delta: +30-50%
Monotonic Improvement Rate: >70%
Improvement Rate: >80%
```

This indicates the model effectively improves responses with feedback.

### Poor Self-Refinement

```
Reward Delta: <10% or negative
Monotonic Improvement Rate: <50%
Improvement Rate: <60%
```

This suggests the model may:
- Not be learning from feedback
- Be generating degraded responses
- Need more training or different hyperparameters

### Key Questions to Answer

1. **Does the model improve at all?** Check `improvement_rate`
2. **Is improvement consistent?** Check `monotonic_improvement_rate`
3. **How much does it improve?** Check `reward_delta`
4. **When does improvement stop?** Check `turns_to_plateau`

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size or use sequential scoring
--max_samples 5  # Test with fewer samples first
```

For generated mode, reduce `max_new_tokens`:
```bash
--max_new_tokens 256
```

### Model Not Found

```bash
# Check available checkpoints
ls -lh /outputs/

# Use base model for testing
--model_path Qwen/Qwen2.5-VL-7B-Instruct
```

### Dataset Not Found

```bash
# Check dataset location
ls -lh /outputs/fire_preprocessed_v2/

# Use train set for testing if test set unavailable
--dataset_path /outputs/fire_preprocessed_v2/fire_sharegpt_train.jsonl
```

### Flash Attention Errors

```bash
# Disable flash attention
--no_flash_attn
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_MODE` | `generated` | Evaluation mode: `ground_truth` or `generated` |
| `DATASET_PATH` | `/outputs/fire_preprocessed_v2/fire_sharegpt_test.jsonl` | Test dataset path |
| `MODEL_PATH` | - | Fine-tuned model checkpoint (required for generated mode) |
| `REWARD_MODEL_ID` | `Skywork/Skywork-VL-Reward-7B` | Reward model HuggingFace ID |
| `MAX_SAMPLES` | `0` | Max samples to evaluate (0 = all) |
| `MAX_TURNS` | `3` | Max refinement turns (generated mode) |
| `TEMPERATURE` | `0.7` | Generation temperature |
| `TOP_P` | `0.9` | Nucleus sampling probability |
| `MAX_NEW_TOKENS` | `512` | Max tokens per response |
| `PLATEAU_THRESHOLD` | `0.5` | Min improvement for plateau detection |

---

## References

- [Skywork-VL-Reward-7B](https://huggingface.co/Skywork/Skywork-VL-Reward-7B) - Multimodal reward model
- [arXiv 2502.05605](https://arxiv.org/abs/2502.05605) - Self-refinement evaluation methodology
- [VL-RewardBench](https://vl-rewardbench.github.io/) - Benchmark for VL reward models
- [FIRE Dataset](https://huggingface.co/datasets/PengxiangLi/FIRE) - Feedback-based self-refinement data
