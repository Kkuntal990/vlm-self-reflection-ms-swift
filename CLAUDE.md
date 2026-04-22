# Claude Code Configuration - RuFlo V3

## Behavioral Rules (Always Enforced)

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
- NEVER save working files, text/mds, or tests to the root folder
- Never continuously check status after spawning a swarm — wait for results
- ALWAYS read a file before editing it
- NEVER commit secrets, credentials, or .env files

## File Organization

- NEVER save to root folder — use the directories below
- Use `/src` for source code files
- Use `/tests` for test files
- Use `/docs` for documentation and markdown files
- Use `/config` for configuration files
- Use `/scripts` for utility scripts
- Use `/examples` for example code

## Project Architecture

- Follow Domain-Driven Design with bounded contexts
- Keep files under 500 lines
- Use typed interfaces for all public APIs
- Prefer TDD London School (mock-first) for new code
- Use event sourcing for state changes
- Ensure input validation at system boundaries

### Project Config

- **Topology**: hierarchical-mesh
- **Max Agents**: 15
- **Memory**: hybrid
- **HNSW**: Enabled
- **Neural**: Enabled

## Build & Test

```bash
# Build
npm run build

# Test
npm test

# Lint
npm run lint
```

- ALWAYS run tests after making code changes
- ALWAYS verify build succeeds before committing

## Security Rules

- NEVER hardcode API keys, secrets, or credentials in source files
- NEVER commit .env files or any file containing secrets
- Always validate user input at system boundaries
- Always sanitize file paths to prevent directory traversal
- Run `npx @claude-flow/cli@latest security scan` after security-related changes

## Concurrency: 1 MESSAGE = ALL RELATED OPERATIONS

- All operations MUST be concurrent/parallel in a single message
- Use Claude Code's Task tool for spawning agents, not just MCP
- ALWAYS batch ALL todos in ONE TodoWrite call (5-10+ minimum)
- ALWAYS spawn ALL agents in ONE message with full instructions via Task tool
- ALWAYS batch ALL file reads/writes/edits in ONE message
- ALWAYS batch ALL Bash commands in ONE message

## Swarm Orchestration

- MUST initialize the swarm using CLI tools when starting complex tasks
- MUST spawn concurrent agents using Claude Code's Task tool
- Never use CLI tools alone for execution — Task tool agents do the actual work
- MUST call CLI tools AND Task tool in ONE message for complex work

### 3-Tier Model Routing (ADR-026)

| Tier | Handler | Latency | Cost | Use Cases |
|------|---------|---------|------|-----------|
| **1** | Agent Booster (WASM) | <1ms | $0 | Simple transforms (var→const, add types) — Skip LLM |
| **2** | Haiku | ~500ms | $0.0002 | Simple tasks, low complexity (<30%) |
| **3** | Sonnet/Opus | 2-5s | $0.003-0.015 | Complex reasoning, architecture, security (>30%) |

- Always check for `[AGENT_BOOSTER_AVAILABLE]` or `[TASK_MODEL_RECOMMENDATION]` before spawning agents
- Use Edit tool directly when `[AGENT_BOOSTER_AVAILABLE]`

## Swarm Configuration & Anti-Drift

- ALWAYS use hierarchical topology for coding swarms
- Keep maxAgents at 6-8 for tight coordination
- Use specialized strategy for clear role boundaries
- Use `raft` consensus for hive-mind (leader maintains authoritative state)
- Run frequent checkpoints via `post-task` hooks
- Keep shared memory namespace for all agents

```bash
npx @claude-flow/cli@latest swarm init --topology hierarchical --max-agents 8 --strategy specialized
```

## Swarm Execution Rules

- ALWAYS use `run_in_background: true` for all agent Task calls
- ALWAYS put ALL agent Task calls in ONE message for parallel execution
- After spawning, STOP — do NOT add more tool calls or check status
- Never poll TaskOutput or check swarm status — trust agents to return
- When agent results arrive, review ALL results before proceeding

## V3 CLI Commands

### Core Commands

| Command | Subcommands | Description |
|---------|-------------|-------------|
| `init` | 4 | Project initialization |
| `agent` | 8 | Agent lifecycle management |
| `swarm` | 6 | Multi-agent swarm coordination |
| `memory` | 11 | AgentDB memory with HNSW search |
| `task` | 6 | Task creation and lifecycle |
| `session` | 7 | Session state management |
| `hooks` | 17 | Self-learning hooks + 12 workers |
| `hive-mind` | 6 | Byzantine fault-tolerant consensus |

### Quick CLI Examples

```bash
npx @claude-flow/cli@latest init --wizard
npx @claude-flow/cli@latest agent spawn -t coder --name my-coder
npx @claude-flow/cli@latest swarm init --v3-mode
npx @claude-flow/cli@latest memory search --query "authentication patterns"
npx @claude-flow/cli@latest doctor --fix
```

## Available Agents (60+ Types)

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Specialized
`security-architect`, `security-auditor`, `memory-specialist`, `performance-engineer`

### Swarm Coordination
`hierarchical-coordinator`, `mesh-coordinator`, `adaptive-coordinator`

### GitHub & Repository
`pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager`

### SPARC Methodology
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`

## Memory Commands Reference

```bash
# Store (REQUIRED: --key, --value; OPTIONAL: --namespace, --ttl, --tags)
npx @claude-flow/cli@latest memory store --key "pattern-auth" --value "JWT with refresh" --namespace patterns

# Search (REQUIRED: --query; OPTIONAL: --namespace, --limit, --threshold)
npx @claude-flow/cli@latest memory search --query "authentication patterns"

# List (OPTIONAL: --namespace, --limit)
npx @claude-flow/cli@latest memory list --namespace patterns --limit 10

# Retrieve (REQUIRED: --key; OPTIONAL: --namespace)
npx @claude-flow/cli@latest memory retrieve --key "pattern-auth" --namespace patterns
```

## Quick Setup

```bash
claude mcp add claude-flow -- npx -y @claude-flow/cli@latest
npx @claude-flow/cli@latest daemon start
npx @claude-flow/cli@latest doctor --fix
```

## Claude Code vs CLI Tools

- Claude Code's Task tool handles ALL execution: agents, file ops, code generation, git
- CLI tools handle coordination via Bash: swarm init, memory, hooks, routing
- NEVER use CLI tools as a substitute for Task tool agents

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues

---

# VLM Self-Reflection - Developer Guide

## Overview

Production Kubernetes platform for VLM fine-tuning with **ms-swift**, **PyTorch DDP**, and **LoRA**.

**Training Paradigms**: FIRE behavior cloning (8-GPU), Standard SFT (2-GPU), Full fine-tuning
**Models**: Qwen2.5-VL-7B, Qwen3-8B, Qwen3-VL-32B
**Tech Stack**: ms-swift, PyTorch DDP, Kubernetes, LoRA, HuggingFace

## How to run the code ?

Always use 'uv' for running.

### Linting Commands

```bash
# Format all Python files
ruff format scripts/

# Lint all Python files
ruff check scripts/

# Lint and auto-fix
ruff check scripts/ --fix

# Type check (optional)
mypy scripts/
```

### MUST Follow

1. **Imports**: Standard library first, then third-party, then local. Alphabetized within groups.
2. **Type Hints**: Required on all function parameters and return types.
3. **Docstrings**: Google-style with Args/Returns sections.
4. **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants.
5. **Logging**: Use the standard logging pattern with `logging.basicConfig()` and `logger = logging.getLogger(__name__)`.
6. **Entry Points**: All scripts MUST use `if __name__ == "__main__": main()` guard.
7. **Dataclasses**: Use `@dataclass` for structured results with `to_dict()` methods.

### MUST NOT Change

1. **Lazy Imports in Model Classes**: Do NOT move imports from `__init__` methods to module level in model wrapper classes (e.g., `SkyworkVLRewardScorer`, `VLMInferenceEngine`). This is intentional for performance.
2. **sys.path.insert Pattern**: Do NOT change local import pattern in scripts directory.

### Detailed Rules

See `.claude/rules/` for comprehensive style guides:

- [.claude/rules/python-style.md](.claude/rules/python-style.md) - Core Python conventions
- [.claude/rules/ml-patterns.md](.claude/rules/ml-patterns.md) - ML-specific patterns (dataclasses, model loading, etc.)

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

## Repository Layout

Each file's one-line purpose. See the file's top-of-module docstring for full details.

### `scripts/data_prep/` — dataset preprocessing

| File | Purpose |
| ---- | ------- |
| `analyze_dataset_lengths.py` | Token-length statistics for FIRE ShareGPT to pick `max_length`. |
| `categorize_fire_samples.py` | Classify FIRE samples by answer type (MCQ / yes-no / open). |
| `prepare_fire_feedback_with_thought.py` | Inject original Thought+Answer into FIRE feedback user turns. |
| `prepare_fire_with_mappings.py` | Build FIRE ShareGPT JSONL with local image-path mappings. |
| `prepare_fire_with_thought.py` | FIRE ShareGPT JSONL preserving Thought+Answer in assistant turns. |
| `prepare_grpo_mcq_yesno.py` | Build MCQ/Yes-No GRPO training set from PixelReasoner/MME-CoT/RLAIF-V/VISCO. |
| `prepare_lvlm_nlf.py` | Convert LVLM_NLF into ms-swift messages format. |
| `prepare_volcano_sharegpt.py` | Convert Volcano dataset into ShareGPT format. |
| `prepare_vqa_mix.py` | Build a balanced VQA mix with format-specific system prompts. |
| `verify_fire_images.py` | Check that referenced FIRE images exist on the pod. |

### `scripts/training/` — training entry points

| File | Purpose |
| ---- | ------- |
| `env.sh` | Shared training environment variables (paths, tokens, NCCL). |
| `training_env.sh` | Alternate env file used by some job yamls. |
| `run_full_sft_qwen3vl_fire_8gpu.sh` | 8-GPU full fine-tune of Qwen VL on FIRE. |
| `run_full_sft_qwen3vl_fire_multinode.sh` | Multinode variant of the 8-GPU full SFT. |
| `run_lora_sft_qwen25vl.sh` | LoRA SFT for Qwen2.5-VL. |
| `run_sft_single_gpu.sh` | Single-GPU SFT for local smoke tests. |

### `scripts/evaluation/` — inference, judges, benchmark runners

| File | Purpose |
| ---- | ------- |
| `run_vlm_benchmarks.sh` | Top-level driver for VLMEvalKit benchmarks (BLINK, HallusionBench, …). |
| `run_llava_ov_benchmarks.sh` | Driver for LLaVA-OneVision lmms-eval benchmarks. |
| `prepare_blink_vlmevalkit.py` | Convert BLINK HF dataset into VLMEvalKit TSV format. |
| `aggregate_blink_results.py` | Aggregate per-subtask BLINK CSVs into a summary. |
| `analyze_blink_turns.py` | Per-turn accuracy analysis of BLINK self-reflective logs. |
| `register_vlmevalkit_model.py` | Register a fine-tuned checkpoint in VLMEvalKit config. |
| `qwen2vl_self_reflective.py` | VLMEvalKit wrapper for Qwen2.5-VL self-reflective inference. |
| `llava_hf_wrapper.py` | VLMEvalKit wrapper for HF-format LLaVA-1.5 checkpoints. |
| `self_reflective_inference_v2.py` | Multi-turn self-reflective inference loop with natural roles. |
| `generate_refinements.py` | Generate self-refinement responses from a fine-tuned VLM. |
| `evaluate_self_refinement.py` | Score refinements via reward model and/or VLM judge. |
| `evaluate_answer_matching.py` | Extract + match final answers against ground truth. |
| `analyze_refinement_metrics.py` | Compute and plot self-refinement metrics. |
| `compare_benchmark_results.py` | Cross-framework / cross-model benchmark comparison. |
| `score_with_reward_model.py` | Score responses with Skywork-VL-Reward-7B. |
| `vlm_judge.py` | Generic VLM-as-judge interface used by evaluations. |
| `judges/llava_critic_r1.py` | LLaVA-Critic-R1 judge implementation. |
| `eval_llava_ov_lmms.py` | LLaVA-OneVision evaluation via lmms-eval + HF transformers. |
| `run_lmms_eval_patched.py` | lmms-eval launcher with LLaVA-NeXT compatibility patches. |
| `verify_dataset_alignment.py` | Spot-check dataset/image alignment against ground truth. |

### `scripts/run_inference.py`

ms-swift-compatible inference entry point for VLM self-refinement.

### `k8s/` — Kubernetes manifests

- `pvc-*.yaml`, `secret-hf-token.yaml.template` — cluster storage and secrets.
- `pytorchjob-full-sft-qwen2-5vl-fire-8gpu.yaml`, `job-full-*.yaml`, `job-lora-*.yaml` — training jobs (full SFT / LoRA, single- and multi-turn variants).
- `job-prepare-grpo-mcq-yesno.yaml`, `job-upload-fire-images-hf.yaml` — data prep jobs.
- `job-eval-vlmevalkit-*.yaml` — VLMEvalKit eval jobs (base, self-reflective, GRPO, BLINK, hallu, thought, LLaVA). One file per experiment variant.
- `job-eval-lmms-eval.yaml`, `job-eval-llava-ov-lmms.yaml` — lmms-eval jobs.
- `job-evaluate-self-refinement*.yaml`, `job-run-inference.yaml` — self-refinement inference / evaluation jobs.
- `jupyter-1gpu-test.yaml`, `pod-temp-cpu.yaml` — interactive dev pods.

### `docs/`

`DATASETS.md`, `DOCKER_BUILD.md`, `EVALUATION.md`, `EVAL_ANSWER_MATCHING.md`, `GHCR_SETUP.md`, `K8S_DEPLOY.md`, `QUICKSTART.md` — topic-specific guides.

### `eval_ground_truth/`

Reference evaluation outputs (`aggregate_metrics.json`, `evaluation_report.txt`, `sample_results.jsonl`) used for regression comparison.

### Container & build

| File | Purpose |
| ---- | ------- |
| `Dockerfile.base` | Base image (CUDA 12.1, PyTorch 2.2.0). |
| `Dockerfile` | App image on top of base (ms-swift + scripts). |
| `build_base.sh` | Build and push the base image. |
| `build_main.sh` | Build and push the app image. |
| `pyproject.toml`, `requirements.txt`, `uv.lock` | Python deps (uv-managed). |
| `ruff.toml` | Lint/format config. |

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
python scripts/data_prep/prepare_fire_sharegpt.py \
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
- [docs/DATASETS.md](docs/DATASETS.md) - Dataset statistics and preprocessing
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - Quick deployment guide
- [docs/DOCKER_BUILD.md](docs/DOCKER_BUILD.md) - Docker build system
- [docs/GHCR_SETUP.md](docs/GHCR_SETUP.md) - GitHub Container Registry setup
- [ms-swift](https://github.com/modelscope/swift) - Framework reference
- [FIRE dataset](https://huggingface.co/datasets/PengxiangLi/FIRE) - Behavior cloning data

## Environment

- **Conda environment**: `vlm-self-reflection-swift`
- **Storage**: `/cache` (models/datasets), `/outputs` (checkpoints)
- **Container registry**: `ghcr.io/kkuntal990` (GHCR default)
