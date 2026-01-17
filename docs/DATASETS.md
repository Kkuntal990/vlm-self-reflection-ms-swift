# Dataset Documentation

This document provides comprehensive information about the datasets, preprocessing pipelines, and data files used in the VLM Self-Reflection project.

## Table of Contents

1. [Overview](#overview)
2. [Current Dataset Statistics](#current-dataset-statistics)
3. [Data Files](#data-files)
4. [Image Mapping System](#image-mapping-system)
5. [Preprocessing Scripts](#preprocessing-scripts)
6. [Source Datasets](#source-datasets)
7. [Known Issues](#known-issues)
8. [Usage Examples](#usage-examples)

---

## Overview

The VLM Self-Reflection project uses the **FIRE dataset** (Feedback for Iterative Refinement and Evaluation) as its primary training data source. The FIRE dataset contains multi-round student-teacher conversations where a vision-language model (student) receives feedback from a teacher model and iteratively refines its responses.

**HuggingFace Dataset**: [PengxiangLi/FIRE](https://huggingface.co/datasets/PengxiangLi/FIRE)

### Data Pipeline

```
FIRE Dataset (HuggingFace)
         │
         ▼
┌─────────────────────────────┐
│  Image Mapping Pipeline     │
│  - HuggingFace datasets     │
│  - Local images             │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Preprocessing Pipeline     │
│  - ShareGPT format          │
│  - Multi-round conversations│
└─────────────────────────────┘
         │
         ▼
  Training-ready JSONL files
```

---

## Current Dataset Statistics

### Version 2 (Latest) - `data/fire_preprocessed_v2/`

| Split | Samples Processed | Samples Skipped | Coverage | Total Rounds |
|-------|-------------------|-----------------|----------|--------------|
| **Train** | 88,608 | 16,333 | **84.4%** | 232,652 |
| **Test** | 3,839 | 7,167 | **34.9%** | 9,095 |

#### Rounds Breakdown (Train - Processed)

| Rounds | Count |
|--------|-------|
| 2 rounds | 39,048 |
| 3 rounds | 44,036 |
| 4 rounds | 5,215 |
| 5 rounds | 265 |
| 6 rounds | 44 |

#### Rounds Breakdown (Test - Processed)

| Rounds | Count |
|--------|-------|
| 2 rounds | 2,589 |
| 3 rounds | 1,107 |
| 4 rounds | 123 |
| 5 rounds | 16 |
| 6 rounds | 4 |

### Version 1 - `data/fire_preprocessed/`

| Split | Samples Processed | Samples Skipped | Coverage |
|-------|-------------------|-----------------|----------|
| **Train** | 85,244 | 19,697 | 81.2% |
| **Test** | 3,839 | 7,167 | 34.9% |

---

## Data Files

### Preprocessed Datasets

| File | Location | Lines | Description |
|------|----------|-------|-------------|
| `fire_sharegpt_train.jsonl` | `data/fire_preprocessed_v2/` | 88,608 | Training data (ShareGPT format) |
| `fire_sharegpt_test.jsonl` | `data/fire_preprocessed_v2/` | 3,839 | Test data (ShareGPT format) |
| `stats.json` | `data/fire_preprocessed_v2/` | - | Preprocessing statistics |

### Image Mapping Files

| File | Size | Mappings | Description |
|------|------|----------|-------------|
| `fire_image_mapping.json` | 16 MB | 78,964 | HuggingFace dataset mappings |
| `local_image_mapping.json` | 4.8 MB | 22,510 | Local file mappings |
| `fire_image_mapping_complete.json` | 16 MB | 81,217 | Combined mappings |

### Other Data Files

| File | Description |
|------|-------------|
| `fire_conversations_by_rounds.json` | Analysis of conversations grouped by round count |

---

## Image Mapping System

The project uses a two-tier image mapping system to locate images from various sources:

### 1. HuggingFace Dataset Mappings (`fire_image_mapping.json`)

Maps FIRE image paths to HuggingFace dataset indices for streaming/loading.

**Supported Datasets**:

| Dataset | HuggingFace ID | Mappings |
|---------|----------------|----------|
| ALLaVA-4V | `FreedomIntelligence/ALLaVA-4V` | 20,257 |
| COCO | `detection-datasets/coco` | 17,057 |
| GQA | `lmms-lab/GQA` | 12,597 |
| Visual Genome | `visual_genome` | 8,849 |
| TextVQA | `lmms-lab/textvqa` | 6,120 |
| OCR-VQA | `howard-hou/OCR-VQA` | 5,222 |
| ChartQA | `ahmed-masry/ChartQA` | 3,768 |
| SynthDog-EN | `naver-clova-ix/synthdog-en` | 2,791 |
| ScienceQA | `derek-thomas/ScienceQA` | 942 |
| MathVista | `AI4Math/MathVista` | 820 |
| MME | `lmms-lab/MME` | 541 |

### 2. Local Image Mappings (`local_image_mapping.json`)

Maps FIRE image paths to locally downloaded files.

| Source | Local Directory | Mappings |
|--------|-----------------|----------|
| Vision-FLAN | `/outputs/images_191task_1k` | 20,257 |
| GeoQA+ | `/outputs/geoqa_plus/images` | 2,253 |

### Creating/Updating Mappings

```bash
# Build HuggingFace mappings (run on cluster)
python scripts/data_prep/build_fire_image_mapping.py \
    --fire_dataset PengxiangLi/FIRE \
    --output /outputs/fire_image_mapping.json \
    --cache_dir /cache

# Build local mappings (for manually downloaded datasets)
python scripts/data_prep/build_local_image_mapping.py \
    --fire_paths /workspace/image_sources.txt \
    --output /outputs/local_image_mapping.json

# Merge both mappings
python scripts/data_prep/merge_mappings.py \
    --hf_mapping /outputs/fire_image_mapping.json \
    --local_mapping /outputs/local_image_mapping.json \
    --output /outputs/fire_image_mapping_complete.json
```

---

## Preprocessing Scripts

### `scripts/data_prep/prepare_fire_sharegpt.py`

Main preprocessing script that converts FIRE dataset to ShareGPT format for ms-swift training.

**Key Features**:
- Converts multi-round conversations to ShareGPT format
- Supports image loading via mapping files or local directories
- Configurable system prompt
- Filter by source datasets
- Progress tracking with statistics

**Usage**:

```bash
# Full preprocessing with mapping file
python scripts/data_prep/prepare_fire_sharegpt.py \
    --output_dir /outputs/fire_bc \
    --mapping_file /outputs/fire_image_mapping_complete.json \
    --splits train test

# Quick test (no images)
python scripts/data_prep/prepare_fire_sharegpt.py \
    --output_dir ./test_output \
    --skip-images \
    --max_samples 100

# Filter specific sources only
python scripts/data_prep/prepare_fire_sharegpt.py \
    --output_dir /outputs/fire_coco_only \
    --mapping_file /outputs/fire_image_mapping.json \
    --filter_sources coco textvqa
```

**Output Format** (ShareGPT):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful vision-language assistant..."},
    {"role": "user", "content": "<image>\nDescribe this image."},
    {"role": "assistant", "content": "The image shows..."},
    {"role": "user", "content": "Your answer could be improved by..."},
    {"role": "assistant", "content": "Upon reflection, I see..."}
  ],
  "images": ["/path/to/image.jpg"]
}
```

### `scripts/data_prep/build_fire_image_mapping.py`

Builds mappings from FIRE image paths to HuggingFace dataset indices.

**Key Features**:
- Loads HuggingFace datasets in non-streaming mode for indexed access
- Handles multi-config datasets (e.g., MMMU with 30 subject configs)
- Supports composite ID fields (e.g., DocVQA)
- Progress tracking per dataset

**Dataset Configuration** (partial):

```python
DATASET_SOURCES = {
    ("coco", "train2017"): {"hf_id": "detection-datasets/coco", "split": "train", "id_field": "image_id"},
    ("gqa", "images"): {"hf_id": "lmms-lab/GQA", "config": "train_all_images", "split": "train"},
    ("docvqa", "documents"): {"hf_id": "lmms-lab/DocVQA", "split": "validation", "id_field": "composite_docvqa"},
    ("mmmu", "test-images"): {"hf_id": "MMMU/MMMU", "split": "validation", "id_field": "id", "all_configs": True},
    # ... more datasets
}
```

### `scripts/data_prep/build_local_image_mapping.py`

Creates mappings for manually downloaded datasets.

**Configuration**:

```python
LOCAL_DATASETS = {
    "vision_flan": {
        "fire_prefix": "allava_vflan/images/images_191task_1k/",
        "local_dir": "/outputs/images_191task_1k",
    },
    "geoqa_plus": {
        "fire_prefix": "geoqa+/images/",
        "local_dir": "/outputs/geoqa_plus/images",
    },
}
```

### `scripts/data_prep/merge_mappings.py`

Merges HuggingFace and local mappings into a single file.

### `scripts/data_prep/prepare_fire_feedback_sft.py`

Alternative preprocessing for feedback-based SFT training format.

### `scripts/data_prep/analyze_dataset_lengths.py`

Analyzes token lengths in preprocessed datasets for batch optimization.

---

## Source Datasets

### Supported (Working)

| Dataset | Type | Status | Notes |
|---------|------|--------|-------|
| COCO | Object Detection | ✅ Working | train2017, val2014 |
| GQA | Visual QA | ✅ Working | |
| TextVQA | Text-in-Image QA | ✅ Working | |
| ALLaVA-4V (Vision-FLAN) | Multi-task | ✅ Working | Local download |
| ChartQA | Chart QA | ✅ Working | |
| OCR-VQA | OCR | ✅ Working | |
| Visual Genome | Scene Understanding | ✅ Working | VG_100K, VG_100K_2 |
| ScienceQA | Science QA | ✅ Working | |
| MathVista | Math Reasoning | ✅ Working | |
| SynthDog-EN | Document | ✅ Working | |
| MME | Evaluation | ✅ Working | Multiple categories |
| DocVQA | Document QA | ✅ Working | Composite ID |
| MMMU | Multi-subject | ✅ Working | 30 configs |
| GeoQA+ | Geometry | ✅ Working | Local download only |
| AI2D | Diagrams | ✅ Working | |
| SEED-Bench | Evaluation | ✅ Working | |
| MM-Vet | Evaluation | ✅ Working | |
| LLaVA-in-the-Wild | In-the-wild | ✅ Working | |
| DVQA | Chart QA | ✅ Working | |

### Not Available on HuggingFace

| Dataset | Issue | Solution |
|---------|-------|----------|
| MathVerse | Only testmini (99 samples) on HF | Manual download required |
| SAM | Not available | Disabled |
| Web-Landmark | Not available | Disabled |
| Web-Celebrity | Not available | Disabled |
| WikiArt | Partial | Some images missing |

---

## Known Issues

### Test Set Low Coverage (34.9%)

The test set has significantly lower coverage than train due to missing datasets:

| Missing Source | Estimated Samples |
|----------------|-------------------|
| MathVerse | ~863 |
| DVQA (test) | ~969 |
| SEED-Bench | ~800 |
| DocVQA | ~731 |
| MMMU | ~700 |
| GeoQA+ (test) | ~693 |
| MM-Vet | ~188 |
| LLaVA-in-the-Wild | ~36 |

**Improving Coverage**:
1. Download missing datasets manually
2. Update `LOCAL_DATASETS` in `build_local_image_mapping.py`
3. Rebuild local mappings
4. Merge with HuggingFace mappings
5. Rerun preprocessing

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Image not found: coco/train2017/...` | Missing COCO images | Run mapping builder with full COCO |
| `Image not found: geoqa+/images/...` | GeoQA+ not in HF | Use local mapping |
| `Image not found: sam/images/...` | SAM not available | Cannot be resolved |

---

## Usage Examples

### Full Preprocessing Pipeline (Kubernetes)

```bash
# 1. Create mapping (CPU job, 4-6 hours first time)
kubectl apply -f k8s/job-build-mapping.yaml

# 2. Preprocess dataset (CPU job)
kubectl apply -f k8s/job-preprocess-fire-cpu.yaml

# 3. Verify outputs
kubectl exec -it <pod> -- ls -lh /outputs/fire_bc/
```

### Local Testing

```bash
# Quick test without images
python scripts/data_prep/prepare_fire_sharegpt.py \
    --output_dir ./test_output \
    --skip-images \
    --max_samples 50 \
    --splits train

# Verify output format
head -1 ./test_output/fire_sharegpt_train.jsonl | python -m json.tool
```

### Analyzing Dataset

```bash
# Check stats
cat data/fire_preprocessed_v2/stats.json | python -m json.tool

# Count samples per split
wc -l data/fire_preprocessed_v2/*.jsonl

# Analyze token lengths
python scripts/data_prep/analyze_dataset_lengths.py \
    --input data/fire_preprocessed_v2/fire_sharegpt_train.jsonl
```

---

## Related Documentation

- [README.md](../README.md) - Project overview and training instructions
- [QUICKSTART.md](QUICKSTART.md) - Quick deployment guide
- [DOCKER_BUILD.md](DOCKER_BUILD.md) - Container build system
- [GHCR_SETUP.md](GHCR_SETUP.md) - Registry setup
