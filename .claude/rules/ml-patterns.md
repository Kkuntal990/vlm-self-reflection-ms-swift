# ML-Specific Patterns for VLM Self-Reflection

## Dataclass Usage

MUST use `@dataclass` for structured results:
```python
from dataclasses import dataclass, asdict

@dataclass
class SampleResult:
    """Result for a single evaluated sample."""
    sample_id: str
    score: float
    is_valid: bool
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
```

For complex nested structures, use field(default_factory=):
```python
from dataclasses import dataclass, field

@dataclass
class AggregateMetrics:
    """Aggregate metrics across samples."""
    num_samples: int
    avg_score: float
    score_percentiles: Dict[str, float] = field(default_factory=dict)
```

## Model Loading Pattern

For HuggingFace models, use this standard pattern:
```python
class ModelWrapper:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attn: bool = True,
    ):
        # Lazy imports inside __init__ for heavy ML libraries
        from transformers import AutoProcessor, SomeModel

        logger.info(f"Loading model from {model_id}")

        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        try:
            self.model = SomeModel.from_pretrained(
                model_id,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(f"Flash attention failed ({e}), falling back to eager")
            self.model = SomeModel.from_pretrained(
                model_id,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=dtype,
                attn_implementation="eager",
            )

        self.model.eval()
        logger.info("Model loaded successfully")
```

## Dataset Processing Pattern

For JSONL loading:
```python
def load_dataset(dataset_path: str, max_samples: int = 0) -> List[Dict]:
    """Load dataset from JSONL file.

    Args:
        dataset_path: Path to JSONL file
        max_samples: Maximum samples to load (0 = all)

    Returns:
        List of sample dictionaries
    """
    samples = []

    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i}: {e}")

    logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
    return samples
```

For JSONL writing:
```python
def save_results(results: List[Dict], output_path: str) -> None:
    """Save results to JSONL file."""
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Saved {len(results)} results to {output_path}")
```

## Argument Parsing Pattern

Group arguments by purpose:
```python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Description here",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset file",
    )

    # Optional configuration
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for results",
    )

    # Hardware configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention",
    )

    return parser.parse_args()
```

## Progress Reporting

Use tqdm for progress bars:
```python
from tqdm import tqdm

for sample in tqdm(samples, desc="Processing samples"):
    # ... process
```

For nested progress:
```python
for batch in tqdm(batches, desc="Batches"):
    for sample in tqdm(batch, desc="Samples", leave=False):
        # ... process
```

## Result Saving Pattern

```python
def save_all_results(
    results: List[SampleResult],
    metrics: AggregateMetrics,
    output_dir: Path,
) -> None:
    """Save all results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample results as JSONL
    results_path = output_dir / "sample_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    logger.info(f"Saved sample results to {results_path}")

    # Aggregate metrics as JSON (pretty printed)
    metrics_path = output_dir / "aggregate_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info(f"Saved aggregate metrics to {metrics_path}")
```

## Environment Variable Pattern

Set defaults early in module (after imports):
```python
import os

# Set HuggingFace timeouts to avoid network issues
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

# Disable tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
```

## Statistics Tracking Pattern

Track processing statistics throughout:
```python
stats = {
    "total_samples": len(samples),
    "processed": 0,
    "skipped": 0,
    "errors": [],
}

for sample in samples:
    try:
        # ... process
        stats["processed"] += 1
    except Exception as e:
        stats["skipped"] += 1
        if len(stats["errors"]) < 100:  # Cap error messages
            stats["errors"].append(f"{sample['id']}: {str(e)}")

logger.info(f"Stats: {stats['processed']}/{stats['total_samples']} processed, "
            f"{stats['skipped']} skipped")
```

## Batch Processing Pattern

For GPU-efficient processing:
```python
def process_in_batches(
    samples: List[Dict],
    batch_size: int = 8,
) -> List[Dict]:
    """Process samples in batches for efficiency."""
    results = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        batch_results = process_batch(batch)
        results.extend(batch_results)

        if (i + batch_size) % 100 == 0:
            logger.info(f"Processed {i + batch_size}/{len(samples)} samples")

    return results
```

## Image Handling Pattern

For loading images with error handling:
```python
from PIL import Image

def load_image(image_path: str) -> Optional[Image.Image]:
    """Load image from path with error handling.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image or None if loading failed
    """
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None
```

## Factory Pattern for Extensibility

Use registries for pluggable components:
```python
from typing import Type

JUDGE_REGISTRY: Dict[str, Type[BaseJudge]] = {}

def register_judge(name: str):
    """Decorator to register a judge class."""
    def decorator(cls: Type[BaseJudge]) -> Type[BaseJudge]:
        JUDGE_REGISTRY[name] = cls
        return cls
    return decorator

def create_judge(judge_type: str, **kwargs) -> BaseJudge:
    """Factory function to create judge instance."""
    if judge_type not in JUDGE_REGISTRY:
        available = ", ".join(JUDGE_REGISTRY.keys())
        raise ValueError(f"Unknown judge: {judge_type}. Available: {available}")
    return JUDGE_REGISTRY[judge_type](**kwargs)
```
