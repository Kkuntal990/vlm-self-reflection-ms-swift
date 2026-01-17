# Python Style Rules for VLM Self-Reflection

## File Structure

MUST include in order:
1. Shebang: `#!/usr/bin/env python3`
2. Module docstring with usage examples
3. Standard library imports (alphabetical)
4. Third-party imports (alphabetical)
5. Local imports (using sys.path.insert pattern)
6. Constants
7. Classes and functions
8. Entry point guard

## Import Rules

### Standard Imports
- MUST be at module level
- MUST be grouped: stdlib, third-party, local
- MUST be alphabetically ordered within groups
- MUST use `from pathlib import Path` (not os.path)

### Lazy Imports (EXCEPTION - DO NOT CHANGE)
Heavy ML dependencies MAY be imported inside class `__init__` methods when:
- The import is for a large ML library (transformers, trl, torch model loading)
- The class is a model wrapper that loads on instantiation
- This prevents slow imports when the class is not used

Example (ALLOWED):
```python
class SkyworkVLRewardScorer:
    def __init__(self, ...):
        # Lazy import to avoid loading heavy libraries until needed
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from trl import AutoModelForCausalLMWithValueHead
```

### Local Import Pattern
Use this pattern for importing from scripts/ directory:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score_with_reward_model import SkyworkVLRewardScorer
```

## Type Hints

MUST use type hints on:
- All function parameters
- All function return values
- Class attributes (via dataclass or explicit annotations)

Use typing module for:
- `Dict`, `List`, `Optional`, `Union`, `Tuple`, `Any`
- `Type[T]` for class references

Example:
```python
def load_dataset(dataset_path: str, max_samples: int = 0) -> List[Dict]:
    """Load dataset from JSONL file."""
    ...
```

## Docstrings

MUST use Google-style docstrings:
```python
def function_name(param1: str, param2: int) -> Dict:
    """Short description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong
    """
```

Module docstrings SHOULD include usage examples:
```python
"""
Module description.

Usage:
    python scripts/my_script.py --arg1 value1 --arg2 value2

Reference:
    - Link to relevant paper or documentation
"""
```

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `load_test_dataset` |
| Variables | snake_case | `image_path` |
| Classes | PascalCase | `VLMInferenceEngine` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_SYSTEM_PROMPT` |
| Private methods | _single_underscore | `_load_model` |
| Type variables | PascalCase with T suffix | `ResultT` |

## Logging

MUST use this exact pattern at module level:
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
```

Use logger levels appropriately:
- `logger.info()` - Progress updates, success messages
- `logger.warning()` - Recoverable issues, fallbacks
- `logger.error()` - Failures that affect output
- `logger.debug()` - Verbose debugging info

## Entry Points

MUST use this pattern for scripts:
```python
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="...")
    # ... arguments
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    # ... main logic

if __name__ == "__main__":
    main()
```

## Error Handling

Prefer specific exceptions over generic `Exception`:
```python
try:
    sample = json.loads(line.strip())
except json.JSONDecodeError as e:
    logger.warning(f"Failed to parse line {i}: {e}")
```

Track error statistics for batch processing:
```python
stats = {"processed": 0, "errors": 0, "error_messages": []}
```

## String Formatting

Use f-strings for string formatting:
```python
logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
```

## Line Length

Prefer lines under 100 characters. For long argument lists, use trailing commas:
```python
result = SampleResult(
    sample_id=sample["id"],
    score=computed_score,
    is_valid=True,
)
```
