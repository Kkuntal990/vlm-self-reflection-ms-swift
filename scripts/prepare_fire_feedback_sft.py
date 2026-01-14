#!/usr/bin/env python3
"""
Prepare FIRE dataset in Messages format for Feedback SFT.

Converts FIRE multi-round student-teacher conversations into messages format
where the model learns to generate FEEDBACK (teacher responses) instead of
student answers.

Format: Messages format with per-message loss control
- First assistant message: Question + image token (loss: false)
- User messages: Student responses
- Subsequent assistant messages: Teacher feedback (loss: true)

Image Loading Modes (pick one):
  1. --source_images_dir: Reference pre-downloaded images (no duplication)
  2. --mapping_file: Load via mapping JSON (from build_fire_image_mapping.py)
  3. --skip-images: Placeholder paths for testing

Usage Examples:
    # Mode 1: Pre-downloaded images (COCO only)
    python scripts/prepare_fire_feedback_sft.py \
        --output_dir /outputs/fire_feedback_coco \
        --source_images_dir /outputs \
        --filter_sources coco \
        --splits train

    # Mode 2: Mapping file (all datasets)
    python scripts/prepare_fire_feedback_sft.py \
        --output_dir /outputs/fire_feedback_full \
        --mapping_file /outputs/fire_image_mapping.json \
        --splits train test

    # Mode 3: Testing only
    python scripts/prepare_fire_feedback_sft.py \
        --output_dir /outputs/fire_feedback_test \
        --skip-images \
        --max_samples 100
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Union

# Set HuggingFace timeouts to avoid network issues
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")  # 10 minutes for downloads
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")  # 1 minute for metadata checks

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default system prompt for feedback generation
DEFAULT_SYSTEM_PROMPT = "You are a teacher providing feedback on student responses to visual questions. Given an image and a student's answer, provide constructive feedback identifying what is correct, what needs improvement, and specific suggestions."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FIRE dataset to Messages format for Feedback SFT"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/fire_feedback",
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/outputs/fire_feedback/images",
        help="Directory to save extracted images",
    )
    parser.add_argument(
        "--source_images_dir",
        type=str,
        default=None,
        help="Directory containing pre-downloaded source images (e.g., /outputs with coco/, mathvista/ subdirs). Samples without images will be skipped.",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default=None,
        help="Path to FIRE image mapping JSON (from build_fire_image_mapping.py). Enables efficient loading from HuggingFace datasets by index.",
    )
    parser.add_argument(
        "--filter_sources",
        type=str,
        nargs="+",
        default=None,
        help="Only process samples from specific image sources (e.g., 'coco' 'mathvista'). Useful for processing available images while others download.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process per split (0 = all)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets (memory efficient)",
    )
    parser.add_argument(
        "--image_quality",
        type=int,
        default=95,
        help="JPEG quality for saved images (1-100)",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="PengxiangLi/FIRE",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image processing (for testing FIRE parsing logic only)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help=f"System prompt to include in each conversation (default: '{DEFAULT_SYSTEM_PROMPT}')",
    )
    return parser.parse_args()


def save_image(image: Union[Image.Image, str, bytes, dict], path: str, quality: int = 95) -> bool:
    """Save image to disk as JPEG, handling multiple input formats.

    Args:
        image: PIL Image, file path, base64 string, bytes, or dict with 'path'/'bytes'
        path: Destination file path
        quality: JPEG quality (1-100)

    Returns:
        True on success, False on failure
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            # Could be file path, URL, or base64
            if image.startswith('data:image'):
                # Base64 with data URI prefix
                base64_data = image.split(',', 1)[1]
                image_bytes = base64.b64decode(base64_data)
                pil_image = Image.open(io.BytesIO(image_bytes))
            elif image.startswith('http://') or image.startswith('https://'):
                # URL - would need requests library
                logger.warning(f"Image URLs not supported yet: {image[:50]}")
                return False
            elif os.path.exists(image):
                # File path
                pil_image = Image.open(image)
            else:
                # Try as base64 without prefix
                try:
                    image_bytes = base64.b64decode(image)
                    pil_image = Image.open(io.BytesIO(image_bytes))
                except Exception:
                    logger.warning(f"Could not parse image string: {image[:50]}")
                    return False
        elif isinstance(image, bytes):
            # Raw bytes
            pil_image = Image.open(io.BytesIO(image))
        elif isinstance(image, dict):
            # HuggingFace datasets sometimes use dict format
            if 'path' in image:
                pil_image = Image.open(image['path'])
            elif 'bytes' in image:
                pil_image = Image.open(io.BytesIO(image['bytes']))
            else:
                logger.warning(f"Unknown dict image format: {list(image.keys())}")
                return False
        else:
            logger.warning(f"Unsupported image type: {type(image)}")
            return False

        # Convert to RGB if needed and save
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(path, "JPEG", quality=quality)
        return True
    except Exception as e:
        logger.warning(f"Failed to save image to {path}: {e}")
        return False


def extract_question_text(question) -> str:
    """Extract clean question text from FIRE question field.

    The FIRE dataset stores questions as dicts with 'value' key,
    and may include <image> tokens which we preserve.
    """
    if isinstance(question, dict) and "value" in question:
        text = question["value"]
    elif isinstance(question, str):
        text = question
    else:
        text = str(question)

    return text.strip()


def extract_answer_from_response(response_value: str) -> str:
    """Extract answer from student response, ignoring thought.

    Student responses in FIRE format look like:
    'Thought: ...\nAnswer: ...\n\n'

    We only want the Answer part.
    """
    if not response_value:
        return ""

    # Split by "Answer:" and take everything after it
    if "Answer:" in response_value:
        parts = response_value.split("Answer:", 1)
        answer = parts[1].strip()
        return answer

    # If no "Answer:" marker, return as is
    return response_value.strip()


def extract_feedback_text(feedback_value: str) -> str:
    """Extract feedback text from teacher response, stripping score.

    Teacher feedback in FIRE format looks like:
    'Score: 6.\nFeedback: ...\n'

    We extract just the feedback part, without the score.
    """
    if not feedback_value:
        return ""

    # Extract feedback after "Feedback:" marker
    if "Feedback:" in feedback_value:
        feedback = feedback_value.split("Feedback:", 1)[1].strip()
        return feedback

    # If no "Feedback:" marker, return as is
    return feedback_value.strip()


class LazyImageLoader:
    """Lazy image loader that loads images on-demand from HuggingFace datasets.

    This avoids loading all images into memory at once, preventing OOM errors.
    Dataset handles are cached, but images are loaded only when requested.
    """

    def __init__(self, mapping_file: Path, cache_dir: str = "/cache"):
        """Initialize lazy loader with mapping file.

        Args:
            mapping_file: Path to mapping JSON from build_fire_image_mapping.py
            cache_dir: HuggingFace cache directory
        """
        logger.info(f"Loading mapping from {mapping_file}")
        with open(mapping_file, 'r') as f:
            self.mapping = json.load(f)

        self.cache_dir = cache_dir
        self.dataset_cache = {}  # Cache dataset handles, not images
        self.stats = {"hits": 0, "misses": 0, "errors": 0}

        logger.info(f"Mapping loaded: {len(self.mapping)} paths mapped")

    def get(self, fire_path: str):
        """Get image for a FIRE path, loading on-demand.

        Args:
            fire_path: FIRE image path (e.g., "coco/train2014/COCO_train2014_000000123456.jpg")

        Returns:
            PIL Image or None if not found
        """
        if fire_path not in self.mapping:
            self.stats["misses"] += 1
            return None

        entry = self.mapping[fire_path]

        # Handle local file paths (for manually downloaded datasets)
        if "local_path" in entry:
            try:
                local_path = entry["local_path"]
                self.stats["hits"] += 1
                return Image.open(local_path)
            except Exception as e:
                logger.error(f"Failed to load local image {entry['local_path']}: {e}")
                self.stats["errors"] += 1
                return None

        # Handle HuggingFace datasets
        dataset_id = entry["dataset"]
        config = entry.get("config")
        split = entry["split"]
        index = entry["index"]

        # Cache key for dataset
        cache_key = (dataset_id, config, split)

        # Load dataset if not cached
        if cache_key not in self.dataset_cache:
            try:
                logger.info(f"Loading dataset: {dataset_id} ({split})")
                if config:
                    ds = load_dataset(dataset_id, config, split=split, cache_dir=self.cache_dir, trust_remote_code=True)
                else:
                    ds = load_dataset(dataset_id, split=split, cache_dir=self.cache_dir, trust_remote_code=True)
                self.dataset_cache[cache_key] = ds
                logger.info(f"Dataset cached: {dataset_id} ({len(ds)} samples)")
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_id}: {e}")
                self.stats["errors"] += 1
                return None

        # Get image from cached dataset
        try:
            ds = self.dataset_cache[cache_key]
            if index < len(ds) and 'image' in ds[index]:
                self.stats["hits"] += 1
                return ds[index]['image']
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Failed to load image {fire_path} at index {index}: {e}")
            self.stats["errors"] += 1
            return None

    def get_stats(self):
        """Get loader statistics."""
        total = self.stats["hits"] + self.stats["misses"] + self.stats["errors"]
        return {
            "total_requests": total,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "errors": self.stats["errors"],
            "hit_rate": f"{self.stats['hits']/total*100:.1f}%" if total > 0 else "0%",
            "datasets_cached": len(self.dataset_cache)
        }


def load_images_from_source_dir(source_dir: Path, needed_paths: set) -> dict:
    """Load images from pre-downloaded directory structure.

    Args:
        source_dir: Root directory containing dataset subdirs (e.g., coco/, gqa/)
        needed_paths: Set of FIRE image paths needed

    Returns:
        Dict mapping FIRE paths to absolute path strings
    """
    logger.info(f"Loading images from source directory: {source_dir}")

    source_images = {}
    missing = []

    for img_path in needed_paths:
        full_path = source_dir / img_path
        if full_path.exists():
            source_images[img_path] = str(full_path)
        else:
            missing.append(img_path)

    logger.info(f"Found {len(source_images)}/{len(needed_paths)} images")
    if missing:
        logger.warning(f"Missing {len(missing)} images")
        for path in missing[:5]:
            logger.warning(f"  - {path}")
        if len(missing) > 5:
            logger.warning(f"  ... and {len(missing) - 5} more")

    return source_images


def count_rounds_in_sample(sample: dict) -> int:
    """Count the number of conversation rounds in a FIRE sample.

    A round is a student response (optionally followed by teacher feedback).

    Args:
        sample: FIRE dataset sample

    Returns:
        Number of conversation rounds (0 if none)
    """
    conversations = sample.get("conversations", [])
    if not conversations:
        return 0

    rounds = 0
    for turn in conversations:
        if turn.get("role") == "student" and turn.get("type") == "response":
            rounds += 1

    return rounds


def parse_fire_to_messages(
    sample: dict,
    sample_id: str,
    image_path: str,
    system_prompt: str = "",
) -> dict | None:
    """Convert a single FIRE sample into Messages format for Feedback SFT.

    Messages format with loss control:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "assistant", "content": "question + <image>", "loss": false},
            {"role": "user", "content": "student answer 1"},
            {"role": "assistant", "content": "feedback 1", "loss": true},
            {"role": "user", "content": "student answer 2"},
            {"role": "assistant", "content": "feedback 2", "loss": true},
            ...
        ],
        "images": ["/path/to/image.jpg"]
    }

    Args:
        sample: FIRE dataset row containing question, image, conversations
        sample_id: Unique identifier for logging/debugging
        image_path: Absolute path to saved image file
        system_prompt: System prompt to include in the conversation

    Returns:
        Messages format dict or None if parsing fails
    """
    question_text = extract_question_text(sample.get("question", ""))
    conversations = sample.get("conversations", [])

    if not conversations:
        logger.debug(f"{sample_id}: No conversations found")
        return None

    if not question_text:
        logger.warning(f"{sample_id}: Empty question text")
        return None

    # Parse student-teacher rounds into (answer, feedback) pairs
    rounds = []
    i = 0
    while i < len(conversations):
        turn = conversations[i]

        # Expect student response
        if turn.get("role") == "student" and turn.get("type") == "response":
            response_value = turn.get("value", "")
            answer = extract_answer_from_response(response_value)

            if answer:
                # Check for following teacher feedback
                feedback = None
                if i + 1 < len(conversations):
                    next_turn = conversations[i + 1]
                    if (
                        next_turn.get("role") == "teacher"
                        and next_turn.get("type") == "feedback"
                    ):
                        feedback_value = next_turn.get("value", "")
                        feedback = extract_feedback_text(feedback_value)
                        i += 1  # Skip the feedback turn

                rounds.append((answer, feedback))

        i += 1

    if not rounds:
        logger.debug(f"{sample_id}: No valid rounds parsed")
        return None

    # Need at least one round with feedback to train on
    if not any(feedback for _, feedback in rounds):
        logger.debug(f"{sample_id}: No feedback found in any round")
        return None

    # Build Messages format conversation
    messages = []

    # System prompt
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    # First assistant message: Question with image token (loss: false)
    messages.append({
        "role": "assistant",
        "content": question_text,
        "loss": False
    })

    # Build conversation rounds: student answer (user) -> feedback (assistant)
    for idx, (answer, feedback) in enumerate(rounds):
        # Student answer as user message
        messages.append({
            "role": "user",
            "content": answer
        })

        # Teacher feedback as assistant message (only if feedback exists)
        if feedback:
            messages.append({
                "role": "assistant",
                "content": feedback,
                "loss": True
            })

    # Return Messages format with images field
    return {
        "messages": messages,
        "images": [image_path]
    }


def process_split(
    dataset_id: str,
    split: str,
    output_dir: Path,
    image_dir: Path,
    max_samples: int,
    streaming: bool,
    image_quality: int,
    skip_images: bool = False,
    system_prompt: str = "",
    source_images_dir: Path | None = None,
    mapping_file: Path | None = None,
    filter_sources: list[str] | None = None,
) -> dict:
    """Process a single dataset split.

    Args:
        dataset_id: HuggingFace dataset identifier
        split: Dataset split name (train/test)
        output_dir: Directory for output JSONL
        image_dir: Directory for saved images
        max_samples: Maximum samples to process (0 = all)
        streaming: Whether to use HF streaming mode
        image_quality: JPEG quality for saved images
        skip_images: Skip image processing for testing
        system_prompt: System prompt to include in each conversation
        source_images_dir: Directory with pre-downloaded images
        mapping_file: JSON mapping from build_fire_image_mapping.py
        filter_sources: Only process these sources (e.g., ['coco'])

    Returns:
        Dictionary of processing statistics
    """
    logger.info(f"Processing {split} split from {dataset_id}...")

    # Create directories
    split_image_dir = image_dir / split
    split_image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FIRE dataset
    logger.info(f"Loading FIRE dataset (streaming={streaming})...")
    dataset = load_dataset(dataset_id, split=split, streaming=streaming)

    # First pass: collect image paths we need
    logger.info(f"Collecting needed image paths from FIRE...")
    if streaming:
        if max_samples > 0:
            fire_samples = list(dataset.take(max_samples))
        else:
            # Take all samples - streaming datasets don't have len() so we iterate fully
            logger.info(f"Loading all samples from streaming dataset (this may take time)...")
            fire_samples = list(dataset)
    else:
        if max_samples > 0:
            fire_samples = list(dataset.select(range(min(max_samples, len(dataset)))))
        else:
            fire_samples = list(dataset)

    needed_image_paths = set()
    for sample in fire_samples:
        img_path = sample.get("image")
        if img_path and isinstance(img_path, str):
            # Apply source filter if specified
            if filter_sources:
                # Check if image path starts with any of the filter sources
                source = img_path.split('/')[0]
                if source not in filter_sources:
                    continue
            needed_image_paths.add(img_path)


    # Initialize stats
    stats = {
        "split": split,
        "samples_processed": 0,
        "samples_skipped": 0,
        "samples_no_image": 0,
        "samples_no_conversations": 0,
        "samples_no_feedback": 0,
        "samples_filtered_by_source": 0,
        "total_feedback_turns": 0,
        "images_saved": 0,
        "errors": [],
        # Round-based breakdown
        "rounds_breakdown": {
            "processed": {},  # {num_rounds: count}
            "skipped": {}     # {num_rounds: count}
        },
    }

    logger.info(f"Need {len(needed_image_paths)} unique images")
    if filter_sources:
        logger.info(f"Filtering to sources: {filter_sources}")

    # Load images based on mode
    source_images = {}
    lazy_loader = None

    if skip_images:
        logger.warning("Skipping image loading (--skip-images mode)")
        logger.warning("Image paths will be placeholders for testing!")
    elif source_images_dir:
        source_images = load_images_from_source_dir(source_images_dir, needed_image_paths)
    elif mapping_file:
        # Use lazy loader to avoid loading all images into memory at once
        lazy_loader = LazyImageLoader(mapping_file)
        logger.info("Using lazy image loading (on-demand, memory efficient)")
    else:
        logger.error("No image source specified!")
        logger.error("Use one of: --source_images_dir, --mapping_file, or --skip-images")
        raise ValueError("Must specify image source: --source_images_dir, --mapping_file, or --skip-images")

    output_file = output_dir / f"fire_feedback_{split}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        desc = f"Processing {split}"
        if max_samples > 0:
            desc += f" (max {max_samples})"

        for idx, sample in enumerate(tqdm(fire_samples, desc=desc)):
            if max_samples > 0 and idx >= max_samples:
                break

            sample_id = f"{split}_{idx:06d}"

            try:
                # Count rounds in this sample
                num_rounds = count_rounds_in_sample(sample)

                # Get image path from FIRE sample
                image_path_ref = sample.get("image")
                if image_path_ref is None or not isinstance(image_path_ref, str):
                    stats["samples_skipped"] += 1
                    stats["samples_no_image"] += 1
                    # Track rounds for skipped samples
                    stats["rounds_breakdown"]["skipped"][num_rounds] = stats["rounds_breakdown"]["skipped"].get(num_rounds, 0) + 1
                    if len(stats["errors"]) < 100:
                        stats["errors"].append(f"{sample_id}: No image path")
                    continue

                # Apply source filter if specified
                if filter_sources:
                    source = image_path_ref.split('/')[0]
                    if source not in filter_sources:
                        stats["samples_skipped"] += 1
                        stats["samples_filtered_by_source"] += 1
                        # Track rounds for skipped samples
                        stats["rounds_breakdown"]["skipped"][num_rounds] = stats["rounds_breakdown"]["skipped"].get(num_rounds, 0) + 1
                        continue

                # Get image for this sample
                if skip_images:
                    # Placeholder mode for testing
                    image_save_path = f"/placeholder/images/{split}/{sample_id}.jpg"
                    stats["images_saved"] += 1
                else:
                    # Look up image from loaded sources or lazy loader
                    if lazy_loader:
                        # Lazy loading mode - load on demand
                        image = lazy_loader.get(image_path_ref)

                        # Try alternative path formats if not found
                        if image is None and image_path_ref.startswith('images/'):
                            alt_path = f"mathvista/{image_path_ref}"
                            image = lazy_loader.get(alt_path)
                            if image is not None:
                                image_path_ref = alt_path
                    else:
                        # Pre-loaded mode (source_images_dir)
                        image = source_images.get(image_path_ref)

                        # Try alternative path formats if not found
                        if image is None and image_path_ref.startswith('images/'):
                            alt_path = f"mathvista/{image_path_ref}"
                            image = source_images.get(alt_path)
                            if image is not None:
                                image_path_ref = alt_path

                    if image is None:
                        stats["samples_skipped"] += 1
                        stats["samples_no_image"] += 1
                        # Track rounds for skipped samples
                        stats["rounds_breakdown"]["skipped"][num_rounds] = stats["rounds_breakdown"]["skipped"].get(num_rounds, 0) + 1
                        if len(stats["errors"]) < 100:
                            stats["errors"].append(f"{sample_id}: Image not found: {image_path_ref}")
                        continue

                    # Handle based on image type
                    if isinstance(image, str):
                        # Path string from source_images_dir - reference directly
                        image_save_path = image
                        stats["images_saved"] += 1
                    else:
                        # PIL Image from mapping - save to disk
                        image_save_path = str(split_image_dir / f"{sample_id}.jpg")
                        if not save_image(image, image_save_path, image_quality):
                            stats["samples_skipped"] += 1
                            # Track rounds for skipped samples
                            stats["rounds_breakdown"]["skipped"][num_rounds] = stats["rounds_breakdown"]["skipped"].get(num_rounds, 0) + 1
                            if len(stats["errors"]) < 100:
                                stats["errors"].append(f"{sample_id}: Image save failed")
                            continue
                        stats["images_saved"] += 1

                # Generate Messages format
                messages_example = parse_fire_to_messages(
                    sample, sample_id, image_save_path, system_prompt
                )

                if not messages_example:
                    stats["samples_skipped"] += 1
                    stats["samples_no_conversations"] += 1
                    # Track rounds for skipped samples
                    stats["rounds_breakdown"]["skipped"][num_rounds] = stats["rounds_breakdown"]["skipped"].get(num_rounds, 0) + 1
                    if len(stats["errors"]) < 100:
                        stats["errors"].append(f"{sample_id}: No valid conversations")
                    continue

                # Count feedback turns (assistant messages with loss: true)
                feedback_count = sum(
                    1 for msg in messages_example["messages"]
                    if msg.get("role") == "assistant" and msg.get("loss") is True
                )

                if feedback_count == 0:
                    stats["samples_skipped"] += 1
                    stats["samples_no_feedback"] += 1
                    stats["rounds_breakdown"]["skipped"][num_rounds] = stats["rounds_breakdown"]["skipped"].get(num_rounds, 0) + 1
                    if len(stats["errors"]) < 100:
                        stats["errors"].append(f"{sample_id}: No feedback turns")
                    continue

                # Write to JSONL
                f.write(json.dumps(messages_example, ensure_ascii=False) + "\n")

                stats["samples_processed"] += 1
                stats["total_feedback_turns"] += feedback_count
                # Track rounds for processed samples
                stats["rounds_breakdown"]["processed"][num_rounds] = stats["rounds_breakdown"]["processed"].get(num_rounds, 0) + 1

            except Exception as e:
                stats["samples_skipped"] += 1
                # Track rounds for skipped samples (use num_rounds if available, else 0)
                try:
                    num_rounds = count_rounds_in_sample(sample)
                except:
                    num_rounds = 0
                stats["rounds_breakdown"]["skipped"][num_rounds] = stats["rounds_breakdown"]["skipped"].get(num_rounds, 0) + 1
                if len(stats["errors"]) < 100:
                    stats["errors"].append(f"{sample_id}: {str(e)}")
                logger.warning(f"Error processing sample {idx}: {e}")
                continue

    logger.info(f"Wrote {stats['samples_processed']} conversations to {output_file}")

    # Log lazy loader stats if used
    if lazy_loader:
        loader_stats = lazy_loader.get_stats()
        logger.info("=" * 60)
        logger.info("Lazy Loader Statistics:")
        logger.info(f"  Total requests: {loader_stats['total_requests']}")
        logger.info(f"  Hits: {loader_stats['hits']}")
        logger.info(f"  Misses: {loader_stats['misses']}")
        logger.info(f"  Errors: {loader_stats['errors']}")
        logger.info(f"  Hit rate: {loader_stats['hit_rate']}")
        logger.info(f"  Datasets cached: {loader_stats['datasets_cached']}")
        logger.info("=" * 60)

    return stats


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)

    logger.info("=" * 60)
    logger.info("FIRE Feedback SFT Dataset Preparation")
    logger.info("Messages Format with Loss Control")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Max samples per split: {args.max_samples if args.max_samples > 0 else 'all'}")
    logger.info(f"Streaming: {args.streaming}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Image dir: {image_dir}")
    logger.info("=" * 60)

    all_stats = {}

    source_images_dir = Path(args.source_images_dir) if args.source_images_dir else None
    mapping_file = Path(args.mapping_file) if args.mapping_file else None

    for split in args.splits:
        stats = process_split(
            dataset_id=args.dataset_id,
            split=split,
            output_dir=output_dir,
            image_dir=image_dir,
            max_samples=args.max_samples,
            streaming=args.streaming,
            image_quality=args.image_quality,
            skip_images=args.skip_images,
            system_prompt=args.system_prompt,
            source_images_dir=source_images_dir,
            mapping_file=mapping_file,
            filter_sources=args.filter_sources,
        )
        all_stats[split] = stats

    # Write stats file
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Stats written to {stats_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)

    total_processed = 0
    total_skipped = 0
    total_feedback_turns = 0

    for split, stats in all_stats.items():
        avg_feedback = (
            stats["total_feedback_turns"] / max(1, stats["samples_processed"])
        )
        logger.info(f"\n{split.upper()} SPLIT:")
        logger.info(f"  Samples processed: {stats['samples_processed']}")
        logger.info(f"  Samples skipped: {stats['samples_skipped']}")
        logger.info(f"    - Filtered by source: {stats.get('samples_filtered_by_source', 0)}")
        logger.info(f"    - No image: {stats['samples_no_image']}")
        logger.info(f"    - No conversations: {stats['samples_no_conversations']}")
        logger.info(f"    - No feedback: {stats.get('samples_no_feedback', 0)}")
        logger.info(f"  Total feedback turns: {stats['total_feedback_turns']}")
        logger.info(f"  Avg feedback/sample: {avg_feedback:.2f}")
        logger.info(f"  Images saved: {stats['images_saved']}")

        # Round-based breakdown
        logger.info(f"\n  Rounds Breakdown (Processed):")
        processed_breakdown = stats.get("rounds_breakdown", {}).get("processed", {})
        if processed_breakdown:
            for num_rounds in sorted(processed_breakdown.keys()):
                count = processed_breakdown[num_rounds]
                logger.info(f"    {num_rounds} round(s): {count} samples")
        else:
            logger.info(f"    No samples processed")

        logger.info(f"\n  Rounds Breakdown (Skipped):")
        skipped_breakdown = stats.get("rounds_breakdown", {}).get("skipped", {})
        if skipped_breakdown:
            for num_rounds in sorted(skipped_breakdown.keys()):
                count = skipped_breakdown[num_rounds]
                logger.info(f"    {num_rounds} round(s): {count} samples")
        else:
            logger.info(f"    No samples skipped")

        total_processed += stats["samples_processed"]
        total_skipped += stats["samples_skipped"]
        total_feedback_turns += stats["total_feedback_turns"]

    logger.info("\n" + "-" * 60)
    logger.info(f"TOTAL: {total_processed} conversations with {total_feedback_turns} feedback turns")
    logger.info(f"       ({total_skipped} samples skipped)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
