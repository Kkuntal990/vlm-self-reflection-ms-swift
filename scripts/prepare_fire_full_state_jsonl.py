#!/usr/bin/env python3
"""
Prepare FIRE dataset for behavior cloning with ms-swift.

This script implements offline imitation learning (behavior cloning) by converting
FIRE dataset trajectories into ms-swift JSONL format where each timestep becomes
a separate training example:

    State = (image, question, all previous attempts, all previous feedback)
    Action = next student answer
    Loss  = cross-entropy on expert action given state

This follows the RePer learning paradigm but with:
    - Pre-logged trajectories from FIRE
    - External critic (not trained)
    - Offline training

Usage:
    python scripts/prepare_fire_full_state_jsonl.py \\
        --output_dir /outputs/fire_bc \\
        --image_dir /cache/fire_images \\
        --max_samples 0 \\
        --max_history_rounds 6 \\
        --splits train test \\
        --streaming
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FIRE dataset to ms-swift behavior cloning format"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/fire_bc",
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/cache/fire_images",
        help="Directory to save extracted images",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process per split (0 = all)",
    )
    parser.add_argument(
        "--max_history_rounds",
        type=int,
        default=6,
        help="Maximum history rounds to include in state (for context length control)",
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
    and may include <image> tokens which we handle separately.
    """
    if isinstance(question, dict) and "value" in question:
        text = question["value"]
    elif isinstance(question, str):
        text = question
    else:
        text = str(question)

    # Remove <image> token if present (we add it in state construction)
    text = text.replace("<image>", "").strip()
    return text


def parse_fire_to_bc_examples(
    sample: dict,
    sample_id: str,
    image_path: str,
    max_history_rounds: int = 6,
) -> list[dict]:
    """Convert a single FIRE sample into multiple behavior cloning examples.

    This is the core function implementing the behavior cloning paradigm:
    - For each timestep t, we construct state s_t from (image, question, history)
    - The action a_t is the next student response
    - Each (s_t, a_t) pair becomes one training example

    Args:
        sample: FIRE dataset row containing question, image, conversations
        sample_id: Unique identifier for logging/debugging
        image_path: Absolute path to saved image file
        max_history_rounds: Maximum (attempt, feedback) pairs to include in state
                           (sliding window for context length control)

    Returns:
        List of ms-swift format training examples, one per student response
    """
    examples = []
    question_text = extract_question_text(sample.get("question", ""))
    conversations = sample.get("conversations", [])

    if not conversations:
        logger.debug(f"{sample_id}: No conversations found")
        return examples

    if not question_text:
        logger.warning(f"{sample_id}: Empty question text")
        return examples

    # Parse conversations into (attempt, feedback) rounds
    # FIRE format: alternating student responses and teacher feedback
    rounds = []
    i = 0
    while i < len(conversations):
        turn = conversations[i]

        # Expect student response
        if turn.get("role") == "student" and turn.get("type") == "response":
            attempt = turn.get("value", "").strip()
            feedback = None

            # Check for following teacher feedback
            if i + 1 < len(conversations):
                next_turn = conversations[i + 1]
                if (
                    next_turn.get("role") == "teacher"
                    and next_turn.get("type") == "feedback"
                ):
                    feedback = next_turn.get("value", "").strip()
                    i += 1

            if attempt:  # Only add non-empty attempts
                rounds.append((attempt, feedback))
        i += 1

    if not rounds:
        logger.debug(f"{sample_id}: No valid rounds parsed from conversations")
        return examples

    # Generate one training example per student response (timestep)
    # This is the behavior cloning objective: predict action given state
    for timestep, (current_attempt, _) in enumerate(rounds):
        # Build state: <image> + question + history of (attempt, feedback) pairs
        state_parts = [f"<image>\nQuestion:\n{question_text}"]

        # Add history using sliding window if trajectory is long
        history_start = max(0, timestep - max_history_rounds)
        for h in range(history_start, timestep):
            prev_attempt, prev_feedback = rounds[h]
            attempt_num = h + 1

            state_parts.append(f"\n\nAttempt {attempt_num}:\n{prev_attempt}")
            if prev_feedback:
                state_parts.append(f"\n\nFeedback {attempt_num}:\n{prev_feedback}")

        state = "".join(state_parts)
        action = current_attempt

        # ms-swift JSONL format for multimodal SFT
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": state},
                    {"role": "assistant", "content": action},
                ],
                "images": [image_path],
            }
        )

    return examples


def process_split(
    dataset_id: str,
    split: str,
    output_dir: Path,
    image_dir: Path,
    max_samples: int,
    max_history_rounds: int,
    streaming: bool,
    image_quality: int,
    skip_images: bool = False,
) -> dict:
    """Process a single dataset split.

    Args:
        dataset_id: HuggingFace dataset identifier
        split: Dataset split name (train/test)
        output_dir: Directory for output JSONL
        image_dir: Directory for saved images
        max_samples: Maximum samples to process (0 = all)
        max_history_rounds: Max history in state construction
        streaming: Whether to use HF streaming mode
        image_quality: JPEG quality for saved images

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
        fire_samples = list(dataset.take(max_samples if max_samples > 0 else 1000))
    else:
        if max_samples > 0:
            fire_samples = list(dataset.select(range(min(max_samples, len(dataset)))))
        else:
            fire_samples = list(dataset)

    needed_image_paths = set()
    for sample in fire_samples:
        img_path = sample.get("image")
        if img_path and isinstance(img_path, str):
            needed_image_paths.add(img_path)

    logger.info(f"Need {len(needed_image_paths)} unique images from COCO")

    # Load images from COCO (or skip if requested)
    coco_images = {}

    if skip_images:
        logger.warning(f"Skipping image loading (--skip-images flag set)")
        logger.warning(f"Image paths will be placeholders for testing only!")
    else:
        logger.info(f"Loading COCO dataset to fetch needed images...")
        logger.warning(f"This may take a long time for the first run...")

        # Try to load COCO with the images feature
        try:
            coco_train = load_dataset("detection-datasets/coco", split="train", streaming=True)
            coco_val = load_dataset("detection-datasets/coco", split="val", streaming=True)

            # Search for needed images
            for coco_sample in tqdm(coco_train, desc="Searching COCO train"):
                if 'image' in coco_sample and 'image_id' in coco_sample:
                    filename = f"{coco_sample['image_id']:012d}.jpg"
                    img_path = f"coco/train2017/{filename}"
                    if img_path in needed_image_paths:
                        coco_images[img_path] = coco_sample['image']
                        if len(coco_images) >= len(needed_image_paths):
                            break

            # Search validation set if needed
            if len(coco_images) < len(needed_image_paths):
                for coco_sample in tqdm(coco_val, desc="Searching COCO val"):
                    if 'image' in coco_sample and 'image_id' in coco_sample:
                        filename = f"{coco_sample['image_id']:012d}.jpg"
                        img_path = f"coco/val2017/{filename}"
                        if img_path in needed_image_paths:
                            coco_images[img_path] = coco_sample['image']
                            if len(coco_images) >= len(needed_image_paths):
                                break
        except Exception as e:
            logger.error(f"Failed to load COCO dataset: {e}")
            logger.warning(f"Cannot process samples without COCO images")

        logger.info(f"Found {len(coco_images)} images in COCO")

    output_file = output_dir / f"fire_bc_{split}.jsonl"
    stats = {
        "split": split,
        "samples_processed": 0,
        "samples_skipped": 0,
        "samples_no_image": 0,
        "samples_no_conversations": 0,
        "examples_generated": 0,
        "total_rounds": 0,
        "images_saved": 0,
        "errors": [],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        desc = f"Processing {split}"
        if max_samples > 0:
            desc += f" (max {max_samples})"

        for idx, sample in enumerate(tqdm(fire_samples, desc=desc)):
            if max_samples > 0 and idx >= max_samples:
                break

            sample_id = f"{split}_{idx:06d}"

            try:
                # Get image path from FIRE sample
                image_path_ref = sample.get("image")
                if image_path_ref is None or not isinstance(image_path_ref, str):
                    stats["samples_skipped"] += 1
                    stats["samples_no_image"] += 1
                    if len(stats["errors"]) < 100:
                        stats["errors"].append(f"{sample_id}: No image path")
                    continue

                # Handle image based on skip_images flag
                if skip_images:
                    # Use placeholder path for testing
                    image_save_path = f"/placeholder/images/{split}/{sample_id}.jpg"
                    stats["images_saved"] += 1
                else:
                    # Look up actual image from COCO
                    image = coco_images.get(image_path_ref)
                    if image is None:
                        stats["samples_skipped"] += 1
                        stats["samples_no_image"] += 1
                        if len(stats["errors"]) < 100:
                            stats["errors"].append(f"{sample_id}: Image not found in COCO: {image_path_ref}")
                        continue

                    # Save image to disk
                    image_save_path = str(split_image_dir / f"{sample_id}.jpg")
                    if not save_image(image, image_save_path, image_quality):
                        stats["samples_skipped"] += 1
                        if len(stats["errors"]) < 100:
                            stats["errors"].append(f"{sample_id}: Image save failed")
                        continue

                    stats["images_saved"] += 1

                # Generate behavior cloning examples
                examples = parse_fire_to_bc_examples(
                    sample, sample_id, image_save_path, max_history_rounds
                )

                if not examples:
                    stats["samples_skipped"] += 1
                    stats["samples_no_conversations"] += 1
                    if len(stats["errors"]) < 100:
                        stats["errors"].append(f"{sample_id}: No valid conversations")
                    continue

                # Write examples to JSONL
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

                stats["samples_processed"] += 1
                stats["examples_generated"] += len(examples)
                stats["total_rounds"] += len(examples)

            except Exception as e:
                stats["samples_skipped"] += 1
                if len(stats["errors"]) < 100:
                    stats["errors"].append(f"{sample_id}: {str(e)}")
                logger.warning(f"Error processing sample {idx}: {e}")
                continue

    logger.info(f"Wrote {stats['examples_generated']} examples to {output_file}")
    return stats


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)

    logger.info("=" * 60)
    logger.info("FIRE Behavior Cloning Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Max samples per split: {args.max_samples if args.max_samples > 0 else 'all'}")
    logger.info(f"Max history rounds: {args.max_history_rounds}")
    logger.info(f"Streaming: {args.streaming}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Image dir: {image_dir}")
    logger.info("=" * 60)

    all_stats = {}

    for split in args.splits:
        stats = process_split(
            dataset_id=args.dataset_id,
            split=split,
            output_dir=output_dir,
            image_dir=image_dir,
            max_samples=args.max_samples,
            max_history_rounds=args.max_history_rounds,
            streaming=args.streaming,
            image_quality=args.image_quality,
            skip_images=args.skip_images,
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

    total_examples = 0
    total_processed = 0
    total_skipped = 0

    for split, stats in all_stats.items():
        avg_examples = (
            stats["examples_generated"] / max(1, stats["samples_processed"])
        )
        logger.info(f"\n{split.upper()} SPLIT:")
        logger.info(f"  Samples processed: {stats['samples_processed']}")
        logger.info(f"  Samples skipped: {stats['samples_skipped']}")
        logger.info(f"    - No image: {stats['samples_no_image']}")
        logger.info(f"    - No conversations: {stats['samples_no_conversations']}")
        logger.info(f"  BC examples generated: {stats['examples_generated']}")
        logger.info(f"  Avg examples/sample: {avg_examples:.2f}")
        logger.info(f"  Images saved: {stats['images_saved']}")

        total_examples += stats["examples_generated"]
        total_processed += stats["samples_processed"]
        total_skipped += stats["samples_skipped"]

    logger.info("\n" + "-" * 60)
    logger.info(f"TOTAL: {total_examples} BC examples from {total_processed} samples")
    logger.info(f"       ({total_skipped} samples skipped)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
