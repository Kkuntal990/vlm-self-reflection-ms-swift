#!/usr/bin/env python3
"""
Prepare FIRE dataset in ShareGPT format for ms-swift.

Converts FIRE multi-round student-teacher conversations into ShareGPT format:
- Each FIRE sample becomes one multi-round conversation
- Question starts as 'human' with <image> token
- Student answers become 'gpt' responses
- Teacher feedback becomes 'human' responses
- Ignores thought and score fields

Usage:
    python scripts/prepare_fire_sharegpt.py \
        --output_dir /outputs/fire_sharegpt \
        --image_dir /cache/fire_images \
        --max_samples 0 \
        --splits train test \
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
from typing import Union

# Set HuggingFace download timeout to avoid network issues
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")  # 5 minutes

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default system prompt - can be customized via command line or by editing here
DEFAULT_SYSTEM_PROMPT = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FIRE dataset to ShareGPT format for ms-swift"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/fire_sharegpt",
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
    and may include <image> tokens which we preserve for ShareGPT format.
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
    """Extract feedback text and prepend score information.

    Teacher feedback in FIRE format looks like:
    'Score: 6.\nFeedback: ...\n'

    We extract the score and feedback, then format as:
    'A score of 6 is given to the answer. ...'
    """
    if not feedback_value:
        return ""

    score = None
    feedback = ""

    # Extract score
    if "Score:" in feedback_value:
        score_part = feedback_value.split("Score:", 1)[1]
        # Extract the number (handle formats like "Score: 6." or "Score: 6")
        score_str = score_part.split("\n")[0].strip().rstrip(".")
        try:
            score = int(score_str)
        except ValueError:
            # If score is not a simple integer, try to parse as float
            try:
                score = float(score_str)
            except ValueError:
                score = None

    # Extract feedback
    if "Feedback:" in feedback_value:
        feedback = feedback_value.split("Feedback:", 1)[1].strip()
    elif score is None:
        # If no "Feedback:" marker and no score, return as is
        return feedback_value.strip()

    # Build final feedback with score prepended
    if score is not None and feedback:
        return f"A score of {score} is given to the answer. {feedback}"
    elif feedback:
        return feedback
    else:
        return feedback_value.strip()


def parse_fire_to_sharegpt(
    sample: dict,
    sample_id: str,
    image_path: str,
    system_prompt: str = "",
) -> dict | None:
    """Convert a single FIRE sample into ShareGPT format.

    ShareGPT format pairs human/assistant exchanges:
    {
        "system": "system prompt (optional)",
        "conversation": [
            {"human": "query1", "assistant": "response1"},
            {"human": "query2", "assistant": "response2"}
        ],
        "images": ["/path/to/image.jpg"]
    }

    Args:
        sample: FIRE dataset row containing question, image, conversations
        sample_id: Unique identifier for logging/debugging
        image_path: Absolute path to saved image file
        system_prompt: System prompt to include in the conversation

    Returns:
        ShareGPT format dict or None if parsing fails
    """
    question_text = extract_question_text(sample.get("question", ""))
    conversations = sample.get("conversations", [])

    if not conversations:
        logger.debug(f"{sample_id}: No conversations found")
        return None

    if not question_text:
        logger.warning(f"{sample_id}: Empty question text")
        return None

    # Parse student-teacher rounds into (question, answer) pairs
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

    # Build ShareGPT conversation as paired exchanges
    conversation = []

    # First exchange: question -> first answer
    conversation.append({
        "human": question_text,
        "assistant": rounds[0][0]
    })

    # Subsequent exchanges: feedback -> next answer
    for i in range(1, len(rounds)):
        prev_feedback = rounds[i - 1][1]
        current_answer = rounds[i][0]

        if prev_feedback:  # Only add if there was feedback
            conversation.append({
                "human": prev_feedback,
                "assistant": current_answer
            })

    # Return ShareGPT format with system prompt and images field
    return {
        "system": system_prompt,
        "conversation": conversation,
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

    # Categorize images by source
    coco_paths = {p for p in needed_image_paths if p.startswith('coco/')}
    mathvista_paths = {p for p in needed_image_paths if p.startswith('mathvista/')}
    other_paths = needed_image_paths - coco_paths - mathvista_paths

    logger.info(f"Need {len(needed_image_paths)} unique images total:")
    logger.info(f"  - COCO: {len(coco_paths)}")
    logger.info(f"  - MathVista: {len(mathvista_paths)}")
    if other_paths:
        logger.warning(f"  - Unknown sources: {len(other_paths)}")

    # Load images from source datasets (or skip if requested)
    source_images = {}

    if skip_images:
        logger.warning(f"Skipping image loading (--skip-images flag set)")
        logger.warning(f"Image paths will be placeholders for testing only!")
    else:
        # Load COCO images if needed
        if coco_paths:
            logger.info(f"Loading COCO dataset to fetch {len(coco_paths)} images...")
            logger.warning(f"This may take a long time for the first run...")

            try:
                coco_train = load_dataset("detection-datasets/coco", split="train", streaming=True)
                coco_val = load_dataset("detection-datasets/coco", split="val", streaming=True)

                # Search for needed images in COCO train
                for coco_sample in tqdm(coco_train, desc="Searching COCO train"):
                    if 'image' in coco_sample and 'image_id' in coco_sample:
                        filename = f"{coco_sample['image_id']:012d}.jpg"
                        img_path = f"coco/train2017/{filename}"
                        if img_path in coco_paths:
                            source_images[img_path] = coco_sample['image']
                            if len([p for p in source_images if p.startswith('coco/')]) >= len(coco_paths):
                                break

                # Search COCO validation set if needed
                coco_found = len([p for p in source_images if p.startswith('coco/')])
                if coco_found < len(coco_paths):
                    for coco_sample in tqdm(coco_val, desc="Searching COCO val"):
                        if 'image' in coco_sample and 'image_id' in coco_sample:
                            filename = f"{coco_sample['image_id']:012d}.jpg"
                            img_path = f"coco/val2017/{filename}"
                            if img_path in coco_paths:
                                source_images[img_path] = coco_sample['image']
                                if len([p for p in source_images if p.startswith('coco/')]) >= len(coco_paths):
                                    break

                coco_found = len([p for p in source_images if p.startswith('coco/')])
                logger.info(f"Found {coco_found}/{len(coco_paths)} COCO images")
            except Exception as e:
                logger.error(f"Failed to load COCO dataset: {e}")
                logger.warning(f"Cannot process samples without COCO images")

        # Load MathVista images if needed
        if mathvista_paths:
            logger.info(f"Loading MathVista dataset to fetch {len(mathvista_paths)} images...")

            try:
                # MathVista dataset - try different possible dataset paths
                mathvista_dataset = None
                dataset_attempts = [
                    "AI4Math/MathVista",
                    "MathVista/MathVista",
                    "mathvista/MathVista"
                ]

                for dataset_name in dataset_attempts:
                    try:
                        logger.info(f"Trying to load {dataset_name}...")
                        mathvista_dataset = load_dataset(dataset_name, split="testmini", streaming=True)
                        logger.info(f"Successfully loaded {dataset_name}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load {dataset_name}: {e}")
                        continue

                if mathvista_dataset is None:
                    logger.error("Could not load MathVista dataset from any known source")
                    logger.warning(f"Cannot process {len(mathvista_paths)} samples without MathVista images")
                else:
                    # Extract filename to path mapping for MathVista
                    mathvista_filename_map = {}
                    for path in mathvista_paths:
                        # Extract filename from mathvista/images/123.jpg
                        filename = path.split('/')[-1]
                        mathvista_filename_map[filename] = path

                    # Search MathVista dataset
                    for mv_sample in tqdm(mathvista_dataset, desc="Searching MathVista"):
                        # MathVista samples have 'image' field and potentially 'pid' or similar
                        if 'image' in mv_sample:
                            # Try to match by filename - MathVista might have different field names
                            # Common patterns: 'pid', 'question_id', 'image_path', etc.
                            sample_id = None
                            for id_field in ['pid', 'question_id', 'id', 'image_id']:
                                if id_field in mv_sample:
                                    sample_id = str(mv_sample[id_field])
                                    break

                            if sample_id:
                                # Try matching with .jpg extension
                                for ext in ['.jpg', '.png', '.jpeg', '']:
                                    test_filename = f"{sample_id}{ext}"
                                    if test_filename in mathvista_filename_map:
                                        img_path = mathvista_filename_map[test_filename]
                                        source_images[img_path] = mv_sample['image']
                                        break

                        # Check if we found all MathVista images
                        mathvista_found = len([p for p in source_images if p.startswith('mathvista/')])
                        if mathvista_found >= len(mathvista_paths):
                            break

                    mathvista_found = len([p for p in source_images if p.startswith('mathvista/')])
                    logger.info(f"Found {mathvista_found}/{len(mathvista_paths)} MathVista images")

            except Exception as e:
                logger.error(f"Failed to load MathVista dataset: {e}")
                logger.warning(f"Cannot process samples without MathVista images")

        logger.info(f"Total images loaded: {len(source_images)}/{len(needed_image_paths)}")

    output_file = output_dir / f"fire_sharegpt_{split}.jsonl"
    stats = {
        "split": split,
        "samples_processed": 0,
        "samples_skipped": 0,
        "samples_no_image": 0,
        "samples_no_conversations": 0,
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
                    # Look up actual image from source datasets
                    image = source_images.get(image_path_ref)
                    if image is None:
                        stats["samples_skipped"] += 1
                        stats["samples_no_image"] += 1
                        if len(stats["errors"]) < 100:
                            stats["errors"].append(f"{sample_id}: Image not found: {image_path_ref}")
                        continue

                    # Save image to disk
                    image_save_path = str(split_image_dir / f"{sample_id}.jpg")
                    if not save_image(image, image_save_path, image_quality):
                        stats["samples_skipped"] += 1
                        if len(stats["errors"]) < 100:
                            stats["errors"].append(f"{sample_id}: Image save failed")
                        continue

                    stats["images_saved"] += 1

                # Generate ShareGPT format
                sharegpt_example = parse_fire_to_sharegpt(
                    sample, sample_id, image_save_path, system_prompt
                )

                if not sharegpt_example:
                    stats["samples_skipped"] += 1
                    stats["samples_no_conversations"] += 1
                    if len(stats["errors"]) < 100:
                        stats["errors"].append(f"{sample_id}: No valid conversations")
                    continue

                # Write to JSONL
                f.write(json.dumps(sharegpt_example, ensure_ascii=False) + "\n")

                stats["samples_processed"] += 1
                # Count conversation rounds
                stats["total_rounds"] += len(sharegpt_example["conversation"])

            except Exception as e:
                stats["samples_skipped"] += 1
                if len(stats["errors"]) < 100:
                    stats["errors"].append(f"{sample_id}: {str(e)}")
                logger.warning(f"Error processing sample {idx}: {e}")
                continue

    logger.info(f"Wrote {stats['samples_processed']} conversations to {output_file}")
    return stats


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)

    logger.info("=" * 60)
    logger.info("FIRE ShareGPT Format Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Max samples per split: {args.max_samples if args.max_samples > 0 else 'all'}")
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
            streaming=args.streaming,
            image_quality=args.image_quality,
            skip_images=args.skip_images,
            system_prompt=args.system_prompt,
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
    total_rounds = 0

    for split, stats in all_stats.items():
        avg_rounds = (
            stats["total_rounds"] / max(1, stats["samples_processed"])
        )
        logger.info(f"\n{split.upper()} SPLIT:")
        logger.info(f"  Samples processed: {stats['samples_processed']}")
        logger.info(f"  Samples skipped: {stats['samples_skipped']}")
        logger.info(f"    - No image: {stats['samples_no_image']}")
        logger.info(f"    - No conversations: {stats['samples_no_conversations']}")
        logger.info(f"  Total conversation rounds: {stats['total_rounds']}")
        logger.info(f"  Avg rounds/sample: {avg_rounds:.2f}")
        logger.info(f"  Images saved: {stats['images_saved']}")

        total_processed += stats["samples_processed"]
        total_skipped += stats["samples_skipped"]
        total_rounds += stats["total_rounds"]

    logger.info("\n" + "-" * 60)
    logger.info(f"TOTAL: {total_processed} conversations with {total_rounds} rounds")
    logger.info(f"       ({total_skipped} samples skipped)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
