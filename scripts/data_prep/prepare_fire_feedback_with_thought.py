#!/usr/bin/env python3
"""
Add original Thought+Answer back into FIRE feedback dataset's user messages.

Reads fire_feedback_train_v2.jsonl (where user role has student answers without
thought) and replaces user content with the full "Thought: ...\nAnswer: ..."
from the original FIRE dataset.

Usage:
    python scripts/data_prep/prepare_fire_feedback_with_thought.py \
        --input_file /outputs/fire_preprocessed_v3/fire_feedback_train_v2.jsonl \
        --output_file /outputs/fire_preprocessed_v3/fire_feedback_train_v2_with_thought.jsonl \
        --max_samples 0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path


os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from datasets import load_dataset
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FILTERED_SOURCES = {"synthdog-en"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add Thought+Answer to FIRE feedback dataset user messages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/outputs/fire_preprocessed_v3/fire_feedback_train_v2.jsonl",
        help="Path to existing feedback JSONL file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/outputs/fire_preprocessed_v3/fire_feedback_train_v2_with_thought.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="PengxiangLi/FIRE",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )
    return parser.parse_args()


def extract_question_text(question: dict | str) -> str:
    """Extract question text from FIRE question field."""
    if isinstance(question, dict) and "value" in question:
        text = question["value"]
    elif isinstance(question, str):
        text = question
    else:
        text = str(question)
    return text.strip()


def get_student_responses(sample: dict) -> list[str]:
    """Extract all full student responses (with Thought+Answer) from FIRE sample.

    Args:
        sample: Raw FIRE dataset sample

    Returns:
        List of full student response strings in order
    """
    conversations = sample.get("conversations", [])
    responses = []

    for turn in conversations:
        if turn.get("role") == "student" and turn.get("type") == "response":
            value = turn.get("value", "").strip()
            if value:
                responses.append(value)

    return responses


def build_fire_lookup(dataset_id: str) -> dict[str, list[str]]:
    """Build lookup from (question, image) to student responses.

    Args:
        dataset_id: HuggingFace dataset ID

    Returns:
        Dict mapping "question|||image_path" to list of full student responses
    """
    logger.info("Loading FIRE dataset from HuggingFace...")
    dataset = load_dataset(dataset_id, split="train")
    logger.info(f"Loaded {len(dataset)} samples")

    lookup = {}

    for idx in tqdm(range(len(dataset)), desc="Building lookup"):
        sample = dataset[idx]

        fire_image_path = sample.get("image", "")
        if not fire_image_path or not isinstance(fire_image_path, str):
            continue

        if any(src in fire_image_path for src in FILTERED_SOURCES):
            continue

        question_text = extract_question_text(sample.get("question", ""))
        # Strip <image> tag and normalize for matching (feedback file has it stripped)
        question_normalized = question_text.replace("<image>", "").strip()
        if not question_normalized:
            continue

        responses = get_student_responses(sample)
        if not responses:
            continue

        key = f"{question_normalized}|||{fire_image_path}"
        lookup[key] = responses

    logger.info(f"Built lookup with {len(lookup)} entries")
    return lookup


def normalize_question(content: str) -> str:
    """Normalize question text from feedback format for matching.

    The assistant message contains the question, sometimes with <image> tag.
    Strip the tag and whitespace for matching.
    """
    return content.replace("<image>", "").strip()


def extract_image_suffix(image_path: str) -> str:
    """Extract the FIRE-style image path suffix from the full path.

    E.g., '/outputs/image_base/coco/train2017/000000496160.jpg'
    -> 'coco/train2017/000000496160.jpg'
    """
    # Remove the image_base prefix
    markers = ["/image_base/", "/outputs/image_base/"]
    for marker in markers:
        if marker in image_path:
            return image_path.split(marker, 1)[1]
    return image_path


def reverse_map_image_path(suffix: str) -> list[str]:
    """Generate possible FIRE image paths from the actual on-disk suffix.

    Reverses the mappings in get_mapped_image_path:
    - COCO train2014/val2014: remove COCO_ prefix
    - SEED-Bench: add .jpg extension
    - ScienceQA: add test/ subdirectory

    Args:
        suffix: On-disk image path suffix

    Returns:
        List of possible FIRE-style paths (original suffix + alternates)
    """
    candidates = [suffix]

    # Reverse COCO mapping: COCO_train2014_000000123456.jpg -> 000000123456.jpg
    if suffix.startswith("coco/train2014/COCO_train2014_"):
        filename = suffix.split("COCO_train2014_")[1]
        candidates.append(f"coco/train2014/{filename}")
    elif suffix.startswith("coco/val2014/COCO_val2014_"):
        filename = suffix.split("COCO_val2014_")[1]
        candidates.append(f"coco/val2014/{filename}")

    # Reverse SEED-Bench: no extension -> add .jpg
    if suffix.startswith("seedbench/") and not suffix.endswith(".jpg"):
        candidates.append(suffix + ".jpg")

    # Reverse ScienceQA: scienceqa/images/ -> scienceqa/images/test/
    if suffix.startswith("scienceqa/images/") and "/test/" not in suffix:
        candidates.append(suffix.replace("scienceqa/images/", "scienceqa/images/test/"))

    return candidates


def main() -> None:
    """Main function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("FIRE Feedback Dataset - Add Thought to User Messages")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Output: {args.output_file}")
    logger.info("=" * 60)

    # Build lookup from FIRE dataset
    lookup = build_fire_lookup(args.dataset_id)

    # Process existing feedback file
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    with open(input_path) as f_count:
        total_lines = sum(1 for _ in f_count)
    if args.max_samples > 0:
        total_lines = min(args.max_samples, total_lines)

    stats = {
        "total": total_lines,
        "matched": 0,
        "unmatched": 0,
        "user_messages_replaced": 0,
    }

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for _ in tqdm(range(total_lines), desc="Processing"):
            line = f_in.readline()
            sample = json.loads(line)

            # Extract question from assistant message [1]
            question_msg = sample["messages"][1]
            question_text = normalize_question(question_msg["content"])

            # Extract image path and reverse-map to FIRE path
            image_path = sample["images"][0]
            image_suffix = extract_image_suffix(image_path)
            fire_candidates = reverse_map_image_path(image_suffix)

            # Try to find matching FIRE sample
            responses = None
            for fire_path in fire_candidates:
                key = f"{question_text}|||{fire_path}"
                if key in lookup:
                    responses = lookup[key]
                    break

            if responses is None:
                # No match found, keep original
                stats["unmatched"] += 1
                f_out.write(line)
                continue

            stats["matched"] += 1

            # Replace user messages with full thought+answer
            response_idx = 0
            for msg in sample["messages"]:
                if msg["role"] == "user" and response_idx < len(responses):
                    msg["content"] = responses[response_idx]
                    stats["user_messages_replaced"] += 1
                    response_idx += 1

            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples: {stats['total']}")
    logger.info(f"Matched: {stats['matched']}")
    logger.info(f"Unmatched: {stats['unmatched']}")
    logger.info(f"User messages replaced: {stats['user_messages_replaced']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
