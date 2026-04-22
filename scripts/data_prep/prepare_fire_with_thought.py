#!/usr/bin/env python3
"""
Prepare FIRE dataset preserving original Thought+Answer in assistant responses.

Creates fire_messages_train_last_loss_only_with_thought.jsonl with the same
structure as fire_messages_train_last_loss_only.jsonl but keeping the full
"Thought: ...\nAnswer: ..." student response instead of just the answer.

Usage:
    python scripts/data_prep/prepare_fire_with_thought.py \
        --output_dir /outputs/fire_preprocessed_v3 \
        --image_base_dir /outputs/image_base \
        --splits train
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

# Same system prompt as the original preprocessing
VL_ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate, "
    "detailed, and grounded answers based on the image and the user's instructions. "
    "When given feedback, critique, or scores, revise your response to improve "
    "correctness, specificity, and completeness."
)

# Sources to filter out due to data quality issues
FILTERED_SOURCES = {"synthdog-en"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare FIRE dataset with original Thought+Answer responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/fire_preprocessed_v3",
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs/image_base",
        help="Base directory where images are stored",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="PengxiangLi/FIRE",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "test"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples per split (0 = all)",
    )
    return parser.parse_args()


def get_mapped_image_path(fire_path: str, image_base_dir: Path) -> str | None:
    """Get the actual image path on disk, applying path mappings.

    Args:
        fire_path: FIRE image path (e.g., "coco/train2014/000000123456.jpg")
        image_base_dir: Base directory where images are stored

    Returns:
        Absolute path to image or None if not found
    """
    full_path = image_base_dir / fire_path
    if full_path.exists():
        return str(full_path)

    alternates = []

    if fire_path.startswith("coco/train2014/"):
        filename = fire_path.split("/")[-1]
        if not filename.startswith("COCO_"):
            alternates.append(f"coco/train2014/COCO_train2014_{filename}")
    elif fire_path.startswith("coco/val2014/"):
        filename = fire_path.split("/")[-1]
        if not filename.startswith("COCO_"):
            alternates.append(f"coco/val2014/COCO_val2014_{filename}")

    if fire_path.startswith("seedbench/") and fire_path.endswith(".jpg"):
        alternates.append(fire_path[:-4])

    if fire_path.startswith("scienceqa/images/test/"):
        alternates.append(fire_path.replace("scienceqa/images/test/", "scienceqa/images/"))

    for alt_path in alternates:
        alt_full_path = image_base_dir / alt_path
        if alt_full_path.exists():
            return str(alt_full_path)

    return None


def extract_question_text(question: dict | str) -> str:
    """Extract question text from FIRE question field as-is."""
    if isinstance(question, dict) and "value" in question:
        text = question["value"]
    elif isinstance(question, str):
        text = question
    else:
        text = str(question)
    return text.strip()


def extract_full_response(response_value: str) -> str:
    """Return the full student response with Thought+Answer, just stripped.

    Args:
        response_value: Raw student response value from FIRE dataset

    Returns:
        Cleaned full response preserving Thought and Answer
    """
    if not response_value:
        return ""
    return response_value.strip()


def extract_feedback_text(feedback_value: str) -> str:
    """Extract feedback text from teacher response (strips Score: prefix)."""
    if not feedback_value:
        return ""
    if "Feedback:" in feedback_value:
        return feedback_value.split("Feedback:", 1)[1].strip()
    return feedback_value.strip()


def process_sample(sample: dict, image_base_dir: Path) -> dict | None:
    """Process a single FIRE sample into messages format with full thought.

    Args:
        sample: Raw FIRE dataset sample
        image_base_dir: Base directory for image resolution

    Returns:
        Dict with messages and images, or None if invalid/skipped
    """
    fire_image_path = sample.get("image")
    if not fire_image_path or not isinstance(fire_image_path, str):
        return None

    if any(src in fire_image_path for src in FILTERED_SOURCES):
        return None

    actual_image_path = get_mapped_image_path(fire_image_path, image_base_dir)
    if actual_image_path is None:
        return None

    question_text = extract_question_text(sample.get("question", ""))
    conversations = sample.get("conversations", [])

    if not conversations or not question_text:
        return None

    # Parse student-teacher rounds keeping full response
    rounds = []
    i = 0
    while i < len(conversations):
        turn = conversations[i]

        if turn.get("role") == "student" and turn.get("type") == "response":
            response_value = turn.get("value", "")
            full_response = extract_full_response(response_value)

            if full_response:
                feedback = None
                if i + 1 < len(conversations):
                    next_turn = conversations[i + 1]
                    if next_turn.get("role") == "teacher" and next_turn.get("type") == "feedback":
                        feedback_value = next_turn.get("value", "")
                        feedback = extract_feedback_text(feedback_value)
                        i += 1

                rounds.append({"answer": full_response, "feedback": feedback})

        i += 1

    if not rounds:
        return None

    # Build messages in last_loss_only format
    messages = []

    messages.append(
        {
            "role": "system",
            "content": VL_ASSISTANT_SYSTEM_PROMPT,
        }
    )

    messages.append(
        {
            "role": "user",
            "content": question_text,
        }
    )

    # First assistant response (loss=false since not last)
    messages.append(
        {
            "role": "assistant",
            "content": rounds[0]["answer"],
            "loss": False,
        }
    )

    # Subsequent feedback -> refined answer exchanges
    for j in range(1, len(rounds)):
        prev_feedback = rounds[j - 1]["feedback"]
        current_answer = rounds[j]["answer"]

        if prev_feedback:
            messages.append(
                {
                    "role": "user",
                    "content": prev_feedback,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": current_answer,
                    "loss": False,
                }
            )

    # Set last assistant turn to loss=true
    for k in range(len(messages) - 1, -1, -1):
        if messages[k]["role"] == "assistant":
            messages[k]["loss"] = True
            break

    return {
        "messages": messages,
        "images": [actual_image_path],
    }


def process_split(
    dataset_id: str,
    split: str,
    output_dir: Path,
    image_base_dir: Path,
    max_samples: int,
) -> dict:
    """Process a single dataset split.

    Args:
        dataset_id: HuggingFace dataset ID
        split: Dataset split name
        output_dir: Output directory for JSONL files
        image_base_dir: Base directory for images
        max_samples: Max samples to process (0 = all)

    Returns:
        Statistics dictionary
    """
    logger.info(f"Processing {split} split...")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading FIRE dataset {split} split...")
    dataset = load_dataset(dataset_id, split=split)

    total_samples = len(dataset)
    if max_samples > 0:
        total_samples = min(max_samples, total_samples)

    logger.info(f"Processing {total_samples} samples...")

    stats = {
        "split": split,
        "total_samples": total_samples,
        "processed": 0,
        "skipped": 0,
    }

    output_file = output_dir / f"fire_messages_{split}_last_loss_only_with_thought.jsonl"

    with open(output_file, "w") as f:
        for idx in tqdm(range(total_samples), desc=f"Processing {split}"):
            sample = dataset[idx]
            result = process_sample(sample, image_base_dir)

            if result is None:
                stats["skipped"] += 1
                continue

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            stats["processed"] += 1

    logger.info(f"Wrote {stats['processed']} samples to {output_file}")
    logger.info(f"Skipped {stats['skipped']} samples")

    return stats


def main() -> None:
    """Main function."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    image_base_dir = Path(args.image_base_dir)

    logger.info("=" * 60)
    logger.info("FIRE Dataset with Thought+Answer Responses")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Image base dir: {image_base_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")
    logger.info("=" * 60)

    if not image_base_dir.exists():
        logger.error(f"Image base directory not found: {image_base_dir}")
        sys.exit(1)

    for split in args.splits:
        process_split(
            dataset_id=args.dataset_id,
            split=split,
            output_dir=output_dir,
            image_base_dir=image_base_dir,
            max_samples=args.max_samples,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
