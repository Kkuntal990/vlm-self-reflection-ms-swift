#!/usr/bin/env python3
"""
Prepare FIRE dataset with image path mappings.

Creates two output formats:
1. fire_messages_{split}.jsonl - ShareGPT format for training
2. fire_feedback_{split}.jsonl - Feedback-specific format

Applies path mappings for datasets where FIRE paths differ from actual paths:
- COCO train2014/val2014: adds COCO_ prefix
- SEED-Bench: removes .jpg extension
- ScienceQA: removes test/ subdirectory

Usage:
    python scripts/data_prep/prepare_fire_with_mappings.py \
        --output_dir /outputs/fire_preprocessed \
        --image_base_dir /outputs/image_base \
        --splits train test
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

# System prompt for VL Assistant (fire_messages - answer + refine training)
VL_ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate, "
    "detailed, and grounded answers based on the image and the user's instructions. "
    "When given feedback, critique, or scores, revise your response to improve "
    "correctness, specificity, and completeness."
)

# System prompt for Feedback Critic (fire_feedback - feedback generation training)
FEEDBACK_CRITIC_SYSTEM_PROMPT = (
    "You are a helpful assistant that provides constructive feedback on answers "
    "to visual questions. Given an image, a question, and an answer, and the conversation history identify "
    "what is correct, what is incorrect and provide specific critique based on visual evidence."
)

# System prompt for Last Response (fire_last_response - final answer training)
LAST_RESPONSE_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate, "
    "detailed, and grounded answers based on the image and the user's instructions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FIRE dataset with image path mappings")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/fire_preprocessed",
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
        default=["train", "test"],
        choices=["train", "test"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples per split (0 = all)",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip samples with missing images instead of failing",
    )
    return parser.parse_args()


def get_mapped_image_path(fire_path: str, image_base_dir: Path) -> str | None:
    """Get the actual image path on disk, applying path mappings.

    Handles naming mismatches between FIRE paths and actual files:
    - COCO train2014/val2014: adds COCO_ prefix
    - SEED-Bench: removes .jpg extension
    - ScienceQA: removes test/ subdirectory

    Args:
        fire_path: FIRE image path (e.g., "coco/train2014/000000123456.jpg")
        image_base_dir: Base directory where images are stored

    Returns:
        Absolute path to image or None if not found
    """
    # Try original path first
    full_path = image_base_dir / fire_path
    if full_path.exists():
        return str(full_path)

    # Try alternate paths
    alternates = []

    # COCO train2014/val2014: add COCO_ prefix
    if fire_path.startswith("coco/train2014/"):
        filename = fire_path.split("/")[-1]
        if not filename.startswith("COCO_"):
            alternates.append(f"coco/train2014/COCO_train2014_{filename}")
    elif fire_path.startswith("coco/val2014/"):
        filename = fire_path.split("/")[-1]
        if not filename.startswith("COCO_"):
            alternates.append(f"coco/val2014/COCO_val2014_{filename}")

    # SEED-Bench: files have no extension
    if fire_path.startswith("seedbench/") and fire_path.endswith(".jpg"):
        alternates.append(fire_path[:-4])  # Remove .jpg

    # ScienceQA: FIRE uses scienceqa/images/test/ but disk has scienceqa/images/
    if fire_path.startswith("scienceqa/images/test/"):
        alternates.append(fire_path.replace("scienceqa/images/test/", "scienceqa/images/"))

    for alt_path in alternates:
        alt_full_path = image_base_dir / alt_path
        if alt_full_path.exists():
            return str(alt_full_path)

    return None


def extract_question_text(question) -> str:
    """Extract question text from FIRE question field as-is."""
    if isinstance(question, dict) and "value" in question:
        text = question["value"]
    elif isinstance(question, str):
        text = question
    else:
        text = str(question)
    return text.strip()


def extract_answer_from_response(response_value: str) -> str:
    """Extract answer from student response, ignoring thought."""
    if not response_value:
        return ""
    if "Answer:" in response_value:
        parts = response_value.split("Answer:", 1)
        return parts[1].strip()
    return response_value.strip()


def extract_feedback_text(feedback_value: str) -> str:
    """Extract feedback text from teacher response."""
    if not feedback_value:
        return ""
    if "Feedback:" in feedback_value:
        return feedback_value.split("Feedback:", 1)[1].strip()
    return feedback_value.strip()


def parse_fire_sample(sample: dict) -> dict | None:
    """Parse a FIRE sample into structured rounds.

    Returns:
        Dict with question, rounds list, and metadata, or None if invalid
    """
    question_text = extract_question_text(sample.get("question", ""))
    conversations = sample.get("conversations", [])

    if not conversations or not question_text:
        return None

    # Parse student-teacher rounds
    rounds = []
    i = 0
    while i < len(conversations):
        turn = conversations[i]

        if turn.get("role") == "student" and turn.get("type") == "response":
            response_value = turn.get("value", "")
            answer = extract_answer_from_response(response_value)

            if answer:
                feedback = None
                if i + 1 < len(conversations):
                    next_turn = conversations[i + 1]
                    if next_turn.get("role") == "teacher" and next_turn.get("type") == "feedback":
                        feedback_value = next_turn.get("value", "")
                        feedback = extract_feedback_text(feedback_value)
                        i += 1

                rounds.append({"answer": answer, "feedback": feedback})

        i += 1

    if not rounds:
        return None

    return {
        "question": question_text,
        "rounds": rounds,
        "source": sample.get("source", "unknown"),
    }


def create_messages_format(
    parsed: dict,
    image_path: str,
    system_prompt: str,
) -> dict:
    """Create messages format for training.

    Format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "<image>\\n{question}"},
            {"role": "assistant", "content": "...", "loss": true},
            {"role": "user", "content": "{feedback}"},
            {"role": "assistant", "content": "...", "loss": true},
            ...
        ],
        "images": ["/path/to/image.jpg"]
    }
    """
    messages = []

    # System message
    messages.append(
        {
            "role": "system",
            "content": system_prompt,
        }
    )

    # First user message with question (includes <image> tag from FIRE)
    messages.append(
        {
            "role": "user",
            "content": parsed["question"],
        }
    )

    # First assistant response
    messages.append(
        {
            "role": "assistant",
            "content": parsed["rounds"][0]["answer"],
            "loss": True,
        }
    )

    # Subsequent feedback -> refined answer exchanges
    for i in range(1, len(parsed["rounds"])):
        prev_feedback = parsed["rounds"][i - 1]["feedback"]
        current_answer = parsed["rounds"][i]["answer"]

        if prev_feedback:
            # User provides feedback
            messages.append(
                {
                    "role": "user",
                    "content": prev_feedback,
                }
            )
            # Assistant provides refined answer
            messages.append(
                {
                    "role": "assistant",
                    "content": current_answer,
                    "loss": True,
                }
            )

    return {
        "messages": messages,
        "images": [image_path],
    }


def create_feedback_format(
    parsed: dict,
    image_path: str,
    system_prompt: str,
) -> dict:
    """Create feedback format (assistant asks question, gives feedback).

    Format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "assistant", "content": "{question}", "loss": false},
            {"role": "user", "content": "{answer}"},
            {"role": "assistant", "content": "{feedback}", "loss": true},
            {"role": "user", "content": "{refined_answer}"},
            {"role": "assistant", "content": "{feedback}", "loss": true},
            ...
        ],
        "images": ["/path/to/image.jpg"]
    }
    """
    messages = []

    # System message
    messages.append(
        {
            "role": "system",
            "content": system_prompt,
        }
    )

    # Assistant asks the question (no loss)
    messages.append(
        {
            "role": "assistant",
            "content": parsed["question"],
            "loss": False,
        }
    )

    # Process each round: user provides answer, assistant provides feedback
    for round_data in parsed["rounds"]:
        answer = round_data["answer"]
        feedback = round_data["feedback"]

        # User provides answer
        messages.append(
            {
                "role": "user",
                "content": answer,
            }
        )

        # Assistant provides feedback (if available)
        if feedback:
            messages.append(
                {
                    "role": "assistant",
                    "content": feedback,
                    "loss": True,
                }
            )

    return {
        "messages": messages,
        "images": [image_path],
    }


def create_last_response_format(
    parsed: dict,
    image_path: str,
    system_prompt: str,
) -> dict:
    """Create last response format (question -> final answer only).

    Format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "{question}"},
            {"role": "assistant", "content": "{final_answer}", "loss": true}
        ],
        "images": ["/path/to/image.jpg"]
    }
    """
    # Get the final answer (last round)
    final_answer = parsed["rounds"][-1]["answer"]

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": parsed["question"],
        },
        {
            "role": "assistant",
            "content": final_answer,
            "loss": True,
        },
    ]

    return {
        "messages": messages,
        "images": [image_path],
    }


def process_split(
    dataset_id: str,
    split: str,
    output_dir: Path,
    image_base_dir: Path,
    max_samples: int,
    skip_missing: bool,
    messages_prompt: str,
    feedback_prompt: str,
    last_response_prompt: str,
) -> dict:
    """Process a single dataset split."""
    logger.info(f"Processing {split} split...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FIRE dataset
    logger.info(f"Loading FIRE dataset {split} split...")
    dataset = load_dataset(dataset_id, split=split)

    total_samples = len(dataset)
    if max_samples > 0:
        total_samples = min(max_samples, total_samples)

    logger.info(f"Processing {total_samples} samples...")

    # Sources to filter out due to data quality issues
    filtered_sources = {"synthdog-en"}

    stats = {
        "split": split,
        "total_samples": total_samples,
        "processed": 0,
        "skipped_no_image": 0,
        "skipped_no_conversations": 0,
        "skipped_image_not_found": 0,
        "skipped_filtered_source": 0,
        "by_source": {},
    }

    messages_file = output_dir / f"fire_messages_{split}.jsonl"
    feedback_file = output_dir / f"fire_feedback_{split}.jsonl"
    last_response_file = output_dir / f"fire_last_response_{split}.jsonl"

    with (
        open(messages_file, "w") as f_messages,
        open(feedback_file, "w") as f_feedback,
        open(last_response_file, "w") as f_last_response,
    ):
        for idx in tqdm(range(total_samples), desc=f"Processing {split}"):
            sample = dataset[idx]

            # Get image path
            fire_image_path = sample.get("image")
            if not fire_image_path or not isinstance(fire_image_path, str):
                stats["skipped_no_image"] += 1
                continue

            # Filter out problematic sources (e.g., synthdog-en has image-text misalignment)
            if any(src in fire_image_path for src in filtered_sources):
                stats["skipped_filtered_source"] += 1
                continue

            # Get mapped image path
            actual_image_path = get_mapped_image_path(fire_image_path, image_base_dir)
            if actual_image_path is None:
                stats["skipped_image_not_found"] += 1
                if not skip_missing:
                    logger.warning(f"Image not found: {fire_image_path}")
                continue

            # Parse sample
            parsed = parse_fire_sample(sample)
            if parsed is None:
                stats["skipped_no_conversations"] += 1
                continue

            # Create output formats with format-specific prompts
            messages_data = create_messages_format(parsed, actual_image_path, messages_prompt)
            feedback_data = create_feedback_format(parsed, actual_image_path, feedback_prompt)
            last_response_data = create_last_response_format(
                parsed, actual_image_path, last_response_prompt
            )

            # Write to files
            f_messages.write(json.dumps(messages_data, ensure_ascii=False) + "\n")
            f_feedback.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
            f_last_response.write(json.dumps(last_response_data, ensure_ascii=False) + "\n")

            stats["processed"] += 1

            # Track by source
            source = parsed["source"]
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

    logger.info(f"Wrote {stats['processed']} samples to {messages_file}")
    logger.info(f"Wrote {stats['processed']} samples to {feedback_file}")
    logger.info(f"Wrote {stats['processed']} samples to {last_response_file}")

    return stats


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    image_base_dir = Path(args.image_base_dir)

    logger.info("=" * 60)
    logger.info("FIRE Dataset Preparation with Path Mappings")
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

    all_stats = {}

    for split in args.splits:
        stats = process_split(
            dataset_id=args.dataset_id,
            split=split,
            output_dir=output_dir,
            image_base_dir=image_base_dir,
            max_samples=args.max_samples,
            skip_missing=args.skip_missing,
            messages_prompt=VL_ASSISTANT_SYSTEM_PROMPT,
            feedback_prompt=FEEDBACK_CRITIC_SYSTEM_PROMPT,
            last_response_prompt=LAST_RESPONSE_SYSTEM_PROMPT,
        )
        all_stats[split] = stats

    # Write stats
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Stats written to {stats_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    total_processed = 0
    total_skipped = 0

    for split, stats in all_stats.items():
        skipped = (
            stats["skipped_no_image"]
            + stats["skipped_no_conversations"]
            + stats["skipped_image_not_found"]
            + stats["skipped_filtered_source"]
        )
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"    - No image path: {stats['skipped_no_image']}")
        logger.info(f"    - Image not found: {stats['skipped_image_not_found']}")
        logger.info(f"    - No conversations: {stats['skipped_no_conversations']}")
        logger.info(f"    - Filtered source (synthdog-en): {stats['skipped_filtered_source']}")

        total_processed += stats["processed"]
        total_skipped += skipped

    logger.info("\n" + "-" * 60)
    logger.info(f"TOTAL: {total_processed} processed, {total_skipped} skipped")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
