#!/usr/bin/env python3
"""
Prepare LVLM_NLF dataset into messages format for ms-swift training.

Converts HuggingFace dataset YangyiYY/LVLM_NLF into multi-turn messages JSONL
with last-loss-only masking (only the final assistant turn has loss=true).

Source format (LVLM_NLF interaction):
    [
        {"user": "question", "model": "answer", "rating": "4", "critique_NLF": "..."},
        {"user": "feedback", "model": "refined_answer", ...},
        ...
    ]

Target format (fire_messages_train_last_loss_only.jsonl):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "<question>\\n<image>"},
            {"role": "assistant", "content": "...", "loss": false},
            {"role": "user", "content": "<feedback>"},
            {"role": "assistant", "content": "...", "loss": true}
        ],
        "images": ["/path/to/image.jpg"]
    }

Usage:
    python scripts/data_prep/prepare_lvlm_nlf.py \
        --output_dir /outputs/lvlm_nlf_preprocessed \
        --image_base_dir /outputs/image_base/coco/train2017 \
        --max_samples 10

    # Full dataset
    python scripts/data_prep/prepare_lvlm_nlf.py \
        --output_dir /outputs/lvlm_nlf_preprocessed \
        --image_base_dir /outputs/image_base/coco/train2017
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path


os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# System prompt matching existing FIRE pipeline
VL_ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate, "
    "detailed, and grounded answers based on the image and the user's instructions. "
    "When given feedback, critique, or scores, first determine whether the feedback "
    "identifies an actual error or missing information. Revise your response only if "
    "the feedback indicates that your answer is incorrect or incomplete. If the "
    "feedback confirms that your answer is correct, keep it unchanged."
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare LVLM_NLF dataset into messages format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/lvlm_nlf_preprocessed",
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs/image_base/coco/train2017",
        help="Base directory where COCO train2017 images are stored",
    )

    # Optional configuration
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="YangyiYY/LVLM_NLF",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip samples with missing images instead of warning",
    )

    return parser.parse_args()


def convert_sample(
    sample: dict,
    image_base_dir: Path,
    skip_missing: bool,
) -> dict | None:
    """Convert a single LVLM_NLF sample to messages format.

    Args:
        sample: Raw sample from LVLM_NLF dataset
        image_base_dir: Directory containing COCO train2017 images
        skip_missing: If True, suppress warnings for missing images

    Returns:
        Formatted dict with messages and images, or None if invalid
    """
    image_id = sample.get("image_id")
    interaction = sample.get("interaction")

    if not image_id or not interaction or len(interaction) == 0:
        return None

    # Build image path
    image_path = image_base_dir / image_id
    if not image_path.exists():
        if not skip_missing:
            logger.warning(f"Image not found: {image_path}")
        return None

    # Build messages
    messages: list[dict] = []

    # System message
    messages.append(
        {
            "role": "system",
            "content": VL_ASSISTANT_SYSTEM_PROMPT,
        }
    )

    num_turns = len(interaction)

    for i, turn in enumerate(interaction):
        user_text = turn.get("user", "").strip()
        model_text = turn.get("model", "").strip()

        if not user_text or not model_text:
            return None

        # First turn: append <image> tag to the question
        if i == 0:
            user_content = f"{user_text}\n<image>"
        else:
            user_content = user_text

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        # Last assistant turn gets loss=true, all others loss=false
        is_last = i == num_turns - 1
        messages.append(
            {
                "role": "assistant",
                "content": model_text,
                "loss": is_last,
            }
        )

    return {
        "messages": messages,
        "images": [str(image_path)],
    }


def process_dataset(
    dataset_id: str,
    output_dir: Path,
    image_base_dir: Path,
    max_samples: int,
    skip_missing: bool,
) -> dict:
    """Process the LVLM_NLF dataset and write output JSONL.

    Args:
        dataset_id: HuggingFace dataset ID
        output_dir: Output directory for JSONL files
        image_base_dir: Directory containing COCO train2017 images
        max_samples: Maximum samples to process (0 = all)
        skip_missing: Skip samples with missing images

    Returns:
        Statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download raw JSONL files from HuggingFace Hub
    # (load_dataset fails due to inconsistent 'caption' field types across files)
    logger.info(f"Downloading JSONL files from {dataset_id}...")
    repo_files = list_repo_files(dataset_id, repo_type="dataset")
    jsonl_files = [f for f in repo_files if f.endswith(".jsonl")]
    logger.info(f"Found {len(jsonl_files)} JSONL files: {jsonl_files}")

    # Load all samples from all JSONL files
    all_samples: list[dict] = []
    for jsonl_file in jsonl_files:
        local_path = hf_hub_download(dataset_id, jsonl_file, repo_type="dataset")
        count = 0
        with open(local_path) as jf:
            for line in jf:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    all_samples.append(sample)
                    count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {jsonl_file}: {e}")
        logger.info(f"  {jsonl_file}: {count} samples")

    logger.info(f"Total samples loaded: {len(all_samples)}")

    total_samples = len(all_samples)
    if max_samples > 0:
        total_samples = min(max_samples, total_samples)

    logger.info(f"Processing {total_samples} samples...")

    stats = {
        "total_samples": total_samples,
        "processed": 0,
        "skipped_missing_image": 0,
        "skipped_invalid": 0,
        "turn_distribution": {},
    }

    output_file = output_dir / "lvlm_nlf_messages_train_last_loss_only.jsonl"

    with open(output_file, "w") as f:
        for idx in tqdm(range(total_samples), desc="Processing"):
            sample = all_samples[idx]

            result = convert_sample(sample, image_base_dir, skip_missing)

            if result is None:
                # Distinguish between missing image and invalid data
                image_id = sample.get("image_id", "")
                if image_id and not (image_base_dir / image_id).exists():
                    stats["skipped_missing_image"] += 1
                else:
                    stats["skipped_invalid"] += 1
                continue

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            stats["processed"] += 1

            # Track turn distribution
            num_turns = len(sample.get("interaction", []))
            turn_key = str(num_turns)
            stats["turn_distribution"][turn_key] = stats["turn_distribution"].get(turn_key, 0) + 1

    logger.info(f"Wrote {stats['processed']} samples to {output_file}")

    # Write stats
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to {stats_file}")

    return stats


def main() -> None:
    """Main function."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    image_base_dir = Path(args.image_base_dir)

    logger.info("=" * 60)
    logger.info("LVLM_NLF Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Image base dir: {image_base_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")
    logger.info("=" * 60)

    stats = process_dataset(
        dataset_id=args.dataset_id,
        output_dir=output_dir,
        image_base_dir=image_base_dir,
        max_samples=args.max_samples,
        skip_missing=args.skip_missing,
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Skipped (missing image): {stats['skipped_missing_image']}")
    logger.info(f"Skipped (invalid data): {stats['skipped_invalid']}")
    logger.info(f"Turn distribution: {stats['turn_distribution']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
