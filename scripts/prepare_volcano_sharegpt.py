#!/usr/bin/env python3
"""
Prepare Volcano dataset in ShareGPT format for ms-swift.

Converts Volcano multi-round conversations into ShareGPT format.

Usage:
    python scripts/prepare_volcano_sharegpt.py \
        --input_file /path/to/volcano.json \
        --output_dir /outputs/volcano_sharegpt \
        --image_dir /cache/volcano_images \
        --max_samples 10
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "You should produce accurate, detailed, and grounded answers based on the image and the user's instructions. "
    "When given feedback, critique, or scores, revise your response to improve correctness, specificity, and completeness."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Volcano dataset to ShareGPT format for ms-swift"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to local Volcano JSON/JSONL file (optional if using HuggingFace)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="kaist-ai/volcano-train",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/train-00000-of-00001.json",
        help="Path to data file in HuggingFace repo",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/volcano_sharegpt",
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./outputs/volcano_images",
        help="Directory to save images",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for conversations",
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="Skip image download (use placeholders)",
    )
    return parser.parse_args()


def parse_volcano_to_sharegpt(
    sample: dict,
    sample_id: str,
    image_path: str,
    system_prompt: str,
) -> Optional[dict]:
    """Convert a Volcano sample to ShareGPT format.

    Expected Volcano format (adjust based on actual structure):
    {
        "id": "...",
        "image": "...",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]
    }
    """
    try:
        conversations = sample.get("conversations", [])

        if not conversations:
            logger.debug(f"{sample_id}: No conversations found")
            return None

        # Convert to ShareGPT paired format
        sharegpt_conversation = []

        i = 0
        while i < len(conversations):
            # Look for human-assistant pairs
            if i + 1 < len(conversations):
                human_turn = conversations[i]
                assistant_turn = conversations[i + 1]

                human_role = human_turn.get("from", human_turn.get("role", ""))
                assistant_role = assistant_turn.get("from", assistant_turn.get("role", ""))

                # Handle different role naming conventions
                if human_role in ["human", "user"] and assistant_role in ["gpt", "assistant"]:
                    human_text = human_turn.get("value", "")
                    assistant_text = assistant_turn.get("value", "")

                    if human_text and assistant_text:
                        sharegpt_conversation.append({
                            "human": human_text,
                            "assistant": assistant_text
                        })
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        if not sharegpt_conversation:
            logger.debug(f"{sample_id}: No valid conversation pairs")
            return None

        return {
            "system": system_prompt,
            "conversation": sharegpt_conversation,
            "images": [image_path]
        }

    except Exception as e:
        logger.warning(f"{sample_id}: Error parsing - {e}")
        return None


def load_volcano_data(args) -> list:
    """Load Volcano dataset from local file or HuggingFace."""
    samples = []

    if args.input_file and os.path.exists(args.input_file):
        # Load from local file
        logger.info(f"Loading from local file: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.input_file.endswith('.jsonl'):
                for line in f:
                    samples.append(json.loads(line))
            else:
                data = json.load(f)
                samples = data if isinstance(data, list) else [data]
    else:
        # Download from HuggingFace
        logger.info(f"Downloading from {args.repo_id}/{args.data_file}")
        local_file = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.data_file,
            repo_type="dataset"
        )

        with open(local_file, 'r', encoding='utf-8') as f:
            # Try both JSON and JSONL
            try:
                data = json.load(f)
                samples = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                # Try JSONL
                f.seek(0)
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Volcano ShareGPT Format Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Image dir: {image_dir}")
    logger.info(f"Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")
    logger.info("=" * 60)

    # Load data
    samples = load_volcano_data(args)

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    # Process samples
    output_file = output_dir / "volcano_sharegpt_train.jsonl"
    processed = 0
    skipped = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(tqdm(samples, desc="Processing")):
            sample_id = f"volcano_{idx:06d}"

            try:
                # Handle image
                if args.skip_images:
                    image_path = f"/placeholder/images/{sample_id}.jpg"
                else:
                    # TODO: Download actual images if available in dataset
                    image_path = str(image_dir / f"{sample_id}.jpg")
                    # For now, use placeholder
                    logger.warning(f"Image download not implemented, using placeholder")

                # Convert to ShareGPT
                sharegpt_sample = parse_volcano_to_sharegpt(
                    sample, sample_id, image_path, args.system_prompt
                )

                if sharegpt_sample:
                    f.write(json.dumps(sharegpt_sample, ensure_ascii=False) + "\n")
                    processed += 1
                else:
                    skipped += 1

            except Exception as e:
                logger.warning(f"Error processing {sample_id}: {e}")
                skipped += 1

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Output: {output_file}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
