#!/usr/bin/env python3
"""
Convert ShareGPT format to Messages format for ms-swift training.

This script converts FIRE ShareGPT format datasets to Messages format with
explicit `loss: true` on all assistant turns. This enables combining the
converted dataset with other Messages-format datasets (like feedback SFT)
for multi-task training.

Input (ShareGPT):
{
  "system": "...",
  "conversation": [
    {"human": "Question", "assistant": "Answer 1"},
    {"human": "Feedback", "assistant": "Answer 2"}
  ],
  "images": ["path/to/image.jpg"]
}

Output (Messages):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer 1", "loss": true},
    {"role": "user", "content": "Feedback"},
    {"role": "assistant", "content": "Answer 2", "loss": true}
  ],
  "images": ["path/to/image.jpg"]
}

Usage:
    python scripts/data_prep/convert_sharegpt_to_messages.py \
        --input /outputs/fire_preprocessed_v2/fire_sharegpt_train.jsonl \
        --output /outputs/fire_preprocessed_v2/fire_messages_train.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT format to Messages format with loss flags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input ShareGPT JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output Messages JSONL file (default: same dir as input with '_messages' suffix)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to convert (0 = all)",
    )
    return parser.parse_args()


def convert_sharegpt_to_messages(sample: dict) -> dict:
    """Convert a single ShareGPT sample to Messages format.

    Args:
        sample: ShareGPT format sample with system, conversation, images

    Returns:
        Messages format sample with explicit loss flags
    """
    messages = []

    # Add system message if present
    system_content = sample.get("system", "")
    if system_content:
        messages.append({"role": "system", "content": system_content})

    # Convert conversation turns
    conversation = sample.get("conversation", [])
    for turn in conversation:
        # Human message -> user role
        human_content = turn.get("human", "")
        if human_content:
            messages.append({"role": "user", "content": human_content})

        # Assistant message -> assistant role with loss: true
        assistant_content = turn.get("assistant", "")
        if assistant_content:
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "loss": True,
                }
            )

    # Build output sample
    output = {"messages": messages}

    # Preserve images field if present
    if "images" in sample:
        output["images"] = sample["images"]

    return output


def main() -> None:
    """Main function."""
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Default output path: same directory with '_messages' suffix
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / input_path.name.replace(
            "_sharegpt_", "_messages_"
        ).replace("_sharegpt.", "_messages.")

    logger.info("=" * 60)
    logger.info("ShareGPT to Messages Format Conversion")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    if args.max_samples > 0:
        logger.info(f"Max samples: {args.max_samples}")
    logger.info("=" * 60)

    # Count input lines for progress bar
    logger.info("Counting input samples...")
    with open(input_path) as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"Found {total_lines} samples in input")

    # Process samples
    stats = {
        "total": 0,
        "converted": 0,
        "skipped": 0,
        "total_turns": 0,
        "total_assistant_turns": 0,
        "errors": [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as infile, open(output_path, "w") as outfile:
        for i, line in enumerate(tqdm(infile, total=total_lines, desc="Converting")):
            if args.max_samples > 0 and i >= args.max_samples:
                break

            stats["total"] += 1

            try:
                sample = json.loads(line.strip())
                converted = convert_sharegpt_to_messages(sample)

                # Count statistics
                num_messages = len(converted["messages"])
                num_assistant = sum(
                    1 for m in converted["messages"] if m.get("role") == "assistant"
                )
                stats["total_turns"] += num_messages
                stats["total_assistant_turns"] += num_assistant

                # Write converted sample
                outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
                stats["converted"] += 1

            except json.JSONDecodeError as e:
                stats["skipped"] += 1
                if len(stats["errors"]) < 10:
                    stats["errors"].append(f"Line {i}: JSON decode error: {e}")
            except Exception as e:
                stats["skipped"] += 1
                if len(stats["errors"]) < 10:
                    stats["errors"].append(f"Line {i}: {e}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {stats['total']}")
    logger.info(f"Converted: {stats['converted']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Total messages: {stats['total_turns']}")
    logger.info(f"Assistant messages (with loss=true): {stats['total_assistant_turns']}")

    if stats["converted"] > 0:
        avg_messages = stats["total_turns"] / stats["converted"]
        avg_assistant = stats["total_assistant_turns"] / stats["converted"]
        logger.info(f"Avg messages/sample: {avg_messages:.2f}")
        logger.info(f"Avg assistant turns/sample: {avg_assistant:.2f}")

    if stats["errors"]:
        logger.warning("")
        logger.warning("Errors encountered:")
        for error in stats["errors"]:
            logger.warning(f"  {error}")

    logger.info("")
    logger.info(f"Output written to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
