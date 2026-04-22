#!/usr/bin/env python3
"""
Verify dataset alignment by sampling images and comparing with ground truth.

This script:
1. Samples N images from the training dataset
2. Copies them to a debug folder with index naming
3. Creates a summary file with question + GT answer for manual verification

Usage:
    python scripts/evaluation/verify_dataset_alignment.py \
        --dataset_path data/fire_preprocessed_v2/fire_messages_train.jsonl \
        --output_dir /outputs/debug_images \
        --num_samples 100 \
        --random_seed 42
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify dataset image-answer alignment")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/debug_images",
        help="Output directory for debug images and summary",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to verify (default: 100)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for sequential sampling (if not random)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential sampling instead of random",
    )
    return parser.parse_args()


def extract_question(messages: list) -> str:
    """Extract question from messages."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Remove image token for cleaner display
            return content.replace("<image>", "").strip()
    return ""


def extract_final_answer(messages: list) -> str:
    """Extract the final assistant answer from messages."""
    final_answer = ""
    for msg in messages:
        if msg.get("role") == "assistant":
            final_answer = msg.get("content", "")
    return final_answer


def extract_all_turns(messages: list) -> list:
    """Extract all turns (answers and feedback) from messages."""
    turns = []
    current_turn = {}

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            if current_turn:
                turns.append(current_turn)
            current_turn = {"answer": content, "feedback": ""}
        elif (
            role == "user"
            and current_turn
            and not content.startswith("What")
            and "<image>" not in content
        ):
            # This is feedback, not the initial question
            current_turn["feedback"] = content

    if current_turn:
        turns.append(current_turn)

    return turns


def load_dataset(dataset_path: str) -> list:
    """Load dataset from JSONL file."""
    samples = []
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            try:
                sample = json.loads(line.strip())
                sample["_index"] = i
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i}: {e}")
    logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
    return samples


def resolve_image_path(image_path: str | list | dict) -> str | None:
    """Resolve image path from various formats."""
    if isinstance(image_path, list):
        if len(image_path) > 0:
            image_path = image_path[0]
        else:
            return None

    if isinstance(image_path, dict):
        image_path = image_path.get("path", "")

    if not image_path:
        return None

    return image_path


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Dataset Alignment Verification")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info("=" * 60)

    # Load dataset
    samples = load_dataset(args.dataset_path)

    if len(samples) == 0:
        logger.error("No samples loaded!")
        return

    # Sample selection
    if args.sequential:
        selected_indices = list(
            range(args.start_index, min(args.start_index + args.num_samples, len(samples)))
        )
    else:
        random.seed(args.random_seed)
        selected_indices = random.sample(range(len(samples)), min(args.num_samples, len(samples)))
        selected_indices.sort()

    logger.info(f"Selected {len(selected_indices)} samples for verification")

    # Process samples
    verification_data = []
    images_copied = 0
    images_missing = 0

    for i, idx in enumerate(selected_indices):
        sample = samples[idx]

        # Extract info
        messages = sample.get("messages", [])
        images = sample.get("images", [])

        question = extract_question(messages)
        final_answer = extract_final_answer(messages)
        all_turns = extract_all_turns(messages)

        # Get image path
        if images:
            image_path = resolve_image_path(images[0] if isinstance(images, list) else images)
        else:
            image_path = None

        # Copy image to output dir
        output_image_name = f"sample_{i:04d}_idx_{idx:06d}.jpg"
        output_image_path = output_dir / output_image_name

        image_exists = False
        if image_path and os.path.exists(image_path):
            try:
                shutil.copy2(image_path, output_image_path)
                images_copied += 1
                image_exists = True
            except Exception as e:
                logger.warning(f"Failed to copy {image_path}: {e}")
                images_missing += 1
        else:
            images_missing += 1
            if image_path:
                logger.warning(f"Image not found: {image_path}")

        # Store verification data
        verification_data.append(
            {
                "sample_num": i,
                "dataset_index": idx,
                "image_file": output_image_name if image_exists else None,
                "original_image_path": image_path,
                "image_exists": image_exists,
                "question": question,
                "final_answer": final_answer,
                "num_turns": len(all_turns),
                "all_turns": all_turns,
            }
        )

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(selected_indices)} samples")

    # Save verification summary
    summary_path = output_dir / "verification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(verification_data, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    # Create human-readable report
    report_path = output_dir / "verification_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET ALIGNMENT VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {args.dataset_path}\n")
        f.write(f"Total samples in dataset: {len(samples)}\n")
        f.write(f"Samples verified: {len(verification_data)}\n")
        f.write(f"Images copied: {images_copied}\n")
        f.write(f"Images missing: {images_missing}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        for item in verification_data:
            f.write(f"Sample #{item['sample_num']} (Dataset Index: {item['dataset_index']})\n")
            f.write("-" * 60 + "\n")
            f.write(f"Image: {item['image_file'] or 'MISSING'}\n")
            f.write(f"Original Path: {item['original_image_path']}\n")
            f.write(f"Question: {item['question']}\n")
            f.write(f"Final Answer: {item['final_answer']}\n")
            f.write(f"Num Turns: {item['num_turns']}\n")
            f.write("\n")

    logger.info(f"Saved report to {report_path}")

    # Create HTML report for easier viewing
    html_path = output_dir / "verification_report.html"
    with open(html_path, "w") as f:
        f.write(
            """<!DOCTYPE html>
<html>
<head>
    <title>Dataset Alignment Verification</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .sample { border: 1px solid #ccc; margin: 20px 0; padding: 15px; }
        .sample img { max-width: 400px; max-height: 400px; }
        .question { color: #0066cc; font-weight: bold; }
        .answer { color: #006600; }
        .missing { color: #cc0000; }
        .feedback { color: #666; font-style: italic; }
        h2 { margin-top: 0; }
        .meta { color: #999; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Dataset Alignment Verification Report</h1>
    <p><strong>Dataset:</strong> """
            + args.dataset_path
            + """</p>
    <p><strong>Total verified:</strong> """
            + str(len(verification_data))
            + """</p>
    <p><strong>Images copied:</strong> """
            + str(images_copied)
            + """</p>
    <p><strong>Images missing:</strong> """
            + str(images_missing)
            + """</p>
    <hr>
"""
        )

        for item in verification_data:
            f.write(f"""
    <div class="sample">
        <h2>Sample #{item["sample_num"]} (Index: {item["dataset_index"]})</h2>
        <div class="meta">Original: {item["original_image_path"]}</div>
""")
            if item["image_exists"]:
                f.write(f'        <img src="{item["image_file"]}" alt="Sample image">\n')
            else:
                f.write('        <p class="missing">IMAGE MISSING</p>\n')

            f.write(f"""
        <p class="question">Q: {item["question"]}</p>
        <p class="answer">Final Answer: {item["final_answer"]}</p>
        <p>Turns: {item["num_turns"]}</p>
""")

            # Show all turns
            for t_idx, turn in enumerate(item.get("all_turns", [])):
                f.write(
                    f"        <p><strong>Turn {t_idx + 1} Answer:</strong> {turn.get('answer', '')[:200]}...</p>\n"
                )
                if turn.get("feedback"):
                    f.write(
                        f'        <p class="feedback">Feedback: {turn.get("feedback", "")[:200]}...</p>\n'
                    )

            f.write("    </div>\n")

        f.write("""
</body>
</html>
""")

    logger.info(f"Saved HTML report to {html_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Images copied: {images_copied}/{len(verification_data)}")
    logger.info(f"Images missing: {images_missing}/{len(verification_data)}")
    logger.info(f"View HTML report: {html_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
