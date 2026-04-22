#!/usr/bin/env python3
"""Convert BLINK benchmark from HuggingFace to VLMEvalKit TSV format.

Downloads the BLINK-Benchmark/BLINK dataset (val split), saves images to
disk, and creates per-subtask TSV files compatible with VLMEvalKit's
custom dataset loading.

Usage:
    python scripts/evaluation/prepare_blink_vlmevalkit.py \
        --output_dir /outputs/benchmark_data/blink

    # Specific subtasks only
    python scripts/evaluation/prepare_blink_vlmevalkit.py \
        --output_dir /outputs/benchmark_data/blink \
        --subtasks Counting Art_Style Jigsaw

Output structure:
    /outputs/benchmark_data/blink/
    ├── images/
    │   ├── Art_Style/
    │   │   ├── 0000.jpg
    │   │   └── ...
    │   ├── Counting/
    │   └── ...
    └── tsv/
        ├── BLINK_Art_Style.tsv
        ├── BLINK_Counting.tsv
        └── ...
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys

from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# All 14 BLINK subtasks
ALL_SUBTASKS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]

# BLINK image columns in HF dataset
IMAGE_COLUMNS = ["image_1", "image_2", "image_3", "image_4"]

# VLMEvalKit TSV image column names
# First image -> "image", rest -> "image_1", "image_2", "image_3"
TSV_IMAGE_COLUMNS = ["image", "image_1", "image_2", "image_3"]

OPTION_LETTERS = ["A", "B", "C", "D"]


def image_to_base64(img: Image.Image) -> str:
    """Encode a PIL image as base64 PNG string.

    Args:
        img: PIL Image object.

    Returns:
        Base64-encoded PNG string.
    """
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_image(img: Image.Image, path: str) -> str:
    """Save a PIL image to disk as JPEG.

    Args:
        img: PIL Image object.
        path: Output file path.

    Returns:
        Absolute path to saved image.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, "JPEG", quality=90)
    return os.path.abspath(path)


def process_subtask(
    subtask: str,
    output_dir: str,
    use_base64: bool = False,
) -> int:
    """Download and convert one BLINK subtask to VLMEvalKit TSV.

    Args:
        subtask: BLINK subtask name (e.g., "Counting").
        output_dir: Root output directory.
        use_base64: If True, embed images as base64 in TSV instead of paths.

    Returns:
        Number of samples processed.
    """
    from datasets import load_dataset

    logger.info(f"Loading BLINK subtask: {subtask}")
    ds = load_dataset(
        "BLINK-Benchmark/BLINK",
        subtask,
        split="val",
        trust_remote_code=True,
    )
    logger.info(f"  Loaded {len(ds)} samples")

    image_dir = os.path.join(output_dir, "images", subtask)
    tsv_dir = os.path.join(output_dir, "tsv")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(tsv_dir, exist_ok=True)

    tsv_path = os.path.join(tsv_dir, f"BLINK_{subtask}.tsv")

    # Determine which image columns have data
    active_image_cols = []
    for col in IMAGE_COLUMNS:
        # Check first sample to see if column has data
        if col in ds.column_names and ds[0][col] is not None:
            active_image_cols.append(col)

    logger.info(f"  Active image columns: {active_image_cols}")

    # Build TSV header
    tsv_columns = ["index"]
    for i, _ in enumerate(active_image_cols):
        tsv_columns.append(TSV_IMAGE_COLUMNS[i])
    tsv_columns.append("question")
    tsv_columns.extend(OPTION_LETTERS)
    tsv_columns.append("answer")

    rows = []
    for idx, sample in enumerate(ds):
        # Save images
        image_values = []
        for img_idx, col in enumerate(active_image_cols):
            img = sample[col]
            if img is None:
                image_values.append("")
                continue

            if use_base64:
                image_values.append(image_to_base64(img))
            else:
                img_path = os.path.join(image_dir, f"{idx:04d}_img{img_idx + 1}.jpg")
                saved_path = save_image(img, img_path)
                image_values.append(saved_path)

        # Extract question — use the pre-formatted prompt field if available
        question = sample.get("prompt") or sample.get("question", "")

        # Extract choices
        choices = sample.get("choices", [])
        choice_values = [""] * len(OPTION_LETTERS)
        for i, choice in enumerate(choices):
            if i < len(OPTION_LETTERS):
                choice_values[i] = str(choice)

        # Extract answer (single letter like "A" or "B")
        answer = sample.get("answer", "")

        # Build row
        row = [str(idx)]
        row.extend(image_values)
        row.append(question)
        row.extend(choice_values)
        row.append(answer)

        rows.append(row)

    # Write TSV
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(tsv_columns) + "\n")
        for row in rows:
            # Escape tabs and newlines in cell values
            escaped = []
            for cell in row:
                cell = str(cell).replace("\t", " ").replace("\n", " ")
                escaped.append(cell)
            f.write("\t".join(escaped) + "\n")

    logger.info(f"  Wrote {len(rows)} rows to {tsv_path}")
    return len(rows)


def main() -> None:
    """Download BLINK from HuggingFace and create VLMEvalKit TSVs."""
    parser = argparse.ArgumentParser(
        description="Convert BLINK benchmark to VLMEvalKit TSV format."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/benchmark_data/blink",
        help="Root output directory for images and TSVs.",
    )
    parser.add_argument(
        "--subtasks",
        nargs="+",
        default=None,
        help="Specific subtasks to process. Default: all 14.",
    )
    parser.add_argument(
        "--use_base64",
        action="store_true",
        help="Embed images as base64 in TSV instead of file paths.",
    )
    args = parser.parse_args()

    subtasks = args.subtasks or ALL_SUBTASKS

    # Validate subtask names
    for s in subtasks:
        if s not in ALL_SUBTASKS:
            logger.error(f"Unknown subtask: {s}. Valid: {ALL_SUBTASKS}")
            sys.exit(1)

    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Subtasks to process: {subtasks}")
    logger.info(f"Image mode: {'base64' if args.use_base64 else 'file paths'}")

    total = 0
    results = {}
    for subtask in subtasks:
        n = process_subtask(subtask, args.output_dir, args.use_base64)
        results[subtask] = n
        total += n

    # Print summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("BLINK Data Preparation Complete")
    logger.info("=" * 50)
    for subtask, count in results.items():
        logger.info(f"  {subtask:30s} {count:5d} samples")
    logger.info(f"  {'TOTAL':30s} {total:5d} samples")
    logger.info(f"  TSV dir: {os.path.join(args.output_dir, 'tsv')}")
    logger.info(f"  Image dir: {os.path.join(args.output_dir, 'images')}")


if __name__ == "__main__":
    main()
