#!/usr/bin/env python3
"""
Download ALL source images from COCO and MathVista datasets.
Organizes them to match FIRE's image path references exactly.
"""

import argparse
import logging
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def download_coco_images(output_dir: Path):
    """Download ALL COCO images (train + val)."""
    logger.info("=" * 60)
    logger.info("Downloading COCO images")
    logger.info("=" * 60)

    # COCO train
    coco_train_dir = output_dir / "coco" / "train2017"
    coco_train_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading COCO train split...")
    coco_train = load_dataset("detection-datasets/coco", split="train", streaming=True)

    count = 0
    for sample in tqdm(coco_train, desc="Downloading COCO train"):
        if 'image' in sample and 'image_id' in sample:
            filename = f"{sample['image_id']:012d}.jpg"
            output_path = coco_train_dir / filename

            if output_path.exists():
                continue

            image = sample['image']
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(output_path, "JPEG", quality=95)
                count += 1

    logger.info(f"Downloaded {count} COCO train images")

    # COCO val
    coco_val_dir = output_dir / "coco" / "val2017"
    coco_val_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading COCO val split...")
    coco_val = load_dataset("detection-datasets/coco", split="val", streaming=True)

    count = 0
    for sample in tqdm(coco_val, desc="Downloading COCO val"):
        if 'image' in sample and 'image_id' in sample:
            filename = f"{sample['image_id']:012d}.jpg"
            output_path = coco_val_dir / filename

            if output_path.exists():
                continue

            image = sample['image']
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(output_path, "JPEG", quality=95)
                count += 1

    logger.info(f"Downloaded {count} COCO val images")


def download_mathvista_images(output_dir: Path):
    """Download ALL MathVista images."""
    logger.info("=" * 60)
    logger.info("Downloading MathVista images")
    logger.info("=" * 60)

    # Create directory matching FIRE path format: mathvista/images/
    mathvista_dir = output_dir / "mathvista" / "images"
    mathvista_dir.mkdir(parents=True, exist_ok=True)

    # Also create images/ directory for alternate path format in FIRE
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MathVista dataset...")
    try:
        mathvista = load_dataset("AI4Math/MathVista", split="testmini", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load MathVista: {e}")
        return

    count = 0
    for sample in tqdm(mathvista, desc="Downloading MathVista"):
        # Get the actual PIL Image - MathVista has 'decoded_image' field
        image = sample.get('decoded_image') or sample.get('image')
        if not image or not isinstance(image, Image.Image):
            continue

        # Get sample ID
        sample_id = None
        for id_field in ['pid', 'question_id', 'id', 'image_id']:
            if id_field in sample:
                sample_id = str(sample[id_field])
                break

        if not sample_id:
            continue

        filename = f"{sample_id}.jpg"
        output_path_1 = mathvista_dir / filename
        output_path_2 = images_dir / filename

        # Skip if both already exist
        if output_path_1.exists() and output_path_2.exists():
            continue

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save to both locations to handle both path formats
        image.save(output_path_1, "JPEG", quality=95)
        image.save(output_path_2, "JPEG", quality=95)
        count += 1

    logger.info(f"Downloaded {count} MathVista images")


def main():
    parser = argparse.ArgumentParser(
        description="Download ALL source images for FIRE dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs",
        help="Base directory to save images (creates coco/, mathvista/, images/ subdirs)",
    )
    parser.add_argument(
        "--coco_only",
        action="store_true",
        help="Download only COCO images",
    )
    parser.add_argument(
        "--mathvista_only",
        action="store_true",
        help="Download only MathVista images",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("FIRE Source Images Download")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    if not args.mathvista_only:
        download_coco_images(output_dir)

    if not args.coco_only:
        download_mathvista_images(output_dir)

    logger.info("=" * 60)
    logger.info("Download complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
