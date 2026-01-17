#!/usr/bin/env python3
"""
Build local image mapping for manually downloaded datasets.
Maps FIRE image paths to local file paths for Vision-FL

AN and GeoQA+.
"""

import json
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration for local datasets
LOCAL_DATASETS = {
    "vision_flan": {
        "fire_prefix": "allava_vflan/images/images_191task_1k/",
        "local_dir": "/outputs/images_191task_1k",
    },
    "geoqa_plus": {
        "fire_prefix": "geoqa+/images/",
        "local_dir": "/outputs/geoqa_plus/images",
    },
    "geoqa_plus_test": {
        "fire_prefix": "geoqa+/test-images/",
        "local_dir": "/outputs/geoqa_plus/images",
    },
}


def build_local_mapping(fire_image_paths_file: str, output_file: str):
    """Build mapping from FIRE paths to local image files."""

    # Load FIRE image paths
    logger.info(f"Loading FIRE image paths from {fire_image_paths_file}")
    with open(fire_image_paths_file) as f:
        fire_paths = set()
        for line in f:
            line = line.strip()
            if line.startswith("image: "):
                fire_paths.add(line[7:])  # Remove "image: " prefix

    logger.info(f"Found {len(fire_paths)} total FIRE image paths")

    # Build mapping
    mapping = {}

    for dataset_name, config in LOCAL_DATASETS.items():
        fire_prefix = config["fire_prefix"]
        local_dir = Path(config["local_dir"])

        if not local_dir.exists():
            logger.warning(f"Local directory does not exist: {local_dir}")
            continue

        logger.info(f"\nProcessing {dataset_name}...")
        logger.info(f"  FIRE prefix: {fire_prefix}")
        logger.info(f"  Local dir: {local_dir}")

        # Find all relevant FIRE paths
        relevant_paths = {p for p in fire_paths if p.startswith(fire_prefix)}
        logger.info(f"  Found {len(relevant_paths)} FIRE paths with this prefix")

        # Map to local files
        mapped_count = 0
        missing_count = 0

        for fire_path in relevant_paths:
            # Get relative path after prefix
            relative_path = fire_path[len(fire_prefix) :]
            local_file = local_dir / relative_path

            if local_file.exists():
                mapping[fire_path] = {"local_path": str(local_file), "source": dataset_name}
                mapped_count += 1
            else:
                missing_count += 1
                if missing_count <= 10:  # Show first 10 missing files
                    logger.warning(f"  Missing: {fire_path} -> {local_file}")

        coverage = (mapped_count / len(relevant_paths) * 100) if relevant_paths else 0
        logger.info(f"  Mapped: {mapped_count}/{len(relevant_paths)} ({coverage:.1f}%)")
        if missing_count > 0:
            logger.info(f"  Missing: {missing_count}")

    # Save mapping
    logger.info(f"\nSaving local mapping to {output_file}")
    with open(output_file, "w") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"Total paths mapped: {len(mapping)}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("LOCAL IMAGE MAPPING SUMMARY")
    logger.info("=" * 60)
    for dataset_name in LOCAL_DATASETS:
        count = sum(1 for v in mapping.values() if v.get("source") == dataset_name)
        logger.info(f"{dataset_name}: {count} images")
    logger.info(f"Total: {len(mapping)} images")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build local image mapping")
    parser.add_argument(
        "--fire_paths",
        type=str,
        default="/workspace/image_sources.txt",
        help="File containing FIRE image paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/outputs/local_image_mapping.json",
        help="Output mapping JSON file",
    )

    args = parser.parse_args()

    build_local_mapping(args.fire_paths, args.output)
