#!/usr/bin/env python3
"""
Merge HuggingFace and local image mappings.
"""

import json
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def merge_mappings(hf_mapping_file: str, local_mapping_file: str, output_file: str):
    """Merge HuggingFace and local mappings. Local takes precedence."""

    # Load HuggingFace mapping
    logger.info(f"Loading HuggingFace mapping from {hf_mapping_file}")
    if Path(hf_mapping_file).exists():
        with open(hf_mapping_file) as f:
            hf_mapping = json.load(f)
        logger.info(f"  Loaded {len(hf_mapping)} HuggingFace paths")
    else:
        logger.warning("  HuggingFace mapping not found, starting with empty mapping")
        hf_mapping = {}

    # Load local mapping
    logger.info(f"Loading local mapping from {local_mapping_file}")
    if Path(local_mapping_file).exists():
        with open(local_mapping_file) as f:
            local_mapping = json.load(f)
        logger.info(f"  Loaded {len(local_mapping)} local paths")
    else:
        logger.warning("  Local mapping not found, using only HuggingFace mapping")
        local_mapping = {}

    # Merge (local takes precedence)
    complete_mapping = {**hf_mapping, **local_mapping}

    # Save merged mapping
    logger.info(f"\nSaving merged mapping to {output_file}")
    with open(output_file, "w") as f:
        json.dump(complete_mapping, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MERGED MAPPING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"HuggingFace paths: {len(hf_mapping)}")
    logger.info(f"Local paths: {len(local_mapping)}")
    logger.info(f"Total paths: {len(complete_mapping)}")
    logger.info(
        f"Overlapping paths: {len(hf_mapping) + len(local_mapping) - len(complete_mapping)}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge image mappings")
    parser.add_argument(
        "--hf_mapping",
        type=str,
        default="/outputs/fire_image_mapping.json",
        help="HuggingFace mapping file",
    )
    parser.add_argument(
        "--local_mapping",
        type=str,
        default="/outputs/local_image_mapping.json",
        help="Local mapping file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/outputs/fire_image_mapping_complete.json",
        help="Output merged mapping file",
    )

    args = parser.parse_args()

    merge_mappings(args.hf_mapping, args.local_mapping, args.output)
