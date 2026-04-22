#!/usr/bin/env python3
"""
Verify FIRE dataset images exist on the pod.

Two-step process:
1. Extract all image paths from FIRE dataset (cached to JSON)
2. Check if paths exist in image directory (fast, reusable)

Usage:
    # Step 1: Extract paths (only needed once)
    python scripts/data_prep/verify_fire_images.py extract \
        --output_paths /outputs/fire_image_paths.json

    # Step 2: Verify paths exist (run after each download)
    python scripts/data_prep/verify_fire_images.py verify \
        --paths_file /outputs/fire_image_paths.json \
        --image_base_dir /outputs/image_base

    # Or run both in one command
    python scripts/data_prep/verify_fire_images.py all \
        --paths_file /outputs/fire_image_paths.json \
        --image_base_dir /outputs/image_base
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FIRE dataset images exist on pod")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract image paths from FIRE dataset")
    extract_parser.add_argument(
        "--dataset_id",
        type=str,
        default="PengxiangLi/FIRE",
        help="HuggingFace dataset ID",
    )
    extract_parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to process",
    )
    extract_parser.add_argument(
        "--output_paths",
        type=str,
        default="/outputs/fire_image_paths.json",
        help="Output JSON file for extracted paths",
    )
    extract_parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples per split (0 = all)",
    )

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify", help="Verify extracted paths exist in image directory"
    )
    verify_parser.add_argument(
        "--paths_file",
        type=str,
        default="/outputs/fire_image_paths.json",
        help="JSON file with extracted paths",
    )
    verify_parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs/image_base",
        help="Base directory where images are stored",
    )
    verify_parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Path to save detailed JSON report",
    )

    # All command (extract + verify)
    all_parser = subparsers.add_parser("all", help="Extract paths and verify (convenience command)")
    all_parser.add_argument(
        "--dataset_id",
        type=str,
        default="PengxiangLi/FIRE",
        help="HuggingFace dataset ID",
    )
    all_parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to process",
    )
    all_parser.add_argument(
        "--paths_file",
        type=str,
        default="/outputs/fire_image_paths.json",
        help="JSON file for extracted paths (created if not exists)",
    )
    all_parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs/image_base",
        help="Base directory where images are stored",
    )
    all_parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Path to save detailed JSON report",
    )
    all_parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples per split (0 = all)",
    )
    all_parser.add_argument(
        "--force_extract",
        action="store_true",
        help="Force re-extraction even if paths file exists",
    )

    return parser.parse_args()


def extract_paths(
    dataset_id: str,
    splits: list,
    output_path: Path,
    max_samples: int = 0,
) -> dict:
    """Extract all image paths from FIRE dataset.

    Returns:
        Dictionary with paths organized by split and source
    """
    from datasets import load_dataset
    from tqdm import tqdm

    logger.info("=" * 60)
    logger.info("Extracting image paths from FIRE dataset")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_id}")
    logger.info(f"Splits: {splits}")
    logger.info("=" * 60)

    all_paths = {
        "dataset_id": dataset_id,
        "splits": {},
    }

    for split in splits:
        logger.info(f"\nLoading {split} split...")
        ds = load_dataset(dataset_id, split=split)

        total_samples = len(ds)
        if max_samples > 0:
            total_samples = min(max_samples, total_samples)

        logger.info(f"Processing {total_samples} samples...")

        split_data = {
            "total_samples": total_samples,
            "paths": [],  # List of {path, source} dicts
        }

        for idx in tqdm(range(total_samples), desc=f"Extracting {split}"):
            sample = ds[idx]
            image_path = sample.get("image", "")
            source = sample.get("source", "unknown")

            if image_path and isinstance(image_path, str):
                split_data["paths"].append(
                    {
                        "path": image_path,
                        "source": source,
                    }
                )

        all_paths["splits"][split] = split_data
        logger.info(f"Extracted {len(split_data['paths'])} paths from {split}")

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_paths, f, indent=2)

    logger.info(f"\nPaths saved to: {output_path}")

    # Print summary
    total_paths = sum(len(s["paths"]) for s in all_paths["splits"].values())
    logger.info(f"Total paths extracted: {total_paths:,}")

    return all_paths


def get_path_prefix(image_path: str, levels: int = 2) -> str:
    """Extract path prefix (first N levels) from image path."""
    parts = image_path.split("/")
    if len(parts) >= levels:
        return "/".join(parts[:levels])
    return image_path


def get_alternate_paths(image_path: str) -> list[str]:
    """Get alternate path variants to check.

    Handles naming mismatches between FIRE paths and actual files:
    - COCO train2014/val2014: adds COCO_ prefix
    - SEED-Bench: removes .jpg extension (files have no extension)

    Returns:
        List of alternate paths to try
    """
    alternates = []

    # COCO train2014/val2014: add COCO_ prefix
    if image_path.startswith("coco/train2014/"):
        filename = image_path.split("/")[-1]
        if not filename.startswith("COCO_"):
            alternates.append(f"coco/train2014/COCO_train2014_{filename}")
    elif image_path.startswith("coco/val2014/"):
        filename = image_path.split("/")[-1]
        if not filename.startswith("COCO_"):
            alternates.append(f"coco/val2014/COCO_val2014_{filename}")

    # SEED-Bench: files have no extension
    if image_path.startswith("seedbench/") and image_path.endswith(".jpg"):
        alternates.append(image_path[:-4])  # Remove .jpg

    # TextVQA: FIRE uses train_val_images/ but folder renamed to images/
    if image_path.startswith("textvqa/train_val_images/"):
        alternates.append(image_path.replace("textvqa/train_val_images/", "textvqa/images/"))

    # ScienceQA: FIRE uses scienceqa/images/test/ but disk has scienceqa/images/ (no test/)
    if image_path.startswith("scienceqa/images/test/"):
        alternates.append(image_path.replace("scienceqa/images/test/", "scienceqa/images/"))

    return alternates


def check_file_exists(args: tuple) -> tuple:
    """Check if a single file exists. Used for parallel processing.

    Checks original path first, then tries alternate path variants.
    """
    image_path, source, image_base_dir = args
    full_path = image_base_dir / image_path
    exists = full_path.exists()

    # If not found, try alternate path variants
    if not exists:
        for alt_path in get_alternate_paths(image_path):
            if (image_base_dir / alt_path).exists():
                exists = True
                break

    return (image_path, source, exists)


def verify_paths(
    paths_file: Path,
    image_base_dir: Path,
    output_report: Path | None = None,
    num_workers: int = 32,
) -> dict:
    """Verify extracted paths exist in image directory.

    Uses parallel processing for fast verification.

    Returns:
        Dictionary with verification statistics
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    logger.info("=" * 60)
    logger.info("Verifying image paths")
    logger.info("=" * 60)
    logger.info(f"Paths file: {paths_file}")
    logger.info(f"Image base dir: {image_base_dir}")
    logger.info(f"Parallel workers: {num_workers}")
    logger.info("=" * 60)

    # Load paths
    with open(paths_file) as f:
        all_paths = json.load(f)

    all_stats = {}

    for split, split_data in all_paths["splits"].items():
        num_paths = len(split_data["paths"])
        logger.info(f"\nVerifying {split} split ({num_paths:,} paths)...")

        stats = {
            "split": split,
            "total_samples": split_data["total_samples"],
            "images_found": 0,
            "images_missing": 0,
            "by_source": defaultdict(lambda: {"found": 0, "missing": 0, "total": 0}),
            "by_path_prefix": defaultdict(lambda: {"found": 0, "missing": 0, "total": 0}),
            "missing_paths_sample": [],
        }

        # Prepare args for parallel processing
        check_args = [
            (item["path"], item["source"], image_base_dir) for item in split_data["paths"]
        ]

        # Process in parallel with progress bar
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(check_file_exists, arg): arg for arg in check_args}

            for future in tqdm(
                as_completed(futures),
                total=num_paths,
                desc=f"Verifying {split}",
                unit="files",
            ):
                image_path, source, exists = future.result()
                path_prefix = get_path_prefix(image_path, levels=2)

                # Update statistics
                stats["by_source"][source]["total"] += 1
                stats["by_path_prefix"][path_prefix]["total"] += 1

                if exists:
                    stats["images_found"] += 1
                    stats["by_source"][source]["found"] += 1
                    stats["by_path_prefix"][path_prefix]["found"] += 1
                else:
                    stats["images_missing"] += 1
                    stats["by_source"][source]["missing"] += 1
                    stats["by_path_prefix"][path_prefix]["missing"] += 1

                    # Store sample of missing paths
                    prefix_missing = [
                        p for p in stats["missing_paths_sample"] if p.startswith(path_prefix)
                    ]
                    if len(prefix_missing) < 5:
                        stats["missing_paths_sample"].append(image_path)

        # Convert defaultdicts to regular dicts
        stats["by_source"] = dict(stats["by_source"])
        stats["by_path_prefix"] = dict(stats["by_path_prefix"])

        all_stats[split] = stats

        found_pct = (
            stats["images_found"] / len(split_data["paths"]) * 100 if split_data["paths"] else 0
        )
        logger.info(f"  Found: {stats['images_found']:,} ({found_pct:.1f}%)")
        logger.info(f"  Missing: {stats['images_missing']:,}")

    # Print detailed summary
    print_summary(all_stats, image_base_dir)

    # Save report if requested
    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "image_base_dir": str(image_base_dir),
            "paths_file": str(paths_file),
            "splits": all_stats,
            "summary": {
                "total_samples": sum(s["total_samples"] for s in all_stats.values()),
                "total_found": sum(s["images_found"] for s in all_stats.values()),
                "total_missing": sum(s["images_missing"] for s in all_stats.values()),
            },
        }
        with open(output_report, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nDetailed report saved to: {output_report}")

    return all_stats


def print_summary(all_stats: dict, image_base_dir: Path) -> None:
    """Print formatted summary of verification results."""

    print("\n" + "=" * 80)
    print("FIRE IMAGE VERIFICATION REPORT")
    print("=" * 80)
    print(f"Image base directory: {image_base_dir}")
    print("=" * 80)

    # Overall summary
    total_found = sum(s["images_found"] for s in all_stats.values())
    total_missing = sum(s["images_missing"] for s in all_stats.values())
    total_samples = total_found + total_missing

    print("\nOVERALL SUMMARY:")
    print(f"  Total samples checked: {total_samples:,}")
    print(f"  Images found: {total_found:,} ({total_found / total_samples * 100:.1f}%)")
    print(f"  Images missing: {total_missing:,} ({total_missing / total_samples * 100:.1f}%)")

    # Per-split summary
    for split, stats in all_stats.items():
        found = stats["images_found"]
        missing = stats["images_missing"]
        total = found + missing
        pct = found / total * 100 if total > 0 else 0
        print(f"\n{split.upper()} SPLIT:")
        print(f"  Total: {total:,} | Found: {found:,} ({pct:.1f}%) | Missing: {missing:,}")

    # Aggregate by source across all splits
    print("\n" + "-" * 80)
    print("BY SOURCE (across all splits):")
    print("-" * 80)
    print(f"{'Source':<25} {'Found':>10} {'Missing':>10} {'Total':>10} {'Coverage':>10}")
    print("-" * 80)

    source_totals = defaultdict(lambda: {"found": 0, "missing": 0, "total": 0})
    for stats in all_stats.values():
        for source, data in stats["by_source"].items():
            source_totals[source]["found"] += data["found"]
            source_totals[source]["missing"] += data["missing"]
            source_totals[source]["total"] += data["total"]

    for source, data in sorted(source_totals.items(), key=lambda x: -x[1]["missing"]):
        pct = data["found"] / data["total"] * 100 if data["total"] > 0 else 0
        status = "✓" if data["missing"] == 0 else "✗"
        print(
            f"{status} {source:<23} {data['found']:>10,} {data['missing']:>10,} {data['total']:>10,} {pct:>9.1f}%"
        )

    # Aggregate by path prefix
    print("\n" + "-" * 80)
    print("BY IMAGE PATH PREFIX (across all splits):")
    print("-" * 80)
    print(f"{'Path Prefix':<40} {'Found':>10} {'Missing':>10} {'Total':>10} {'Coverage':>10}")
    print("-" * 80)

    prefix_totals = defaultdict(lambda: {"found": 0, "missing": 0, "total": 0})
    for stats in all_stats.values():
        for prefix, data in stats["by_path_prefix"].items():
            prefix_totals[prefix]["found"] += data["found"]
            prefix_totals[prefix]["missing"] += data["missing"]
            prefix_totals[prefix]["total"] += data["total"]

    for prefix, data in sorted(prefix_totals.items(), key=lambda x: -x[1]["missing"]):
        pct = data["found"] / data["total"] * 100 if data["total"] > 0 else 0
        status = "✓" if data["missing"] == 0 else "✗"
        print(
            f"{status} {prefix:<38} {data['found']:>10,} {data['missing']:>10,} {data['total']:>10,} {pct:>9.1f}%"
        )

    # Missing datasets
    print("\n" + "-" * 80)
    print("DATASETS NEEDING DOWNLOAD (0% coverage):")
    print("-" * 80)

    missing_datasets = [
        (prefix, data)
        for prefix, data in prefix_totals.items()
        if data["found"] == 0 and data["missing"] > 0
    ]

    if missing_datasets:
        for prefix, data in sorted(missing_datasets, key=lambda x: -x[1]["missing"]):
            print(f"  {prefix}: {data['missing']:,} images needed")
    else:
        print("  None! All datasets have at least some coverage.")

    # Partial datasets
    print("\n" + "-" * 80)
    print("DATASETS WITH PARTIAL COVERAGE (<100%):")
    print("-" * 80)

    partial_datasets = [
        (prefix, data)
        for prefix, data in prefix_totals.items()
        if 0 < data["found"] < data["total"]
    ]

    if partial_datasets:
        for prefix, data in sorted(partial_datasets, key=lambda x: x[1]["found"] / x[1]["total"]):
            pct = data["found"] / data["total"] * 100
            print(
                f"  {prefix}: {data['found']:,}/{data['total']:,} ({pct:.1f}%) - missing {data['missing']:,}"
            )
    else:
        print("  None! All datasets have 100% coverage.")

    print("\n" + "=" * 80)


def main():
    args = parse_args()

    if args.command == "extract":
        extract_paths(
            dataset_id=args.dataset_id,
            splits=args.splits,
            output_path=Path(args.output_paths),
            max_samples=args.max_samples,
        )

    elif args.command == "verify":
        paths_file = Path(args.paths_file)
        if not paths_file.exists():
            logger.error(f"Paths file not found: {paths_file}")
            logger.error("Run 'extract' command first to create it.")
            sys.exit(1)

        image_base_dir = Path(args.image_base_dir)
        if not image_base_dir.exists():
            logger.error(f"Image base directory not found: {image_base_dir}")
            sys.exit(1)

        output_report = Path(args.output_report) if args.output_report else None

        stats = verify_paths(
            paths_file=paths_file,
            image_base_dir=image_base_dir,
            output_report=output_report,
        )

        # Exit with error if images missing
        total_missing = sum(s["images_missing"] for s in stats.values())
        if total_missing > 0:
            sys.exit(1)

    elif args.command == "all":
        paths_file = Path(args.paths_file)

        # Extract if needed
        if not paths_file.exists() or args.force_extract:
            extract_paths(
                dataset_id=args.dataset_id,
                splits=args.splits,
                output_path=paths_file,
                max_samples=args.max_samples,
            )
        else:
            logger.info(f"Using existing paths file: {paths_file}")

        # Verify
        image_base_dir = Path(args.image_base_dir)
        if not image_base_dir.exists():
            logger.error(f"Image base directory not found: {image_base_dir}")
            sys.exit(1)

        output_report = Path(args.output_report) if args.output_report else None

        stats = verify_paths(
            paths_file=paths_file,
            image_base_dir=image_base_dir,
            output_report=output_report,
        )

        total_missing = sum(s["images_missing"] for s in stats.values())
        if total_missing > 0:
            sys.exit(1)

    else:
        logger.error("Please specify a command: extract, verify, or all")
        sys.exit(1)


if __name__ == "__main__":
    main()
