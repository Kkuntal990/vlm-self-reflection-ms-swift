
#!/usr/bin/env python3
"""
Build a mapping file from FIRE image paths to actual downloaded images.
This allows us to download datasets once and map FIRE paths flexibly.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Map FIRE dataset/split to HuggingFace dataset configuration
DATASET_SPLIT_MAPPING = {
    # COCO variants
    ("coco", "train2014"): {"hf_id": "detection-datasets/coco", "split": "train", "id_field": "image_id"},
    ("coco", "train2017"): {"hf_id": "detection-datasets/coco", "split": "train", "id_field": "image_id"},
    ("coco", "val2014"): {"hf_id": "detection-datasets/coco", "split": "val", "id_field": "image_id"},

    # GQA
    ("gqa", "images"): {"hf_id": "lmms-lab/GQA", "config": "train_all_images", "split": "train", "id_field": None},

    # TextVQA
    ("textvqa", "train_val_images"): {"hf_id": "lmms-lab/textvqa", "split": "train", "id_field": "image_id"},

    # DocVQA
    ("docvqa", "documents"): {"hf_id": "lmms-lab/DocVQA", "config": "DocVQA", "split": "validation", "id_field": "ucsf_document_id"},

    # ALLaVA
    ("allava_vflan", "images"): {"hf_id": "FreedomIntelligence/ALLaVA-4V", "config": "allava_vflan", "split": "caption", "id_field": None},

    # Visual Genome
    ("vg", "VG_100K"): {"hf_id": "visual_genome", "config": "region_descriptions_v1.2.0", "split": "train", "id_field": "image_id"},
    ("vg", "VG_100K_2"): {"hf_id": "visual_genome", "config": "region_descriptions_v1.2.0", "split": "train", "id_field": "image_id"},

    # OCR-VQA
    ("ocr_vqa", "images"): {"hf_id": "howard-hou/OCR-VQA", "split": "train", "id_field": "image_id"},

    # ChartQA
    ("chartqa", "train"): {"hf_id": "ahmed-masry/ChartQA", "split": "train", "id_field": None},
    ("chartqa", "test"): {"hf_id": "ahmed-masry/ChartQA", "split": "test", "id_field": None},

    # MathVista
    ("mathvista", "images"): {"hf_id": "AI4Math/MathVista", "split": "testmini", "id_field": "pid"},

    # ScienceQA
    ("scienceqa", "images"): {"hf_id": "derek-thomas/ScienceQA", "split": "train", "id_field": None},

    # GeoQA+ - DISABLED: Dataset doesn't exist on Hub
    # ("geoqa+", "images"): {"hf_id": "AI4Math/GeoQA_Plus", "split": "train", "id_field": "problem"},
    # ("geoqa+", "test-images"): {"hf_id": "AI4Math/GeoQA_Plus", "split": "test", "id_field": "problem"},

    # SynthDog-EN
    ("synthdog-en", "images"): {"hf_id": "naver-clova-ix/synthdog-en", "split": "train", "id_field": None},
    ("synthdog-en", "test-images"): {"hf_id": "naver-clova-ix/synthdog-en", "split": "validation", "id_field": None},

    # DVQA - DISABLED: Dataset doesn't exist on Hub
    # ("dvqa", "images"): {"hf_id": "lmms-lab/DVQA", "split": "train", "id_field": "image"},

    # AI2D
    ("ai2d", "images"): {"hf_id": "lmms-lab/ai2d", "split": "test", "id_field": "image"},

    # MathVerse (multiple versions)
    ("mathverse", "images_version_1-4"): {"hf_id": "AI4Math/MathVerse", "config": "testmini", "split": "testmini", "id_field": "problem"},
    ("mathverse", "images_version_5"): {"hf_id": "AI4Math/MathVerse", "config": "testmini", "split": "testmini", "id_field": "problem"},
    ("mathverse", "images_version_6"): {"hf_id": "AI4Math/MathVerse", "config": "testmini", "split": "testmini", "id_field": "problem"},

    # SEED-Bench - Re-enabled (should work now, was temporary 502 error)
    ("seedbench", "SEED-Bench-image"): {"hf_id": "lmms-lab/SEED-Bench", "split": "test", "id_field": "question_id"},

    # SAM (Segment Anything) - DISABLED: Dataset doesn't exist on Hub
    # ("sam", "images"): {"hf_id": "facebook/segment-anything-1b", "split": "train", "id_field": None},

    # MMMU - DISABLED: Requires config parameter (30 different configs available)
    # ("mmmu", "test-images"): {"hf_id": "MMMU/MMMU", "split": "test", "id_field": "id"},

    # MME (multiple categories)
    ("mme", "landmark"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "artwork"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "celebrity"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "color"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "count"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "position"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "existence"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "OCR"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},

    # WikiArt - Re-enabled (should work now, was temporary timeout)
    ("wikiart", "images"): {"hf_id": "huggan/wikiart", "split": "train", "id_field": None},

    # Web-Landmark - DISABLED: Dataset doesn't exist on Hub
    # ("web-landmark", "images"): {"hf_id": "google-research-datasets/web-landmarks", "split": "train", "id_field": None},

    # Web-Celebrity - DISABLED: Dataset doesn't exist on Hub
    # ("web-celebrity", "images"): {"hf_id": "google-research-datasets/web-celebrity", "split": "train", "id_field": None},

    # ShareGPT4V TextVQA
    ("share_textvqa", "images"): {"hf_id": "lmms-lab/textvqa", "split": "train", "id_field": "image_id"},

    # LLaVA in the Wild - Re-enabled (dataset is available on HuggingFace)
    ("llava-in-the-wild", "images"): {"hf_id": "liuhaotian/LLaVA-Instruct-150K", "split": "train", "id_field": "id"},

    # MM-Vet - DISABLED: Dataset doesn't exist on Hub
    # ("mm-vet", "images"): {"hf_id": "lmms-lab/MM-Vet", "split": "test", "id_field": None},
}


def collect_fire_paths(fire_dataset_id: str, splits: list, max_samples: int = 0):
    """Collect all image paths from FIRE dataset."""
    logger.info("Collecting image paths from FIRE...")

    fire_paths = set()
    for split in splits:
        logger.info(f"Processing FIRE {split} split...")
        ds = load_dataset(fire_dataset_id, split=split, streaming=True)

        for idx, sample in enumerate(tqdm(ds, desc=f"Scanning {split}")):
            if max_samples > 0 and idx >= max_samples:
                break

            img_path = sample.get("image")
            if img_path and isinstance(img_path, str):
                fire_paths.add(img_path)

    logger.info(f"Collected {len(fire_paths)} unique image paths from FIRE")
    return fire_paths


def build_mapping_for_source(fire_paths: set, source: str, subfolder: str, config: dict, cache_dir: Path):
    """
    Build mapping for a specific dataset/split combination.

    Returns:
        dict: {fire_path: actual_image_path}
    """
    logger.info(f"=" * 60)
    logger.info(f"Building mapping for {source}/{subfolder}")
    logger.info(f"=" * 60)

    hf_id = config["hf_id"]
    split = config["split"]
    id_field = config.get("id_field")
    hf_config = config.get("config")

    # Filter FIRE paths for this source/subfolder
    relevant_paths = {
        path for path in fire_paths
        if path.startswith(f"{source}/{subfolder}/")
    }

    if not relevant_paths:
        logger.info(f"No paths found for {source}/{subfolder}")
        return {}

    logger.info(f"Need to map {len(relevant_paths)} paths")

    # Load dataset (non-streaming for fast indexed access)
    logger.info(f"Loading dataset: {hf_id} (split: {split})")
    try:
        if hf_config:
            ds = load_dataset(hf_id, hf_config, split=split, cache_dir=str(cache_dir), trust_remote_code=True)
        else:
            ds = load_dataset(hf_id, split=split, cache_dir=str(cache_dir), trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {}

    logger.info(f"Dataset loaded: {len(ds)} samples")

    # Build ID to index mapping
    mapping = {}

    if id_field:
        # Build index for fast lookup
        id_to_idx = {}
        for idx, sample in enumerate(tqdm(ds, desc="Building index")):
            sample_id = sample.get(id_field)
            if sample_id is not None:
                id_to_idx[str(sample_id)] = idx

        logger.info(f"Built index with {len(id_to_idx)} entries")

        # Map FIRE paths
        for fire_path in tqdm(relevant_paths, desc="Mapping paths"):
            # Extract image ID from FIRE path
            filename = fire_path.split('/')[-1]
            img_id = filename.rsplit('.', 1)[0]

            # For COCO, convert to int
            if source == "coco":
                try:
                    img_id = str(int(img_id))
                except:
                    pass

            # Look up in index
            if img_id in id_to_idx:
                idx = id_to_idx[img_id]
                # Store the dataset index as the mapped value
                mapping[fire_path] = {
                    "dataset": hf_id,
                    "config": hf_config,
                    "split": split,
                    "index": idx,
                    "id_field": id_field,
                    "id_value": img_id
                }
    else:
        # No ID field - use sequential indexing (all paths map to dataset in order)
        logger.info(f"No ID field - mapping {len(relevant_paths)} paths to {len(ds)} samples sequentially")

        # For datasets without ID field, we assume FIRE paths are just dataset indices
        # Map each path to its sequential index in the dataset
        for idx, fire_path in enumerate(tqdm(sorted(relevant_paths), desc="Mapping paths")):
            if idx < len(ds):
                mapping[fire_path] = {
                    "dataset": hf_id,
                    "config": hf_config,
                    "split": split,
                    "index": idx,
                    "id_field": None,
                    "id_value": None
                }

    logger.info(f"✓ Mapped {len(mapping)}/{len(relevant_paths)} paths")
    return mapping


def calculate_dataset_coverage(mapping: dict, fire_paths: set) -> dict:
    """Calculate mapping coverage per dataset/split combination.

    Returns:
        Dict mapping (source, subfolder) to (mapped_count, total_count, coverage%)
    """
    coverage = {}

    for (source, subfolder) in DATASET_SPLIT_MAPPING.keys():
        # Find all FIRE paths for this source/split
        relevant_paths = {
            path for path in fire_paths
            if path.startswith(f"{source}/{subfolder}/")
        }

        if not relevant_paths:
            continue

        # Count how many are in the mapping
        mapped_count = sum(1 for path in relevant_paths if path in mapping)
        total_count = len(relevant_paths)
        coverage_pct = (mapped_count / total_count) if total_count > 0 else 0

        coverage[(source, subfolder)] = {
            "mapped": mapped_count,
            "total": total_count,
            "coverage": coverage_pct
        }

    return coverage


def main():
    parser = argparse.ArgumentParser(
        description="Build mapping from FIRE image paths to HuggingFace datasets"
    )
    parser.add_argument(
        "--fire_dataset",
        type=str,
        default="PengxiangLi/FIRE",
        help="FIRE dataset ID",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/cache",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--output_mapping",
        type=str,
        default="/outputs/fire_image_mapping.json",
        help="Output mapping file path",
    )
    parser.add_argument(
        "--existing_mapping",
        type=str,
        default=None,
        help="Existing mapping JSON to extend (skips datasets with coverage >= min_coverage)",
    )
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.8,
        help="Skip datasets with coverage >= this threshold (0.0-1.0, default 0.8 = 80%%)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="FIRE splits to process",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Max samples from FIRE (0 = all)",
    )

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)

    logger.info("=" * 60)
    logger.info("FIRE Image Path Mapping Builder")
    logger.info("=" * 60)

    # Step 1: Collect all FIRE paths
    fire_paths = collect_fire_paths(args.fire_dataset, args.splits, args.max_samples)

    # Step 2: Load existing mapping if provided
    existing_mapping = {}
    skip_datasets = set()

    if args.existing_mapping:
        logger.info(f"Loading existing mapping from {args.existing_mapping}")
        with open(args.existing_mapping, 'r') as f:
            existing_mapping = json.load(f)

        # Calculate coverage for each dataset
        coverage = calculate_dataset_coverage(existing_mapping, fire_paths)

        logger.info(f"\nExisting mapping coverage (threshold: {args.min_coverage*100:.0f}%):")
        logger.info("-" * 60)

        for (source, subfolder), stats in sorted(coverage.items()):
            cov_pct = stats["coverage"] * 100
            status = "✓ SKIP" if stats["coverage"] >= args.min_coverage else "✗ RETRY"

            logger.info(f"{source:20s}/{subfolder:20s} {stats['mapped']:6d}/{stats['total']:6d} ({cov_pct:5.1f}%) {status}")

            if stats["coverage"] >= args.min_coverage:
                skip_datasets.add((source, subfolder))

        logger.info("-" * 60)
        logger.info(f"Skipping {len(skip_datasets)} datasets with good coverage")
        logger.info(f"Processing {len(DATASET_SPLIT_MAPPING) - len(skip_datasets)} datasets\n")

    # Step 3: Build mapping for each source/split
    full_mapping = existing_mapping.copy()
    datasets_processed = 0

    for (source, subfolder), config in DATASET_SPLIT_MAPPING.items():
        if (source, subfolder) in skip_datasets:
            logger.info(f"⊘ Skipping {source}/{subfolder} (coverage >= {args.min_coverage*100:.0f}%)")
            continue

        datasets_processed += 1
        mapping = build_mapping_for_source(fire_paths, source, subfolder, config, cache_dir)
        full_mapping.update(mapping)

    # Step 4: Save mapping
    output_path = Path(args.output_mapping)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-backup if overwriting existing file
    if args.existing_mapping and Path(args.existing_mapping).resolve() == output_path.resolve():
        backup_path = f"{output_path}.backup"
        if output_path.exists():
            logger.info(f"Creating backup: {backup_path}")
            shutil.copy2(output_path, backup_path)
            logger.info(f"Backup saved ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    with open(output_path, 'w') as f:
        json.dump(full_mapping, f, indent=2)

    logger.info("=" * 60)
    logger.info("MAPPING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Datasets processed: {datasets_processed}")
    logger.info(f"Datasets skipped: {len(skip_datasets)}")
    logger.info(f"Total paths mapped: {len(full_mapping)}")
    logger.info(f"Total paths in FIRE: {len(fire_paths)}")
    logger.info(f"Coverage: {len(full_mapping)/len(fire_paths)*100:.1f}%")
    logger.info(f"Mapping saved to: {output_path}")

    # Mention backup if created
    if args.existing_mapping and Path(args.existing_mapping).resolve() == output_path.resolve():
        backup_path = Path(f"{output_path}.backup")
        if backup_path.exists():
            logger.info(f"Backup available at: {backup_path}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
