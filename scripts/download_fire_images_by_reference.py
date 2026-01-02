#!/usr/bin/env python3
"""
Download images referenced by FIRE dataset to exact paths.
Parses FIRE dataset to get all image references, then downloads from source datasets.
"""

import argparse
import logging
import json
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Map FIRE image path prefixes to HuggingFace datasets
SOURCE_DATASET_MAPPING = {
    "coco": {
        "hf_id": "detection-datasets/coco",
        "image_field": "image",
        "id_field": "image_id",
    },
    "gqa": {
        "hf_id": "lmms-lab/GQA",
        "config": "train_all_images",
        "image_field": "image",
        "id_field": "imageId",
    },
    "textvqa": {
        "hf_id": "lmms-lab/textvqa",
        "image_field": "image",
        "id_field": "image_id",
    },
    "docvqa": {
        "hf_id": "lmms-lab/DocVQA",
        "config": "DocVQA",
        "image_field": "image",
        "id_field": "questionId",
    },
    "allava_vflan": {
        "hf_id": "FreedomIntelligence/ALLaVA-4V",
        "image_field": "image",
        "id_field": None,
    },
    "vg": {
        "hf_id": "visual_genome",
        "config": "region_descriptions_v1.2.0",
        "image_field": "image",
        "id_field": "image_id",
    },
    "geoqa+": {
        "hf_id": "AI4Math/GeoQA_Plus",
        "image_field": "image",
        "id_field": "problem",
    },
    "ocr_vqa": {
        "hf_id": "howard-hou/OCR-VQA",
        "image_field": "image",
        "id_field": "image_id",
    },
    "chartqa": {
        "hf_id": "ahmed-masry/ChartQA",
        "image_field": "image",
        "id_field": None,
    },
    "synthdog-en": {
        "hf_id": "naver-clova-ix/synthdog-en",
        "image_field": "image",
        "id_field": None,
    },
    "dvqa": {
        "hf_id": "lmms-lab/DVQA",
        "image_field": "image",
        "id_field": "image",
    },
    "scienceqa": {
        "hf_id": "derek-thomas/ScienceQA",
        "image_field": "image",
        "id_field": None,
    },
    "mathverse": {
        "hf_id": "AI4Math/MathVerse",
        "image_field": "image",
        "id_field": "problem",
    },
    "mathvista": {
        "hf_id": "AI4Math/MathVista",
        "image_field": "decoded_image",
        "id_field": "pid",
    },
    "seedbench": {
        "hf_id": "lmms-lab/SEED-Bench",
        "image_field": "image",
        "id_field": "question_id",
    },
    "mmmu": {
        "hf_id": "MMMU/MMMU",
        "image_field": "image_1",
        "id_field": "id",
    },
    "mme": {
        "hf_id": "lmms-lab/MME",
        "image_field": "image",
        "id_field": None,
    },
    "mm-vet": {
        "hf_id": "lmms-lab/MM-Vet",
        "image_field": "image",
        "id_field": None,
    },
    "llava-in-the-wild": {
        "hf_id": "liuhaotian/LLaVA-Instruct-150K",
        "image_field": "image",
        "id_field": "id",
    },
    "sam": {
        "hf_id": "facebook/segment-anything-1b",
        "image_field": "image",
        "id_field": None,
    },
    "wikiart": {
        "hf_id": "huggan/wikiart",
        "image_field": "image",
        "id_field": None,
    },
    "web-landmark": {
        "hf_id": "google-research-datasets/web-landmarks",
        "image_field": "image",
        "id_field": None,
    },
    "web-celebrity": {
        "hf_id": "google-research-datasets/web-celebrity",
        "image_field": "image",
        "id_field": None,
    },
    "share_textvqa": {
        "hf_id": "lmms-lab/textvqa",
        "image_field": "image",
        "id_field": "image_id",
    },
    "ai2d": {
        "hf_id": "lmms-lab/ai2d",
        "image_field": "image",
        "id_field": "image",
    },
}


def collect_fire_image_paths(fire_dataset_id: str, splits: list, max_samples: int = 0):
    """
    Collect all unique image paths from FIRE dataset.

    Returns:
        dict: {source_name: set of image paths}
    """
    logger.info(f"Collecting image paths from FIRE dataset: {fire_dataset_id}")

    image_paths_by_source = defaultdict(set)
    total_samples = 0

    for split in splits:
        logger.info(f"Processing {split} split...")
        ds = load_dataset(fire_dataset_id, split=split, streaming=True)

        for idx, sample in enumerate(tqdm(ds, desc=f"Scanning {split}")):
            if max_samples > 0 and idx >= max_samples:
                break

            img_path = sample.get("image")
            if not img_path or not isinstance(img_path, str):
                continue

            # Extract source from path (e.g., "coco/train2017/..." -> "coco")
            source = img_path.split('/')[0]
            image_paths_by_source[source].add(img_path)
            total_samples += 1

    logger.info(f"Collected {total_samples} total image references")
    for source, paths in sorted(image_paths_by_source.items()):
        logger.info(f"  {source}: {len(paths)} unique images")

    return image_paths_by_source


def extract_image_id_from_path(image_path: str, source: str) -> str:
    """
    Extract the image ID from FIRE's image path.

    Examples:
        coco/train2017/000000496160.jpg -> 000000496160
        gqa/images/n336443.jpg -> n336443
        textvqa/train_val_images/train_images/0004c9478eeda995.jpg -> 0004c9478eeda995
    """
    filename = image_path.split('/')[-1]
    # Remove extension
    img_id = filename.rsplit('.', 1)[0]

    # For COCO, convert to integer
    if source == "coco":
        try:
            return int(img_id)
        except:
            return img_id

    return img_id


def download_images_for_source(
    source_name: str,
    image_paths: set,
    output_dir: Path,
    config: dict
):
    """
    Download all images for a specific source dataset.
    """
    logger.info(f"=" * 60)
    logger.info(f"Downloading {len(image_paths)} images for {source_name}")
    logger.info(f"=" * 60)

    hf_id = config["hf_id"]
    image_field = config["image_field"]
    id_field = config.get("id_field")
    hf_config = config.get("config")

    # Create a mapping of image IDs to FIRE paths
    id_to_fire_paths = {}
    for fire_path in image_paths:
        img_id = extract_image_id_from_path(fire_path, source_name)
        if img_id not in id_to_fire_paths:
            id_to_fire_paths[img_id] = []
        id_to_fire_paths[img_id].append(fire_path)

    logger.info(f"Need to find {len(id_to_fire_paths)} unique image IDs")

    # Load source dataset
    try:
        logger.info(f"Loading source dataset: {hf_id}")
        ds = None

        if hf_config:
            # Try different splits
            for split_name in ["train", "validation", "test", "testmini"]:
                try:
                    ds = load_dataset(hf_id, hf_config, split=split_name, streaming=True, trust_remote_code=True)
                    logger.info(f"  Using split: {split_name}")
                    break
                except Exception as e:
                    continue
        else:
            # Try to load all splits
            try:
                ds_dict = load_dataset(hf_id, streaming=True, trust_remote_code=True)
                # Combine all splits into a list
                all_datasets = []
                for split_name, split_ds in ds_dict.items():
                    logger.info(f"  Found split: {split_name}")
                    all_datasets.append(split_ds)

                # We'll iterate through all splits
                ds = all_datasets if all_datasets else None
            except Exception as e:
                # Fallback to single split
                for split_name in ["train", "validation", "test"]:
                    try:
                        ds = load_dataset(hf_id, split=split_name, streaming=True, trust_remote_code=True)
                        logger.info(f"  Using split: {split_name}")
                        break
                    except:
                        continue

        if ds is None:
            logger.error(f"Failed to load dataset {hf_id}")
            return 0

        # Iterate through dataset and find matching images
        downloaded = 0
        remaining_ids = set(id_to_fire_paths.keys())

        # Handle both single dataset and list of datasets
        datasets_to_search = [ds] if not isinstance(ds, list) else ds

        for dataset in datasets_to_search:
            if not remaining_ids:
                break

            for sample in tqdm(dataset, desc=f"Searching {source_name}"):
                if not remaining_ids:
                    break

                # Get image
                image = sample.get(image_field)
                if not image or not isinstance(image, Image.Image):
                    continue

                # Get image ID from source dataset
                if id_field and id_field in sample:
                    sample_id = sample[id_field]
                    # For COCO, ensure it's an int
                    if source_name == "coco" and isinstance(sample_id, str):
                        try:
                            sample_id = int(sample_id)
                        except:
                            continue
                else:
                    # If no ID field, we can't match - skip this source
                    continue

                # Check if this ID is needed
                if sample_id in remaining_ids:
                    # Save image to all FIRE paths that reference it
                    for fire_path in id_to_fire_paths[sample_id]:
                        output_path = output_dir / fire_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        # Convert to RGB for JPEG
                        if fire_path.endswith('.jpg') and image.mode != "RGB":
                            image = image.convert("RGB")

                        # Save image
                        try:
                            if fire_path.endswith('.jpg'):
                                image.save(output_path, "JPEG", quality=95)
                            else:
                                image.save(output_path, "PNG")
                            downloaded += 1
                        except Exception as e:
                            logger.warning(f"Failed to save {fire_path}: {e}")

                    remaining_ids.remove(sample_id)

        logger.info(f"✓ Downloaded {downloaded} images for {source_name}")
        logger.info(f"  Missing: {len(remaining_ids)} images")
        if remaining_ids and len(remaining_ids) <= 20:
            logger.info(f"  Missing IDs: {list(remaining_ids)}")

        return downloaded

    except Exception as e:
        logger.error(f"✗ Failed to download {source_name}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download images for FIRE dataset matching exact path references"
    )
    parser.add_argument(
        "--fire_dataset",
        type=str,
        default="PengxiangLi/FIRE",
        help="FIRE dataset ID on HuggingFace",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs",
        help="Base output directory for images",
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
        help="Max samples to process from FIRE (0 = all)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=None,
        help="Specific sources to download (default: all found in FIRE)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FIRE Image Download by Reference")
    logger.info("=" * 60)
    logger.info(f"FIRE dataset: {args.fire_dataset}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Max samples: {args.max_samples if args.max_samples > 0 else 'ALL'}")
    logger.info("=" * 60)

    # Step 1: Collect all image paths from FIRE
    image_paths_by_source = collect_fire_image_paths(
        args.fire_dataset, args.splits, args.max_samples
    )

    # Filter to requested sources if specified
    if args.sources:
        image_paths_by_source = {
            k: v for k, v in image_paths_by_source.items()
            if k in args.sources
        }

    # Step 2: Download images for each source
    total_downloaded = 0
    successful_sources = []
    failed_sources = []

    for source_name in sorted(image_paths_by_source.keys()):
        image_paths = image_paths_by_source[source_name]

        if source_name not in SOURCE_DATASET_MAPPING:
            logger.warning(f"No mapping for source: {source_name}, skipping...")
            failed_sources.append(source_name)
            continue

        config = SOURCE_DATASET_MAPPING[source_name]
        try:
            count = download_images_for_source(
                source_name, image_paths, output_dir, config
            )
            total_downloaded += count
            if count > 0:
                successful_sources.append(source_name)
            else:
                failed_sources.append(source_name)
        except Exception as e:
            logger.error(f"Error downloading {source_name}: {e}")
            failed_sources.append(source_name)

    # Summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images downloaded: {total_downloaded}")
    logger.info(f"Successful sources ({len(successful_sources)}): {', '.join(successful_sources)}")
    if failed_sources:
        logger.info(f"Failed sources ({len(failed_sources)}): {', '.join(failed_sources)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
