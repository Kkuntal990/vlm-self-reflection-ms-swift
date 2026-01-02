#!/usr/bin/env python3
"""
Download ALL images from all source datasets used by FIRE.
Downloads everything without worrying about splits or exact path matching.
"""

import argparse
import logging
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mapping of FIRE source names to HuggingFace dataset IDs
DATASET_MAPPING = {
    # "coco": {
    #     "hf_id": "detection-datasets/coco",
    #     "image_field": "image",
    #     "id_field": "image_id",
    #     "format_id": lambda x: f"{x:012d}",  # COCO uses 12-digit padded IDs
    # },
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
    "chartqa": {
        "hf_id": "ahmed-masry/ChartQA",
        "image_field": "image",
        "id_field": None,  # Will use index if no ID
    },
    "mathvista": {
        "hf_id": "AI4Math/MathVista",
        "image_field": "decoded_image",
        "id_field": "pid",
    },
    "scienceqa": {
        "hf_id": "derek-thomas/ScienceQA",
        "image_field": "image",
        "id_field": None,
    },
    "ocr_vqa": {
        "hf_id": "howard-hou/OCR-VQA",
        "image_field": "image",
        "id_field": "image_id",
    },
    "geoqa+": {
        "hf_id": "AI4Math/GeoQA_Plus",
        "image_field": "image",
        "id_field": "problem",
    },
    "dvqa": {
        "hf_id": "lmms-lab/DVQA",
        "image_field": "image",
        "id_field": "image",  # DVQA has image name
    },
    "ai2d": {
        "hf_id": "lmms-lab/ai2d",
        "image_field": "image",
        "id_field": "image",
    },
    "vg": {
        "hf_id": "visual_genome",
        "config": "region_descriptions_v1.2.0",
        "image_field": "image",
        "id_field": "image_id",
    },
    "mathverse": {
        "hf_id": "AI4Math/MathVerse",
        "image_field": "image",
        "id_field": "problem",
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
    "synthdog-en": {
        "hf_id": "naver-clova-ix/synthdog-en",
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
    "share_textvqa": {
        "hf_id": "lmms-lab/textvqa",  # Same as textvqa
        "image_field": "image",
        "id_field": "image_id",
    },
    "allava_vflan": {
        "hf_id": "FreedomIntelligence/ALLaVA-4V",
        "image_field": "image",
        "id_field": None,
    },
}


def download_dataset_images(source_name: str, config: dict, output_dir: Path, max_images: int = 0):
    """
    Download ALL images from a single source dataset.

    Args:
        source_name: Name of the source (e.g., 'coco', 'gqa')
        config: Configuration dict with dataset info
        output_dir: Base output directory
        max_images: Maximum images to download (0 = all)
    """
    logger.info(f"=" * 60)
    logger.info(f"Downloading {source_name}")
    logger.info(f"=" * 60)

    hf_id = config["hf_id"]
    image_field = config["image_field"]
    id_field = config.get("id_field")
    format_id = config.get("format_id", lambda x: str(x))
    hf_config = config.get("config")

    # Create output directory for this source
    source_dir = output_dir / source_name
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset in streaming mode to handle large datasets
        logger.info(f"Loading dataset: {hf_id}")
        if hf_config:
            # Try common split names in order
            for split_name in ["train", "validation", "test", "testmini"]:
                try:
                    ds = load_dataset(hf_id, hf_config, split=split_name, streaming=True, trust_remote_code=True)
                    logger.info(f"  Using split: {split_name}")
                    break
                except:
                    continue
            else:
                raise ValueError(f"No valid split found for {hf_id}")
        else:
            # Try to load all splits
            try:
                ds = load_dataset(hf_id, streaming=True, trust_remote_code=True)
                # Combine all splits
                all_samples = []
                for split_name, split_ds in ds.items():
                    logger.info(f"  Found split: {split_name}")
                    all_samples.append(split_ds)
                # Use first split for now (we'll iterate through all)
                ds = all_samples[0] if all_samples else None
            except:
                # Fallback to trying common splits
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

        count = 0
        skipped = 0
        errors = 0

        for idx, sample in enumerate(tqdm(ds, desc=f"Downloading {source_name}")):
            if max_images > 0 and count >= max_images:
                break

            # Get image
            image = sample.get(image_field)
            if not image:
                skipped += 1
                if idx < 5:  # Log first few for debugging
                    logger.debug(f"Sample {idx}: No image field '{image_field}'")
                continue

            # Handle different image types
            if not isinstance(image, Image.Image):
                skipped += 1
                if idx < 5:
                    logger.debug(f"Sample {idx}: Image is {type(image)}, not PIL Image")
                continue

            # Get ID or use index
            if id_field and id_field in sample:
                img_id = format_id(sample[id_field])
            else:
                img_id = str(idx)

            # Determine file extension
            ext = ".jpg" if image.mode == "RGB" else ".png"
            output_path = source_dir / f"{img_id}{ext}"

            # Skip if exists
            if output_path.exists():
                continue

            # Convert to RGB for JPEG
            if ext == ".jpg" and image.mode != "RGB":
                image = image.convert("RGB")

            # Save image
            try:
                image.save(output_path, "JPEG" if ext == ".jpg" else "PNG", quality=95)
                count += 1
            except Exception as e:
                errors += 1
                if errors <= 5:  # Log first few errors
                    logger.warning(f"Failed to save image {img_id}: {e}")
                continue

        logger.info(f"✓ Downloaded {count} images from {source_name} (skipped: {skipped}, errors: {errors})")
        return count

    except Exception as e:
        logger.error(f"✗ Failed to download {source_name}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download ALL images from all FIRE source datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs",
        help="Base output directory for all images",
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=None,
        help="Specific sources to download (default: all)",
    )
    parser.add_argument(
        "--max_images_per_source",
        type=int,
        default=0,
        help="Max images per source (0 = all, for testing use 10)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which sources to download
    sources_to_download = args.sources if args.sources else list(DATASET_MAPPING.keys())

    logger.info("=" * 60)
    logger.info("FIRE Source Datasets Download")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Sources to download: {len(sources_to_download)}")
    logger.info(f"Max images per source: {args.max_images_per_source if args.max_images_per_source > 0 else 'ALL'}")
    logger.info("=" * 60)

    # Download each source
    total_downloaded = 0
    successful_sources = []
    failed_sources = []

    for source_name in sources_to_download:
        if source_name not in DATASET_MAPPING:
            logger.warning(f"Unknown source: {source_name}, skipping...")
            continue

        config = DATASET_MAPPING[source_name]
        try:
            count = download_dataset_images(
                source_name, config, output_dir, args.max_images_per_source
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
