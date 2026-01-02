
#!/usr/bin/env python3
"""
Build a mapping file from FIRE image paths to actual downloaded images.
This allows us to download datasets once and map FIRE paths flexibly.
"""

import argparse
import json
import logging
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
    ("gqa", "images"): {"hf_id": "lmms-lab/GQA", "config": "train_all_images", "split": "train", "id_field": "imageId"},

    # TextVQA
    ("textvqa", "train_val_images"): {"hf_id": "lmms-lab/textvqa", "split": "train", "id_field": "image_id"},

    # DocVQA
    ("docvqa", "documents"): {"hf_id": "lmms-lab/DocVQA", "config": "DocVQA", "split": "validation", "id_field": "image"},

    # ALLaVA
    ("allava_vflan", "images"): {"hf_id": "FreedomIntelligence/ALLaVA-4V", "config": "allava_vflan", "split": "train", "id_field": None},

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

    # GeoQA+
    ("geoqa+", "images"): {"hf_id": "AI4Math/GeoQA_Plus", "split": "train", "id_field": "problem"},
    ("geoqa+", "test-images"): {"hf_id": "AI4Math/GeoQA_Plus", "split": "test", "id_field": "problem"},

    # SynthDog-EN
    ("synthdog-en", "images"): {"hf_id": "naver-clova-ix/synthdog-en", "split": "train", "id_field": None},
    ("synthdog-en", "test-images"): {"hf_id": "naver-clova-ix/synthdog-en", "split": "validation", "id_field": None},

    # DVQA
    ("dvqa", "images"): {"hf_id": "lmms-lab/DVQA", "split": "train", "id_field": "image"},

    # AI2D
    ("ai2d", "images"): {"hf_id": "lmms-lab/ai2d", "split": "train", "id_field": "image"},

    # MathVerse (multiple versions)
    ("mathverse", "images_version_1-4"): {"hf_id": "AI4Math/MathVerse", "split": "train", "id_field": "problem"},
    ("mathverse", "images_version_5"): {"hf_id": "AI4Math/MathVerse", "split": "train", "id_field": "problem"},
    ("mathverse", "images_version_6"): {"hf_id": "AI4Math/MathVerse", "split": "train", "id_field": "problem"},

    # SEED-Bench
    ("seedbench", "SEED-Bench-image"): {"hf_id": "lmms-lab/SEED-Bench", "split": "test", "id_field": "question_id"},

    # SAM (Segment Anything)
    ("sam", "images"): {"hf_id": "facebook/segment-anything-1b", "split": "train", "id_field": None},

    # MMMU
    ("mmmu", "test-images"): {"hf_id": "MMMU/MMMU", "split": "test", "id_field": "id"},

    # MME (multiple categories)
    ("mme", "landmark"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "artwork"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "celebrity"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "color"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "count"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "position"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "existence"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},
    ("mme", "OCR"): {"hf_id": "lmms-lab/MME", "split": "test", "id_field": None},

    # WikiArt
    ("wikiart", "images"): {"hf_id": "huggan/wikiart", "split": "train", "id_field": None},

    # Web-Landmark
    ("web-landmark", "images"): {"hf_id": "google-research-datasets/web-landmarks", "split": "train", "id_field": None},

    # Web-Celebrity
    ("web-celebrity", "images"): {"hf_id": "google-research-datasets/web-celebrity", "split": "train", "id_field": None},

    # ShareGPT4V TextVQA
    ("share_textvqa", "images"): {"hf_id": "lmms-lab/textvqa", "split": "train", "id_field": "image_id"},

    # LLaVA in the Wild
    ("llava-in-the-wild", "images"): {"hf_id": "liuhaotian/LLaVA-Instruct-150K", "split": "train", "id_field": "id"},

    # MM-Vet
    ("mm-vet", "images"): {"hf_id": "lmms-lab/MM-Vet", "split": "test", "id_field": None},
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

    # Step 2: Build mapping for each source/split
    full_mapping = {}

    for (source, subfolder), config in DATASET_SPLIT_MAPPING.items():
        mapping = build_mapping_for_source(fire_paths, source, subfolder, config, cache_dir)
        full_mapping.update(mapping)

    # Step 3: Save mapping
    output_path = Path(args.output_mapping)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(full_mapping, f, indent=2)

    logger.info("=" * 60)
    logger.info("MAPPING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total paths mapped: {len(full_mapping)}")
    logger.info(f"Total paths in FIRE: {len(fire_paths)}")
    logger.info(f"Coverage: {len(full_mapping)/len(fire_paths)*100:.1f}%")
    logger.info(f"Mapping saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
