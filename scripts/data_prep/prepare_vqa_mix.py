#!/usr/bin/env python3
"""
Prepare balanced VQA training mix to fix data distribution mismatch.

Addresses benchmark degradation (MME -135, SEEDBench -1.1%) caused by FIRE's
75% open-ended / 2.9% MCQ distribution vs MCQ/yes-no heavy benchmarks.

Steps:
  1. Filter FIRE messages: remove bounding_box + region_description categories
  2. Convert 30% of open_ended + short_answer FIRE samples to single-turn
  3. Process LVLM-NLF: keep all yes/no + short, 10K long multi-turn,
     10K long as single-turn, 10K long as reversed feedback
  4. Download A-OKVQA and TallyQA from LLaVA-OneVision-Data (Cauldron format)
  5. Combine all datasets and shuffle

Output: mixed_training_v1.jsonl

Usage:
    # Full run on pod (downloads datasets, saves images)
    python scripts/data_prep/prepare_vqa_mix.py \\
        --output_dir /outputs/mixed_training_v1 \\
        --image_dir /outputs/image_base \\
        --fire_messages /outputs/fire_preprocessed_v3/fire_messages_train_last_loss_only_with_thought.jsonl \\
        --fire_categories /outputs/fire_preprocessed_v3/fire_sample_categories.jsonl \\
        --fire_feedback /outputs/fire_preprocessed_v3/fire_feedback_train_v2_with_thought.jsonl \\
        --lvlm_nlf /outputs/lvlm_nlf_preprocessed/lvlm_nlf_messages_train_last_loss_only_balanced.jsonl

    # Quick test (10 samples per dataset)
    python scripts/data_prep/prepare_vqa_mix.py \\
        --output_dir /outputs/mixed_training_v1_test \\
        --image_dir /outputs/image_base \\
        --fire_messages /outputs/fire_preprocessed_v3/fire_messages_train_last_loss_only_with_thought.jsonl \\
        --fire_categories /outputs/fire_preprocessed_v3/fire_sample_categories.jsonl \\
        --fire_feedback /outputs/fire_preprocessed_v3/fire_feedback_train_v2_with_thought.jsonl \\
        --lvlm_nlf /outputs/lvlm_nlf_preprocessed/lvlm_nlf_messages_train_last_loss_only_balanced.jsonl \\
        --max_samples 10
"""

import argparse
import io
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Categories to remove from FIRE (not benchmarked)
FIRE_REMOVE_CATEGORIES = {"bounding_box", "region_description"}

# Categories eligible for single-turn conversion in FIRE
FIRE_SINGLE_TURN_CATEGORIES = {"open_ended", "short_answer"}


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Prepare balanced VQA training mix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Existing dataset paths
    parser.add_argument(
        "--fire_messages",
        type=str,
        required=True,
        help="Path to fire_messages_train_last_loss_only_with_thought.jsonl",
    )
    parser.add_argument(
        "--fire_categories",
        type=str,
        required=True,
        help="Path to fire_sample_categories.jsonl",
    )
    parser.add_argument(
        "--fire_feedback",
        type=str,
        required=True,
        help="Path to fire_feedback_train_v2_with_thought.jsonl",
    )
    parser.add_argument(
        "--lvlm_nlf",
        type=str,
        required=True,
        help="Path to lvlm_nlf_messages_train_last_loss_only_balanced.jsonl",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/mixed_training_v1",
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/outputs/image_base",
        help="Base directory for saving new dataset images",
    )

    # Sampling configuration
    parser.add_argument(
        "--lvlm_long_multiturn",
        type=int,
        default=10000,
        help="Number of long-descriptive LVLM samples to keep as multi-turn",
    )
    parser.add_argument(
        "--lvlm_long_singleturn",
        type=int,
        default=10000,
        help="Number of long-descriptive LVLM samples to convert to single-turn",
    )
    parser.add_argument(
        "--lvlm_long_feedback",
        type=int,
        default=10000,
        help="Number of long-descriptive LVLM samples to reverse as feedback",
    )
    parser.add_argument(
        "--aokvqa_max",
        type=int,
        default=17000,
        help="Maximum A-OKVQA samples (0 = all)",
    )
    parser.add_argument(
        "--tallyqa_max",
        type=int,
        default=5000,
        help="Maximum TallyQA samples",
    )
    parser.add_argument(
        "--single_turn_fraction",
        type=float,
        default=0.30,
        help="Fraction of eligible FIRE samples to convert to single-turn",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples per dataset for testing (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip_new_datasets",
        action="store_true",
        help="Skip downloading A-OKVQA and TallyQA (for testing existing data steps)",
    )

    return parser.parse_args()


# =============================================================================
# Image utilities
# =============================================================================


def _save_image(
    image: Image.Image,
    image_dir: Path,
    dataset_name: str,
    idx: int,
) -> str:
    """Save a PIL image to disk and return its absolute path.

    Args:
        image: PIL image to save
        image_dir: Base directory for image storage
        dataset_name: Dataset subdirectory name
        idx: Sample index used for filename

    Returns:
        Absolute path to saved image
    """
    out_dir = image_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if image.mode != "RGB":
        image = image.convert("RGB")

    abs_path = str(image_dir / dataset_name / f"{idx:06d}.jpg")
    image.save(abs_path, format="JPEG", quality=90)
    return abs_path


def _open_image(raw: Any) -> Image.Image | None:
    """Open a PIL image from various raw formats returned by HF datasets.

    Args:
        raw: Raw image value (PIL.Image, bytes, or dict with "bytes" key)

    Returns:
        PIL Image in RGB mode, or None if loading failed
    """
    try:
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, bytes):
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if isinstance(raw, dict) and "bytes" in raw:
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to open image: {e}")
    return None


# =============================================================================
# Step 1: Filter FIRE messages and convert some to single-turn
# =============================================================================


def filter_fire_messages(
    fire_messages_path: str,
    fire_categories_path: str,
    single_turn_fraction: float,
    seed: int,
    max_samples: int,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Filter FIRE messages and convert eligible samples to single-turn.

    Removes bounding_box and region_description categories. Converts a fraction
    of open_ended and short_answer samples to single-turn format (question +
    final answer only, no feedback rounds).

    Args:
        fire_messages_path: Path to FIRE messages JSONL
        fire_categories_path: Path to FIRE categories JSONL
        single_turn_fraction: Fraction of eligible samples to convert
        seed: Random seed
        max_samples: Max samples for testing (0 = all)

    Returns:
        Tuple of (multi_turn_samples, single_turn_samples, stats_dict)
    """
    logger.info("Loading FIRE categories...")
    categories: dict[int, str] = {}
    with open(fire_categories_path) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                categories[d["index"]] = d["category"]
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse category line: {e}")

    logger.info(f"Loaded {len(categories)} category labels")

    rng = random.Random(seed)

    # Pre-select which eligible indices get converted to single-turn
    # When max_samples is set, restrict to indices within that range
    eligible_indices = {
        idx
        for idx, cat in categories.items()
        if cat in FIRE_SINGLE_TURN_CATEGORIES and (max_samples <= 0 or idx < max_samples)
    }

    n_convert = int(len(eligible_indices) * single_turn_fraction)
    single_turn_indices = (
        set(rng.sample(sorted(eligible_indices), n_convert)) if n_convert > 0 else set()
    )

    logger.info(
        f"Eligible for single-turn: {len(eligible_indices)}, converting: {len(single_turn_indices)}"
    )

    multi_turn: list[dict] = []
    single_turn: list[dict] = []
    stats: dict[str, Any] = {
        "total": 0,
        "removed_bounding_box": 0,
        "removed_region_description": 0,
        "kept_multi_turn": 0,
        "converted_single_turn": 0,
        "categories_kept": {},
    }

    logger.info("Filtering FIRE messages...")
    with open(fire_messages_path) as f:
        for idx, line in enumerate(tqdm(f, desc="FIRE filter")):
            if max_samples > 0 and idx >= max_samples:
                break

            stats["total"] += 1
            cat = categories.get(idx, "unknown")

            if cat in FIRE_REMOVE_CATEGORIES:
                stats[f"removed_{cat}"] += 1
                continue

            try:
                sample = json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse FIRE line {idx}: {e}")
                continue

            if idx in single_turn_indices:
                converted = _convert_to_single_turn(sample)
                if converted is not None:
                    single_turn.append(converted)
                    stats["converted_single_turn"] += 1
                else:
                    multi_turn.append(sample)
                    stats["kept_multi_turn"] += 1
            else:
                multi_turn.append(sample)
                stats["kept_multi_turn"] += 1

            stats["categories_kept"][cat] = stats["categories_kept"].get(cat, 0) + 1

    logger.info(
        f"FIRE filter: {stats['kept_multi_turn']} multi-turn, "
        f"{stats['converted_single_turn']} single-turn, "
        f"{stats['removed_bounding_box']} bounding_box removed, "
        f"{stats['removed_region_description']} region_description removed"
    )
    logger.info(f"Category distribution: {stats['categories_kept']}")

    return multi_turn, single_turn, stats


def _convert_to_single_turn(sample: dict) -> dict | None:
    """Convert a multi-turn sample to single-turn.

    Keeps only the first user question and the last assistant response.
    Removes system prompt, all feedback rounds, and intermediate responses.

    Args:
        sample: Sample with messages and images

    Returns:
        Single-turn sample or None if conversion fails
    """
    messages = sample.get("messages", [])
    images = sample.get("images", [])

    # Find first user question (skip system)
    first_question = None
    for msg in messages:
        if msg["role"] == "user":
            first_question = msg["content"]
            break

    # Find last assistant response
    last_answer = None
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            last_answer = msg["content"]
            break

    if first_question is None or last_answer is None:
        return None

    new_messages = [
        {"role": "user", "content": first_question},
        {"role": "assistant", "content": last_answer, "loss": True},
    ]

    return {"messages": new_messages, "images": images[:1] if images else []}


# =============================================================================
# Step 2: Process LVLM-NLF with multiple strategies
# =============================================================================


def _classify_lvlm_answer(sample: dict) -> str:
    """Classify a LVLM-NLF sample by answer type.

    Args:
        sample: Sample dict with messages

    Returns:
        Category string: "yes_no", "short_phrase", or "long_descriptive"
    """
    last_resp = ""
    for msg in sample["messages"]:
        if msg["role"] == "assistant" and msg.get("loss"):
            last_resp = msg["content"]

    words = last_resp.strip().split()
    if not words:
        return "short_phrase"

    first_word = words[0].rstrip(".,!").lower()
    if first_word in ("yes", "no"):
        return "yes_no"
    elif len(words) <= 10:
        return "short_phrase"
    else:
        return "long_descriptive"


def _convert_to_feedback_turn(sample: dict) -> dict | None:
    """Convert a multi-turn LVLM-NLF sample to a feedback-generation sample.

    Reverses roles: given the question and a student's initial answer, the model
    should produce feedback. Uses the first user feedback turn as the target.

    Input format (LVLM-NLF multi-turn):
        system → user(question) → assistant(answer1, loss=false) →
        user(feedback) → assistant(answer2, loss=true)

    Output format (feedback generation):
        user(question) → assistant(answer1, loss=false) →
        user("Provide feedback on the above response.") →
        assistant(feedback_text, loss=true)

    Args:
        sample: LVLM-NLF sample with messages and images

    Returns:
        Feedback-generation sample or None if conversion fails
    """
    messages = sample.get("messages", [])
    images = sample.get("images", [])

    # Extract components: first question, first answer, first feedback
    first_question = None
    first_answer = None
    first_feedback = None

    state = "find_question"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if state == "find_question" and role == "user":
            first_question = content
            state = "find_answer"
        elif state == "find_answer" and role == "assistant":
            first_answer = content
            state = "find_feedback"
        elif state == "find_feedback" and role == "user":
            first_feedback = content
            break

    if first_question is None or first_answer is None or first_feedback is None:
        return None

    # Skip if feedback is too short to be useful
    if len(first_feedback.split()) < 5:
        return None

    new_messages = [
        {"role": "user", "content": first_question},
        {"role": "assistant", "content": first_answer, "loss": False},
        {"role": "user", "content": "Provide feedback on the above response."},
        {"role": "assistant", "content": first_feedback, "loss": True},
    ]

    return {"messages": new_messages, "images": images}


def process_lvlm_nlf(
    lvlm_path: str,
    long_multiturn: int,
    long_singleturn: int,
    long_feedback: int,
    seed: int,
    max_samples: int,
) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    """Process LVLM-NLF into multi-turn, single-turn, and feedback subsets.

    Strategy:
      - All yes/no samples (~3.5K): kept as multi-turn
      - All short phrase samples (~9K): kept as multi-turn
      - Long descriptive: split into three non-overlapping pools:
        - long_multiturn: kept as-is (multi-turn)
        - long_singleturn: converted to single-turn (question + final answer)
        - long_feedback: reversed roles (model generates feedback)

    Args:
        lvlm_path: Path to LVLM-NLF JSONL
        long_multiturn: Number of long samples to keep as multi-turn
        long_singleturn: Number of long samples to convert to single-turn
        long_feedback: Number of long samples to reverse as feedback
        seed: Random seed
        max_samples: Max samples for testing (0 = all)

    Returns:
        Tuple of (multiturn_list, singleturn_list, feedback_list, stats_dict)
    """
    logger.info("Loading LVLM-NLF...")

    # First pass: categorize
    yesno_samples: list[dict] = []
    short_samples: list[dict] = []
    long_samples: list[dict] = []

    with open(lvlm_path) as f:
        for i, line in enumerate(tqdm(f, desc="LVLM-NLF load")):
            if max_samples > 0 and i >= max_samples:
                break
            try:
                sample = json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LVLM-NLF line {i}: {e}")
                continue

            cat = _classify_lvlm_answer(sample)

            if cat == "yes_no":
                yesno_samples.append(sample)
            elif cat == "short_phrase":
                short_samples.append(sample)
            else:
                long_samples.append(sample)

    logger.info(
        f"LVLM-NLF loaded: {len(yesno_samples)} yes/no, "
        f"{len(short_samples)} short, {len(long_samples)} long"
    )

    rng = random.Random(seed)

    # Shuffle long samples then split into three non-overlapping pools
    rng.shuffle(long_samples)
    total_long_needed = long_multiturn + long_singleturn + long_feedback
    if total_long_needed > len(long_samples):
        logger.warning(
            f"Requested {total_long_needed} long samples but only {len(long_samples)} "
            f"available; reducing proportionally"
        )
        scale = len(long_samples) / total_long_needed
        long_multiturn = int(long_multiturn * scale)
        long_singleturn = int(long_singleturn * scale)
        long_feedback = len(long_samples) - long_multiturn - long_singleturn

    pool_multiturn = long_samples[:long_multiturn]
    pool_singleturn = long_samples[long_multiturn : long_multiturn + long_singleturn]
    pool_feedback = long_samples[
        long_multiturn + long_singleturn : long_multiturn + long_singleturn + long_feedback
    ]

    # Multi-turn output: yes/no + short + long_multiturn pool
    multiturn_out = yesno_samples + short_samples + pool_multiturn

    # Single-turn output: convert long_singleturn pool
    singleturn_out: list[dict] = []
    singleturn_failed = 0
    for s in pool_singleturn:
        converted = _convert_to_single_turn(s)
        if converted is not None:
            singleturn_out.append(converted)
        else:
            singleturn_failed += 1

    # Feedback output: reverse roles for long_feedback pool
    feedback_out: list[dict] = []
    feedback_failed = 0
    for s in pool_feedback:
        converted = _convert_to_feedback_turn(s)
        if converted is not None:
            feedback_out.append(converted)
        else:
            feedback_failed += 1

    stats: dict[str, Any] = {
        "total_loaded": len(yesno_samples) + len(short_samples) + len(long_samples),
        "yesno": len(yesno_samples),
        "short_phrase": len(short_samples),
        "long_available": len(long_samples),
        "long_multiturn": len(pool_multiturn),
        "long_singleturn": len(singleturn_out),
        "long_singleturn_failed": singleturn_failed,
        "long_feedback": len(feedback_out),
        "long_feedback_failed": feedback_failed,
        "multiturn_total": len(multiturn_out),
        "singleturn_total": len(singleturn_out),
        "feedback_total": len(feedback_out),
    }

    logger.info(
        f"LVLM-NLF output: {len(multiturn_out)} multi-turn "
        f"({len(yesno_samples)} yes/no + {len(short_samples)} short + {len(pool_multiturn)} long), "
        f"{len(singleturn_out)} single-turn, {len(feedback_out)} feedback"
    )

    return multiturn_out, singleturn_out, feedback_out, stats


# =============================================================================
# Step 3: Load FIRE feedback (pass-through)
# =============================================================================


def load_fire_feedback(
    feedback_path: str,
    max_samples: int,
) -> list[dict]:
    """Load FIRE feedback dataset as-is.

    Args:
        feedback_path: Path to fire_feedback JSONL
        max_samples: Max samples for testing (0 = all)

    Returns:
        List of feedback samples
    """
    logger.info("Loading FIRE feedback...")
    samples: list[dict] = []

    with open(feedback_path) as f:
        for i, line in enumerate(tqdm(f, desc="FIRE feedback")):
            if max_samples > 0 and i >= max_samples:
                break
            try:
                samples.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse feedback line {i}: {e}")

    logger.info(f"Loaded {len(samples)} FIRE feedback samples")
    return samples


# =============================================================================
# Step 4: A-OKVQA from LLaVA-OneVision-Data (Cauldron format)
# =============================================================================


def _convert_cauldron_to_msswift(
    conversations: list[dict],
    image_path: str,
) -> dict | None:
    """Convert LLaVA-OneVision / Cauldron conversations to ms-swift messages.

    Input format (Cauldron texts):
        [{"user": "human", "text": "Question\\n<image>"},
         {"user": "gpt", "text": "Answer"}]

    Output format (ms-swift messages):
        {"messages": [{"role": "user", "content": "Question\\n<image>"},
                      {"role": "assistant", "content": "Answer", "loss": true}],
         "images": ["/path/to/image.jpg"]}

    Args:
        conversations: List of conversation turns from Cauldron
        image_path: Absolute path to saved image

    Returns:
        ms-swift format sample or None if conversion fails
    """
    if not conversations:
        return None

    messages = []
    for turn in conversations:
        role_from = turn.get("from", "")
        value = turn.get("value", "").strip()

        if not role_from or not value:
            continue

        if role_from == "human":
            messages.append({"role": "user", "content": value})
        elif role_from == "gpt":
            messages.append({"role": "assistant", "content": value, "loss": True})

    # Must have at least one user + one assistant message
    has_user = any(m["role"] == "user" for m in messages)
    has_assistant = any(m["role"] == "assistant" for m in messages)
    if not has_user or not has_assistant:
        return None

    return {"messages": messages, "images": [image_path]}


def _parse_cauldron_texts(texts: list[dict]) -> list[dict]:
    """Parse Cauldron texts field into conversation pairs.

    Args:
        texts: Raw texts list from Cauldron dataset row

    Returns:
        List of {"from": "human"|"gpt", "value": "..."} dicts
    """
    conv_pairs = []
    for t in texts:
        role = t.get("user", "")
        text = t.get("text", "").strip()
        if not role or not text:
            continue
        conv_pairs.append({"from": role, "value": text})
    return conv_pairs


def load_aokvqa(
    image_dir: Path,
    max_samples: int,
) -> tuple[list[dict], dict[str, int]]:
    """Load A-OKVQA from the Cauldron (train split, MCQ with rationales).

    Args:
        image_dir: Base directory for image storage
        max_samples: Maximum samples (0 = all)

    Returns:
        Tuple of (samples_list, stats_dict)
    """
    logger.info("Loading A-OKVQA from HuggingFaceM4/the_cauldron ...")
    dataset = load_dataset(
        "HuggingFaceM4/the_cauldron",
        "aokvqa",
        split="train",
    )

    total = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)
    logger.info(f"Processing {total} / {len(dataset)} A-OKVQA samples ...")

    samples: list[dict] = []
    stats = {"processed": 0, "skipped_no_image": 0, "skipped_conversion": 0}

    for idx in tqdm(range(total), desc="A-OKVQA"):
        row = dataset[idx]
        images_list = row.get("images", [])

        if not images_list:
            stats["skipped_no_image"] += 1
            continue

        image = _open_image(images_list[0])
        if image is None:
            stats["skipped_no_image"] += 1
            continue

        image_path = _save_image(image, image_dir, "aokvqa", idx)

        texts = row.get("texts", [])
        conv_pairs = _parse_cauldron_texts(texts)

        converted = _convert_cauldron_to_msswift(conv_pairs, image_path)
        if converted is None:
            stats["skipped_conversion"] += 1
            continue

        samples.append(converted)
        stats["processed"] += 1

    logger.info(
        f"A-OKVQA: {stats['processed']} ok, "
        f"{stats['skipped_no_image']} no image, "
        f"{stats['skipped_conversion']} conversion failed"
    )
    return samples, stats


# =============================================================================
# Step 5: TallyQA from LLaVA-OneVision-Data (Cauldron format)
# =============================================================================


def load_tallyqa(
    image_dir: Path,
    max_samples: int,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    """Load TallyQA from the Cauldron (train split, counting questions).

    Args:
        image_dir: Base directory for image storage
        max_samples: Maximum samples (0 = all, subsampled if needed)
        seed: Random seed for subsampling

    Returns:
        Tuple of (samples_list, stats_dict)
    """
    logger.info("Loading TallyQA from HuggingFaceM4/the_cauldron ...")
    dataset = load_dataset(
        "HuggingFaceM4/the_cauldron",
        "tallyqa",
        split="train",
    )

    # Subsample if dataset is larger than target
    indices = list(range(len(dataset)))
    if max_samples > 0 and len(dataset) > max_samples:
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, max_samples))
        logger.info(f"Subsampled {max_samples} / {len(dataset)} TallyQA indices")

    total = len(indices)
    logger.info(f"Processing {total} TallyQA samples ...")

    samples: list[dict] = []
    stats = {"processed": 0, "skipped_no_image": 0, "skipped_conversion": 0}

    for _out_idx, dataset_idx in enumerate(tqdm(indices, desc="TallyQA")):
        row = dataset[dataset_idx]
        images_list = row.get("images", [])

        if not images_list:
            stats["skipped_no_image"] += 1
            continue

        image = _open_image(images_list[0])
        if image is None:
            stats["skipped_no_image"] += 1
            continue

        image_path = _save_image(image, image_dir, "tallyqa", dataset_idx)

        texts = row.get("texts", [])
        conv_pairs = _parse_cauldron_texts(texts)

        converted = _convert_cauldron_to_msswift(conv_pairs, image_path)
        if converted is None:
            stats["skipped_conversion"] += 1
            continue

        samples.append(converted)
        stats["processed"] += 1

    logger.info(
        f"TallyQA: {stats['processed']} ok, "
        f"{stats['skipped_no_image']} no image, "
        f"{stats['skipped_conversion']} conversion failed"
    )
    return samples, stats


# =============================================================================
# Step 6: Combine and shuffle
# =============================================================================


def save_jsonl(samples: list[dict], output_path: Path, desc: str) -> None:
    """Save samples to JSONL file.

    Args:
        samples: List of sample dicts
        output_path: Path to output file
        desc: Description for logging
    """
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    logger.info(f"Saved {len(samples)} {desc} to {output_path}")


def verify_samples(samples: list[dict], name: str, n: int = 5) -> None:
    """Verify format of random samples from a dataset.

    Args:
        samples: List of samples to verify
        name: Dataset name for logging
        n: Number of samples to check
    """
    if not samples:
        logger.warning(f"[{name}] No samples to verify!")
        return

    rng = random.Random(0)
    check = rng.sample(samples, min(n, len(samples)))

    for i, s in enumerate(check):
        messages = s.get("messages", [])
        images = s.get("images", [])

        has_user = any(m["role"] == "user" for m in messages)
        has_assistant = any(m["role"] == "assistant" for m in messages)
        has_loss = any(m.get("loss") for m in messages if m["role"] == "assistant")

        if not has_user or not has_assistant or not has_loss:
            logger.warning(
                f"[{name}] Sample {i}: missing user={not has_user}, "
                f"assistant={not has_assistant}, loss={not has_loss}"
            )
        if not images:
            logger.warning(f"[{name}] Sample {i}: no images")

    logger.info(f"[{name}] Verified {len(check)} samples — format OK")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main function."""
    args = parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    all_stats: dict[str, Any] = {}

    # --- Step 1: Filter FIRE messages + single-turn conversion ---
    fire_multi, fire_single, fire_stats = filter_fire_messages(
        fire_messages_path=args.fire_messages,
        fire_categories_path=args.fire_categories,
        single_turn_fraction=args.single_turn_fraction,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    all_stats["fire_messages"] = fire_stats

    save_jsonl(
        fire_multi,
        output_dir / "fire_messages_filtered_multi_turn.jsonl",
        "FIRE multi-turn",
    )
    save_jsonl(
        fire_single,
        output_dir / "fire_messages_filtered_single_turn.jsonl",
        "FIRE single-turn",
    )

    # --- Step 2: Load FIRE feedback (pass-through) ---
    fire_feedback = load_fire_feedback(args.fire_feedback, args.max_samples)
    all_stats["fire_feedback"] = {"count": len(fire_feedback)}

    # --- Step 3: Process LVLM-NLF ---
    lvlm_multi, lvlm_single, lvlm_feedback, lvlm_stats = process_lvlm_nlf(
        lvlm_path=args.lvlm_nlf,
        long_multiturn=args.lvlm_long_multiturn,
        long_singleturn=args.lvlm_long_singleturn,
        long_feedback=args.lvlm_long_feedback,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    all_stats["lvlm_nlf"] = lvlm_stats

    save_jsonl(lvlm_multi, output_dir / "lvlm_nlf_multiturn.jsonl", "LVLM multi-turn")
    save_jsonl(lvlm_single, output_dir / "lvlm_nlf_singleturn.jsonl", "LVLM single-turn")
    save_jsonl(lvlm_feedback, output_dir / "lvlm_nlf_feedback.jsonl", "LVLM feedback")

    # --- Step 4 & 5: New VQA datasets ---
    aokvqa_samples: list[dict] = []
    tallyqa_samples: list[dict] = []

    if not args.skip_new_datasets:
        aokvqa_max = args.aokvqa_max
        tallyqa_max = args.tallyqa_max
        if args.max_samples > 0:
            aokvqa_max = min(aokvqa_max, args.max_samples)
            tallyqa_max = min(tallyqa_max, args.max_samples)

        aokvqa_samples, aokvqa_stats = load_aokvqa(
            image_dir=image_dir,
            max_samples=aokvqa_max,
        )
        all_stats["aokvqa"] = aokvqa_stats

        save_jsonl(aokvqa_samples, output_dir / "vqa_mix_aokvqa.jsonl", "A-OKVQA")

        tallyqa_samples, tallyqa_stats = load_tallyqa(
            image_dir=image_dir,
            max_samples=tallyqa_max,
            seed=args.seed,
        )
        all_stats["tallyqa"] = tallyqa_stats

        save_jsonl(tallyqa_samples, output_dir / "vqa_mix_tallyqa.jsonl", "TallyQA")
    else:
        logger.info("Skipping new dataset downloads (--skip_new_datasets)")

    # --- Step 6: Combine and shuffle ---
    logger.info("Combining all datasets...")
    combined = (
        fire_multi
        + fire_single
        + fire_feedback
        + lvlm_multi
        + lvlm_single
        + lvlm_feedback
        + aokvqa_samples
        + tallyqa_samples
    )
    rng.shuffle(combined)

    combined_path = output_dir / "mixed_training_v1.jsonl"
    save_jsonl(combined, combined_path, "combined mixed training")

    # --- Verification ---
    logger.info("Running format verification...")
    verify_samples(fire_multi, "FIRE multi-turn")
    verify_samples(fire_single, "FIRE single-turn")
    verify_samples(fire_feedback, "FIRE feedback")
    verify_samples(lvlm_multi, "LVLM multi-turn")
    verify_samples(lvlm_single, "LVLM single-turn")
    verify_samples(lvlm_feedback, "LVLM feedback")
    if aokvqa_samples:
        verify_samples(aokvqa_samples, "A-OKVQA")
    if tallyqa_samples:
        verify_samples(tallyqa_samples, "TallyQA")

    # --- Summary ---
    summary = {
        "fire_multi_turn": len(fire_multi),
        "fire_single_turn": len(fire_single),
        "fire_feedback": len(fire_feedback),
        "lvlm_multi_turn": len(lvlm_multi),
        "lvlm_single_turn": len(lvlm_single),
        "lvlm_feedback": len(lvlm_feedback),
        "aokvqa": len(aokvqa_samples),
        "tallyqa": len(tallyqa_samples),
        "total_combined": len(combined),
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET MIX SUMMARY")
    logger.info("=" * 60)
    for name, count in summary.items():
        pct = 100 * count / len(combined) if combined else 0
        logger.info(f"  {name:30s}: {count:>7d} ({pct:5.1f}%)")
    logger.info(f"  {'TOTAL':30s}: {len(combined):>7d}")
    logger.info("=" * 60)

    # Save stats
    stats_path = output_dir / "mix_stats.json"
    full_stats = {"summary": summary, "detailed": all_stats}
    with open(stats_path, "w") as f:
        json.dump(full_stats, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")

    logger.info(f"\nFinal combined dataset: {combined_path}")
    logger.info(f"Total samples: {len(combined)}")


if __name__ == "__main__":
    main()
