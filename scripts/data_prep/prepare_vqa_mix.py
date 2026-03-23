#!/usr/bin/env python3
"""
Prepare balanced VQA training mix with format-specific system prompts.

Produces 8 separate JSONL files (+ existing FIRE feedback = 9 total):
  1. fire_messages_filtered.jsonl       — FIRE multi-turn (Thought/Answer prompt)
  2. fire_messages_single_turn.jsonl    — FIRE single-turn (Thought/Answer prompt)
  3. lvlm_nlf_multiturn.jsonl           — LVLM multi-turn (existing prompt)
  4. lvlm_nlf_single_turn.jsonl         — LVLM single-turn (generic VQA prompt)
  5. lvlm_nlf_feedback.jsonl            — LVLM feedback (critic prompt)
  6. vqa_aokvqa.jsonl                   — A-OKVQA MCQ (generic VQA prompt)
  7. vqa_scienceqa.jsonl                — ScienceQA MCQ (generic VQA prompt)
  8. vqa_tallyqa.jsonl                  — TallyQA counting (generic VQA prompt)

Usage:
    # Full run on pod
    python scripts/data_prep/prepare_vqa_mix.py \\
        --output_dir /outputs/mixed_training_v1 \\
        --image_dir /outputs/image_base \\
        --fire_messages /outputs/fire_preprocessed_v3/fire_messages_train_last_loss_only_with_thought.jsonl \\
        --fire_categories /outputs/fire_preprocessed_v3/fire_sample_categories.jsonl \\
        --lvlm_nlf /outputs/lvlm_nlf_preprocessed/lvlm_nlf_messages_train_last_loss_only_balanced.jsonl

    # Quick test (10 samples per dataset)
    python scripts/data_prep/prepare_vqa_mix.py \\
        --output_dir /outputs/mixed_training_v1_test \\
        --image_dir /outputs/image_base \\
        --fire_messages /outputs/fire_preprocessed_v3/fire_messages_train_last_loss_only_with_thought.jsonl \\
        --fire_categories /outputs/fire_preprocessed_v3/fire_sample_categories.jsonl \\
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

# =============================================================================
# System prompts (format-specific)
# =============================================================================

FIRE_MULTITURN_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate, "
    "detailed, and grounded answers based on the image and the user's instructions. "
    "First think about your reasoning process, then provide the answer. "
    "Your response should follow this format:\n"
    "Thought: [your reasoning about the image and question]\n"
    "Answer: [your final answer].\n"
    "When given feedback, critique, or scores, revise your response to improve "
    "correctness, specificity, and completeness."
)

LVLM_MULTITURN_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate, "
    "detailed, and grounded answers based on the image and the user's instructions. "
    "When given feedback, critique, or scores, revise your response to improve "
    "correctness, specificity, and completeness."
)

FIRE_SINGLETURN_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate, "
    "detailed, and grounded answers based on the image and the user's instructions. "
    "First think about your reasoning process, then provide the answer. "
    "Your response should follow this format:\n"
    "Thought: [your reasoning about the image and question]\n"
    "Answer: [your final answer]"
)

GENERIC_VQA_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate "
    "and concise answers based on the image and the user's instructions."
)

FEEDBACK_CRITIC_SYSTEM_PROMPT = (
    "You are a visual question answering critic. Given an image, a question, "
    "and the conversation history of the user's answers and prior feedback, "
    "provide constructive feedback on the user's latest answer identifying "
    "what is correct, what is incorrect, and how to improve. Ground your "
    "feedback in what is visible in the image."
)

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
        description="Prepare balanced VQA training mix with format-specific prompts",
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
        "--scienceqa_max",
        type=int,
        default=0,
        help="Maximum ScienceQA samples (0 = all)",
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
        help="Skip downloading A-OKVQA, ScienceQA, TallyQA",
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
# Utilities
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
    issues = 0

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
            issues += 1
        if not images:
            logger.warning(f"[{name}] Sample {i}: no images")
            issues += 1

    if issues == 0:
        logger.info(f"[{name}] Verified {len(check)} samples — format OK")


# =============================================================================
# Step 1 & 2: Filter FIRE messages + single-turn conversion
# =============================================================================


def process_fire_messages(
    fire_messages_path: str,
    fire_categories_path: str,
    single_turn_fraction: float,
    seed: int,
    max_samples: int,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Filter FIRE messages and split into multi-turn and single-turn.

    Removes bounding_box and region_description. Converts a fraction of
    open_ended and short_answer to single-turn. Replaces system prompts
    with format-specific versions.

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

    # Pre-select indices for single-turn conversion (only within max_samples range)
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
                converted = _fire_to_single_turn(sample)
                if converted is not None:
                    single_turn.append(converted)
                    stats["converted_single_turn"] += 1
                else:
                    # Fallback: keep as multi-turn
                    _replace_system_prompt(sample, FIRE_MULTITURN_SYSTEM_PROMPT)
                    multi_turn.append(sample)
                    stats["kept_multi_turn"] += 1
            else:
                _replace_system_prompt(sample, FIRE_MULTITURN_SYSTEM_PROMPT)
                multi_turn.append(sample)
                stats["kept_multi_turn"] += 1

            stats["categories_kept"][cat] = stats["categories_kept"].get(cat, 0) + 1

    logger.info(
        f"FIRE: {stats['kept_multi_turn']} multi-turn, "
        f"{stats['converted_single_turn']} single-turn, "
        f"{stats['removed_bounding_box']} bbox removed, "
        f"{stats['removed_region_description']} region removed"
    )

    return multi_turn, single_turn, stats


def _replace_system_prompt(sample: dict, new_prompt: str) -> dict:
    """Replace the system prompt in a sample's messages.

    Args:
        sample: Sample dict with messages
        new_prompt: New system prompt text

    Returns:
        Modified sample (mutated in place)
    """
    messages = sample.get("messages", [])
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = new_prompt
    else:
        messages.insert(0, {"role": "system", "content": new_prompt})
    return sample


def _fire_to_single_turn(sample: dict) -> dict | None:
    """Convert a multi-turn FIRE sample to single-turn.

    Keeps first user question + last assistant response. Uses FIRE single-turn
    system prompt (Thought/Answer format, no feedback instruction).

    Args:
        sample: FIRE sample with messages and images

    Returns:
        Single-turn sample or None if conversion fails
    """
    messages = sample.get("messages", [])
    images = sample.get("images", [])

    first_question = None
    for msg in messages:
        if msg["role"] == "user":
            first_question = msg["content"]
            break

    last_answer = None
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            last_answer = msg["content"]
            break

    if first_question is None or last_answer is None:
        return None

    return {
        "messages": [
            {"role": "system", "content": FIRE_SINGLETURN_SYSTEM_PROMPT},
            {"role": "user", "content": first_question},
            {"role": "assistant", "content": last_answer, "loss": True},
        ],
        "images": images[:1] if images else [],
    }


# =============================================================================
# Step 3: Process LVLM-NLF → 3 files
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


def _lvlm_to_single_turn(sample: dict) -> dict | None:
    """Convert LVLM-NLF multi-turn to single-turn with generic VQA prompt.

    Args:
        sample: LVLM-NLF sample

    Returns:
        Single-turn sample or None if conversion fails
    """
    messages = sample.get("messages", [])
    images = sample.get("images", [])

    first_question = None
    for msg in messages:
        if msg["role"] == "user":
            first_question = msg["content"]
            break

    last_answer = None
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            last_answer = msg["content"]
            break

    if first_question is None or last_answer is None:
        return None

    return {
        "messages": [
            {"role": "system", "content": GENERIC_VQA_SYSTEM_PROMPT},
            {"role": "user", "content": first_question},
            {"role": "assistant", "content": last_answer, "loss": True},
        ],
        "images": images[:1] if images else [],
    }


def _lvlm_to_feedback(sample: dict) -> dict | None:
    """Convert LVLM-NLF multi-turn to feedback format (reversed roles).

    Input: system → user(Q) → assistant(A1,loss=false) → user(feedback) → ...
    Output: critic_system → assistant(Q,loss=false) → user(A1) →
            assistant(feedback,loss=true)

    Args:
        sample: LVLM-NLF sample

    Returns:
        Feedback-format sample or None if conversion fails
    """
    messages = sample.get("messages", [])
    images = sample.get("images", [])

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

    if len(first_feedback.split()) < 5:
        return None

    return {
        "messages": [
            {"role": "system", "content": FEEDBACK_CRITIC_SYSTEM_PROMPT},
            {"role": "assistant", "content": first_question, "loss": False},
            {"role": "user", "content": first_answer},
            {"role": "assistant", "content": first_feedback, "loss": True},
        ],
        "images": images,
    }


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
      - All yes/no + all short phrase: kept as multi-turn (existing prompt)
      - Long descriptive split into 3 non-overlapping pools

    Args:
        lvlm_path: Path to LVLM-NLF JSONL
        long_multiturn: Long samples to keep as multi-turn
        long_singleturn: Long samples to convert to single-turn
        long_feedback: Long samples to reverse as feedback
        seed: Random seed
        max_samples: Max samples for testing (0 = all)

    Returns:
        Tuple of (multiturn_list, singleturn_list, feedback_list, stats_dict)
    """
    logger.info("Loading LVLM-NLF...")

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
    rng.shuffle(long_samples)

    # Scale down if not enough long samples
    total_long_needed = long_multiturn + long_singleturn + long_feedback
    if total_long_needed > len(long_samples):
        logger.warning(
            f"Requested {total_long_needed} long samples but only "
            f"{len(long_samples)} available; reducing proportionally"
        )
        scale = len(long_samples) / total_long_needed
        long_multiturn = int(long_multiturn * scale)
        long_singleturn = int(long_singleturn * scale)
        long_feedback = len(long_samples) - long_multiturn - long_singleturn

    pool_mt = long_samples[:long_multiturn]
    pool_st = long_samples[long_multiturn : long_multiturn + long_singleturn]
    pool_fb = long_samples[
        long_multiturn + long_singleturn : long_multiturn + long_singleturn + long_feedback
    ]

    # Multi-turn: yes/no + short + long pool (replace system prompt)
    multiturn_out = yesno_samples + short_samples + pool_mt
    for s in multiturn_out:
        _replace_system_prompt(s, LVLM_MULTITURN_SYSTEM_PROMPT)

    # Single-turn: convert with generic VQA prompt
    singleturn_out: list[dict] = []
    st_failed = 0
    for s in pool_st:
        converted = _lvlm_to_single_turn(s)
        if converted is not None:
            singleturn_out.append(converted)
        else:
            st_failed += 1

    # Feedback: reverse roles with critic prompt
    feedback_out: list[dict] = []
    fb_failed = 0
    for s in pool_fb:
        converted = _lvlm_to_feedback(s)
        if converted is not None:
            feedback_out.append(converted)
        else:
            fb_failed += 1

    stats: dict[str, Any] = {
        "yesno": len(yesno_samples),
        "short_phrase": len(short_samples),
        "long_available": len(long_samples),
        "long_multiturn": len(pool_mt),
        "long_singleturn": len(singleturn_out),
        "long_singleturn_failed": st_failed,
        "long_feedback": len(feedback_out),
        "long_feedback_failed": fb_failed,
        "multiturn_total": len(multiturn_out),
    }

    logger.info(
        f"LVLM-NLF output: {len(multiturn_out)} multi-turn, "
        f"{len(singleturn_out)} single-turn, {len(feedback_out)} feedback"
    )

    return multiturn_out, singleturn_out, feedback_out, stats


# =============================================================================
# Steps 4-6: Cauldron datasets (A-OKVQA, ScienceQA, TallyQA)
# =============================================================================


def _cauldron_to_msswift(
    entry: dict,
    image_path: str,
) -> dict | None:
    """Convert a Cauldron texts entry to ms-swift single-turn format.

    Cauldron texts format: {"user": "Question", "assistant": "Answer", "source": "..."}

    Args:
        entry: Single texts dict with "user" and "assistant" keys
        image_path: Absolute path to saved image

    Returns:
        ms-swift format sample or None if conversion fails
    """
    question = entry.get("user", "").strip()
    answer = entry.get("assistant", "").strip()

    if not question or not answer:
        return None

    # Ensure <image> token is present
    if "<image>" not in question:
        question = question + "\n<image>"

    return {
        "messages": [
            {"role": "system", "content": GENERIC_VQA_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "loss": True},
        ],
        "images": [image_path],
    }


def load_cauldron_dataset(
    config: str,
    image_dir: Path,
    image_subdir: str,
    max_samples: int,
    seed: int,
    expand_multi_qa: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    """Load a dataset from HuggingFaceM4/the_cauldron and convert to ms-swift.

    Args:
        config: Cauldron config name (aokvqa, scienceqa, tallyqa)
        image_dir: Base directory for image storage
        image_subdir: Subdirectory name for this dataset's images
        max_samples: Maximum samples (0 = all)
        seed: Random seed for subsampling
        expand_multi_qa: If True, expand rows with multiple Q&A pairs

    Returns:
        Tuple of (samples_list, stats_dict)
    """
    logger.info(f"Loading {config} from HuggingFaceM4/the_cauldron ...")
    dataset = load_dataset(
        "HuggingFaceM4/the_cauldron",
        config,
        split="train",
    )
    logger.info(f"  {config}: {len(dataset)} rows loaded")

    stats = {"processed": 0, "skipped_no_image": 0, "skipped_conversion": 0}
    samples: list[dict] = []

    if expand_multi_qa:
        # Each row has multiple texts entries, each is a separate Q&A
        # Format: {"user": "Q", "assistant": "A", "source": "..."}
        all_entries: list[tuple[int, dict]] = []
        for row_idx in range(len(dataset)):
            texts = dataset[row_idx].get("texts", [])
            for entry in texts:
                if entry.get("user") and entry.get("assistant"):
                    all_entries.append((row_idx, entry))

        logger.info(f"  {config}: expanded to {len(all_entries)} Q&A entries")

        if max_samples > 0 and len(all_entries) > max_samples:
            rng = random.Random(seed)
            all_entries = rng.sample(all_entries, max_samples)
            logger.info(f"  {config}: subsampled to {max_samples}")

        # Track saved images to avoid re-saving for same-image entries
        saved_images: dict[int, str] = {}

        for _out_idx, (row_idx, entry) in enumerate(tqdm(all_entries, desc=config)):
            if row_idx not in saved_images:
                images_list = dataset[row_idx].get("images", [])
                if not images_list:
                    stats["skipped_no_image"] += 1
                    continue
                image = _open_image(images_list[0])
                if image is None:
                    stats["skipped_no_image"] += 1
                    continue
                saved_images[row_idx] = _save_image(image, image_dir, image_subdir, row_idx)

            image_path = saved_images.get(row_idx)
            if image_path is None:
                stats["skipped_no_image"] += 1
                continue

            converted = _cauldron_to_msswift(entry, image_path)
            if converted is None:
                stats["skipped_conversion"] += 1
                continue

            samples.append(converted)
            stats["processed"] += 1
    else:
        # Standard: one or few Q&A entries per row, process first entry
        indices = list(range(len(dataset)))
        if max_samples > 0 and len(indices) > max_samples:
            rng = random.Random(seed)
            indices = sorted(rng.sample(indices, max_samples))
            logger.info(f"  {config}: subsampled to {max_samples}")

        for dataset_idx in tqdm(indices, desc=config):
            row = dataset[dataset_idx]
            images_list = row.get("images", [])

            if not images_list:
                stats["skipped_no_image"] += 1
                continue

            image = _open_image(images_list[0])
            if image is None:
                stats["skipped_no_image"] += 1
                continue

            image_path = _save_image(image, image_dir, image_subdir, dataset_idx)

            texts = row.get("texts", [])
            if not texts:
                stats["skipped_conversion"] += 1
                continue

            converted = _cauldron_to_msswift(texts[0], image_path)
            if converted is None:
                stats["skipped_conversion"] += 1
                continue

            samples.append(converted)
            stats["processed"] += 1

    logger.info(
        f"  {config}: {stats['processed']} ok, "
        f"{stats['skipped_no_image']} no image, "
        f"{stats['skipped_conversion']} conversion failed"
    )
    return samples, stats


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main function."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    all_stats: dict[str, Any] = {}

    # --- Step 1 & 2: FIRE messages ---
    fire_multi, fire_single, fire_stats = process_fire_messages(
        fire_messages_path=args.fire_messages,
        fire_categories_path=args.fire_categories,
        single_turn_fraction=args.single_turn_fraction,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    all_stats["fire_messages"] = fire_stats

    save_jsonl(fire_multi, output_dir / "fire_messages_filtered.jsonl", "FIRE multi-turn")
    save_jsonl(
        fire_single,
        output_dir / "fire_messages_single_turn.jsonl",
        "FIRE single-turn",
    )

    # --- Step 3: LVLM-NLF ---
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
    save_jsonl(lvlm_single, output_dir / "lvlm_nlf_single_turn.jsonl", "LVLM single-turn")
    save_jsonl(lvlm_feedback, output_dir / "lvlm_nlf_feedback.jsonl", "LVLM feedback")

    # --- Steps 4-6: New VQA datasets ---
    aokvqa_samples: list[dict] = []
    scienceqa_samples: list[dict] = []
    tallyqa_samples: list[dict] = []

    if not args.skip_new_datasets:
        aokvqa_max = args.aokvqa_max
        scienceqa_max = args.scienceqa_max
        tallyqa_max = args.tallyqa_max
        if args.max_samples > 0:
            aokvqa_max = min(aokvqa_max, args.max_samples) if aokvqa_max > 0 else args.max_samples
            scienceqa_max = (
                min(scienceqa_max, args.max_samples) if scienceqa_max > 0 else args.max_samples
            )
            tallyqa_max = (
                min(tallyqa_max, args.max_samples) if tallyqa_max > 0 else args.max_samples
            )

        aokvqa_samples, aokvqa_stats = load_cauldron_dataset(
            config="aokvqa",
            image_dir=image_dir,
            image_subdir="aokvqa",
            max_samples=aokvqa_max,
            seed=args.seed,
        )
        all_stats["aokvqa"] = aokvqa_stats
        save_jsonl(aokvqa_samples, output_dir / "vqa_aokvqa.jsonl", "A-OKVQA")

        scienceqa_samples, scienceqa_stats = load_cauldron_dataset(
            config="scienceqa",
            image_dir=image_dir,
            image_subdir="scienceqa_train",
            max_samples=scienceqa_max,
            seed=args.seed,
        )
        all_stats["scienceqa"] = scienceqa_stats
        save_jsonl(scienceqa_samples, output_dir / "vqa_scienceqa.jsonl", "ScienceQA")

        tallyqa_samples, tallyqa_stats = load_cauldron_dataset(
            config="tallyqa",
            image_dir=image_dir,
            image_subdir="tallyqa",
            max_samples=tallyqa_max,
            seed=args.seed,
            expand_multi_qa=True,
        )
        all_stats["tallyqa"] = tallyqa_stats
        save_jsonl(tallyqa_samples, output_dir / "vqa_tallyqa.jsonl", "TallyQA")
    else:
        logger.info("Skipping new dataset downloads (--skip_new_datasets)")

    # --- Verification ---
    logger.info("Running format verification...")
    verify_samples(fire_multi, "FIRE multi-turn")
    verify_samples(fire_single, "FIRE single-turn")
    verify_samples(lvlm_multi, "LVLM multi-turn")
    verify_samples(lvlm_single, "LVLM single-turn")
    verify_samples(lvlm_feedback, "LVLM feedback")
    if aokvqa_samples:
        verify_samples(aokvqa_samples, "A-OKVQA")
    if scienceqa_samples:
        verify_samples(scienceqa_samples, "ScienceQA")
    if tallyqa_samples:
        verify_samples(tallyqa_samples, "TallyQA")

    # --- Summary ---
    summary = {
        "fire_multi_turn": len(fire_multi),
        "fire_single_turn": len(fire_single),
        "lvlm_multi_turn": len(lvlm_multi),
        "lvlm_single_turn": len(lvlm_single),
        "lvlm_feedback": len(lvlm_feedback),
        "aokvqa": len(aokvqa_samples),
        "scienceqa": len(scienceqa_samples),
        "tallyqa": len(tallyqa_samples),
    }
    total = sum(summary.values())

    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET MIX SUMMARY")
    logger.info("=" * 60)
    for name, count in summary.items():
        pct = 100 * count / total if total else 0
        logger.info(f"  {name:30s}: {count:>7d} ({pct:5.1f}%)")
    logger.info(f"  {'TOTAL':30s}: {total:>7d}")
    logger.info("=" * 60)
    logger.info(
        "\nNote: FIRE feedback (~100K) already exists at "
        "fire_feedback_train_v2_with_thought.jsonl — pass as separate DATASET_PATH"
    )

    # Save stats
    stats_path = output_dir / "mix_stats.json"
    full_stats = {"summary": summary, "total": total, "detailed": all_stats}
    with open(stats_path, "w") as f:
        json.dump(full_stats, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
