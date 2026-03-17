#!/usr/bin/env python3
"""
Prepare MCQ and Yes/No GRPO training dataset.

Downloads and preprocesses three datasets focused on question types where
Qwen2.5-VL underperforms, then runs model inference to generate answer1.

Datasets:
  - PixelReasoner-SFT-Data  (MCQ, 7 850 samples, train + val)
  - MME-CoT                 (MCQ, 1 130 samples, val only — high quality)
  - RLAIF-V                 (yes/no subset, train + val)
  - VISCO                   (MCQ, 1 645 samples, train + val)

Output JSONL format (one sample per line):
  {
    "question": "...",
    "ground_truth": "A",
    "answer_type": "mcq",
    "choices": "(A) cat (B) dog (C) bird (D) fish",
    "answer1": "B",
    "a1_is_correct": false,
    "image_path": "pixel_reasoner/000042.jpg",
    "dataset_name": "pixel_reasoner"
  }

Usage:
    # Schema mapping only, no GPU:
    python scripts/data_prep/prepare_grpo_mcq_yesno.py \\
        --output_dir /outputs/grpo_mcq_yesno \\
        --image_dir  /outputs/grpo_mcq_yesno/images \\
        --skip_inference \\
        --max_samples 20

    # Full run with inference:
    python scripts/data_prep/prepare_grpo_mcq_yesno.py \\
        --output_dir  /outputs/grpo_mcq_yesno \\
        --image_dir   /outputs/grpo_mcq_yesno/images \\
        --model_id    /outputs/<checkpoint> \\
        --batch_size  16
"""

import argparse
import base64
import io
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Answer extraction patterns
_BOXED_PATTERN = re.compile(r"\\boxed\{([A-Za-z])\}")
_MCQ_LETTER_PATTERN = re.compile(r"\b([A-F])\b")
# Matches yes/no AND true/false (all normalised to "Yes"/"No")
_YESNO_PATTERN = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)

# Maps any accepted token → canonical "Yes" or "No"
_YESNO_CANONICAL: Dict[str, str] = {
    "yes": "Yes",
    "true": "Yes",
    "no": "No",
    "false": "No",
}

# Hedging words that invalidate a yes/no answer
_HEDGING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bmaybe\b",
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\bprobably\b",
        r"\bnot sure\b",
        r"\bI think\b",
        r"\bI believe\b",
        r"\blikely\b",
    ]
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Prepare MCQ and Yes/No GRPO training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/grpo_mcq_yesno",
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/outputs/grpo_mcq_yesno/images",
        help="Directory for saved images (referenced by relative path in JSONL)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="",
        help="Local checkpoint path for answer1 inference (required unless --skip_inference)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples per dataset (0 = all)",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of samples held out for validation (0.0–1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip model inference; answer1 is set to '' placeholder",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="pixel_reasoner,mme_cot,rlaif_v,visco",
        help="Comma-separated list of datasets to process",
    )
    return parser.parse_args()


# =============================================================================
# Image utilities
# =============================================================================


def _save_image(image: Image.Image, image_dir: Path, dataset_name: str, idx: int) -> str:
    """Save a PIL image to disk and return its relative path.

    Args:
        image: PIL image to save
        image_dir: Base directory for image storage
        dataset_name: Dataset subdirectory name
        idx: Sample index used for filename

    Returns:
        Relative path from image_dir (e.g. "pixel_reasoner/000042.jpg")
    """
    out_dir = image_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if image.mode != "RGB":
        image = image.convert("RGB")

    rel_path = f"{dataset_name}/{idx:06d}.jpg"
    image.save(image_dir / rel_path, format="JPEG", quality=90)
    return rel_path


def _open_image(raw: Any) -> Optional[Image.Image]:
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


def _format_mcq_choices(letters: List[str], texts: List[str]) -> str:
    """Format option letters and texts into '(A) text (B) text ...' string.

    Args:
        letters: Option letters, e.g. ["A", "B", "C", "D"]
        texts: Option texts, e.g. ["cat", "dog", "bird", "fish"]

    Returns:
        Formatted choices string
    """
    return " ".join(f"({l}) {t}" for l, t in zip(letters, texts))


# =============================================================================
# PixelReasoner-SFT-Data
# =============================================================================


def _parse_pixel_reasoner_sample(
    sample: Dict[str, Any],
    image_dir: Path,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """Parse one PixelReasoner-SFT-Data row.

    The message_list has a user turn (image + question + options) and an
    assistant turn with the answer in \\boxed{A} format.

    Args:
        sample: Raw HF dataset row
        image_dir: Base directory for image storage
        idx: Sample index

    Returns:
        Intermediate GRPO sample dict, or None if parsing fails
    """
    message_list = sample.get("message_list", [])
    if not message_list:
        return None

    question_text = ""
    choices_text = ""
    ground_truth = ""
    image: Optional[Image.Image] = None

    for msg in message_list:
        role = msg.get("role", "")
        content = msg.get("content", [])

        if role == "user":
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    image = _open_image(item.get("image"))
                elif item.get("type") == "text":
                    text = item.get("text", "")
                    lines = text.strip().split("\n")
                    choice_lines = [l for l in lines if re.match(r"^[A-F]:", l.strip())]
                    question_lines = [l for l in lines if not re.match(r"^[A-F]:", l.strip())]
                    question_text = " ".join(question_lines).strip()

                    letters, texts = [], []
                    for cl in choice_lines:
                        parts = cl.split(":", 1)
                        if len(parts) == 2:
                            letters.append(parts[0].strip())
                            texts.append(parts[1].strip())
                    if letters:
                        choices_text = _format_mcq_choices(letters, texts)

        elif role == "assistant":
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    m = _BOXED_PATTERN.search(item.get("text", ""))
                    if m:
                        ground_truth = m.group(1).upper()

    if not question_text or not ground_truth or image is None:
        return None

    image_path = _save_image(image, image_dir, "pixel_reasoner", idx)
    return {
        "question": question_text,
        "ground_truth": ground_truth,
        "answer_type": "mcq",
        "choices": choices_text,
        "image_path": image_path,
        "dataset_name": "pixel_reasoner",
    }


def load_pixel_reasoner(image_dir: Path, max_samples: int) -> List[Dict[str, Any]]:
    """Load PixelReasoner-SFT-Data (MCQ, train split).

    Args:
        image_dir: Base directory for image storage
        max_samples: Cap per dataset (0 = all)

    Returns:
        List of intermediate GRPO sample dicts
    """
    logger.info("Loading TIGER-Lab/PixelReasoner-SFT-Data ...")
    dataset = load_dataset("TIGER-Lab/PixelReasoner-SFT-Data", split="train")
    total = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)
    logger.info(f"Processing {total} / {len(dataset)} samples ...")

    samples: List[Dict[str, Any]] = []
    stats = {"processed": 0, "skipped": 0}

    for idx in tqdm(range(total), desc="PixelReasoner"):
        parsed = _parse_pixel_reasoner_sample(dataset[idx], image_dir, idx)
        if parsed is None:
            stats["skipped"] += 1
            continue
        samples.append(parsed)
        stats["processed"] += 1

    logger.info(f"PixelReasoner: {stats['processed']} ok, {stats['skipped']} skipped")
    return samples


# =============================================================================
# MME-CoT
# =============================================================================


def _parse_mme_cot_sample(
    sample: Dict[str, Any],
    image_dir: Path,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """Parse one MME-CoT row.

    Columns A–L contain option texts; `answer` contains the ground truth letter.

    Args:
        sample: Raw HF dataset row
        image_dir: Base directory for image storage
        idx: Sample index

    Returns:
        Intermediate GRPO sample dict, or None if parsing fails
    """
    question = sample.get("question", "").strip()
    ground_truth = sample.get("answer", "").strip().upper()
    if not question or not ground_truth:
        return None

    # Collect non-null options (columns A through L)
    letters, texts = [], []
    for letter in "ABCDEFGHIJKL":
        val = sample.get(letter)
        if val and str(val).strip():
            letters.append(letter)
            texts.append(str(val).strip())

    choices_text = _format_mcq_choices(letters, texts) if letters else ""

    image = _open_image(sample.get("image"))
    if image is None:
        return None

    image_path = _save_image(image, image_dir, "mme_cot", idx)
    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer_type": "mcq",
        "choices": choices_text,
        "image_path": image_path,
        "dataset_name": "mme_cot",
    }


def load_mme_cot(image_dir: Path, max_samples: int) -> List[Dict[str, Any]]:
    """Load MME-CoT (MCQ, test split — used entirely as validation set).

    Args:
        image_dir: Base directory for image storage
        max_samples: Cap (0 = all)

    Returns:
        List of intermediate GRPO sample dicts
    """
    logger.info("Loading CaraJ/MME-CoT (test split, val-only) ...")
    dataset = load_dataset("CaraJ/MME-CoT", split="test")
    total = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)
    logger.info(f"Processing {total} / {len(dataset)} samples ...")

    samples: List[Dict[str, Any]] = []
    stats = {"processed": 0, "skipped": 0}

    for idx in tqdm(range(total), desc="MME-CoT"):
        parsed = _parse_mme_cot_sample(dataset[idx], image_dir, idx)
        if parsed is None:
            stats["skipped"] += 1
            continue
        samples.append(parsed)
        stats["processed"] += 1

    logger.info(f"MME-CoT: {stats['processed']} ok, {stats['skipped']} skipped")
    return samples


# =============================================================================
# RLAIF-V (yes/no subset)
# =============================================================================


def _extract_yesno_from_chosen(chosen: str) -> Optional[str]:
    """Extract canonical Yes/No ground truth from an RLAIF-V chosen response.

    Accepts "Yes", "No", "True", "False" (case-insensitive) as the first word
    of the response and normalises them to "Yes"/"No" so that the GRPO verifier
    (which only extracts yes|no from model output) can compare them correctly.

    Args:
        chosen: Full chosen response text

    Returns:
        "Yes" or "No", or None if the response does not start with a recognised
        yes/no/true/false token.
    """
    if not chosen:
        return None
    first_word = re.sub(r"[^a-zA-Z]", "", chosen.strip().split()[0]).lower()
    return _YESNO_CANONICAL.get(first_word)


def _parse_rlaif_v_sample(
    sample: Dict[str, Any],
    image_dir: Path,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """Parse one RLAIF-V row, keeping only yes/no questions.

    Args:
        sample: Raw HF dataset row
        image_dir: Base directory for image storage
        idx: Sample index

    Returns:
        Intermediate GRPO sample dict, or None if not yes/no or parse fails
    """
    question = sample.get("question", "").strip()
    chosen = sample.get("chosen", "")
    if not question or not chosen:
        return None

    ground_truth = _extract_yesno_from_chosen(chosen)
    if ground_truth is None:
        return None

    image = _open_image(sample.get("image"))
    if image is None:
        return None

    image_path = _save_image(image, image_dir, "rlaif_v", idx)
    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer_type": "yesno",
        "choices": "",
        "image_path": image_path,
        "dataset_name": "rlaif_v",
    }


def load_rlaif_v(image_dir: Path, max_samples: int) -> List[Dict[str, Any]]:
    """Load RLAIF-V (yes/no subset of the train split).

    Args:
        image_dir: Base directory for image storage
        max_samples: Cap on accepted yes/no samples (0 = all)

    Returns:
        List of intermediate GRPO sample dicts
    """
    logger.info("Loading openbmb/RLAIF-V-Dataset (yes/no subset) ...")
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
    logger.info(f"Total RLAIF-V rows: {len(dataset)} — scanning for yes/no ...")

    samples: List[Dict[str, Any]] = []
    stats = {"processed": 0, "not_yesno": 0, "parse_fail": 0}

    for idx in tqdm(range(len(dataset)), desc="RLAIF-V"):
        if max_samples > 0 and stats["processed"] >= max_samples:
            break

        parsed = _parse_rlaif_v_sample(dataset[idx], image_dir, idx)
        if parsed is None:
            if _extract_yesno_from_chosen(dataset[idx].get("chosen", "")) is None:
                stats["not_yesno"] += 1
            else:
                stats["parse_fail"] += 1
            continue

        samples.append(parsed)
        stats["processed"] += 1

    logger.info(
        f"RLAIF-V: {stats['processed']} yes/no samples | "
        f"{stats['not_yesno']} non-yesno skipped | "
        f"{stats['parse_fail']} parse failures"
    )
    return samples


# =============================================================================
# VISCO
# =============================================================================

# Matches the "Choices:" block at the end of a VISCO question string.
# Captures everything after the label, e.g. "(A) (0, 0)\n(B) (-1, 0)\n(C) (2, 0)"
_VISCO_CHOICES_BLOCK = re.compile(r"Choices?\s*:?\s*\n?(.*)", re.DOTALL | re.IGNORECASE)
# Splits individual options "(A) text" out of the choices block.
_VISCO_OPTION = re.compile(r"\(([A-F])\)\s*(.*?)(?=\s*\([A-F]\)|$)", re.DOTALL)


def _parse_visco_choices(question: str) -> tuple[str, List[str], List[str]]:
    """Split a VISCO question into clean question text + MCQ letters + texts.

    VISCO embeds choices at the end of the question string, e.g.:
        "What is the center of symmetry?\nChoices:\n(A) (0, 0)\n(B) (-1, 0)\n(C) (2, 0)"

    Args:
        question: Raw question string from the dataset

    Returns:
        Tuple of (clean_question, letters, texts) where letters and texts are
        parallel lists of option identifiers and their content strings.
    """
    m = _VISCO_CHOICES_BLOCK.search(question)
    if not m:
        return question.strip(), [], []

    clean_question = question[: m.start()].strip()
    choices_block = m.group(1)

    letters, texts = [], []
    for opt in _VISCO_OPTION.finditer(choices_block):
        letters.append(opt.group(1).upper())
        texts.append(opt.group(2).strip())

    return clean_question, letters, texts


def _label_to_letter(label: str, letters: List[str], texts: List[str]) -> Optional[str]:
    """Map a VISCO label value to its MCQ option letter.

    The `label` field contains the answer *text* (e.g. "(0, 0)"), not the
    letter.  This function finds which option's text matches after
    case-insensitive whitespace normalization.

    Args:
        label: Ground-truth label value from the dataset row
        letters: Option letters from _parse_visco_choices
        texts: Option texts from _parse_visco_choices

    Returns:
        The matching letter (e.g. "A"), or None if no match found
    """
    norm_label = re.sub(r"\s+", " ", label.strip().lower())
    for letter, text in zip(letters, texts):
        if re.sub(r"\s+", " ", text.strip().lower()) == norm_label:
            return letter
    return None


def _parse_visco_sample(
    sample: Dict[str, Any],
    image_dir: Path,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """Parse one VISCO row.

    VISCO images are base64-encoded strings. Ground truth is a label value
    that must be matched against the embedded MCQ options to get the letter.

    Args:
        sample: Raw HF dataset row
        image_dir: Base directory for image storage
        idx: Sample index

    Returns:
        Intermediate GRPO sample dict, or None if parsing fails
    """
    raw_question = sample.get("question", "").strip()
    label = str(sample.get("label", "")).strip()
    if not raw_question or not label:
        return None

    clean_question, letters, texts = _parse_visco_choices(raw_question)
    if not letters:
        return None

    ground_truth = _label_to_letter(label, letters, texts)
    if ground_truth is None:
        return None

    # Decode base64 image
    raw_image = sample.get("image", "")
    if not raw_image:
        return None
    try:
        if isinstance(raw_image, str):
            image_bytes = base64.b64decode(raw_image)
        else:
            image_bytes = raw_image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.warning(f"VISCO[{idx}] image decode failed: {e}")
        return None

    choices_text = _format_mcq_choices(letters, texts)
    image_path = _save_image(image, image_dir, "visco", idx)

    return {
        "question": clean_question,
        "ground_truth": ground_truth,
        "answer_type": "mcq",
        "choices": choices_text,
        "image_path": image_path,
        "dataset_name": "visco",
    }


def load_visco(image_dir: Path, max_samples: int) -> List[Dict[str, Any]]:
    """Load VISCO dataset (MCQ with step-level critiques).

    Args:
        image_dir: Base directory for image storage
        max_samples: Cap (0 = all)

    Returns:
        List of intermediate GRPO sample dicts
    """
    logger.info("Loading uclanlp/VISCO ...")
    dataset = load_dataset("uclanlp/VISCO", split="train")
    total = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)
    logger.info(f"Processing {total} / {len(dataset)} samples ...")

    samples: List[Dict[str, Any]] = []
    stats = {"processed": 0, "no_match": 0, "skipped": 0}

    for idx in tqdm(range(total), desc="VISCO"):
        parsed = _parse_visco_sample(dataset[idx], image_dir, idx)
        if parsed is None:
            raw_q = dataset[idx].get("question", "")
            stats["no_match" if "(A)" in raw_q else "skipped"] += 1
            continue
        samples.append(parsed)
        stats["processed"] += 1

    logger.info(
        f"VISCO: {stats['processed']} ok | "
        f"{stats['no_match']} label-match failures | "
        f"{stats['skipped']} skipped"
    )
    return samples


# =============================================================================
# Answer extraction helpers (for labelling answer1)
# =============================================================================


def _extract_mcq_answer(text: str) -> str:
    """Extract MCQ letter from raw model output.

    Args:
        text: Raw model output

    Returns:
        Uppercase letter A–F, or "" if not found
    """
    m = _BOXED_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    m = _MCQ_LETTER_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    return ""


def _extract_yesno_answer(text: str) -> str:
    """Extract canonical Yes/No from raw model output.

    Accepts "yes", "no", "true", "false" (case-insensitive) and normalises
    them to "Yes"/"No".  Hedged answers (maybe, probably, …) return "".

    Args:
        text: Raw model output

    Returns:
        "Yes", "No", or "" if hedged or no recognised token found
    """
    if any(p.search(text) for p in _HEDGING_PATTERNS):
        return ""
    m = _YESNO_PATTERN.search(text)
    if m:
        return _YESNO_CANONICAL.get(m.group(1).lower(), "")
    return ""


def _is_correct(answer1: str, ground_truth: str, answer_type: str) -> bool:
    """Check if answer1 matches ground_truth.

    For yesno, both sides are normalised through _YESNO_CANONICAL so that
    "True"/"Yes" and "False"/"No" compare as equal.

    Args:
        answer1: Extracted model answer
        ground_truth: Reference answer
        answer_type: "mcq" or "yesno"

    Returns:
        True if the answers match
    """
    if not answer1:
        return False
    if answer_type == "mcq":
        return answer1.upper() == ground_truth.upper()
    if answer_type == "yesno":
        norm_a1 = _YESNO_CANONICAL.get(answer1.lower(), answer1.lower())
        norm_gt = _YESNO_CANONICAL.get(ground_truth.lower(), ground_truth.lower())
        return norm_a1 == norm_gt
    return False


# =============================================================================
# Inference
# =============================================================================


def generate_answer1_batch(
    samples: List[Dict[str, Any]],
    model: Any,
    processor: Any,
    image_dir: Path,
    batch_size: int,
    device: str,
) -> List[Dict[str, Any]]:
    """Run model inference to populate answer1 and a1_is_correct for each sample.

    Processes samples individually (Qwen2.5-VL does not support variable-length
    image batching easily). Uses a large outer batch only for tqdm granularity.

    Args:
        samples: Intermediate sample dicts without answer1
        model: Qwen2_5_VLForConditionalGeneration model
        processor: AutoProcessor
        image_dir: Base directory for loading images by relative path
        batch_size: Number of samples processed per tqdm step
        device: CUDA device string (e.g. "cuda:0")

    Returns:
        Completed sample dicts with answer1 and a1_is_correct
    """
    import torch

    results: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        chunk = samples[start : start + batch_size]

        for sample in chunk:
            img_path = image_dir / sample["image_path"]
            img = _open_image(str(img_path))
            if img is None:
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    pass

            # Build single-turn prompt
            content: List[Dict[str, Any]] = []
            if img is not None:
                content.append({"type": "image"})

            question_with_hint = sample["question"]
            if sample["answer_type"] == "mcq" and sample.get("choices"):
                question_with_hint += f"\n\nChoices: {sample['choices']}\n\nAnswer with a single letter only."
            elif sample["answer_type"] == "yesno":
                question_with_hint += "\n\nAnswer with Yes or No only."
            content.append({"type": "text", "text": question_with_hint})

            messages = [{"role": "user", "content": content}]

            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                proc_kwargs: Dict[str, Any] = {"text": text, "return_tensors": "pt"}
                if img is not None:
                    proc_kwargs["images"] = [img]

                inputs = processor(**proc_kwargs).to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=False,
                    )

                new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
                raw = processor.decode(new_tokens, skip_special_tokens=True).strip()

                if sample["answer_type"] == "mcq":
                    answer1 = _extract_mcq_answer(raw)
                else:
                    answer1 = _extract_yesno_answer(raw)

                a1_correct = _is_correct(answer1, sample["ground_truth"], sample["answer_type"])
                results.append({**sample, "answer1": answer1, "a1_is_correct": a1_correct})

            except Exception as e:
                logger.warning(f"Inference failed: {e}")
                results.append({**sample, "answer1": "", "a1_is_correct": False})

    return results


# =============================================================================
# I/O helpers
# =============================================================================


def _save_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
    """Write samples to a JSONL file.

    Args:
        samples: List of sample dicts
        path: Output file path
    """
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples)} samples → {path}")


def _split_train_val(
    samples: List[Dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Randomly split samples into train and val sets.

    Args:
        samples: All samples
        val_fraction: Fraction to use for validation
        seed: Random seed

    Returns:
        Tuple of (train_samples, val_samples)
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main entry point for MCQ and Yes/No GRPO dataset preparation."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    requested = {d.strip() for d in args.datasets.split(",")}

    logger.info("=" * 60)
    logger.info("MCQ + Yes/No GRPO Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Datasets:        {requested}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info(f"Image dir:       {image_dir}")
    logger.info(f"Max samples:     {args.max_samples or 'all (per dataset)'}")
    logger.info(f"Val fraction:    {args.val_fraction}")
    logger.info(f"Skip inference:  {args.skip_inference}")
    logger.info("=" * 60)

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    # --- PixelReasoner (MCQ, split into train + val) ---
    if "pixel_reasoner" in requested:
        pr = load_pixel_reasoner(image_dir, args.max_samples)
        tr, va = _split_train_val(pr, args.val_fraction, args.seed)
        train_samples.extend(tr)
        val_samples.extend(va)
        logger.info(f"PixelReasoner  → train {len(tr)}, val {len(va)}")

    # --- MME-CoT (MCQ, val only — highest quality) ---
    if "mme_cot" in requested:
        mme = load_mme_cot(image_dir, args.max_samples)
        val_samples.extend(mme)
        logger.info(f"MME-CoT        → val only: {len(mme)}")

    # --- RLAIF-V yes/no subset (split into train + val) ---
    if "rlaif_v" in requested:
        rv = load_rlaif_v(image_dir, args.max_samples)
        tr, va = _split_train_val(rv, args.val_fraction, args.seed)
        train_samples.extend(tr)
        val_samples.extend(va)
        logger.info(f"RLAIF-V yes/no → train {len(tr)}, val {len(va)}")

    # --- VISCO (MCQ, split into train + val) ---
    if "visco" in requested:
        vi = load_visco(image_dir, args.max_samples)
        tr, va = _split_train_val(vi, args.val_fraction, args.seed)
        train_samples.extend(tr)
        val_samples.extend(va)
        logger.info(f"VISCO          → train {len(tr)}, val {len(va)}")

    logger.info(
        f"\nBefore inference: train={len(train_samples)}, val={len(val_samples)}"
    )

    # --- Inference: generate answer1 ---
    if not args.skip_inference:
        if not args.model_id:
            logger.error("--model_id is required unless --skip_inference is set")
            sys.exit(1)

        import os as _os

        _os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info(f"Loading model: {args.model_id}")
        processor = AutoProcessor.from_pretrained(args.model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        device = str(next(model.parameters()).device)
        logger.info(f"Model on {device}")

        logger.info("Running inference on train set ...")
        train_samples = generate_answer1_batch(
            train_samples, model, processor, image_dir, args.batch_size, device
        )
        logger.info("Running inference on val set ...")
        val_samples = generate_answer1_batch(
            val_samples, model, processor, image_dir, args.batch_size, device
        )
    else:
        for s in train_samples + val_samples:
            s["answer1"] = ""
            s["a1_is_correct"] = False
        logger.info("Inference skipped — answer1 set to '' placeholder")

    # --- Save ---
    _save_jsonl(train_samples, output_dir / "train.jsonl")
    _save_jsonl(val_samples, output_dir / "val.jsonl")

    # --- Summary ---
    def _balance(samples: List[Dict[str, Any]]) -> str:
        n = len(samples)
        if n == 0:
            return "empty"
        correct = sum(1 for s in samples if s.get("a1_is_correct"))
        return f"{correct}/{n} correct ({100 * correct // n}%)"

    def _types(samples: List[Dict[str, Any]]) -> str:
        mcq = sum(1 for s in samples if s.get("answer_type") == "mcq")
        yesno = sum(1 for s in samples if s.get("answer_type") == "yesno")
        return f"mcq={mcq} yesno={yesno}"

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Train: {len(train_samples):6d} | {_types(train_samples)} | {_balance(train_samples)}")
    logger.info(f"Val:   {len(val_samples):6d} | {_types(val_samples)} | {_balance(val_samples)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
