#!/usr/bin/env python3
"""
Answer extraction and matching evaluation for multi-turn self-refinement.

Extracts core answers from verbose model outputs and ground truth,
then classifies matches with confidence levels for two-tier verification.

Usage:
    python scripts/evaluation/evaluate_answer_matching.py \
        --dataset_path outputs/inference_v5/qwen-thought-mt-v1.jsonl \
        --output_dir outputs/inference_v5/eval_results
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTION_TYPES = ["MCQ", "INTEGER", "FLOAT", "FREE_FORM"]
CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_NEEDS_REVIEW = "NEEDS_REVIEW"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Result of extracting an answer from text."""

    extracted: Optional[str]
    method: str  # Which regex/rule matched
    confidence: str  # HIGH or NEEDS_REVIEW

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SampleEvalResult:
    """Evaluation result for a single sample."""

    sample_index: int
    question_type: str
    question_snippet: str
    gt_raw: str
    initial_raw: str
    final_raw: str
    gt_extracted: Optional[str]
    gt_extraction_method: str
    initial_extracted: Optional[str]
    initial_extraction_method: str
    final_extracted: Optional[str]
    final_extraction_method: str
    initial_match: Optional[bool]  # None = uncertain
    final_match: Optional[bool]
    confidence: str
    review_reason: str = ""
    choices_text: str = ""  # For MCQ: the raw choices

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AggregateMetrics:
    """Aggregate evaluation metrics."""

    total_samples: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    high_confidence: int = 0
    needs_review: int = 0
    initial_correct_high: int = 0
    initial_incorrect_high: int = 0
    final_correct_high: int = 0
    final_incorrect_high: int = 0
    by_type_high_initial: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_type_high_final: Dict[str, Dict[str, int]] = field(default_factory=dict)
    transitions_high: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Question type classification
# ---------------------------------------------------------------------------


def classify_question(question: str) -> str:
    """Classify question type from the hint text.

    Args:
        question: The full question text including hint.

    Returns:
        One of: MCQ, INTEGER, FLOAT, FREE_FORM
    """
    q_lower = question.lower()
    if "option letter" in q_lower:
        return "MCQ"
    elif "integer answer" in q_lower:
        return "INTEGER"
    elif "floating-point" in q_lower:
        return "FLOAT"
    else:
        return "FREE_FORM"


def extract_choices(question: str) -> Dict[str, str]:
    """Extract MCQ choices from question text.

    Handles both formats:
    - Parenthesized: (A) Yes  (B) No
    - Colon: A:Yes  B:No

    Args:
        question: The full question text.

    Returns:
        Dict mapping letter to choice text, e.g. {"A": "Yes", "B": "No"}
    """
    choices = {}

    # Format 1: (A) text
    for m in re.finditer(
        r"\(([A-H])\)\s*(.+?)(?=\s*\([A-H]\)|$)", question, re.DOTALL
    ):
        letter = m.group(1)
        text = m.group(2).strip().rstrip("\n").strip()
        choices[letter] = text

    if choices:
        return choices

    # Format 2: A:text (newline separated)
    for m in re.finditer(
        r"(?:^|\n)\s*([A-H])\s*:\s*(.+?)(?=\n\s*[A-H]\s*:|$)",
        question,
        re.DOTALL,
    ):
        letter = m.group(1)
        text = m.group(2).strip().rstrip("\n").strip()
        choices[letter] = text

    if choices:
        return choices

    # Format 3: A.text (period separator, newline separated)
    for m in re.finditer(
        r"(?:^|\n)\s*([A-H])\.\s*(.+?)(?=\n\s*[A-H]\.|$)",
        question,
        re.DOTALL,
    ):
        letter = m.group(1)
        text = m.group(2).strip().rstrip("\n").strip()
        choices[letter] = text

    return choices


# ---------------------------------------------------------------------------
# MCQ extraction
# ---------------------------------------------------------------------------


def extract_mcq_from_model(answer: str) -> ExtractionResult:
    """Extract option letter from model answer.

    Args:
        answer: The model's verbose answer text.

    Returns:
        ExtractionResult with the extracted letter.
    """
    # 1. \boxed{X}
    m = re.search(r"\\boxed\{([A-H])\}", answer)
    if m:
        return ExtractionResult(m.group(1), "boxed", CONFIDENCE_HIGH)

    # 2. Answer: (X) or Answer: X
    m = re.search(r"[Aa]nswer:\s*\(?([A-H])\)?", answer)
    if m:
        return ExtractionResult(m.group(1), "answer_colon", CONFIDENCE_HIGH)

    # 3. "the answer is (X)" / "the answer is X"
    m = re.search(r"the answer is\s*\(?([A-H])\)?", answer, re.IGNORECASE)
    if m:
        return ExtractionResult(m.group(1), "answer_is", CONFIDENCE_HIGH)

    # 4. "the correct answer is (X)"
    m = re.search(r"the correct answer (?:is|should be)\s*\(?([A-H])\)?", answer, re.IGNORECASE)
    if m:
        return ExtractionResult(m.group(1), "correct_answer_is", CONFIDENCE_HIGH)

    # 5. "Therefore, (X)" or "Therefore, X" near end
    m = re.search(r"[Tt]herefore.*?\(?([A-H])\)?\s*\.?\s*$", answer[-300:])
    if m:
        return ExtractionResult(m.group(1), "therefore", CONFIDENCE_HIGH)

    # 6. Last standalone (X) in text
    matches = re.findall(r"\(([A-H])\)", answer)
    if matches:
        return ExtractionResult(matches[-1], "last_paren", CONFIDENCE_HIGH)

    # 7. "choose (X)" or "select (X)"
    m = re.search(r"(?:choose|select|pick)\s*\(?([A-H])\)?", answer, re.IGNORECASE)
    if m:
        return ExtractionResult(m.group(1), "choose", CONFIDENCE_HIGH)

    # 8. "is X." or "is (X)." near end of text
    m = re.search(r"\bis\s*:?\s*\(?([A-H])\)?\s*\.?\s*$", answer[-200:], re.IGNORECASE)
    if m:
        return ExtractionResult(m.group(1), "is_letter_end", CONFIDENCE_HIGH)

    # 9. Chinese answer patterns: 答案是X, 选X, 答案：X, 应该选X
    m = re.search(
        r"(?:答案[是为]|选择?|应该选|正确答案是|所以选)\s*[:：]?\s*\(?([A-H])\)?",
        answer,
    )
    if m:
        return ExtractionResult(m.group(1), "chinese_answer", CONFIDENCE_HIGH)

    # 10. Last isolated letter near end
    m = re.search(r"\b([A-H])\s*\.?\s*$", answer[-100:])
    if m:
        return ExtractionResult(m.group(1), "last_letter", CONFIDENCE_NEEDS_REVIEW)

    # 11. Model gives answer VALUE instead of letter — match against choices
    # This is handled at the match stage, not extraction
    return ExtractionResult(None, "no_match", CONFIDENCE_NEEDS_REVIEW)


def extract_mcq_from_gt(
    gt: str, choices: Dict[str, str]
) -> ExtractionResult:
    """Extract option letter from GT answer.

    Args:
        gt: The ground truth answer text.
        choices: Dict of letter->choice text.

    Returns:
        ExtractionResult with the extracted letter.
    """
    gt_stripped = gt.strip()

    # 1. Single letter
    if len(gt_stripped) == 1 and gt_stripped.upper() in "ABCDEFGH":
        return ExtractionResult(gt_stripped.upper(), "single_letter", CONFIDENCE_HIGH)

    # 2. Parenthesized: "(A)" or "(D) 21.6"
    m = re.match(r"^\(([A-H])\)", gt_stripped)
    if m:
        return ExtractionResult(m.group(1), "paren_start", CONFIDENCE_HIGH)

    # 3. Letter-comma-text: "A, 15" or "C, 27°"
    m = re.match(r"^([A-H]),\s", gt_stripped)
    if m:
        return ExtractionResult(m.group(1), "letter_comma", CONFIDENCE_HIGH)

    # 4. Text-comma-letter at end: "...decrease, C."
    m = re.search(r",\s*([A-H])\.?\s*$", gt_stripped)
    if m:
        return ExtractionResult(m.group(1), "text_comma_letter", CONFIDENCE_HIGH)

    # 5. Text with (X) embedded
    m = re.search(r"\(([A-H])\)", gt_stripped)
    if m:
        return ExtractionResult(m.group(1), "embedded_paren", CONFIDENCE_HIGH)

    # 6. Yes/No mapping
    if gt_stripped.lower() in ("yes", "no"):
        target = gt_stripped.lower()
        for letter, text in choices.items():
            if text.strip().lower() == target:
                return ExtractionResult(letter, "yes_no_map", CONFIDENCE_HIGH)
        # Fallback: common convention A=Yes, B=No
        if target == "yes" and "A" in choices:
            return ExtractionResult("A", "yes_no_convention", CONFIDENCE_NEEDS_REVIEW)
        if target == "no" and "B" in choices:
            return ExtractionResult("B", "no_convention", CONFIDENCE_NEEDS_REVIEW)

    # 7. GT starts with Yes/No then has more text (e.g., "No, the number of...")
    m = re.match(r"^(yes|no)\b", gt_stripped, re.IGNORECASE)
    if m:
        target = m.group(1).lower()
        for letter, text in choices.items():
            if text.strip().lower() == target:
                return ExtractionResult(letter, "yes_no_prefix_map", CONFIDENCE_HIGH)

    # 8. Embedded standalone letter (not part of a word)
    letters_found = re.findall(r"\b([A-H])\b", gt_stripped)
    # Filter out common English words that are single letters (A, I)
    meaningful = [
        l for l in letters_found if l in choices
    ]
    if len(meaningful) == 1:
        return ExtractionResult(
            meaningful[0], "embedded_letter", CONFIDENCE_NEEDS_REVIEW
        )

    # 9. GT is a value — match against choice text (handles 2000+ cases)
    best_match = _match_gt_value_to_choice(gt_stripped, choices)
    if best_match is not None:
        return ExtractionResult(best_match[0], best_match[1], CONFIDENCE_HIGH)

    # 10. No clear letter found
    return ExtractionResult(None, "no_match", CONFIDENCE_NEEDS_REVIEW)


def _match_gt_value_to_choice(
    gt_value: str, choices: Dict[str, str]
) -> Optional[Tuple[str, str]]:
    """Match a GT answer value against MCQ choice values.

    Args:
        gt_value: The GT answer (may be a value, not a letter).
        choices: Dict of letter -> choice text.

    Returns:
        Tuple of (letter, method) or None if no match.
    """
    if not choices:
        return None

    gt_norm = _normalize_for_matching(gt_value)

    # Pass 1: Exact normalized match
    for letter, text in choices.items():
        if _normalize_for_matching(text) == gt_norm:
            return (letter, "value_exact_match")

    # Pass 2: Numeric match — extract numbers from GT and choices
    gt_nums = re.findall(r"-?[\d,]+\.?\d*", gt_value)
    if gt_nums:
        gt_num = gt_nums[0].replace(",", "")
        for letter, text in choices.items():
            choice_nums = re.findall(r"-?[\d,]+\.?\d*", text)
            if choice_nums:
                choice_num = choice_nums[0].replace(",", "")
                try:
                    if abs(float(gt_num) - float(choice_num)) < 0.01:
                        return (letter, "value_numeric_match")
                except ValueError:
                    pass

    # Pass 3: Substring containment (GT value in choice or choice in GT)
    for letter, text in choices.items():
        text_norm = _normalize_for_matching(text)
        if len(gt_norm) > 2 and len(text_norm) > 2:
            if gt_norm == text_norm or (len(gt_norm) <= 20 and gt_norm in text_norm):
                return (letter, "value_substring_match")
            if len(text_norm) <= 20 and text_norm in gt_norm:
                return (letter, "value_substring_match")

    # Pass 4: For verbose GT, extract the core value and retry
    # e.g., "The area of △ABC' is 6." → try to match "6" against choices
    m = re.search(r"\b(?:is|=|equals?)\s+(.+?)\.?\s*$", gt_value, re.IGNORECASE)
    if m:
        core = m.group(1).strip()
        core_norm = _normalize_for_matching(core)
        for letter, text in choices.items():
            if _normalize_for_matching(text) == core_norm:
                return (letter, "value_core_extract_match")
        # Also try numeric on the core
        core_nums = re.findall(r"-?[\d,]+\.?\d*", core)
        if core_nums:
            for letter, text in choices.items():
                choice_nums = re.findall(r"-?[\d,]+\.?\d*", text)
                if choice_nums and core_nums[0].replace(",", "") == choice_nums[0].replace(",", ""):
                    return (letter, "value_core_numeric_match")

    return None


def _normalize_for_matching(text: str) -> str:
    """Normalize text for value comparison.

    Args:
        text: Raw text.

    Returns:
        Normalized lowercase string.
    """
    normalized = text.strip().lower()
    # Remove LaTeX wrappers
    normalized = re.sub(r"[\$\\(\\)\[\]]", "", normalized)
    normalized = re.sub(r"\\text\{([^}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"\1/\2", normalized)
    normalized = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", normalized)
    normalized = normalized.replace("\\pi", "pi")
    normalized = normalized.replace("\\,", "")
    # Remove common punctuation
    normalized = normalized.strip(".,;:!?°%$ ")
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------


def extract_number(text: str, allow_float: bool = True) -> ExtractionResult:
    """Extract a number from text.

    Args:
        text: The text to extract from.
        allow_float: Whether to allow decimal numbers.

    Returns:
        ExtractionResult with the extracted number as string.
    """
    text_stripped = text.strip()

    # 1. \boxed{N}
    m = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", text)
    if m:
        val = m.group(1).replace(",", "")
        return ExtractionResult(val, "boxed", CONFIDENCE_HIGH)

    # 2. Pure number
    if allow_float:
        m = re.match(r"^(-?[\d,]+\.?\d*)$", text_stripped)
    else:
        m = re.match(r"^(-?[\d,]+)$", text_stripped)
    if m:
        val = m.group(1).replace(",", "")
        return ExtractionResult(val, "pure_number", CONFIDENCE_HIGH)

    # 3. "Answer: N"
    if allow_float:
        m = re.search(r"[Aa]nswer:\s*(-?[\d,]+\.?\d*)", text)
    else:
        m = re.search(r"[Aa]nswer:\s*(-?[\d,]+)", text)
    if m:
        val = m.group(1).replace(",", "")
        return ExtractionResult(val, "answer_colon", CONFIDENCE_HIGH)

    # 4. "the answer is N"
    if allow_float:
        m = re.search(
            r"(?:the answer|the value|the result) (?:is|=)\s*(-?[\d,]+\.?\d*)",
            text,
            re.IGNORECASE,
        )
    else:
        m = re.search(
            r"(?:the answer|the value|the result) (?:is|=)\s*(-?[\d,]+)",
            text,
            re.IGNORECASE,
        )
    if m:
        val = m.group(1).replace(",", "")
        return ExtractionResult(val, "answer_is", CONFIDENCE_HIGH)

    # 5. "is N." or "is N," at end of sentence — common in verbose answers
    if allow_float:
        m = re.search(r"\bis\s+(-?[\d,]+\.?\d*)\s*[.,]?\s*$", text)
    else:
        m = re.search(r"\bis\s+(-?[\d,]+)\s*[.,]?\s*$", text)
    if m:
        val = m.group(1).replace(",", "")
        return ExtractionResult(val, "is_N_end", CONFIDENCE_HIGH)

    # 6. Last number in text
    if allow_float:
        numbers = re.findall(r"(-?\d[\d,]*\.?\d*)", text)
    else:
        numbers = re.findall(r"(-?\d+)", text)
    if numbers:
        val = numbers[-1].replace(",", "")
        # Confidence based on how many numbers exist and text length
        # Short text or few numbers → HIGH confidence the last number is the answer
        confidence = CONFIDENCE_HIGH if len(numbers) <= 5 or len(text) < 200 else CONFIDENCE_NEEDS_REVIEW
        return ExtractionResult(val, "last_number", confidence)

    return ExtractionResult(None, "no_match", CONFIDENCE_NEEDS_REVIEW)


# ---------------------------------------------------------------------------
# Free-form extraction
# ---------------------------------------------------------------------------


def extract_freeform(text: str) -> ExtractionResult:
    """Extract answer from free-form text.

    Args:
        text: The text to extract from.

    Returns:
        ExtractionResult with the extracted value.
    """
    text_stripped = text.strip()

    # 0. Yes/No prefix — extract the yes/no as the core answer
    yn_match = re.match(r"^(yes|no)\b[,.]?\s*(.*)", text_stripped, re.IGNORECASE)
    if yn_match:
        yn = yn_match.group(1).lower()
        # If text is short or is a VQA-style yes/no answer
        if len(text_stripped) < 200:
            return ExtractionResult(yn, "yes_no_prefix", CONFIDENCE_HIGH)

    # 1. \boxed{...}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return ExtractionResult(m.group(1).strip(), "boxed", CONFIDENCE_HIGH)

    # 2. Pure number
    m = re.match(r"^(-?[\d,]+\.?\d*)\s*[°%]?\s*$", text_stripped)
    if m:
        return ExtractionResult(m.group(1).replace(",", ""), "pure_number", CONFIDENCE_HIGH)

    # 3. Short text (< 30 chars, no numbers) — likely a keyword/phrase answer
    if len(text_stripped) < 30 and not re.search(r"\d", text_stripped):
        return ExtractionResult(
            text_stripped.lower().strip(".,;:!? "),
            "short_text",
            CONFIDENCE_HIGH,
        )

    # 4. Short expression (< 50 chars with math symbols)
    if len(text_stripped) < 50 and re.search(r"[=+\-*/^√π∞()]", text_stripped):
        return ExtractionResult(
            normalize_expression(text_stripped),
            "short_expression",
            CONFIDENCE_NEEDS_REVIEW,
        )

    # 5. "Answer: ..." explicit marker
    m = re.search(r"[Aa]nswer:\s*(.+?)(?:\.|$)", text)
    if m:
        ans = m.group(1).strip()
        if len(ans) < 100:
            return ExtractionResult(ans, "answer_colon", CONFIDENCE_HIGH)

    # 6. Try numeric extraction as fallback
    num_result = extract_number(text, allow_float=True)
    if num_result.extracted is not None:
        # Promote confidence: if the numeric method is HIGH, keep it
        conf = num_result.confidence
        return ExtractionResult(
            num_result.extracted,
            f"numeric_fallback_{num_result.method}",
            conf,
        )

    # 7. For verbose text, extract the last sentence's key value
    sentences = re.split(r"[.!?]\s+", text_stripped)
    if sentences:
        last = sentences[-1].strip()
        if len(last) < 100:
            # Try to extract number from last sentence
            last_nums = re.findall(r"-?[\d,]+\.?\d*", last)
            if last_nums:
                return ExtractionResult(
                    last_nums[-1].replace(",", ""),
                    "last_sentence_number",
                    CONFIDENCE_HIGH,
                )
            return ExtractionResult(last.lower().strip(".,;:!? "), "last_sentence", CONFIDENCE_NEEDS_REVIEW)

    return ExtractionResult(None, "no_match", CONFIDENCE_NEEDS_REVIEW)


def normalize_expression(expr: str) -> str:
    """Normalize a math expression for comparison.

    Args:
        expr: Raw expression string.

    Returns:
        Normalized expression string.
    """
    normalized = expr.strip()
    # Remove LaTeX wrappers
    normalized = re.sub(r"\\[(\[]", "", normalized)
    normalized = re.sub(r"\\[)\]]", "", normalized)
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Normalize common symbols
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\cdot", "*")
    normalized = normalized.replace("\\div", "/")
    normalized = normalized.replace("\\sqrt", "√")
    normalized = normalized.replace("\\pi", "π")
    normalized = normalized.replace("\\infty", "∞")
    # Remove \text{...} wrappers
    normalized = re.sub(r"\\text\{([^}]*)\}", r"\1", normalized)
    # Remove \, spacing
    normalized = normalized.replace("\\,", " ")
    return normalized.lower().strip()


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------


def numbers_match(
    a: str, b: str, rel_tol: float = 0.005, abs_tol: float = 0.01
) -> Optional[bool]:
    """Compare two number strings with tolerance.

    Uses exact match for integers, tolerance for floats.

    Args:
        a: First number string.
        b: Second number string.
        rel_tol: Relative tolerance (for floats only).
        abs_tol: Absolute tolerance (for floats only).

    Returns:
        True if match, False if not, None if can't parse.
    """
    try:
        fa = float(a.replace(",", ""))
        fb = float(b.replace(",", ""))
    except (ValueError, TypeError):
        return None

    if fa == fb:
        return True

    # If both are effectively integers, require exact match
    # This prevents 2015 matching 2016 via tolerance
    a_is_int = "." not in a.replace(",", "") and fa == int(fa)
    b_is_int = "." not in b.replace(",", "") and fb == int(fb)
    if a_is_int and b_is_int:
        return int(fa) == int(fb)

    tolerance = max(abs_tol, rel_tol * abs(fb))
    return abs(fa - fb) <= tolerance


def strings_match(a: str, b: str) -> bool:
    """Case-insensitive string comparison with normalization.

    Removes common VQA filler phrases before comparison.

    Args:
        a: First string.
        b: Second string.

    Returns:
        True if semantically equivalent.
    """
    a_clean = _clean_vqa_text(a)
    b_clean = _clean_vqa_text(b)
    a_norm = re.sub(r"[^a-z0-9.]", "", a_clean)
    b_norm = re.sub(r"[^a-z0-9.]", "", b_clean)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    # Check containment for short answers
    if len(a_norm) > 2 and len(b_norm) > 2:
        if a_norm in b_norm or b_norm in a_norm:
            return True
    # Check word-level overlap for longer answers
    a_words = set(a_clean.split())
    b_words = set(b_clean.split())
    if len(a_words) >= 2 and len(b_words) >= 2:
        overlap = a_words & b_words
        # High overlap ratio means likely the same answer
        min_len = min(len(a_words), len(b_words))
        if min_len > 0 and len(overlap) / min_len >= 0.8:
            return True
    return False


def _clean_vqa_text(text: str) -> str:
    """Remove common VQA filler phrases for cleaner comparison.

    Args:
        text: Raw answer text.

    Returns:
        Cleaned text.
    """
    cleaned = text.lower().strip()
    # Remove common image-reference filler
    filler_patterns = [
        r"\b(in the|in this|of the|from the)\s+(image|picture|photo|figure|diagram|scene)\b",
        r"\bshown in the (image|picture|photo)\b",
        r"\bbased on the (image|picture|photo)\b",
        r"\baccording to the (image|picture|photo)\b",
        r"\bvisible in the (image|picture|photo)\b",
        r"\bin the background\b",
        r"\bin the foreground\b",
        r"\bthe answer is\b",
        r"\bthe correct answer is\b",
        r"\bi think\b",
        r"\bit appears (that |to be )?\b",
    ]
    for pat in filler_patterns:
        cleaned = re.sub(pat, " ", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def match_answers(
    model_ext: ExtractionResult,
    gt_ext: ExtractionResult,
    question_type: str,
) -> Tuple[Optional[bool], str]:
    """Compare extracted model and GT answers.

    Args:
        model_ext: Extraction result from model answer.
        gt_ext: Extraction result from GT answer.
        question_type: The classified question type.

    Returns:
        Tuple of (match_result, confidence). match_result is None if uncertain.
    """
    if model_ext.extracted is None or gt_ext.extracted is None:
        return None, CONFIDENCE_NEEDS_REVIEW

    m_val = model_ext.extracted.strip() if model_ext.extracted else ""
    g_val = gt_ext.extracted.strip() if gt_ext.extracted else ""

    # Empty extractions are undetermined, not matches
    if not m_val or not g_val:
        return None, CONFIDENCE_NEEDS_REVIEW

    if question_type == "MCQ":
        # Both are letters → exact comparison
        if (
            m_val
            and g_val
            and len(m_val) == 1
            and len(g_val) == 1
            and m_val.upper() in "ABCDEFGH"
            and g_val.upper() in "ABCDEFGH"
        ):
            match = m_val.upper() == g_val.upper()
            conf = CONFIDENCE_HIGH
            if model_ext.confidence == CONFIDENCE_NEEDS_REVIEW or gt_ext.confidence == CONFIDENCE_NEEDS_REVIEW:
                conf = CONFIDENCE_NEEDS_REVIEW
            return match, conf
        # One or both are values → needs review (value comparison happened at extraction)
        return None, CONFIDENCE_NEEDS_REVIEW

    elif question_type in ("INTEGER", "FLOAT"):
        result = numbers_match(m_val, g_val)
        if result is None:
            return None, CONFIDENCE_NEEDS_REVIEW
        conf = CONFIDENCE_HIGH
        if model_ext.confidence == CONFIDENCE_NEEDS_REVIEW or gt_ext.confidence == CONFIDENCE_NEEDS_REVIEW:
            conf = CONFIDENCE_NEEDS_REVIEW
        return result, conf

    else:  # FREE_FORM
        # Yes/No comparison — if both sides are yes/no, compare directly
        m_yn = _extract_yes_no(m_val)
        g_yn = _extract_yes_no(g_val)
        if m_yn is not None and g_yn is not None:
            return m_yn == g_yn, CONFIDENCE_HIGH

        # If GT is yes/no but model is descriptive text, infer yes/no from text
        if g_yn is not None and m_yn is None:
            m_yn_inferred = _infer_yes_no_from_text(m_val)
            if m_yn_inferred is not None:
                return g_yn == m_yn_inferred, CONFIDENCE_NEEDS_REVIEW

        # Try numeric comparison first
        num_result = numbers_match(m_val, g_val)
        if num_result is not None:
            conf = CONFIDENCE_HIGH
            if model_ext.confidence == CONFIDENCE_NEEDS_REVIEW or gt_ext.confidence == CONFIDENCE_NEEDS_REVIEW:
                conf = CONFIDENCE_NEEDS_REVIEW
            return num_result, conf

        # Try string comparison
        if strings_match(m_val, g_val):
            return True, CONFIDENCE_NEEDS_REVIEW  # String match always needs review

        return False, CONFIDENCE_NEEDS_REVIEW


def _extract_yes_no(text: str) -> Optional[str]:
    """Extract yes/no from text if it's clearly a yes/no answer.

    Args:
        text: The extracted text.

    Returns:
        'yes', 'no', or None.
    """
    text_lower = text.strip().lower()
    if text_lower in ("yes", "no"):
        return text_lower
    m = re.match(r"^(yes|no)\b", text_lower)
    if m:
        return m.group(1)
    return None


def _infer_yes_no_from_text(text: str) -> Optional[str]:
    """Infer yes/no from descriptive text.

    Args:
        text: Model's descriptive answer text.

    Returns:
        'yes', 'no', or None if unclear.
    """
    text_lower = text.strip().lower()
    # Strong negative signals
    neg_patterns = [
        r"\bnot\b", r"\bno\b", r"\bdon'?t\b", r"\bdoesn'?t\b",
        r"\bisn'?t\b", r"\baren'?t\b", r"\bwasn'?t\b", r"\bweren'?t\b",
        r"\bnone\b", r"\bneither\b", r"\bcannot\b", r"\bcan'?t\b",
    ]
    for pat in neg_patterns:
        if re.search(pat, text_lower):
            return "no"
    return "yes"  # Default: affirmative if no negation found


# ---------------------------------------------------------------------------
# Main evaluation per sample
# ---------------------------------------------------------------------------


def evaluate_sample(sample: Dict) -> SampleEvalResult:
    """Evaluate a single sample.

    Args:
        sample: A sample dict from the JSONL file.

    Returns:
        SampleEvalResult with extraction and matching details.
    """
    question = sample["question"]
    gt_raw = sample["gt_final_answer"]
    final_raw = sample["final_answer"]
    initial_raw = sample["generated_turns"][0]["answer"]
    q_type = classify_question(question)

    choices = {}
    choices_text = ""
    if q_type == "MCQ":
        choices = extract_choices(question)
        choices_text = json.dumps(choices)

    # Extract from GT
    if q_type == "MCQ":
        gt_ext = extract_mcq_from_gt(gt_raw, choices)
    elif q_type == "INTEGER":
        gt_ext = extract_number(gt_raw, allow_float=False)
    elif q_type == "FLOAT":
        gt_ext = extract_number(gt_raw, allow_float=True)
    else:
        gt_ext = extract_freeform(gt_raw)

    # Extract from model final answer
    if q_type == "MCQ":
        final_ext = extract_mcq_from_model(final_raw)
        # Fallback: if model didn't give a letter, try value matching against choices
        if final_ext.extracted is None and choices:
            val_match = _match_gt_value_to_choice(final_raw[-300:], choices)
            if val_match is not None:
                final_ext = ExtractionResult(val_match[0], f"model_value_{val_match[1]}", CONFIDENCE_NEEDS_REVIEW)
    elif q_type == "INTEGER":
        final_ext = extract_number(final_raw, allow_float=False)
    elif q_type == "FLOAT":
        final_ext = extract_number(final_raw, allow_float=True)
    else:
        final_ext = extract_freeform(final_raw)

    # Extract from model initial answer
    if q_type == "MCQ":
        initial_ext = extract_mcq_from_model(initial_raw)
        if initial_ext.extracted is None and choices:
            val_match = _match_gt_value_to_choice(initial_raw[-300:], choices)
            if val_match is not None:
                initial_ext = ExtractionResult(val_match[0], f"model_value_{val_match[1]}", CONFIDENCE_NEEDS_REVIEW)
    elif q_type == "INTEGER":
        initial_ext = extract_number(initial_raw, allow_float=False)
    elif q_type == "FLOAT":
        initial_ext = extract_number(initial_raw, allow_float=True)
    else:
        initial_ext = extract_freeform(initial_raw)

    # Match
    initial_match, initial_conf = match_answers(initial_ext, gt_ext, q_type)
    final_match, final_conf = match_answers(final_ext, gt_ext, q_type)

    # Overall confidence
    if initial_conf == CONFIDENCE_NEEDS_REVIEW or final_conf == CONFIDENCE_NEEDS_REVIEW:
        overall_conf = CONFIDENCE_NEEDS_REVIEW
    else:
        overall_conf = CONFIDENCE_HIGH

    # Build review reason
    reasons = []
    if gt_ext.confidence == CONFIDENCE_NEEDS_REVIEW:
        reasons.append(f"gt_extraction_uncertain({gt_ext.method})")
    if initial_ext.confidence == CONFIDENCE_NEEDS_REVIEW:
        reasons.append(f"initial_extraction_uncertain({initial_ext.method})")
    if final_ext.confidence == CONFIDENCE_NEEDS_REVIEW:
        reasons.append(f"final_extraction_uncertain({final_ext.method})")
    if initial_match is None:
        reasons.append("initial_match_undetermined")
    if final_match is None:
        reasons.append("final_match_undetermined")

    return SampleEvalResult(
        sample_index=sample["sample_index"],
        question_type=q_type,
        question_snippet=question[:200],
        gt_raw=gt_raw[:500],
        initial_raw=initial_raw[:500],
        final_raw=final_raw[:500],
        gt_extracted=gt_ext.extracted,
        gt_extraction_method=gt_ext.method,
        initial_extracted=initial_ext.extracted,
        initial_extraction_method=initial_ext.method,
        final_extracted=final_ext.extracted,
        final_extraction_method=final_ext.method,
        initial_match=initial_match,
        final_match=final_match,
        confidence=overall_conf,
        review_reason="; ".join(reasons),
        choices_text=choices_text,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_metrics(results: List[SampleEvalResult]) -> AggregateMetrics:
    """Compute aggregate metrics from results.

    Args:
        results: List of per-sample evaluation results.

    Returns:
        AggregateMetrics with summary statistics.
    """
    metrics = AggregateMetrics(total_samples=len(results))

    for r in results:
        # Count by type
        metrics.by_type[r.question_type] = metrics.by_type.get(r.question_type, 0) + 1

        if r.confidence == CONFIDENCE_HIGH:
            metrics.high_confidence += 1

            # Count correct/incorrect for high-confidence only
            if r.initial_match is True:
                metrics.initial_correct_high += 1
            elif r.initial_match is False:
                metrics.initial_incorrect_high += 1

            if r.final_match is True:
                metrics.final_correct_high += 1
            elif r.final_match is False:
                metrics.final_incorrect_high += 1

            # Per-type breakdown
            for phase, match_val in [("initial", r.initial_match), ("final", r.final_match)]:
                type_dict = (
                    metrics.by_type_high_initial
                    if phase == "initial"
                    else metrics.by_type_high_final
                )
                if r.question_type not in type_dict:
                    type_dict[r.question_type] = {"correct": 0, "incorrect": 0, "uncertain": 0}
                if match_val is True:
                    type_dict[r.question_type]["correct"] += 1
                elif match_val is False:
                    type_dict[r.question_type]["incorrect"] += 1
                else:
                    type_dict[r.question_type]["uncertain"] += 1

            # Transition analysis (high confidence only)
            if r.initial_match is not None and r.final_match is not None:
                key = f"{'right' if r.initial_match else 'wrong'}_to_{'right' if r.final_match else 'wrong'}"
                metrics.transitions_high[key] = metrics.transitions_high.get(key, 0) + 1
        else:
            metrics.needs_review += 1

    return metrics


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_dataset(dataset_path: str, max_samples: int = 0) -> List[Dict]:
    """Load dataset from JSONL file.

    Args:
        dataset_path: Path to JSONL file.
        max_samples: Maximum samples to load (0 = all).

    Returns:
        List of sample dictionaries.
    """
    samples = []
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i}: {e}")
    logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
    return samples


def save_results(
    results: List[SampleEvalResult],
    metrics: AggregateMetrics,
    output_dir: Path,
) -> None:
    """Save all results to output directory.

    Args:
        results: Per-sample results.
        metrics: Aggregate metrics.
        output_dir: Output directory path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # All results
    all_path = output_dir / "eval_results_rulebased.jsonl"
    with open(all_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    logger.info(f"Saved {len(results)} results to {all_path}")

    # Needs-review results (for Opus verification)
    review_path = output_dir / "eval_needs_review.jsonl"
    review_count = 0
    with open(review_path, "w") as f:
        for r in results:
            if r.confidence == CONFIDENCE_NEEDS_REVIEW:
                f.write(json.dumps(r.to_dict()) + "\n")
                review_count += 1
    logger.info(f"Saved {review_count} needs-review samples to {review_path}")

    # High-confidence results (for Opus spot-checking)
    high_path = output_dir / "eval_high_confidence.jsonl"
    high_count = 0
    with open(high_path, "w") as f:
        for r in results:
            if r.confidence == CONFIDENCE_HIGH:
                f.write(json.dumps(r.to_dict()) + "\n")
                high_count += 1
    logger.info(f"Saved {high_count} high-confidence samples to {high_path}")

    # Metrics
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Summary extraction method stats
    method_stats: Dict[str, Counter] = {
        "gt": Counter(),
        "initial": Counter(),
        "final": Counter(),
    }
    for r in results:
        method_stats["gt"][r.gt_extraction_method] += 1
        method_stats["initial"][r.initial_extraction_method] += 1
        method_stats["final"][r.final_extraction_method] += 1

    stats_path = output_dir / "eval_extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump({k: dict(v.most_common()) for k, v in method_stats.items()}, f, indent=2)
    logger.info(f"Saved extraction stats to {stats_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate answer matching for multi-turn self-refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to inference JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to evaluate (0 = all)",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()
    samples = load_dataset(args.dataset_path, args.max_samples)
    output_dir = Path(args.output_dir)

    logger.info("Evaluating samples...")
    results = []
    for sample in samples:
        result = evaluate_sample(sample)
        results.append(result)

    metrics = compute_metrics(results)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {metrics.total_samples}")
    logger.info(f"HIGH confidence: {metrics.high_confidence} ({100*metrics.high_confidence/metrics.total_samples:.1f}%)")
    logger.info(f"NEEDS_REVIEW: {metrics.needs_review} ({100*metrics.needs_review/metrics.total_samples:.1f}%)")
    logger.info("")
    logger.info("By question type:")
    for qt in QUESTION_TYPES:
        count = metrics.by_type.get(qt, 0)
        logger.info(f"  {qt}: {count}")
    logger.info("")
    logger.info("HIGH confidence results (preliminary, pending Opus verification):")
    total_high = metrics.high_confidence
    if total_high > 0:
        logger.info(f"  Initial accuracy: {metrics.initial_correct_high}/{total_high} = {100*metrics.initial_correct_high/total_high:.1f}%")
        logger.info(f"  Final accuracy:   {metrics.final_correct_high}/{total_high} = {100*metrics.final_correct_high/total_high:.1f}%")
    logger.info("")
    logger.info("Transitions (high confidence only):")
    for k, v in sorted(metrics.transitions_high.items()):
        logger.info(f"  {k}: {v}")

    save_results(results, metrics, output_dir)
    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
