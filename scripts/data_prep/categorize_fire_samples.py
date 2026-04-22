#!/usr/bin/env python3
"""
Categorize FIRE dataset samples by answer type.

Reads fire_messages_train_last_loss_only_with_thought.jsonl and assigns each
sample one of 6 categories: open_ended, yes_no, short_answer, mcq,
bounding_box, region_description.

Outputs a JSONL file with (sample_index, category) pairs.

Usage:
    python scripts/data_prep/categorize_fire_samples.py \
        --input_file /outputs/fire_preprocessed_v3/fire_messages_train_last_loss_only_with_thought.jsonl \
        --output_file /outputs/fire_preprocessed_v3/fire_sample_categories.jsonl
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Categorize FIRE dataset samples by answer type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/outputs/fire_preprocessed_v3/fire_messages_train_last_loss_only_with_thought.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/outputs/fire_preprocessed_v3/fire_sample_categories.jsonl",
        help="Path to output JSONL file with (index, category) pairs",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )
    return parser.parse_args()


def extract_answer_text(messages: list[dict]) -> str:
    """Extract the final answer text (Answer: portion) from messages.

    Args:
        messages: List of message dicts

    Returns:
        The answer text from the last assistant turn with loss=True
    """
    for msg in reversed(messages):
        if msg["role"] == "assistant" and msg.get("loss") is True:
            content = msg["content"]
            if "Answer:" in content:
                return content.split("Answer:", 1)[1].strip()
            return content.strip()
    return ""


def extract_question_text(messages: list[dict]) -> str:
    """Extract the first user question from messages.

    Args:
        messages: List of message dicts

    Returns:
        The question text
    """
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def categorize_sample(question: str, answer: str) -> str:
    """Categorize a sample based on question and answer patterns.

    Priority order (checked first wins):
    1. bounding_box - answer has coordinate format
    2. region_description - question asks to describe a bounding box region
    3. mcq - question has explicit A/B/C/D options
    4. yes_no - answer starts with Yes/No
    5. short_answer - answer < 50 chars
    6. open_ended - everything else

    Args:
        question: The question text
        answer: The final answer text (after Answer: prefix)

    Returns:
        Category string
    """
    q_clean = question.replace("<image>", "").strip()
    q_lower = q_clean.lower()
    ans_stripped = answer.strip()

    # 1. Bounding box: answer contains coordinate pattern [0.xx, 0.yy, ...]
    if re.search(
        r"\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]", ans_stripped
    ):
        return "bounding_box"

    # 2. Region description: question references a bounding box region to describe
    if re.search(
        r"(provide a short description for this region|describe the region|what is in the region)\s*[:\.]?\s*\[",
        q_lower,
    ):
        return "region_description"
    if re.search(r"description.*\[\d+\.?\d*\s*,", q_lower):
        return "region_description"

    # 3. MCQ: question has option lines (A. xxx / A) xxx) or explicit instruction
    has_option_lines = bool(re.search(r"(?:^|\n)\s*[A-E][\.\)]\s", question))
    has_option_instruction = bool(
        re.search(
            r"answer with the option|from the given choices|choose.*option|select.*option",
            q_lower,
        )
    )
    if has_option_lines or has_option_instruction:
        return "mcq"

    # 4. Yes/No: answer starts with Yes/No (word boundary)
    ans_lower = ans_stripped.lower()
    if re.match(r"^(yes|no)[\s,.\!]", ans_lower) or ans_lower in ("yes", "no"):
        return "yes_no"

    # 5. Short answer: brief factual answer (< 50 chars)
    if len(ans_stripped) < 50:
        return "short_answer"

    # 6. Open-ended: long descriptive answer
    return "open_ended"


def main() -> None:
    """Main function."""
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    logger.info("=" * 60)
    logger.info("FIRE Dataset Sample Categorization")
    logger.info("=" * 60)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    category_counter: Counter = Counter()

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for i, line in enumerate(f_in):
            if args.max_samples > 0 and i >= args.max_samples:
                break

            sample = json.loads(line)
            messages = sample["messages"]

            question = extract_question_text(messages)
            answer = extract_answer_text(messages)

            category = categorize_sample(question, answer)
            category_counter[category] += 1

            f_out.write(json.dumps({"index": i, "category": category}, ensure_ascii=False) + "\n")

    total = sum(category_counter.values())

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    for cat, count in category_counter.most_common():
        pct = 100 * count / total
        logger.info(f"  {cat:25s} {count:>7d}  ({pct:5.1f}%)")
    logger.info(f"  {'TOTAL':25s} {total:>7d}")
    logger.info("=" * 60)
    logger.info(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
