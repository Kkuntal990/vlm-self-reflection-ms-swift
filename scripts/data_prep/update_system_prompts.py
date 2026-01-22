#!/usr/bin/env python3
"""
Update system prompts in training data files.

This script updates the system prompts in JSONL training files for the
self-reflection training pipeline.

Usage:
    python scripts/data_prep/update_system_prompts.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# New system prompts
FEEDBACK_CRITIC_PROMPT = """You are a visual feedback critic. Given an image, a question, and a proposed answer, your job is to evaluate the answer using ONLY visual evidence.

Rules:
- First state what you see in the image relevant to the question.
- Then judge the answer against that evidence.
- Output must be one line starting with either:
  "CORRECT: ..." or "ERROR: ..."
- Do NOT provide a new answer. Only critique.
- If the question is not visually answerable from the image, say:
  "ERROR: The image does not provide evidence to answer this question."
"""

VL_ASSISTANT_PROMPT = """You are a helpful vision-language assistant.
Use the image as primary evidence.
Answer succinctly and follow the user's requested format (e.g., option letter).
If the user provides feedback, revise the answer to address the specific error.
Do not repeat the feedback; output only the final revised answer."""


def update_system_prompt(input_path: str, output_path: str, new_prompt: str) -> int:
    """Update system prompt in a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        new_prompt: New system prompt to use

    Returns:
        Number of samples processed
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            sample = json.loads(line.strip())

            # Update system prompt (first message)
            if "messages" in sample and len(sample["messages"]) > 0:
                if sample["messages"][0].get("role") == "system":
                    sample["messages"][0]["content"] = new_prompt

            f_out.write(json.dumps(sample) + "\n")
            count += 1

    logger.info(f"Processed {count} samples: {input_path} -> {output_path}")
    return count


def main():
    """Main function to update system prompts."""
    base_dir = Path(__file__).parent.parent.parent / "data"

    # Update fire_feedback_train.jsonl
    logger.info("Updating fire_feedback system prompt...")
    update_system_prompt(
        input_path=base_dir / "fire_feedback/fire_feedback_train.jsonl",
        output_path=base_dir / "fire_feedback_v2/fire_feedback_train.jsonl",
        new_prompt=FEEDBACK_CRITIC_PROMPT,
    )

    # Update fire_messages_train.jsonl
    logger.info("Updating fire_messages system prompt...")
    update_system_prompt(
        input_path=base_dir / "fire_preprocessed_v2/fire_messages_train.jsonl",
        output_path=base_dir / "fire_preprocessed_v3/fire_messages_train.jsonl",
        new_prompt=VL_ASSISTANT_PROMPT,
    )

    # Also update test files if they exist
    feedback_test = base_dir / "fire_feedback/fire_feedback_test.jsonl"
    if feedback_test.exists():
        logger.info("Updating fire_feedback test set...")
        update_system_prompt(
            input_path=feedback_test,
            output_path=base_dir / "fire_feedback_v2/fire_feedback_test.jsonl",
            new_prompt=FEEDBACK_CRITIC_PROMPT,
        )

    messages_test = base_dir / "fire_preprocessed_v2/fire_messages_test.jsonl"
    if messages_test.exists():
        logger.info("Updating fire_messages test set...")
        update_system_prompt(
            input_path=messages_test,
            output_path=base_dir / "fire_preprocessed_v3/fire_messages_test.jsonl",
            new_prompt=VL_ASSISTANT_PROMPT,
        )

    logger.info("Done! New datasets created:")
    logger.info(f"  - {base_dir / 'fire_feedback_v2/'}")
    logger.info(f"  - {base_dir / 'fire_preprocessed_v3/'}")


if __name__ == "__main__":
    main()
