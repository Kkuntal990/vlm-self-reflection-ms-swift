#!/usr/bin/env python3
"""
Self-reflective inference using mode-tagged prompts.

This script generates multi-turn self-reflective dialogues where the model:
1. Generates an initial answer with [ANSWER] mode
2. Generates its own feedback with [FEEDBACK] mode
3. Refines the answer with [REFINE] mode
4. Repeats until matching ground truth turn count

The model stays in assistant role throughout - mode tags switch task intent
without role flipping.

Usage:
    python scripts/evaluation/self_reflective_inference.py \
        --model_path /outputs/checkpoint-final \
        --dataset_path data/fire_preprocessed_v2/fire_messages_test.jsonl \
        --output_path outputs/self_reflective_results.jsonl \
        --image_base_dir /outputs \
        --max_samples 10

Reference:
    - FIRE dataset: https://huggingface.co/datasets/PengxiangLi/FIRE
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================
# Mode Prompt Templates
# ============================================

# ===========================================
# Original Training Format (fire_messages + fire_feedback)
# ===========================================

# For generating answers - just the question
ANSWER_PROMPT_TEMPLATE = """{question}"""

# For refining answers - feedback + explicit instruction to output only answer
REFINE_PROMPT_TEMPLATE = """{feedback}

Based on the above feedback, provide your revised answer only:"""

# System prompt for VL Assistant (from fire_messages training)
VL_ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate, detailed, and grounded answers based on the image and the user's instructions. When given feedback, critique, or scores, revise your response to improve correctness, specificity, and completeness."
)

# System prompt for Feedback Critic (from fire_feedback training)
FEEDBACK_CRITIC_SYSTEM_PROMPT = (
    "You are a helpful assistant that provides constructive feedback on answers to visual questions. Given an image, a question, and an answer, and the conversation history identify what is correct, what is incorrect and provide specific critique based on visual evidence."
)

# Default uses VL assistant for answer generation
DEFAULT_SYSTEM_PROMPT = VL_ASSISTANT_SYSTEM_PROMPT


# ============================================
# Data Classes
# ============================================


@dataclass
class TurnResult:
    """Result for a single turn in the refinement loop."""

    answer: str
    feedback: str  # Empty for final turn

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SampleResult:
    """Result for a single sample's self-reflective dialogue."""

    sample_index: int
    image_path: str
    question: str
    generated_turns: list[dict]
    final_answer: str
    gt_final_answer: str
    num_turns: int
    messages_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ============================================
# Model Wrapper
# ============================================


class SelfReflectionEngine:
    """Inference engine for self-reflective generation with mode-tagged prompts.

    Attributes:
        model_path: Path to the fine-tuned model checkpoint
        model: The loaded model
        processor: The model's processor
        device: Device for inference
        system_prompt: System prompt to use for all generations
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attn: bool = True,
        system_prompt: str | None = None,
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to fine-tuned model checkpoint or HuggingFace ID
            device: Device to run inference on
            dtype: Model data type
            use_flash_attn: Whether to use flash attention
            system_prompt: System prompt (None uses default, "" disables)
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        # Set system prompt
        if system_prompt is None:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
        elif system_prompt == "":
            self.system_prompt = None
        else:
            self.system_prompt = system_prompt

        logger.info(f"Loading model from {model_path}")

        # Lazy imports for heavy ML libraries
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Load model with flash attention fallback
        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(f"Flash attention failed ({e}), falling back to eager")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=dtype,
                attn_implementation="eager",
            )

        self.model.eval()
        logger.info("Model loaded successfully")

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response given the message history.

        Args:
            messages: List of messages in chat format
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling
            system_prompt: Override system prompt for this call (None uses default)

        Returns:
            Generated response text
        """
        from qwen_vl_utils import process_vision_info

        # Use override system prompt if provided, otherwise use default
        active_system_prompt = system_prompt if system_prompt is not None else self.system_prompt

        # Prepend system prompt if set
        full_messages = []
        if active_system_prompt:
            full_messages.append({"role": "system", "content": active_system_prompt})
        full_messages.extend(messages)

        # Process inputs
        text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(full_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode only the generated part
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()


# ============================================
# Dataset Loading
# ============================================


def load_dataset(dataset_path: str, max_samples: int = 0) -> list[dict]:
    """Load dataset in Messages format.

    Args:
        dataset_path: Path to JSONL file
        max_samples: Maximum samples to load (0 = all)

    Returns:
        List of sample dictionaries
    """
    samples = []

    with open(dataset_path) as f:
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


def resolve_image_path(image_path: str, image_base_dir: str) -> str | None:
    """Resolve relative image path to absolute path.

    Args:
        image_path: Relative or absolute image path
        image_base_dir: Base directory for relative paths

    Returns:
        Absolute path if exists, None otherwise
    """
    if os.path.isabs(image_path):
        full_path = image_path
    else:
        full_path = os.path.join(image_base_dir, image_path)

    if not os.path.exists(full_path):
        logger.warning(f"Image not found: {full_path}")
        return None

    return full_path


def parse_sample(sample: dict) -> tuple[str, list[str], list[str], str | None]:
    """Parse a sample in Messages format.

    Args:
        sample: Sample dictionary with 'messages' and 'images' keys

    Returns:
        Tuple of (question, gt_assistant_responses, images, system_prompt)
    """
    messages = sample.get("messages", [])
    images = sample.get("images", [])

    question = None
    gt_responses = []
    system_prompt = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_prompt = content
        elif role == "user" and question is None:
            # First user message is the question
            question = content
        elif role == "assistant":
            gt_responses.append(content)

    return question, gt_responses, images, system_prompt


# ============================================
# Self-Reflective Generation
# ============================================


def generate_self_reflective_dialogue(
    engine: SelfReflectionEngine,
    sample: dict,
    sample_index: int,
    image_base_dir: str,
    generation_config: dict,
) -> SampleResult | None:
    """Generate a self-reflective dialogue for a sample.

    The model generates:
    - Turn 0: Initial answer using [ANSWER] prompt
    - Turn 1+: Feedback using [FEEDBACK], then refined answer using [REFINE]

    Args:
        engine: Self-reflection inference engine
        sample: Input sample with messages and images
        sample_index: Index of sample in dataset
        image_base_dir: Base directory for image paths
        generation_config: Generation parameters

    Returns:
        SampleResult with generated dialogue, or None if failed
    """
    max_new_tokens = generation_config.get("max_new_tokens", 512)
    temperature = generation_config.get("temperature", 0.7)
    top_p = generation_config.get("top_p", 0.9)

    # Parse sample
    question, gt_responses, images, _ = parse_sample(sample)

    if not question:
        logger.warning(f"Sample {sample_index}: No question found")
        return None

    if not images:
        logger.warning(f"Sample {sample_index}: No images found")
        return None

    if not gt_responses:
        logger.warning(f"Sample {sample_index}: No ground truth responses found")
        return None

    # Resolve image path
    image_path = resolve_image_path(images[0], image_base_dir)
    if not image_path:
        return None

    num_turns = len(gt_responses)
    gt_final_answer = gt_responses[-1]

    # Track generated turns and message history
    generated_turns = []
    messages_history = []

    # Clean question (remove any <image> tag if present)
    clean_question = question.replace("<image>", "").strip()

    for turn_idx in range(num_turns):
        if turn_idx == 0:
            # Turn 0: Generate initial answer with [ANSWER] prompt
            answer_prompt = ANSWER_PROMPT_TEMPLATE.format(question=clean_question)

            # First message includes the image
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": answer_prompt},
                ],
            }
            messages_history.append(user_message)

            # Generate answer
            answer = engine.generate(
                messages=messages_history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # Add assistant response to history
            messages_history.append({"role": "assistant", "content": answer})

            # Store turn (no feedback for turn 0 yet, will be added if there are more turns)
            generated_turns.append({"answer": answer, "feedback": ""})

        else:
            # Turn 1+: Generate feedback on previous answer, then refine

            # Get previous answer
            prev_answer = generated_turns[-1]["answer"]

            # =====================================================
            # Generate feedback using FEEDBACK CRITIC system prompt
            # This matches fire_feedback training format:
            # - System: feedback critic prompt
            # - Assistant: question + image (we simulate by putting in context)
            # - User: answer to evaluate
            # - Assistant: generates feedback
            # =====================================================
            feedback_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": f"{clean_question}\n\n{prev_answer}"},
                    ],
                }
            ]

            # Generate feedback with feedback critic system prompt
            feedback = engine.generate(
                messages=feedback_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                system_prompt=FEEDBACK_CRITIC_SYSTEM_PROMPT,
            )

            # Record in history for logging (show what happened)
            messages_history.append(
                {
                    "role": "user",
                    "content": f"[FEEDBACK REQUEST]\nQuestion: {clean_question}\nAnswer: {prev_answer}",
                }
            )
            messages_history.append({"role": "assistant", "content": feedback})

            # Update previous turn with the feedback
            generated_turns[-1]["feedback"] = feedback

            # =====================================================
            # Generate refined answer using VL ASSISTANT system prompt
            # This matches fire_messages training format:
            # - User: feedback
            # - Assistant: revised answer
            # =====================================================
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(feedback=feedback)

            messages_history.append({"role": "user", "content": refine_prompt})

            refined_answer = engine.generate(
                messages=messages_history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            messages_history.append({"role": "assistant", "content": refined_answer})

            # Store this turn
            generated_turns.append({"answer": refined_answer, "feedback": ""})

    # Get final generated answer
    final_answer = generated_turns[-1]["answer"] if generated_turns else ""

    # Serialize messages history (convert image objects to strings for JSON)
    serializable_history = []
    for msg in messages_history:
        if isinstance(msg.get("content"), list):
            # Convert complex content to string representation
            content_parts = []
            for part in msg["content"]:
                if part.get("type") == "image":
                    content_parts.append(f"[IMAGE: {part.get('image', '')}]")
                elif part.get("type") == "text":
                    content_parts.append(part.get("text", ""))
            serializable_history.append({"role": msg["role"], "content": "\n".join(content_parts)})
        else:
            serializable_history.append(msg)

    return SampleResult(
        sample_index=sample_index,
        image_path=image_path,
        question=clean_question,
        generated_turns=generated_turns,
        final_answer=final_answer,
        gt_final_answer=gt_final_answer,
        num_turns=num_turns,
        messages_history=serializable_history,
    )


# ============================================
# CLI and Main
# ============================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Self-reflective inference with mode-tagged prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint or HuggingFace ID",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save generated results (JSONL)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/fire_preprocessed_v2/fire_messages_test.jsonl",
        help="Path to test dataset (Messages JSONL format)",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs",
        help="Base directory for resolving relative image paths",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )

    # Generation configuration
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature (lower = less hallucination)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability",
    )

    # Model configuration
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Override system prompt (empty string to disable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention",
    )

    return parser.parse_args()


def main():
    """Main function for self-reflective inference."""
    args = parse_args()

    # Initialize engine
    engine = SelfReflectionEngine(
        model_path=args.model_path,
        device=args.device,
        use_flash_attn=not args.no_flash_attn,
        system_prompt=args.system_prompt,
    )

    # Load dataset
    samples = load_dataset(args.dataset_path, args.max_samples)

    # Generation config
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    # Process samples
    results = []
    failed = 0

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, sample in enumerate(tqdm(samples, desc="Self-reflective inference")):
            try:
                result = generate_self_reflective_dialogue(
                    engine=engine,
                    sample=sample,
                    sample_index=i,
                    image_base_dir=args.image_base_dir,
                    generation_config=gen_config,
                )

                if result:
                    f.write(json.dumps(result.to_dict()) + "\n")
                    results.append(result)
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                failed += 1

    # Print summary
    print("\n" + "=" * 60)
    print("SELF-REFLECTIVE INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_path}")

    if results:
        avg_turns = sum(r.num_turns for r in results) / len(results)
        print(f"Average turns per sample: {avg_turns:.2f}")

        # Show a preview of first result
        print("\n--- First Result Preview ---")
        first = results[0]
        print(f"Question: {first.question[:100]}...")
        print(f"Num turns: {first.num_turns}")
        print(f"Final answer: {first.final_answer[:100]}...")
        print(f"GT final answer: {first.gt_final_answer[:100]}...")

    print("=" * 60)


if __name__ == "__main__":
    main()
