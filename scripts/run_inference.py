#!/usr/bin/env python3
"""
MS-SWIFT compatible inference script for VLM self-refinement.

This script provides a standalone CLI for running inference on the FIRE test dataset
using any Qwen2.5-VL checkpoint (base model or fine-tuned).

Two Generation Modes:
1. **Continuation Mode**: Given partial conversation (question + image + responses + feedbacks),
   generate only the final response. Output includes full conversation history up to generation.
2. **Full Generation Mode**: Given only question + image, generate all assistant responses
   using ground-truth feedback from the dataset.

Usage:
    # Mode 1: Continue from partial conversation (generate final response only)
    python scripts/run_inference.py \
        --model_path /outputs/checkpoint-final \
        --dataset_path /outputs/fire_sharegpt_test.jsonl \
        --image_base_dir /outputs/fire_images_v2 \
        --mode continuation \
        --output_path /outputs/inference_results.jsonl

    # Mode 2: Generate all responses from scratch
    python scripts/run_inference.py \
        --model_path Qwen/Qwen2.5-VL-7B-Instruct \
        --dataset_path /outputs/fire_sharegpt_test.jsonl \
        --image_base_dir /outputs/fire_images_v2 \
        --mode full_generation \
        --max_turns 3 \
        --output_path /outputs/inference_results.jsonl

Reference:
    - https://github.com/modelscope/swift (ms-swift framework)
    - FIRE dataset: https://huggingface.co/datasets/PengxiangLi/FIRE
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Default system prompt for self-refinement tasks
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. Answer questions about images accurately. "
    "When given feedback about your previous response, carefully analyze the feedback and "
    "provide an improved, more accurate response that addresses the issues raised."
)


class VLMInference:
    """Inference engine for Qwen2.5-VL models.

    This class handles loading and running inference with Qwen2.5-VL models
    for generating self-refinement responses.

    Attributes:
        model_path: Path to the model checkpoint or HuggingFace ID
        model: The loaded model
        processor: The model's processor
        device: Device for inference
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
            model_path: Path to model checkpoint or HuggingFace ID
            device: Device to run inference on
            dtype: Model data type
            use_flash_attn: Whether to use flash attention
            system_prompt: System prompt to use. If None, uses DEFAULT_SYSTEM_PROMPT.
                          Pass empty string "" to disable system prompt.
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        # Set system prompt (None means use default, "" means no system prompt)
        if system_prompt is None:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
        else:
            self.system_prompt = system_prompt if system_prompt else None

        logger.info(f"Loading model from {model_path}")

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Load model with flash attention if available
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
        if self.system_prompt:
            logger.info(f"System prompt: {self.system_prompt[:100]}...")
        else:
            logger.info("No system prompt configured")

    def generate(
        self,
        messages: list[dict],
        generation_config: dict[str, Any] | None = None,
    ) -> str:
        """Generate a response given messages in chat format.

        Args:
            messages: List of message dicts with 'role' and 'content'
            generation_config: Optional generation parameters

        Returns:
            Generated response text
        """
        from qwen_vl_utils import process_vision_info

        config = generation_config or {}
        max_new_tokens = config.get("max_new_tokens", 512)
        temperature = config.get("temperature", 0.7)
        top_p = config.get("top_p", 0.9)
        do_sample = config.get("do_sample", True)

        # Prepend system message if configured
        if self.system_prompt:
            full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        else:
            full_messages = messages

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

    def generate_continuation(
        self,
        sample: dict,
        image_base_dir: str,
        generation_config: dict[str, Any] | None = None,
    ) -> dict | None:
        """Generate final response from partial conversation (Mode 1).

        Given a complete conversation from the dataset, feed all turns EXCEPT
        the last assistant response, then generate only the final response.
        This tests whether the model can produce a good final response given
        full conversation context.

        For a conversation with N turns:
        - Feed: question + image + response1 + feedback1 + ... + response(N-1) + feedback(N-1)
        - Generate: response N (the last turn's response)

        Args:
            sample: Sample dict with 'conversation' and 'images' keys
                Format: {
                    "conversation": [
                        {"human": "question", "assistant": "response1"},
                        {"human": "feedback1", "assistant": "response2"},
                        {"human": "feedback2", "assistant": "response3"}  # GT response3 NOT fed
                    ],
                    "images": ["path/to/image.jpg"]
                }
            image_base_dir: Base directory for resolving relative image paths
            generation_config: Optional generation parameters

        Returns:
            Dict with generated response and full conversation context:
            {
                "sample_id": str,
                "mode": "continuation",
                "image_path": str,
                "conversation_history": [...],  # History up to generation (no GT for last turn)
                "final_prompt": str,  # The feedback that triggered generation
                "generated_response": str,
                "ground_truth_response": str,  # GT response for comparison
                "context_turns": int,  # Number of complete turns before generation
                "generation_config": {...}
            }
        """
        conversation = sample.get("conversation", [])
        images = sample.get("images", [])
        sample_id = sample.get("id", f"sample_{sample.get('sample_index', 'unknown')}")

        if not images:
            logger.warning(f"Sample {sample_id} has no images, skipping")
            return None

        if not conversation:
            logger.warning(f"Sample {sample_id} has no conversation, skipping")
            return None

        if len(conversation) < 1:
            logger.warning(f"Sample {sample_id} has empty conversation, skipping")
            return None

        # Resolve image path
        image_path = images[0]
        if not os.path.isabs(image_path):
            image_path = os.path.join(image_base_dir, image_path)

        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}, skipping sample {sample_id}")
            return None

        # Build messages from conversation history
        # We feed ALL turns EXCEPT the last assistant response
        messages = []
        conversation_history = []  # Human-readable history for output
        context_turns = 0
        final_prompt = None
        ground_truth_response = None

        num_turns = len(conversation)

        for i, turn in enumerate(conversation):
            human_content = turn.get("human", "")
            is_last_turn = i == num_turns - 1

            # Add the human/user message (question or feedback)
            if i == 0:
                # First turn includes image
                clean_question = human_content.replace("<image>", "").strip()
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": clean_question},
                    ],
                })
                conversation_history.append({
                    "role": "user",
                    "content": clean_question,
                    "has_image": True,
                })
            else:
                # Subsequent turns are text-only (feedback)
                messages.append({"role": "user", "content": human_content})
                conversation_history.append({
                    "role": "user",
                    "content": human_content,
                    "has_image": False,
                })

            # Add assistant response ONLY if this is NOT the last turn
            # For the last turn, we want to GENERATE the response, not feed GT
            if is_last_turn:
                # This is the turn we need to generate
                final_prompt = human_content
                # Save ground truth for comparison
                ground_truth_response = turn.get("assistant", "")
            else:
                # Feed the GT assistant response for earlier turns
                assistant_response = turn.get("assistant", "")
                if assistant_response:
                    messages.append({"role": "assistant", "content": assistant_response})
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_response,
                    })
                    context_turns += 1

        # Generate the response for the last turn
        config = generation_config or {}
        response = self.generate(messages, config)

        # Add generated response to history
        conversation_history.append({
            "role": "assistant",
            "content": response,
            "is_generated": True,
        })

        return {
            "sample_id": sample_id,
            "mode": "continuation",
            "image_path": image_path,
            "conversation_history": conversation_history,
            "final_prompt": final_prompt,
            "generated_response": response,
            "ground_truth_response": ground_truth_response,
            "context_turns": context_turns,
            "total_turns": num_turns,
            "generation_config": config,
        }

    def generate_full_dialogue(
        self,
        sample: dict,
        image_base_dir: str,
        max_turns: int = 3,
        use_gt_feedback: bool = True,
        generation_config: dict[str, Any] | None = None,
    ) -> dict | None:
        """Generate all assistant responses from scratch (Mode 2).

        Given a question and image, generate all assistant responses using
        ground-truth feedback from the dataset.

        Args:
            sample: Sample dict with 'conversation' and 'images' keys
            image_base_dir: Base directory for resolving relative image paths
            max_turns: Maximum number of refinement turns
            use_gt_feedback: Whether to use ground truth feedback
            generation_config: Optional generation parameters

        Returns:
            Dict with generated conversation:
            {
                "sample_id": str,
                "mode": "full_generation",
                "image_path": str,
                "generated_conversation": [...],
                "ground_truth_conversation": [...],
                "num_turns": int,
                "generation_config": {...}
            }
        """
        conversation = sample.get("conversation", [])
        images = sample.get("images", [])
        sample_id = sample.get("id", f"sample_{sample.get('sample_index', 'unknown')}")

        if not images:
            logger.warning(f"Sample {sample_id} has no images, skipping")
            return None

        if not conversation:
            logger.warning(f"Sample {sample_id} has no conversation, skipping")
            return None

        # Resolve image path
        image_path = images[0]
        if not os.path.isabs(image_path):
            image_path = os.path.join(image_base_dir, image_path)

        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}, skipping sample {sample_id}")
            return None

        config = generation_config or {}
        original_question = conversation[0]["human"] if conversation else ""
        clean_question = original_question.replace("<image>", "").strip()

        # Build conversation history for generation
        generated_conversation = []
        messages_history = []

        for turn_idx in range(max_turns):
            if turn_idx == 0:
                # First turn: generate initial response
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": clean_question},
                    ],
                }]

                response = self.generate(messages, config)

                generated_conversation.append({
                    "human": original_question,
                    "assistant": response,
                    "is_generated": True,
                })

                # Update history for next turn
                messages_history = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": clean_question},
                        ],
                    },
                    {"role": "assistant", "content": response},
                ]
            else:
                # Subsequent turns: need feedback first
                if use_gt_feedback and turn_idx < len(conversation):
                    # Use ground truth feedback from dataset
                    feedback = conversation[turn_idx]["human"]
                else:
                    # No more ground truth feedback available
                    logger.debug(f"No ground truth feedback for turn {turn_idx}, stopping")
                    break

                # Add feedback to history and generate
                messages_history.append({"role": "user", "content": feedback})
                response = self.generate(messages_history, config)

                generated_conversation.append({
                    "human": feedback,
                    "assistant": response,
                    "is_generated": True,
                    "feedback_source": "ground_truth" if use_gt_feedback else "generated",
                })

                # Update history
                messages_history.append({"role": "assistant", "content": response})

        # Keep ground truth for comparison
        gt_conversation = []
        for turn in conversation[:max_turns]:
            gt_conversation.append({
                "human": turn.get("human", ""),
                "assistant": turn.get("assistant", ""),
                "is_generated": False,
            })

        return {
            "sample_id": sample_id,
            "mode": "full_generation",
            "image_path": image_path,
            "original_question": clean_question,
            "generated_conversation": generated_conversation,
            "ground_truth_conversation": gt_conversation,
            "num_turns": len(generated_conversation),
            "generation_config": config,
        }


def load_dataset(dataset_path: str, max_samples: int = 0) -> list[dict]:
    """Load test dataset in ShareGPT format.

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
                sample["sample_index"] = i
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i}: {e}")

    logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
    return samples


def save_results(results: list[dict], output_path: str):
    """Save results to JSONL file.

    Args:
        results: List of result dictionaries
        output_path: Path to output file
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Saved {len(results)} results to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MS-SWIFT compatible inference for VLM self-refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Continuation mode (generate final response from partial history)
  python scripts/run_inference.py \\
      --model_path Qwen/Qwen2.5-VL-7B-Instruct \\
      --dataset_path /outputs/fire_sharegpt_test.jsonl \\
      --mode continuation \\
      --output_path /outputs/continuation_results.jsonl

  # Full generation mode (generate all responses)
  python scripts/run_inference.py \\
      --model_path /outputs/checkpoint-final \\
      --dataset_path /outputs/fire_sharegpt_test.jsonl \\
      --mode full_generation \\
      --max_turns 3 \\
      --output_path /outputs/full_gen_results.jsonl
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model ID or local checkpoint path",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to FIRE ShareGPT test JSONL file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save inference results (JSONL)",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["continuation", "full_generation"],
        required=True,
        help="Inference mode: 'continuation' (generate final response) or "
        "'full_generation' (generate all responses)",
    )

    # Image directory
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs/fire_images_v2",
        help="Base directory for resolving relative image paths",
    )

    # Full generation specific
    parser.add_argument(
        "--max_turns",
        type=int,
        default=3,
        help="Maximum turns for full_generation mode",
    )
    parser.add_argument(
        "--use_generated_feedback",
        action="store_true",
        help="Use generated feedback instead of ground truth (not implemented)",
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Use sampling for generation",
    )

    # Processing
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )

    # Hardware configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention",
    )

    # System prompt configuration
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt. If not specified, uses default self-refinement prompt. "
        "Pass empty string '' to disable system prompt entirely.",
    )
    parser.add_argument(
        "--no_system_prompt",
        action="store_true",
        help="Disable system prompt entirely (equivalent to --system_prompt '')",
    )

    return parser.parse_args()


def main():
    """Main function for running inference."""
    args = parse_args()

    # Determine system prompt
    if args.no_system_prompt:
        system_prompt = ""  # Empty string disables system prompt
    else:
        system_prompt = args.system_prompt  # None uses default, string uses custom

    # Initialize inference engine
    engine = VLMInference(
        model_path=args.model_path,
        device=args.device,
        use_flash_attn=not args.no_flash_attn,
        system_prompt=system_prompt,
    )

    # Load dataset
    samples = load_dataset(args.dataset_path, args.max_samples)

    if not samples:
        logger.error("No samples loaded, exiting")
        sys.exit(1)

    # Generation config
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
    }

    # Run inference
    results = []
    failed = 0

    logger.info(f"Running inference in '{args.mode}' mode on {len(samples)} samples")

    skipped_reasons = {"no_image": 0, "no_conversation": 0, "image_not_found": 0, "error": 0}

    for sample in tqdm(samples, desc=f"Inference ({args.mode})"):
        try:
            if args.mode == "continuation":
                result = engine.generate_continuation(
                    sample=sample,
                    image_base_dir=args.image_base_dir,
                    generation_config=gen_config,
                )
            else:  # full_generation
                result = engine.generate_full_dialogue(
                    sample=sample,
                    image_base_dir=args.image_base_dir,
                    max_turns=args.max_turns,
                    use_gt_feedback=not args.use_generated_feedback,
                    generation_config=gen_config,
                )

            if result:
                result["sample_index"] = sample.get("sample_index", len(results))
                results.append(result)
            else:
                # Track why sample was skipped (None returned)
                failed += 1
                # Determine reason from sample data
                if not sample.get("images"):
                    skipped_reasons["no_image"] += 1
                elif not sample.get("conversation"):
                    skipped_reasons["no_conversation"] += 1
                else:
                    skipped_reasons["image_not_found"] += 1

            # Periodic saving for long runs
            if len(results) > 0 and len(results) % 100 == 0:
                partial_path = args.output_path + ".partial"
                save_results(results, partial_path)
                logger.info(f"Checkpoint saved: {len(results)} results")

        except Exception as e:
            logger.error(f"Failed to process sample {sample.get('sample_index', '?')}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            skipped_reasons["error"] += 1

    # Save final results
    save_results(results, args.output_path)

    # Clean up partial file if exists
    partial_path = args.output_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_path}")
    print(f"Total samples: {len(samples)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Skipped/Failed: {failed}")

    if failed > 0:
        print("\nSkip reasons:")
        for reason, count in skipped_reasons.items():
            if count > 0:
                print(f"  - {reason}: {count}")

    print(f"\nOutput saved to: {args.output_path}")

    if results:
        if args.mode == "continuation":
            avg_context = sum(r.get("context_turns", 0) for r in results) / len(results)
            print(f"Average context turns: {avg_context:.2f}")
        else:
            avg_turns = sum(r.get("num_turns", 0) for r in results) / len(results)
            print(f"Average turns generated: {avg_turns:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
