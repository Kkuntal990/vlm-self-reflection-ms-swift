#!/usr/bin/env python3
"""
Generate self-refinement responses using fine-tuned VLM.

This script generates multi-turn self-refinement dialogues using a fine-tuned
Qwen2.5-VL model. Given an image and question, it:
1. Generates an initial response
2. Uses teacher feedback (from dataset or generated) to prompt refinement
3. Generates refined responses for N turns
4. Outputs the full conversation for reward scoring

The script supports two feedback modes:
- Ground truth: Uses feedback from the FIRE dataset
- Generated: Uses an LLM to generate feedback (not implemented yet)

Usage:
    python scripts/generate_refinements.py \
        --model_path /outputs/checkpoint-final \
        --dataset_path /outputs/fire_sharegpt_test.jsonl \
        --output_path /outputs/generated_refinements.jsonl \
        --max_turns 3 \
        --max_samples 100

Reference:
    - https://github.com/modelscope/swift (ms-swift framework)
    - FIRE dataset: https://huggingface.co/datasets/PengxiangLi/FIRE
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class VLMInferenceEngine:
    """Inference engine for fine-tuned Qwen2.5-VL model.

    This class handles loading and running inference with a fine-tuned
    Qwen2.5-VL model for generating self-refinement dialogues.

    Attributes:
        model_path: Path to the fine-tuned model checkpoint
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
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to fine-tuned model checkpoint or HuggingFace ID
            device: Device to run inference on
            dtype: Model data type
            use_flash_attn: Whether to use flash attention
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        logger.info(f"Loading model from {model_path}")

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Load model
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

    def generate_response(
        self,
        question: str,
        image_path: str,
        conversation_history: list[dict] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a response for the given question and image.

        Args:
            question: The question or feedback prompt
            image_path: Path to the image file
            conversation_history: Previous conversation turns
                Format: [{"role": "user/assistant", "content": "..."}]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling

        Returns:
            Generated response text
        """
        from qwen_vl_utils import process_vision_info

        # Build messages
        messages = []

        if conversation_history:
            messages.extend(conversation_history)
            # Add current question/feedback
            messages.append({"role": "user", "content": question})
        else:
            # First turn: include image
            clean_question = question.replace("<image>", "").strip()
            first_content = [
                {"type": "image", "image": image_path},
                {"type": "text", "text": clean_question},
            ]
            messages.append({"role": "user", "content": first_content})

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

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


def load_test_dataset(dataset_path: str, max_samples: int = 0) -> list[dict]:
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
            sample = json.loads(line.strip())
            samples.append(sample)

    logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
    return samples


def generate_refinement_dialogue(
    engine: VLMInferenceEngine,
    sample: dict,
    max_turns: int = 3,
    use_gt_feedback: bool = True,
    generation_config: dict | None = None,
) -> dict:
    """Generate a multi-turn self-refinement dialogue.

    Args:
        engine: VLM inference engine
        sample: Input sample with question, image, and optionally ground truth conversation
        max_turns: Maximum number of refinement turns
        use_gt_feedback: Whether to use ground truth feedback from dataset
        generation_config: Optional generation parameters

    Returns:
        Dict with generated conversation and metadata
    """
    config = generation_config or {}
    max_new_tokens = config.get("max_new_tokens", 512)
    temperature = config.get("temperature", 0.7)
    top_p = config.get("top_p", 0.9)

    conversation = sample.get("conversation", [])
    images = sample.get("images", [])

    if not images:
        logger.warning("Sample has no images, skipping")
        return None

    image_path = images[0]
    original_question = conversation[0]["human"] if conversation else ""

    # Build conversation history for generation
    generated_conversation = []
    messages_history = []

    for turn_idx in range(max_turns):
        if turn_idx == 0:
            # First turn: generate initial response
            response = engine.generate_response(
                question=original_question,
                image_path=image_path,
                conversation_history=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            generated_conversation.append(
                {
                    "human": original_question,
                    "assistant": response,
                    "is_generated": True,
                }
            )

            # Update history for next turn
            clean_question = original_question.replace("<image>", "").strip()
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

            # Generate refined response
            response = engine.generate_response(
                question=feedback,
                image_path=image_path,
                conversation_history=messages_history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            generated_conversation.append(
                {
                    "human": feedback,
                    "assistant": response,
                    "is_generated": True,
                    "feedback_source": "ground_truth" if use_gt_feedback else "generated",
                }
            )

            # Update history
            messages_history.append({"role": "user", "content": feedback})
            messages_history.append({"role": "assistant", "content": response})

    # Also keep ground truth for comparison
    gt_conversation = []
    for turn in conversation[:max_turns]:
        gt_conversation.append(
            {
                "human": turn["human"],
                "assistant": turn["assistant"],
                "is_generated": False,
            }
        )

    return {
        "sample_id": sample.get("id", "unknown"),
        "images": images,
        "original_question": original_question,
        "generated_conversation": generated_conversation,
        "ground_truth_conversation": gt_conversation,
        "num_turns": len(generated_conversation),
        "generation_config": config,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate self-refinement dialogues with fine-tuned VLM"
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to test dataset (ShareGPT JSONL format)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save generated conversations",
    )

    # Generation configuration
    parser.add_argument(
        "--max_turns",
        type=int,
        default=3,
        help="Maximum refinement turns to generate",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response",
    )
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

    # Feedback mode
    parser.add_argument(
        "--use_generated_feedback",
        action="store_true",
        help="Use LLM-generated feedback instead of ground truth (not implemented)",
    )

    # Model configuration
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

    return parser.parse_args()


def main():
    """Main function for generating refinement dialogues."""
    args = parse_args()

    # Initialize inference engine
    engine = VLMInferenceEngine(
        model_path=args.model_path,
        device=args.device,
        use_flash_attn=not args.no_flash_attn,
    )

    # Load dataset
    samples = load_test_dataset(args.dataset_path, args.max_samples)

    # Generation config
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    # Generate dialogues
    results = []
    failed = 0

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, sample in enumerate(tqdm(samples, desc="Generating refinements")):
            try:
                result = generate_refinement_dialogue(
                    engine=engine,
                    sample=sample,
                    max_turns=args.max_turns,
                    use_gt_feedback=not args.use_generated_feedback,
                    generation_config=gen_config,
                )

                if result:
                    result["sample_index"] = i
                    f.write(json.dumps(result) + "\n")
                    results.append(result)
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                failed += 1

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Successfully generated: {len(results)}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_path}")

    # Compute statistics
    if results:
        avg_turns = sum(r["num_turns"] for r in results) / len(results)
        print(f"Average turns per sample: {avg_turns:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
