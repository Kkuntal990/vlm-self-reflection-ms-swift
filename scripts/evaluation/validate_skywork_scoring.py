#!/usr/bin/env python3
"""
Validate Skywork-VL-Reward-7B scoring against model card example.

This script reproduces the exact example from the Skywork model card to verify
that our score extraction method produces correct results.

Reference: https://huggingface.co/Skywork/Skywork-VL-Reward-7B
"""

import argparse
import json
import logging
import os
import sys

import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def score_with_model_card_method(model, processor, messages):
    """Score using EXACTLY the method from Skywork model card.

    This is the reference implementation from:
    https://huggingface.co/Skywork/Skywork-VL-Reward-7B
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Model card exact code:
    # values = model(**inputs, return_dict=True, use_cache=False)[-1]
    # scores = values.gather(dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1))
    # score = scores[0].item()

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, use_cache=False)
        values = outputs[-1]  # Model card method

        # Model card gather method
        scores = values.gather(
            dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1)
        )
        score = scores[0].item()

    return score


def score_with_our_method(model, processor, messages):
    """Score using OUR implementation method from score_with_reward_model.py."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, use_cache=False)

        # Our extraction method
        values = None
        if hasattr(outputs, "value"):
            values = outputs.value
        elif isinstance(outputs, (tuple, list)):
            values = outputs[-1]

        # Handle shape
        if values.dim() == 3 and values.size(-1) == 1:
            values = values.squeeze(-1)

        # Our indexing method
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
        seq_lengths = torch.clamp(seq_lengths, min=0)
        batch_idx = torch.arange(values.size(0), device=values.device)
        scores = values[batch_idx, seq_lengths]
        score = float(scores[0].item())

    return score


def test_with_sample(
    model, processor, sample_path: str, image_base_dir: str = None, max_samples: int = 0
):
    """Test scoring with a real FIRE sample."""
    logger.info(f"Loading sample from {sample_path}")

    # Find first sample with valid image
    sample = None
    image_path = None
    samples_checked = 0

    with open(sample_path) as f:
        for line in f:
            samples_checked += 1
            if max_samples > 0 and samples_checked > max_samples:
                logger.error(f"Checked {max_samples} samples, none with valid images")
                return

            candidate = json.loads(line.strip())
            conversation = candidate.get("conversation", [])
            images = candidate.get("images", [])

            if not conversation or not images:
                continue

            img_path = images[0]
            if image_base_dir and not os.path.isabs(img_path):
                img_path = os.path.join(image_base_dir, img_path)

            if os.path.exists(img_path):
                sample = candidate
                image_path = img_path
                logger.info(f"Found valid sample at index {samples_checked - 1}")
                break
            else:
                logger.debug(
                    f"Skipping sample {samples_checked - 1}: image not found at {img_path}"
                )

    if sample is None:
        logger.error("No sample with valid image found")
        return

    conversation = sample.get("conversation", [])

    original_question = conversation[0]["human"]

    # Clean question
    clean_question = original_question.replace("<image>", "").strip()

    print("\n" + "=" * 70)
    print("TESTING WITH FIRE SAMPLE")
    print("=" * 70)
    print(f"Image path: {image_path}")
    print(f"Question: {clean_question[:100]}...")
    print(f"Number of turns: {len(conversation)}")
    print()

    # Score each turn
    prev_score = None
    for turn_idx, turn in enumerate(conversation):
        response = turn["assistant"]

        # Build messages for this turn (isolated scoring - just question + response)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": clean_question},
                ],
            },
            {"role": "assistant", "content": response},
        ]

        # Print exact message structure being sent
        print(f"Turn {turn_idx}:")
        print("  Messages structure:")
        print(
            f"    [0] role: 'user', content: [image: '{image_path}', text: '{clean_question[:50]}...']"
        )
        print(f"    [1] role: 'assistant', content: '{response}'")

        try:
            score_modelcard = score_with_model_card_method(model, processor, messages)
            score_ours = score_with_our_method(model, processor, messages)

            print(f"  Model Card Method Score: {score_modelcard:.4f}")
            print(f"  Our Method Score:        {score_ours:.4f}")
            print(f"  Difference:              {abs(score_modelcard - score_ours):.6f}")

            if prev_score is not None:
                delta = score_modelcard - prev_score
                print(f"  Score Delta from prev:   {'+' if delta >= 0 else ''}{delta:.4f}")

            prev_score = score_modelcard

            if turn_idx > 0:
                # Show the feedback that led to this response
                print(f"  (Feedback: {turn['human'][:80]}...)")
            print()

        except Exception as e:
            logger.error(f"Error scoring turn {turn_idx}: {e}")
            import traceback

            traceback.print_exc()


def test_synthetic_example(model, processor):
    """Test with a simple synthetic example to verify basic functionality."""
    print("\n" + "=" * 70)
    print("SYNTHETIC EXAMPLE TEST")
    print("=" * 70)

    # Note: Synthetic tests require actual image files to work.
    # Use --sample_path with real FIRE data for accurate testing.
    print("Note: Synthetic tests require actual image files to work.")
    print("Use --sample_path with real FIRE data for accurate testing.")


def main():
    parser = argparse.ArgumentParser(description="Validate Skywork-VL-Reward-7B scoring")
    parser.add_argument(
        "--sample_path",
        type=str,
        help="Path to JSONL file with test samples",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs/fire_images_v2",
        help="Base directory for resolving relative image paths",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to check for valid images (0 = unlimited)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Skywork/Skywork-VL-Reward-7B",
        help="Reward model HuggingFace ID",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention",
    )
    args = parser.parse_args()

    # Load model using TRL (same as our implementation)
    logger.info(f"Loading model from {args.model_id}")

    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.utils import cached_file
    from trl import AutoModelForCausalLMWithValueHead

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Load base model
    attn_impl = "eager" if args.no_flash_attn else "flash_attention_2"
    try:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        logger.warning(f"Flash attention failed ({e}), falling back to eager")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

    # Add value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

    # Load value head weights
    vhead_file = cached_file(path_or_repo_id=args.model_id, filename="value_head.safetensors")
    with safe_open(vhead_file, framework="pt", device="cpu") as f:
        vhead_params = {key: f.get_tensor(key) for key in f}
    model.load_state_dict(vhead_params, strict=False)

    model.requires_grad_(False)
    model.eval()

    logger.info("Model loaded successfully")

    # Debug: Check model output structure
    print("\n" + "=" * 70)
    print("MODEL OUTPUT STRUCTURE DEBUG")
    print("=" * 70)

    # Create a minimal test input
    test_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    text = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, use_cache=False)

        print(f"Output type: {type(outputs)}")
        if isinstance(outputs, (tuple, list)):
            print(f"Output length: {len(outputs)}")
            for i, o in enumerate(outputs):
                if hasattr(o, "shape"):
                    print(f"  outputs[{i}] shape: {o.shape}")
                elif hasattr(o, "logits"):
                    print(f"  outputs[{i}].logits shape: {o.logits.shape}")
                else:
                    print(f"  outputs[{i}] type: {type(o)}")

        if hasattr(outputs, "value"):
            print(f"outputs.value shape: {outputs.value.shape}")

        # The last element should be values
        values = outputs[-1]
        print(f"\nValues tensor (outputs[-1]) shape: {values.shape}")
        print(f"Values tensor dtype: {values.dtype}")
        print(f"Values sample: {values[0, -5:].tolist()}")

    # Run tests
    if args.sample_path:
        test_with_sample(model, processor, args.sample_path, args.image_base_dir, args.max_samples)
    else:
        test_synthetic_example(model, processor)
        print("\nTo test with real data, run:")
        print("  python scripts/validate_skywork_scoring.py \\")
        print("    --sample_path /outputs/fire_preprocessed_v2/fire_sharegpt_test.jsonl \\")
        print("    --image_base_dir /outputs/fire_images_v2")


if __name__ == "__main__":
    main()
