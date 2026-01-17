#!/usr/bin/env python3
"""
Score VLM responses using Skywork-VL-Reward-7B multimodal reward model.

This module provides a reward scoring class that evaluates the quality of
vision-language model responses. It supports scoring individual responses
as well as complete multi-turn self-refinement conversations.

The reward model is based on Qwen2.5-VL-7B-Instruct with a value head,
achieving state-of-the-art performance on VL-RewardBench.

Usage:
    # As a module
    from score_with_reward_model import SkyworkVLRewardScorer

    scorer = SkyworkVLRewardScorer()
    score = scorer.score_response(
        question="What is in this image?",
        response="The image shows a cat.",
        image_path="/path/to/image.jpg"
    )

    # As a script (for testing)
    python scripts/score_with_reward_model.py \
        --image_path /path/to/image.jpg \
        --question "What is in this image?" \
        --response "The image shows a cat."

Reference:
    - https://huggingface.co/Skywork/Skywork-VL-Reward-7B
    - https://arxiv.org/html/2505.07263v1
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class SkyworkVLRewardScorer:
    """Reward scorer using Skywork-VL-Reward-7B model.

    This class wraps the Skywork-VL-Reward-7B model to provide an easy-to-use
    interface for scoring vision-language model responses. The model evaluates
    responses based on helpfulness, accuracy, and relevance to the visual context.

    Attributes:
        model_id: HuggingFace model identifier
        device: Device to run inference on
        dtype: Model data type (bfloat16 recommended)
        model: The loaded model with value head
        processor: The model's processor for tokenization
    """

    def __init__(
        self,
        model_id: str = "Skywork/Skywork-VL-Reward-7B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attn: bool = True,
    ):
        """Initialize the reward model.

        Args:
            model_id: HuggingFace model identifier for Skywork-VL-Reward
            device: Device to load model on ("cuda" or "cpu")
            dtype: Model dtype (torch.bfloat16 recommended for A100)
            use_flash_attn: Whether to use flash attention 2 (requires compatible GPU)
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        logger.info(f"Initializing Skywork-VL-Reward scorer from {model_id}")

        # Lazy import to avoid loading heavy libraries until needed
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from transformers.utils import cached_file
        from trl import AutoModelForCausalLMWithValueHead
        from safetensors import safe_open

        # Load processor
        logger.info("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Load base model
        logger.info("Loading base model...")
        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        try:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(f"Flash attention failed ({e}), falling back to eager")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=dtype,
                attn_implementation="eager",
            )

        # Add value head for reward scoring
        logger.info("Adding value head...")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

        # Load value head weights
        logger.info("Loading value head weights...")
        vhead_file = cached_file(
            path_or_repo_id=model_id,
            filename="value_head.safetensors"
        )
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            vhead_params = {key: f.get_tensor(key) for key in f.keys()}
        self.model.load_state_dict(vhead_params, strict=False)

        # Set to eval mode
        self.model.requires_grad_(False)
        self.model.eval()

        logger.info("Reward model initialized successfully")

    def _build_messages(
        self,
        question: str,
        response: str,
        image_path: str,
        context_history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Build message list in Qwen2.5-VL format.

        Args:
            question: The original visual question
            response: The assistant's response to score
            image_path: Path to the image file
            context_history: Optional list of previous conversation turns
                Format: [{"human": "...", "assistant": "..."}, ...]

        Returns:
            List of messages in Qwen2.5-VL chat format
        """
        messages = []

        # Clean the question (remove <image> placeholder if present)
        clean_question = question.replace("<image>", "").strip()

        # First message includes the image
        first_content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": clean_question},
        ]

        if context_history and len(context_history) > 0:
            # Build conversation with history
            messages.append({"role": "user", "content": first_content})
            messages.append({"role": "assistant", "content": context_history[0]["assistant"]})

            # Add subsequent turns
            for i in range(1, len(context_history)):
                turn = context_history[i]
                # The "human" field contains feedback
                messages.append({"role": "user", "content": turn["human"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})

            # Add the current response to score
            # The feedback leading to this response should be in the last context entry
            if "feedback" in context_history[-1] and context_history[-1]["feedback"]:
                messages.append({"role": "user", "content": context_history[-1]["feedback"]})

            messages.append({"role": "assistant", "content": response})
        else:
            # First turn - no history
            messages.append({"role": "user", "content": first_content})
            messages.append({"role": "assistant", "content": response})

        return messages

    def score_response(
        self,
        question: str,
        response: str,
        image_path: str,
        context_history: Optional[List[Dict]] = None,
    ) -> float:
        """Score a single response.

        Args:
            question: The visual question
            response: The assistant's response to score
            image_path: Path to the image file
            context_history: Optional list of previous turns for multi-turn context
                Format: [{"human": "question/feedback", "assistant": "response", "feedback": "next_feedback"}, ...]

        Returns:
            Reward score (higher is better, unbounded float)
        """
        # Import vision processing utility
        from qwen_vl_utils import process_vision_info

        # Build messages
        messages = self._build_messages(question, response, image_path, context_history)

        # Process inputs using the chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move tensors to a compatible device. When device_map="auto" is used, the model may be sharded;
        # sending inputs to the first parameter device is the safest default.
        try:
            target_device = next(self.model.parameters()).device
        except StopIteration:
            target_device = torch.device(self.device)

        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Get reward score from value head
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True, use_cache=False)

            # TRL value-head models may return values as either an attribute or the last tuple item.
            values = None
            if hasattr(outputs, "value"):
                values = outputs.value
            elif isinstance(outputs, (tuple, list)):
                values = outputs[-1]
            else:
                # Fallback: try common key name
                values = getattr(outputs, "values", None)

            if values is None:
                raise RuntimeError(
                    "Could not extract value head output from model forward pass. "
                    "Inspect the returned `outputs` object to locate the value tensor."
                )

            # Values are typically [bsz, seq_len] or [bsz, seq_len, 1]
            if values.dim() == 3 and values.size(-1) == 1:
                values = values.squeeze(-1)

            if values.dim() != 2:
                raise RuntimeError(f"Unexpected value head tensor shape: {tuple(values.shape)}")

            # Last non-padding token index per sequence
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # [bsz]
            seq_lengths = torch.clamp(seq_lengths, min=0)

            batch_idx = torch.arange(values.size(0), device=values.device)
            scores = values[batch_idx, seq_lengths]  # [bsz]
            score = float(scores[0].item())

        return score

    def score_conversation_turns(
        self,
        sample: Dict,
        return_details: bool = False,
        isolated_scoring: bool = False,
    ) -> Union[List[Dict], Dict]:
        """Score all turns in a multi-turn conversation.

        This method processes a ShareGPT format sample and scores each
        assistant response in the conversation.

        Args:
            sample: ShareGPT format sample with keys:
                - "conversation": List of {"human": "...", "assistant": "..."} dicts
                - "images": List of image paths (uses first image)
                - Optional "system": System prompt (ignored for scoring)
            return_details: If True, return full result dict with metrics
            isolated_scoring: If True, score each response in isolation
                (just image + original question + response, ignoring conversation
                history). This gives a cleaner comparison of response quality
                without context length effects. If False (default), builds up
                context progressively to evaluate responses given prior feedback.

        Returns:
            If return_details=False: List of turn results
                [{"turn_index": 0, "response": "...", "reward_score": 15.2}, ...]
            If return_details=True: Dict with turns and computed metrics
        """
        conversation = sample.get("conversation", [])
        images = sample.get("images", [])

        if not conversation:
            logger.warning("Empty conversation in sample")
            return [] if not return_details else {"turns": [], "metrics": {}}

        if not images:
            logger.warning("No images in sample")
            return [] if not return_details else {"turns": [], "metrics": {}}

        image_path = images[0]

        # Get original question from first turn
        original_question = conversation[0]["human"]

        results = []
        context_history = []

        for turn_idx, turn in enumerate(conversation):
            response = turn["assistant"]

            if isolated_scoring:
                # Isolated mode: score each response with just image + original question
                # No conversation history, giving a clean comparison of response quality
                score = self.score_response(
                    question=original_question,
                    response=response,
                    image_path=image_path,
                    context_history=None,
                )
            elif turn_idx == 0:
                # First turn: just question + response
                score = self.score_response(
                    question=original_question,
                    response=response,
                    image_path=image_path,
                    context_history=None,
                )
            else:
                # Subsequent turns: include history (contextual scoring)
                score = self.score_response(
                    question=original_question,
                    response=response,
                    image_path=image_path,
                    context_history=context_history,
                )

            result = {
                "turn_index": turn_idx,
                "response": response[:200] + "..." if len(response) > 200 else response,
                "reward_score": score,
            }

            if turn_idx > 0:
                result["feedback_received"] = turn["human"][:100] + "..." if len(turn["human"]) > 100 else turn["human"]
                result["score_delta"] = score - results[-1]["reward_score"]

            results.append(result)

            # Update context history for next turn
            next_feedback = conversation[turn_idx + 1]["human"] if turn_idx + 1 < len(conversation) else None
            context_history.append({
                "human": turn["human"],
                "assistant": response,
                "feedback": next_feedback,
            })

        if not return_details:
            return results

        # Compute per-sample metrics
        scores = [r["reward_score"] for r in results]
        metrics = {
            "num_turns": len(results),
            "initial_score": scores[0],
            "final_score": scores[-1],
            "absolute_improvement": scores[-1] - scores[0],  # kept for backward compatibility
            "score_delta": scores[-1] - scores[0],
            "reward_delta": (scores[-1] - scores[0]) / max(abs(scores[0]), 1.0),
            "is_monotonic": all(scores[i] <= scores[i+1] for i in range(len(scores)-1)),
            "improvements_per_turn": [scores[i+1] - scores[i] for i in range(len(scores)-1)],
            "scoring_mode": "isolated" if isolated_scoring else "contextual",
        }

        return {"turns": results, "metrics": metrics}


def parse_args():
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(
        description="Score VLM responses using Skywork-VL-Reward-7B"
    )

    # Single response scoring
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to image file for single response scoring",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question for single response scoring",
    )
    parser.add_argument(
        "--response",
        type=str,
        help="Response to score",
    )

    # Conversation scoring from file
    parser.add_argument(
        "--sample_file",
        type=str,
        help="Path to JSON file with ShareGPT format sample",
    )

    # Model configuration
    parser.add_argument(
        "--model_id",
        type=str,
        default="Skywork/Skywork-VL-Reward-7B",
        help="Reward model HuggingFace ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention",
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Score each response in isolation (just image + question + response, "
             "ignoring conversation history). Provides cleaner comparison without "
             "context length effects.",
    )

    return parser.parse_args()


def main():
    """Main function for testing the reward scorer."""
    args = parse_args()

    # Initialize scorer
    scorer = SkyworkVLRewardScorer(
        model_id=args.model_id,
        device=args.device,
        use_flash_attn=not args.no_flash_attn,
    )

    if args.sample_file:
        # Score a full conversation from file
        logger.info(f"Loading sample from {args.sample_file}")
        with open(args.sample_file, "r") as f:
            sample = json.load(f)

        scoring_mode = "isolated" if args.isolated else "contextual"
        logger.info(f"Scoring mode: {scoring_mode}")

        result = scorer.score_conversation_turns(
            sample, return_details=True, isolated_scoring=args.isolated
        )

        print("\n" + "=" * 60)
        print("CONVERSATION SCORING RESULTS")
        print(f"Scoring Mode: {scoring_mode.upper()}")
        print("=" * 60)

        for turn in result["turns"]:  # type: ignore[index]
            print(f"\nTurn {turn['turn_index']}:")
            print(f"  Response: {turn['response']}")
            print(f"  Score: {turn['reward_score']:.2f}")
            if "score_delta" in turn:
                delta_sign = "+" if turn["score_delta"] >= 0 else ""
                print(f"  Delta: {delta_sign}{turn['score_delta']:.2f}")

        print("\n" + "-" * 60)
        print("METRICS:")
        metrics = result["metrics"]  # type: ignore[index]
        print(f"  Initial Score: {metrics['initial_score']:.2f}")
        print(f"  Final Score: {metrics['final_score']:.2f}")
        print(f"  Absolute Improvement: {metrics['absolute_improvement']:.2f}")
        print(f"  Score Delta (Final - Initial): {metrics['score_delta']:.2f}")
        print(f"  Reward Delta (normalized): {metrics['reward_delta']*100:.1f}%")
        print(f"  Monotonic Improvement: {metrics['is_monotonic']}")
        print("=" * 60)

    elif args.image_path and args.question and args.response:
        # Score a single response
        score = scorer.score_response(
            question=args.question,
            response=args.response,
            image_path=args.image_path,
        )

        print("\n" + "=" * 60)
        print("SINGLE RESPONSE SCORING")
        print("=" * 60)
        print(f"Question: {args.question}")
        print(f"Response: {args.response}")
        print(f"Score: {score:.2f}")
        print("=" * 60)

    else:
        logger.error("Must provide either --sample_file or (--image_path, --question, --response)")
        sys.exit(1)


if __name__ == "__main__":
    main()
