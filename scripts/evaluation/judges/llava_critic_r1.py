#!/usr/bin/env python3
"""
LLaVA-Critic-R1 VLM Judge Implementation.

This module provides the LLaVA-Critic-R1 judge, the SOTA VLM-as-Judge model
based on Qwen2.5-VL architecture, trained using GRPO for critic capability.

Models:
    - lmms-lab/LLaVA-Critic-R1-7B (8B params, SOTA)
    - lmms-lab/llava-critic-7b (7B params, fallback)

References:
    - Paper: https://arxiv.org/abs/2509.00676
    - HuggingFace: https://huggingface.co/lmms-lab/LLaVA-Critic-R1-7B

Usage:
    from judges.llava_critic_r1 import LLaVACriticR1Judge

    judge = LLaVACriticR1Judge(device="cuda")
    result = judge.judge_pairwise(
        image_path="path/to/image.jpg",
        question="What color is the bear?",
        feedback="Look more carefully at the image.",
        answer_a="gray",
        answer_b="white",
        debias=True
    )
"""

import logging
import re
import sys

import torch


# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from vlm_judge import BaseVLMJudge, PairwiseResult, PointwiseResult, register_judge


logger = logging.getLogger(__name__)


@register_judge("llava_critic_r1")
class LLaVACriticR1Judge(BaseVLMJudge):
    """LLaVA-Critic-R1 judge (SOTA, based on Qwen2.5-VL).

    This judge uses the LLaVA-Critic-R1-7B model, which is trained using
    GRPO (Group Relative Policy Optimization) on pairwise critic data.
    It achieves SOTA performance on VLM evaluation tasks.

    The model is based on Qwen2.5-VL-7B architecture, so we use the
    Qwen2VLForConditionalGeneration model class.

    Attributes:
        DEFAULT_MODEL_ID: Default HuggingFace model ID
        model: Loaded model instance
        processor: Loaded processor instance
    """

    DEFAULT_MODEL_ID = "lmms-lab/LLaVA-Critic-R1-7B"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attn: bool = True,
    ):
        """Initialize the LLaVA-Critic-R1 judge.

        Args:
            model_id: HuggingFace model ID or local path. Defaults to DEFAULT_MODEL_ID.
            device: Device to load model on ("cuda" or "cpu")
            dtype: Model dtype (torch.bfloat16 recommended)
            use_flash_attn: Whether to use flash attention 2 (requires GPU)
        """
        super().__init__(model_id or self.DEFAULT_MODEL_ID, device, dtype)
        self.use_flash_attn = use_flash_attn
        self._load_model()

    def _load_model(self):
        """Load the Qwen2.5-VL based model and processor."""
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        logger.info(f"Loading model: {self.model_id}")

        # Determine attention implementation
        attn_impl = "flash_attention_2" if self.use_flash_attn else "eager"

        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                device_map="auto",
                attn_implementation=attn_impl,
            )
        except Exception as e:
            if self.use_flash_attn:
                logger.warning(f"Flash attention failed ({e}), falling back to eager")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    device_map="auto",
                    attn_implementation="eager",
                )
            else:
                raise

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

    def _generate(self, prompt: str, image_path: str) -> str:
        """Generate text from prompt + image using the model.

        Args:
            prompt: Text prompt to send to the model
            image_path: Path to the image file

        Returns:
            Generated text response
        """
        from qwen_vl_utils import process_vision_info

        # Build message in Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info (loads and processes the image)
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (skip input)
        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()

    def get_pairwise_prompt(
        self, question: str, feedback: str, answer_a: str, answer_b: str
    ) -> str:
        """Return the pairwise comparison prompt.

        This prompt is designed to:
        1. Evaluate whether Answer B improves upon Answer A given feedback
        2. Use explicit evaluation criteria in order of importance
        3. Request reasoning BEFORE the decision (chain-of-thought)
        4. Use a strict output format for reliable parsing

        Args:
            question: Original question
            feedback: Feedback given after answer_a
            answer_a: Previous answer (before feedback)
            answer_b: Revised answer (after feedback)

        Returns:
            Formatted prompt string
        """
        return f"""You are an expert vision-language evaluator.

## Task
Given feedback on Answer A, determine if Answer B successfully addresses the feedback.

## Context
**Original Question:** {question}

**Feedback on Answer A:** {feedback}

**Answer A (before feedback):**
{answer_a}

**Answer B (after feedback):**
{answer_b}

## Evaluation Criteria (in order of importance)
1. Feedback Compliance: Does B address the specific issues in the feedback?
2. Visual Grounding: Is B more accurate with respect to the image?
3. Correctness: Is B factually more accurate than A?
4. Completeness: Does B provide a more complete answer?

## Instructions
- Focus on whether B improves upon A given the feedback
- Ignore stylistic differences unless they affect clarity
- If both answers are equally good/bad, output "Tie"

## Output Format (strict)
Reasoning: <2-3 sentences explaining your judgment>
Better: A/B/Tie"""

    def get_pointwise_prompt(self, question: str, context: list[dict], answer: str) -> str:
        """Return the pointwise scoring prompt.

        This prompt is designed to:
        1. Score the answer on a 0-10 scale
        2. Consider visual grounding, correctness, and completeness
        3. Include relevant context from previous turns
        4. Use a strict output format for reliable parsing

        Args:
            question: Original question
            context: List of previous turns (dicts with 'response', 'feedback_received')
            answer: Answer to evaluate

        Returns:
            Formatted prompt string
        """
        # Build context string from last 2 turns (to avoid prompt bloat)
        recent_context = context[-2:] if len(context) > 2 else context
        history_lines = []
        for i, turn in enumerate(recent_context):
            response = turn.get("response", "")[:150]  # Truncate long responses
            history_lines.append(f"Turn {i}: {response}...")

        history_text = "\n".join(history_lines) if history_lines else "(First response)"

        return f"""You are an expert vision-language evaluator.

## Task
Rate the quality of the Current Answer on a 0-10 scale.

## Context
**Question:** {question}

**Recent Conversation History:**
{history_text}

**Current Answer to Evaluate:**
{answer}

## Scoring Rubric
- 10: Perfect answer, fully correct, complete, well-grounded in image
- 7-9: Good answer with minor issues
- 4-6: Acceptable but has notable errors or omissions
- 1-3: Poor answer, significant errors or hallucinations
- 0: Completely wrong or irrelevant

## Instructions
- Consider visual grounding, factual accuracy, and completeness
- If feedback was previously given, assess whether the answer addresses it
- Be consistent in scoring across samples

## Output Format (strict)
Reasoning: <1-2 sentences explaining your score>
Score: <0-10>"""

    def _parse_pairwise_output(self, text: str) -> PairwiseResult:
        """Parse model output to PairwiseResult.

        Attempts strict parsing first, then falls back to keyword detection.

        Args:
            text: Raw model output text

        Returns:
            PairwiseResult with parsed fields
        """
        # Try strict parsing first
        better_match = re.search(r"Better:\s*(A|B|Tie)", text, re.IGNORECASE)
        reasoning_match = re.search(
            r"Reasoning:\s*(.+?)(?=Better:|$)", text, re.DOTALL | re.IGNORECASE
        )

        if better_match:
            better = better_match.group(1).upper()
            if better == "TIE":
                better = "tie"

            reasoning = ""
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()

            return PairwiseResult(
                better=better,
                confidence="high",
                reasoning=reasoning,
                raw_text=text,
                position_bias_detected=False,
                parse_success=True,
            )

        # Fallback: keyword detection
        text_lower = text.lower()

        # Look for explicit statements
        if any(
            phrase in text_lower
            for phrase in [
                "b is better",
                "answer b is better",
                "b is the better",
                "prefer answer b",
                "b improves",
                "b correctly",
            ]
        ):
            return PairwiseResult(
                better="B",
                confidence="low",
                reasoning=text,
                raw_text=text,
                position_bias_detected=False,
                parse_success=False,
            )

        if any(
            phrase in text_lower
            for phrase in [
                "a is better",
                "answer a is better",
                "a is the better",
                "prefer answer a",
            ]
        ):
            return PairwiseResult(
                better="A",
                confidence="low",
                reasoning=text,
                raw_text=text,
                position_bias_detected=False,
                parse_success=False,
            )

        if any(
            phrase in text_lower
            for phrase in [
                "tie",
                "equally good",
                "both are",
                "neither is better",
                "no clear winner",
            ]
        ):
            return PairwiseResult(
                better="tie",
                confidence="low",
                reasoning=text,
                raw_text=text,
                position_bias_detected=False,
                parse_success=False,
            )

        # Default to tie if parsing fails completely
        logger.warning(f"Failed to parse pairwise output: {text[:100]}...")
        return PairwiseResult(
            better="tie",
            confidence="low",
            reasoning=f"Parse failed: {text}",
            raw_text=text,
            position_bias_detected=False,
            parse_success=False,
        )

    def _parse_pointwise_output(self, text: str) -> PointwiseResult:
        """Parse model output to PointwiseResult.

        Attempts strict parsing first, then falls back to number extraction.

        Args:
            text: Raw model output text

        Returns:
            PointwiseResult with parsed fields
        """
        # Try strict parsing first
        score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        reasoning_match = re.search(
            r"Reasoning:\s*(.+?)(?=Score:|$)", text, re.DOTALL | re.IGNORECASE
        )

        if score_match:
            score = float(score_match.group(1))
            # Clamp to valid range
            score = max(0.0, min(10.0, score))

            reasoning = ""
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()

            return PointwiseResult(
                score=score,
                reasoning=reasoning,
                raw_text=text,
                parse_success=True,
            )

        # Fallback: look for any number that could be a score
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
        for num_str in numbers:
            num = float(num_str)
            if 0 <= num <= 10:
                logger.warning(f"Fallback score extraction: {num} from {text[:50]}...")
                return PointwiseResult(
                    score=num,
                    reasoning=text,
                    raw_text=text,
                    parse_success=False,
                )

        # Default to middle score if parsing fails completely
        logger.warning(f"Failed to parse pointwise output: {text[:100]}...")
        return PointwiseResult(
            score=5.0,
            reasoning=f"Parse failed: {text}",
            raw_text=text,
            parse_success=False,
        )


@register_judge("llava_critic")
class LLaVACriticJudge(LLaVACriticR1Judge):
    """Original LLaVA-Critic judge (fallback if R1 has issues).

    This is the original LLaVA-Critic model before the R1 training.
    Use this as a fallback if there are compatibility issues with R1.

    The model architecture and interface are the same as R1, only the
    weights are different.
    """

    DEFAULT_MODEL_ID = "lmms-lab/llava-critic-7b"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attn: bool = True,
    ):
        """Initialize the original LLaVA-Critic judge.

        Args:
            model_id: HuggingFace model ID. Defaults to llava-critic-7b.
            device: Device to load model on
            dtype: Model dtype
            use_flash_attn: Whether to use flash attention 2
        """
        # Use our default if none provided
        if model_id is None:
            model_id = self.DEFAULT_MODEL_ID
        super().__init__(model_id, device, dtype, use_flash_attn)
