#!/usr/bin/env python3
"""
Generic VLM Judge Interface for Self-Refinement Evaluation.

This module provides a modular architecture for VLM-as-Judge evaluation,
allowing easy swapping of different judge models through a factory pattern.

Usage:
    from vlm_judge import create_judge, PairwiseResult, PointwiseResult

    # Create a judge
    judge = create_judge("llava_critic_r1", device="cuda")

    # Run pairwise comparison
    result = judge.judge_pairwise(
        image_path="/path/to/image.jpg",
        question="What color is the bear?",
        feedback="The bear is not gray, look more carefully.",
        answer_a="The bear is gray.",
        answer_b="The bear is white.",
        debias=True  # Run both orderings
    )
    print(f"Better: {result.better}, Confidence: {result.confidence}")

    # Run pointwise scoring
    result = judge.judge_pointwise(
        image_path="/path/to/image.jpg",
        question="What color is the bear?",
        context=[],  # Previous turns
        answer="The bear is white."
    )
    print(f"Score: {result.score}/10")

To add a new judge model:
    1. Create a new class extending BaseVLMJudge
    2. Implement required abstract methods
    3. Register in JUDGE_REGISTRY

See scripts/evaluation/judges/llava_critic_r1.py for example implementation.
"""

import argparse
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Standardized Output
# =============================================================================


@dataclass
class PairwiseResult:
    """Standard output format for pairwise comparison.

    Attributes:
        better: Winner of comparison - "A", "B", or "tie"
        confidence: Confidence level - "high" (consistent both orderings) or "low"
        reasoning: Model's reasoning for the decision
        raw_text: Raw model output text
        position_bias_detected: True if results were inconsistent across orderings
        parse_success: True if output was successfully parsed
    """

    better: str  # "A", "B", or "tie"
    confidence: str  # "high", "low"
    reasoning: str
    raw_text: str
    position_bias_detected: bool = False
    parse_success: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PointwiseResult:
    """Standard output format for pointwise scoring.

    Attributes:
        score: Quality score on 0-10 scale
        reasoning: Model's reasoning for the score
        raw_text: Raw model output text
        parse_success: True if output was successfully parsed
    """

    score: float  # 0-10 scale
    reasoning: str
    raw_text: str
    parse_success: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PairwiseMetrics:
    """Aggregate metrics for pairwise evaluation.

    Attributes:
        pairwise_win_rate: Fraction where B (revised) wins
        regression_rate: Fraction where A (original) wins (regression)
        tie_rate: Fraction of ties
        position_bias_rate: Fraction with detected position bias
        high_confidence_rate: Fraction with consistent results
        per_turn_win_rates: Win rates per turn index
        total_comparisons: Total number of pairwise comparisons
        total_wins: Total number of wins (B > A)
        total_losses: Total number of losses (A > B, regression)
        total_ties: Total number of ties
    """

    pairwise_win_rate: float = 0.0
    regression_rate: float = 0.0
    tie_rate: float = 0.0
    position_bias_rate: float = 0.0
    high_confidence_rate: float = 0.0
    per_turn_win_rates: list[float] = field(default_factory=list)
    total_comparisons: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_ties: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# Abstract Base Class for VLM Judges
# =============================================================================


class BaseVLMJudge(ABC):
    """Abstract base class for VLM judges.

    Extend this class to add new judge models. Subclasses must implement:
        - _load_model(): Load the model and processor
        - _generate(): Generate text from prompt + image
        - get_pairwise_prompt(): Return pairwise comparison prompt template
        - get_pointwise_prompt(): Return pointwise scoring prompt template
        - _parse_pairwise_output(): Parse model output to PairwiseResult
        - _parse_pointwise_output(): Parse model output to PointwiseResult

    The base class provides:
        - judge_pairwise(): Run pairwise comparison with optional debiasing
        - judge_pointwise(): Run pointwise scoring
        - Debiasing logic (run both orderings to detect position bias)
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the judge.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to load model on ("cuda" or "cpu")
            dtype: Model dtype (torch.bfloat16 or torch.float16)
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

    @abstractmethod
    def _load_model(self):
        """Load model and processor.

        Override this method for each model architecture.
        Should set self.model and self.processor.
        """
        pass

    @abstractmethod
    def _generate(self, prompt: str, image_path: str) -> str:
        """Generate text from prompt + image.

        Args:
            prompt: Text prompt to send to the model
            image_path: Path to the image file

        Returns:
            Generated text response from the model
        """
        pass

    @abstractmethod
    def get_pairwise_prompt(
        self, question: str, feedback: str, answer_a: str, answer_b: str
    ) -> str:
        """Return the pairwise comparison prompt.

        Args:
            question: Original question
            feedback: Feedback given after answer_a
            answer_a: Previous answer (before feedback)
            answer_b: Revised answer (after feedback)

        Returns:
            Formatted prompt string for pairwise comparison
        """
        pass

    @abstractmethod
    def get_pointwise_prompt(self, question: str, context: list[dict], answer: str) -> str:
        """Return the pointwise scoring prompt.

        Args:
            question: Original question
            context: List of previous turns (dicts with 'response', 'feedback_received')
            answer: Answer to evaluate

        Returns:
            Formatted prompt string for pointwise scoring
        """
        pass

    @abstractmethod
    def _parse_pairwise_output(self, text: str) -> PairwiseResult:
        """Parse model output to PairwiseResult.

        Args:
            text: Raw model output text

        Returns:
            PairwiseResult with parsed fields
        """
        pass

    @abstractmethod
    def _parse_pointwise_output(self, text: str) -> PointwiseResult:
        """Parse model output to PointwiseResult.

        Args:
            text: Raw model output text

        Returns:
            PointwiseResult with parsed fields
        """
        pass

    def judge_pairwise(
        self,
        image_path: str,
        question: str,
        feedback: str,
        answer_a: str,
        answer_b: str,
        debias: bool = True,
    ) -> PairwiseResult:
        """Run pairwise comparison with optional debiasing.

        Args:
            image_path: Path to the image file
            question: Original question
            feedback: Feedback given after answer_a
            answer_a: Previous answer (before feedback)
            answer_b: Revised answer (after feedback)
            debias: If True, run both orderings to detect position bias

        Returns:
            PairwiseResult with winner, confidence, and reasoning
        """
        if debias:
            return self._judge_pairwise_debiased(image_path, question, feedback, answer_a, answer_b)
        else:
            return self._judge_pairwise_single(image_path, question, feedback, answer_a, answer_b)

    def _judge_pairwise_single(
        self,
        image_path: str,
        question: str,
        feedback: str,
        answer_a: str,
        answer_b: str,
    ) -> PairwiseResult:
        """Run single pairwise comparison (A then B ordering)."""
        prompt = self.get_pairwise_prompt(question, feedback, answer_a, answer_b)
        raw_text = self._generate(prompt, image_path)
        return self._parse_pairwise_output(raw_text)

    def _judge_pairwise_debiased(
        self,
        image_path: str,
        question: str,
        feedback: str,
        answer_a: str,
        answer_b: str,
    ) -> PairwiseResult:
        """Run pairwise comparison in BOTH orderings to detect position bias.

        Position bias is when the model favors the first or second answer
        regardless of content. By running both orderings, we can detect this.

        Returns:
            PairwiseResult with:
            - better="A" if both orderings agree A is better
            - better="B" if both orderings agree B is better
            - better="tie" if orderings disagree (position bias detected)
        """
        # Order 1: A first, B second
        result_ab = self._judge_pairwise_single(image_path, question, feedback, answer_a, answer_b)

        # Order 2: B first, A second (swap the answers in the prompt)
        # When we swap, if model says "A" it means the swapped A (which is original B)
        result_ba = self._judge_pairwise_single(image_path, question, feedback, answer_b, answer_a)

        # Map result_ba back: if BA says "A", that's really "B" (original B was in A position)
        ba_mapped = result_ba.better
        if ba_mapped == "A":
            ba_mapped = "B"
        elif ba_mapped == "B":
            ba_mapped = "A"
        # "tie" stays "tie"

        # Aggregate results: consistent if both orderings agree
        if result_ab.better == "A" and ba_mapped == "A":
            # Both orderings say original A is better
            return PairwiseResult(
                better="A",
                confidence="high",
                reasoning=result_ab.reasoning,
                raw_text=f"Order AB: {result_ab.raw_text}\nOrder BA: {result_ba.raw_text}",
                position_bias_detected=False,
                parse_success=result_ab.parse_success and result_ba.parse_success,
            )
        elif result_ab.better == "B" and ba_mapped == "B":
            # Both orderings say revised B is better
            return PairwiseResult(
                better="B",
                confidence="high",
                reasoning=result_ab.reasoning,
                raw_text=f"Order AB: {result_ab.raw_text}\nOrder BA: {result_ba.raw_text}",
                position_bias_detected=False,
                parse_success=result_ab.parse_success and result_ba.parse_success,
            )
        elif result_ab.better == "tie" and ba_mapped == "tie":
            # Both orderings say tie
            return PairwiseResult(
                better="tie",
                confidence="high",
                reasoning="Both orderings resulted in tie",
                raw_text=f"Order AB: {result_ab.raw_text}\nOrder BA: {result_ba.raw_text}",
                position_bias_detected=False,
                parse_success=result_ab.parse_success and result_ba.parse_success,
            )
        else:
            # Inconsistent results - position bias detected
            return PairwiseResult(
                better="tie",
                confidence="low",
                reasoning=f"Inconsistent: AB={result_ab.better}, BA(mapped)={ba_mapped}",
                raw_text=f"Order AB: {result_ab.raw_text}\nOrder BA: {result_ba.raw_text}",
                position_bias_detected=True,
                parse_success=result_ab.parse_success and result_ba.parse_success,
            )

    def judge_pointwise(
        self,
        image_path: str,
        question: str,
        context: list[dict],
        answer: str,
    ) -> PointwiseResult:
        """Run pointwise scoring.

        Args:
            image_path: Path to the image file
            question: Original question
            context: List of previous turns for context
            answer: Answer to evaluate

        Returns:
            PointwiseResult with score (0-10) and reasoning
        """
        prompt = self.get_pointwise_prompt(question, context, answer)
        raw_text = self._generate(prompt, image_path)
        return self._parse_pointwise_output(raw_text)


# =============================================================================
# Judge Registry and Factory
# =============================================================================

# Registry mapping judge names to classes
# Populated dynamically when judge modules are imported
JUDGE_REGISTRY: dict[str, type[BaseVLMJudge]] = {}


def register_judge(name: str):
    """Decorator to register a judge class in the registry.

    Usage:
        @register_judge("my_judge")
        class MyJudge(BaseVLMJudge):
            ...
    """

    def decorator(cls: type[BaseVLMJudge]) -> type[BaseVLMJudge]:
        JUDGE_REGISTRY[name] = cls
        return cls

    return decorator


def create_judge(judge_type: str, **kwargs) -> BaseVLMJudge:
    """Factory function to create VLM judges by name.

    Args:
        judge_type: Name of the judge (e.g., "llava_critic_r1", "llava_critic")
        **kwargs: Additional arguments passed to judge constructor

    Returns:
        Initialized judge instance

    Raises:
        ValueError: If judge_type is not registered
    """
    # Import judge implementations to populate registry
    _import_judge_implementations()

    if judge_type not in JUDGE_REGISTRY:
        available = list(JUDGE_REGISTRY.keys())
        raise ValueError(f"Unknown judge type: '{judge_type}'. Available judges: {available}")

    judge_class = JUDGE_REGISTRY[judge_type]
    return judge_class(**kwargs)


def list_available_judges() -> list[str]:
    """List all available judge types.

    Returns:
        List of registered judge names
    """
    _import_judge_implementations()
    return list(JUDGE_REGISTRY.keys())


def _import_judge_implementations():
    """Import judge implementations to populate registry.

    This is called lazily to avoid import errors if dependencies are missing.
    """
    if JUDGE_REGISTRY:
        # Already imported
        return

    # Import built-in implementations
    try:
        from judges.llava_critic_r1 import LLaVACriticJudge, LLaVACriticR1Judge

        JUDGE_REGISTRY["llava_critic_r1"] = LLaVACriticR1Judge
        JUDGE_REGISTRY["llava_critic"] = LLaVACriticJudge
        logger.debug("Registered LLaVA-Critic judges")
    except ImportError as e:
        logger.warning(f"Could not import LLaVA-Critic judges: {e}")

    # Add more judge imports here as they are implemented


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_pairwise_metrics(results: list[Any], turn_key: str = "turns") -> PairwiseMetrics:
    """Compute aggregate pairwise metrics from evaluation results.

    Args:
        results: List of sample results (dicts or dataclasses with 'turns')
        turn_key: Key/attribute name for turns list in result objects

    Returns:
        PairwiseMetrics with aggregated statistics
    """
    wins = 0
    losses = 0
    ties = 0
    position_bias_count = 0
    high_confidence_count = 0
    total = 0

    # Per-turn tracking
    turn_wins: dict[int, int] = {}
    turn_totals: dict[int, int] = {}

    for result in results:
        # Handle both dict and dataclass results
        if hasattr(result, turn_key):
            turns = getattr(result, turn_key)
        elif isinstance(result, dict):
            turns = result.get(turn_key, [])
        else:
            continue

        for turn in turns:
            # Handle both dict and dataclass turns
            if isinstance(turn, dict):
                pairwise = turn.get("pairwise")
            elif hasattr(turn, "pairwise"):
                pairwise = getattr(turn, "pairwise", None)
            else:
                continue

            if not pairwise:
                continue

            # Get turn_index (handle both dict and dataclass)
            if isinstance(turn, dict):
                turn_idx = turn.get("turn_index", 0)
            else:
                turn_idx = getattr(turn, "turn_index", 0)

            if turn_idx == 0:
                # Skip turn 0 (no pairwise for initial response)
                continue

            total += 1

            # Get pairwise fields (handle both dict and dataclass)
            if isinstance(pairwise, dict):
                better = pairwise.get("better", "tie")
                confidence = pairwise.get("confidence", "low")
                bias_detected = pairwise.get("position_bias_detected", False)
            else:
                better = getattr(pairwise, "better", "tie")
                confidence = getattr(pairwise, "confidence", "low")
                bias_detected = getattr(pairwise, "position_bias_detected", False)

            if better == "B":
                wins += 1
                turn_wins[turn_idx] = turn_wins.get(turn_idx, 0) + 1
            elif better == "A":
                losses += 1
            else:
                ties += 1

            if bias_detected:
                position_bias_count += 1
            if confidence == "high":
                high_confidence_count += 1

            turn_totals[turn_idx] = turn_totals.get(turn_idx, 0) + 1

    # Compute per-turn win rates
    per_turn_win_rates = []
    for turn_idx in sorted(turn_totals.keys()):
        if turn_totals[turn_idx] > 0:
            rate = turn_wins.get(turn_idx, 0) / turn_totals[turn_idx]
            per_turn_win_rates.append(rate)

    return PairwiseMetrics(
        pairwise_win_rate=wins / total if total > 0 else 0.0,
        regression_rate=losses / total if total > 0 else 0.0,
        tie_rate=ties / total if total > 0 else 0.0,
        position_bias_rate=position_bias_count / total if total > 0 else 0.0,
        high_confidence_rate=high_confidence_count / total if total > 0 else 0.0,
        per_turn_win_rates=per_turn_win_rates,
        total_comparisons=total,
        total_wins=wins,
        total_losses=losses,
        total_ties=ties,
    )


# =============================================================================
# CLI for Testing
# =============================================================================


def main():
    """CLI for testing VLM judges."""
    parser = argparse.ArgumentParser(
        description="Test VLM judges on sample data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with FIRE dataset sample
    python vlm_judge.py \\
        --judge llava_critic_r1 \\
        --sample_path /outputs/fire_preprocessed_v2/fire_sharegpt_test.jsonl \\
        --image_base_dir /outputs/fire_images_v2 \\
        --max_samples 5 \\
        --mode both

    # List available judges
    python vlm_judge.py --list_judges
""",
    )

    parser.add_argument(
        "--list_judges",
        action="store_true",
        help="List available judge types and exit",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="llava_critic_r1",
        help="Judge type to use (default: llava_critic_r1)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Override default model ID for the selected judge",
    )
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
        default=5,
        help="Maximum number of samples to test",
    )
    parser.add_argument(
        "--mode",
        choices=["pairwise", "pointwise", "both"],
        default="both",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--debias",
        action="store_true",
        default=True,
        help="Run pairwise in both orderings (default: True)",
    )
    parser.add_argument(
        "--no_debias",
        action="store_true",
        help="Disable debiasing (single ordering only)",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )

    args = parser.parse_args()

    # Handle --list_judges
    if args.list_judges:
        print("Available VLM judges:")
        for judge_name in list_available_judges():
            print(f"  - {judge_name}")
        return

    # Require sample_path for testing
    if not args.sample_path:
        parser.error("--sample_path is required for testing")

    # Resolve debias flag
    debias = not args.no_debias

    # Initialize judge
    logger.info(f"Initializing judge: {args.judge}")
    judge_kwargs = {
        "device": args.device,
        "use_flash_attn": not args.no_flash_attn,
    }
    if args.model_id:
        judge_kwargs["model_id"] = args.model_id

    try:
        judge = create_judge(args.judge, **judge_kwargs)
        logger.info(f"Loaded judge model: {judge.model_id}")
    except Exception as e:
        logger.error(f"Failed to create judge: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Load samples
    logger.info(f"Loading samples from {args.sample_path}")
    samples = []
    with open(args.sample_path) as f:
        for i, line in enumerate(f):
            if args.max_samples > 0 and i >= args.max_samples:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i}: {e}")

    logger.info(f"Loaded {len(samples)} samples")

    # Process samples
    for sample_idx, sample in enumerate(samples):
        conversation = sample.get("conversation", [])
        images = sample.get("images", [])

        if not conversation or not images:
            logger.warning(f"Sample {sample_idx}: missing conversation or images")
            continue

        # Resolve image path
        image_path = images[0]
        if not os.path.isabs(image_path):
            image_path = os.path.join(args.image_base_dir, image_path)

        if not os.path.exists(image_path):
            logger.warning(f"Sample {sample_idx}: image not found at {image_path}")
            continue

        # Extract question from first turn
        original_question = conversation[0].get("human", "")
        clean_question = original_question.replace("<image>", "").strip()

        print(f"\n{'=' * 70}")
        print(f"SAMPLE {sample_idx}")
        print(f"{'=' * 70}")
        print(f"Image: {image_path}")
        print(f"Question: {clean_question[:100]}...")
        print(f"Turns: {len(conversation)}")

        # Score each turn
        for turn_idx in range(len(conversation)):
            turn = conversation[turn_idx]
            response = turn.get("assistant", "")

            print(f"\n--- Turn {turn_idx} ---")
            print(f"Response: {response[:100]}...")

            if turn_idx == 0:
                # First turn - only pointwise
                if args.mode in ["pointwise", "both"]:
                    try:
                        result = judge.judge_pointwise(
                            image_path=image_path,
                            question=clean_question,
                            context=[],
                            answer=response,
                        )
                        print(f"Pointwise Score: {result.score}/10")
                        print(f"Reasoning: {result.reasoning[:200]}...")
                    except Exception as e:
                        logger.error(f"Pointwise error: {e}")
            else:
                # Subsequent turns - pairwise and/or pointwise
                prev_response = conversation[turn_idx - 1].get("assistant", "")
                feedback = turn.get("human", "")

                if args.mode in ["pairwise", "both"]:
                    try:
                        result = judge.judge_pairwise(
                            image_path=image_path,
                            question=clean_question,
                            feedback=feedback,
                            answer_a=prev_response,
                            answer_b=response,
                            debias=debias,
                        )
                        print(f"Pairwise: {result.better} ({result.confidence} confidence)")
                        if result.position_bias_detected:
                            print("  ⚠️ Position bias detected")
                        print(f"Reasoning: {result.reasoning[:200]}...")
                    except Exception as e:
                        logger.error(f"Pairwise error: {e}")

                if args.mode in ["pointwise", "both"]:
                    try:
                        # Build context from previous turns
                        context = [
                            {"response": conversation[i].get("assistant", "")}
                            for i in range(turn_idx)
                        ]
                        result = judge.judge_pointwise(
                            image_path=image_path,
                            question=clean_question,
                            context=context,
                            answer=response,
                        )
                        print(f"Pointwise Score: {result.score}/10")
                        print(f"Reasoning: {result.reasoning[:200]}...")
                    except Exception as e:
                        logger.error(f"Pointwise error: {e}")

    print(f"\n{'=' * 70}")
    print("TESTING COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
