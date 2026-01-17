#!/usr/bin/env python3
"""
Evaluate VLM self-refinement capability using reward model and/or VLM judge.

This script is the main evaluation pipeline that:
1. Loads test data (FIRE ShareGPT format)
2. Optionally generates new responses with fine-tuned model
3. Scores each turn with Skywork-VL-Reward-7B and/or LLaVA-Critic VLM judge
4. Computes improvement metrics across refinement turns
5. Saves detailed results and generates reports

Evaluation Modes:
- ground_truth: Score existing responses from FIRE dataset
- generated: Generate new responses with fine-tuned model, then score

Judge Options:
- skywork: Skywork-VL-Reward-7B (scalar reward model)
- llava_critic_r1: LLaVA-Critic-R1 VLM judge (pairwise + pointwise)
- llava_critic: Original LLaVA-Critic (fallback)
- none: No judge (useful for generation-only mode)

Key Metrics:
- Reward Delta: (final_score - initial_score) / |initial_score|
- Pairwise Win Rate: % of turns where revised answer is better
- Monotonic Improvement Rate: % samples with strictly increasing scores
- Turns to Plateau: When improvement becomes negligible

Usage:
    # Ground truth evaluation with Skywork reward model
    python scripts/evaluation/evaluate_self_refinement.py \
        --dataset_path /outputs/fire_sharegpt_test.jsonl \
        --mode ground_truth \
        --judge skywork \
        --output_dir /outputs/eval_results

    # Ground truth evaluation with LLaVA-Critic VLM judge
    python scripts/evaluation/evaluate_self_refinement.py \
        --dataset_path /outputs/fire_sharegpt_test.jsonl \
        --mode ground_truth \
        --judge llava_critic_r1 \
        --judge_mode both \
        --debias \
        --output_dir /outputs/eval_results

    # Generated mode (evaluate fine-tuned model)
    python scripts/evaluation/evaluate_self_refinement.py \
        --dataset_path /outputs/fire_sharegpt_test.jsonl \
        --model_path /outputs/checkpoint-final \
        --mode generated \
        --judge llava_critic_r1 \
        --output_dir /outputs/eval_results

Reference:
    - arXiv 2502.05605: Self-refinement evaluation methodology
    - Skywork-VL-Reward: https://huggingface.co/Skywork/Skywork-VL-Reward-7B
    - LLaVA-Critic-R1: https://huggingface.co/lmms-lab/LLaVA-Critic-R1-7B
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


# Add scripts directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from score_with_reward_model import SkyworkVLRewardScorer
from vlm_judge import (
    BaseVLMJudge,
    PairwiseMetrics,
    compute_pairwise_metrics,
    create_judge,
    list_available_judges,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Result for a single evaluated sample."""

    sample_id: str
    sample_index: int
    image_path: str
    original_question: str
    num_turns: int
    turns: list[dict]
    initial_score: float
    final_score: float
    absolute_improvement: float
    reward_delta: float
    is_monotonic: bool
    improvements_per_turn: list[float]
    mode: str  # "ground_truth" or "generated"


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all evaluated samples."""

    num_samples: int
    avg_initial_score: float
    avg_final_score: float
    avg_absolute_improvement: float
    avg_reward_delta: float
    monotonic_improvement_rate: float
    avg_turns_per_sample: float
    avg_improvement_per_turn: float
    avg_turns_to_plateau: float
    plateau_threshold: float
    score_delta_std: float
    score_delta_percentiles: dict[str, float]
    improvement_rate: float  # % of samples that improved at all


def load_dataset(dataset_path: str, max_samples: int = 0) -> list[dict]:
    """Load test dataset in ShareGPT format."""
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


def resolve_image_paths(sample: dict, image_base_dir: str) -> dict:
    """Resolve relative image paths to absolute paths.

    Args:
        sample: Sample dict with 'images' key containing relative paths
        image_base_dir: Base directory for images

    Returns:
        Sample with resolved absolute image paths
    """
    if "images" not in sample or not sample["images"]:
        return sample

    resolved_images = []
    for img_path in sample["images"]:
        if os.path.isabs(img_path):
            # Already absolute
            resolved_images.append(img_path)
        else:
            # Resolve relative path
            resolved = os.path.join(image_base_dir, img_path)
            resolved_images.append(resolved)

    sample["images"] = resolved_images
    return sample


def evaluate_ground_truth(
    samples: list[dict],
    scorer: SkyworkVLRewardScorer | None = None,
    vlm_judge: BaseVLMJudge | None = None,
    image_base_dir: str = "/outputs/fire_images_v2",
    isolated_scoring: bool = False,
    judge_mode: str = "pairwise",
    debias: bool = True,
) -> list[SampleResult]:
    """Evaluate ground truth responses from the dataset.

    This mode scores the existing responses in the FIRE dataset
    to validate that the reward model/VLM judge captures improvement across turns.

    Args:
        samples: List of ShareGPT format samples
        scorer: Initialized Skywork reward model scorer (optional)
        vlm_judge: Initialized VLM judge for pairwise/pointwise evaluation (optional)
        image_base_dir: Base directory for resolving relative image paths
        isolated_scoring: If True, score each response in isolation (just image +
            question + response, ignoring conversation history). If False (default),
            use contextual scoring with full conversation history.
        judge_mode: VLM judge mode - "pairwise", "pointwise", or "both"
        debias: If True, run pairwise comparisons in both orderings

    Returns:
        List of SampleResult objects
    """
    results = []

    for sample in tqdm(samples, desc="Evaluating ground truth"):
        try:
            # Resolve image paths
            sample = resolve_image_paths(sample, image_base_dir)

            conversation = sample.get("conversation", [])
            images = sample.get("images", [])

            if not conversation or not images:
                logger.warning(
                    f"Skipping sample {sample.get('sample_index', '?')}: missing conversation or images"
                )
                continue

            image_path = images[0]
            if not os.path.exists(image_path):
                logger.warning(
                    f"Skipping sample {sample.get('sample_index', '?')}: image not found at {image_path}"
                )
                continue

            # Extract original question
            original_question = conversation[0].get("human", "")
            clean_question = original_question.replace("<image>", "").strip()

            # Initialize turn results
            turns = []
            initial_score = None
            final_score = None

            # Score with Skywork reward model if available
            if scorer is not None:
                scoring_result = scorer.score_conversation_turns(
                    sample, return_details=True, isolated_scoring=isolated_scoring
                )
                turns = scoring_result["turns"]
                metrics = scoring_result["metrics"]
                initial_score = metrics["initial_score"]
                final_score = metrics["final_score"]
            else:
                # Build turn structure without reward scores
                for turn_idx, turn in enumerate(conversation):
                    turn_data = {
                        "turn_index": turn_idx,
                        "response": turn.get("assistant", ""),
                        "reward_score": None,
                    }
                    if turn_idx > 0:
                        turn_data["feedback_received"] = turn.get("human", "")
                    turns.append(turn_data)

            # Add VLM judge evaluations if available
            if vlm_judge is not None:
                for turn_idx in range(len(turns)):
                    turn = turns[turn_idx]
                    response = turn.get("response", "") or conversation[turn_idx].get(
                        "assistant", ""
                    )

                    # Pointwise scoring for all turns
                    if judge_mode in ["pointwise", "both"]:
                        context = [
                            {"response": turns[i].get("response", "")} for i in range(turn_idx)
                        ]
                        try:
                            pointwise_result = vlm_judge.judge_pointwise(
                                image_path=image_path,
                                question=clean_question,
                                context=context,
                                answer=response,
                            )
                            turn["pointwise"] = pointwise_result.to_dict()
                        except Exception as e:
                            logger.warning(f"Pointwise evaluation failed for turn {turn_idx}: {e}")

                    # Pairwise comparison for turns > 0
                    if turn_idx > 0 and judge_mode in ["pairwise", "both"]:
                        prev_response = turns[turn_idx - 1].get("response", "") or conversation[
                            turn_idx - 1
                        ].get("assistant", "")
                        feedback = conversation[turn_idx].get("human", "")
                        try:
                            pairwise_result = vlm_judge.judge_pairwise(
                                image_path=image_path,
                                question=clean_question,
                                feedback=feedback,
                                answer_a=prev_response,
                                answer_b=response,
                                debias=debias,
                            )
                            turn["pairwise"] = pairwise_result.to_dict()
                        except Exception as e:
                            logger.warning(f"Pairwise evaluation failed for turn {turn_idx}: {e}")

            if not turns:
                continue

            # Compute metrics
            if initial_score is None and turns:
                # Use pointwise scores if available
                if "pointwise" in turns[0]:
                    initial_score = turns[0]["pointwise"].get("score", 0)
                    final_score = turns[-1].get("pointwise", {}).get("score", 0)
                else:
                    initial_score = 0
                    final_score = 0

            absolute_improvement = (
                final_score - initial_score
                if (initial_score is not None and final_score is not None)
                else 0
            )
            reward_delta = (
                absolute_improvement / abs(initial_score)
                if initial_score and abs(initial_score) > 0.01
                else 0
            )

            # Compute improvements per turn
            improvements_per_turn = []
            for i in range(1, len(turns)):
                prev_score = turns[i - 1].get("reward_score") or turns[i - 1].get(
                    "pointwise", {}
                ).get("score", 0)
                curr_score = turns[i].get("reward_score") or turns[i].get("pointwise", {}).get(
                    "score", 0
                )
                if prev_score is not None and curr_score is not None:
                    improvements_per_turn.append(curr_score - prev_score)

            # Check monotonicity
            is_monotonic = (
                all(imp >= 0 for imp in improvements_per_turn) if improvements_per_turn else True
            )

            result = SampleResult(
                sample_id=sample.get("id", f"sample_{sample['sample_index']}"),
                sample_index=sample["sample_index"],
                image_path=image_path,
                original_question=clean_question[:200],
                num_turns=len(turns),
                turns=turns,
                initial_score=initial_score or 0,
                final_score=final_score or 0,
                absolute_improvement=absolute_improvement,
                reward_delta=reward_delta,
                is_monotonic=is_monotonic,
                improvements_per_turn=improvements_per_turn,
                mode="ground_truth",
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample.get('sample_index', '?')}: {e}")
            import traceback

            traceback.print_exc()

    return results


def evaluate_generated(
    samples: list[dict],
    model_path: str,
    scorer: SkyworkVLRewardScorer | None = None,
    vlm_judge: BaseVLMJudge | None = None,
    max_turns: int = 3,
    generation_config: dict | None = None,
    image_base_dir: str = "/outputs/fire_images_v2",
    isolated_scoring: bool = False,
    judge_mode: str = "pairwise",
    debias: bool = True,
) -> list[SampleResult]:
    """Evaluate generated responses from fine-tuned model.

    This mode generates new responses using the fine-tuned model,
    uses ground truth feedback to prompt refinements, and scores
    each turn to measure self-refinement capability.

    Args:
        samples: List of ShareGPT format samples
        model_path: Path to fine-tuned model checkpoint
        scorer: Initialized Skywork reward model scorer (optional)
        vlm_judge: Initialized VLM judge for pairwise/pointwise evaluation (optional)
        max_turns: Maximum refinement turns
        generation_config: Generation parameters
        image_base_dir: Base directory for resolving relative image paths
        isolated_scoring: If True, score each response in isolation (just image +
            question + response, ignoring conversation history). If False (default),
            use contextual scoring with full conversation history.
        judge_mode: VLM judge mode - "pairwise", "pointwise", or "both"
        debias: If True, run pairwise comparisons in both orderings

    Returns:
        List of SampleResult objects
    """
    from generate_refinements import VLMInferenceEngine, generate_refinement_dialogue

    # Initialize inference engine
    engine = VLMInferenceEngine(model_path=model_path)

    config = generation_config or {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    results = []

    for sample in tqdm(samples, desc="Evaluating generated responses"):
        try:
            # Resolve image paths
            sample = resolve_image_paths(sample, image_base_dir)

            # Generate refinement dialogue
            gen_result = generate_refinement_dialogue(
                engine=engine,
                sample=sample,
                max_turns=max_turns,
                use_gt_feedback=True,
                generation_config=config,
            )

            if not gen_result:
                continue

            # Convert generated conversation to scoring format
            gen_conversation = gen_result["generated_conversation"]
            images = gen_result["images"]
            image_path = images[0]
            original_question = gen_result.get("original_question", "")
            clean_question = original_question.replace("<image>", "").strip()

            scoring_sample = {
                "conversation": [
                    {"human": turn["human"], "assistant": turn["assistant"]}
                    for turn in gen_conversation
                ],
                "images": images,
            }

            # Initialize turn results
            turns = []
            initial_score = None
            final_score = None

            # Score with Skywork reward model if available
            if scorer is not None:
                scoring_result = scorer.score_conversation_turns(
                    scoring_sample, return_details=True, isolated_scoring=isolated_scoring
                )
                turns = scoring_result["turns"]
                metrics = scoring_result["metrics"]
                initial_score = metrics["initial_score"]
                final_score = metrics["final_score"]
            else:
                # Build turn structure without reward scores
                for turn_idx, turn in enumerate(gen_conversation):
                    turn_data = {
                        "turn_index": turn_idx,
                        "response": turn.get("assistant", ""),
                        "reward_score": None,
                    }
                    if turn_idx > 0:
                        turn_data["feedback_received"] = turn.get("human", "")
                    turns.append(turn_data)

            # Add VLM judge evaluations if available
            if vlm_judge is not None:
                for turn_idx in range(len(turns)):
                    turn = turns[turn_idx]
                    response = turn.get("response", "") or gen_conversation[turn_idx].get(
                        "assistant", ""
                    )

                    # Pointwise scoring for all turns
                    if judge_mode in ["pointwise", "both"]:
                        context = [
                            {"response": turns[i].get("response", "")} for i in range(turn_idx)
                        ]
                        try:
                            pointwise_result = vlm_judge.judge_pointwise(
                                image_path=image_path,
                                question=clean_question,
                                context=context,
                                answer=response,
                            )
                            turn["pointwise"] = pointwise_result.to_dict()
                        except Exception as e:
                            logger.warning(f"Pointwise evaluation failed for turn {turn_idx}: {e}")

                    # Pairwise comparison for turns > 0
                    if turn_idx > 0 and judge_mode in ["pairwise", "both"]:
                        prev_response = turns[turn_idx - 1].get("response", "") or gen_conversation[
                            turn_idx - 1
                        ].get("assistant", "")
                        feedback = gen_conversation[turn_idx].get("human", "")
                        try:
                            pairwise_result = vlm_judge.judge_pairwise(
                                image_path=image_path,
                                question=clean_question,
                                feedback=feedback,
                                answer_a=prev_response,
                                answer_b=response,
                                debias=debias,
                            )
                            turn["pairwise"] = pairwise_result.to_dict()
                        except Exception as e:
                            logger.warning(f"Pairwise evaluation failed for turn {turn_idx}: {e}")

            if not turns:
                continue

            # Compute metrics
            if initial_score is None and turns:
                # Use pointwise scores if available
                if "pointwise" in turns[0]:
                    initial_score = turns[0]["pointwise"].get("score", 0)
                    final_score = turns[-1].get("pointwise", {}).get("score", 0)
                else:
                    initial_score = 0
                    final_score = 0

            absolute_improvement = (
                final_score - initial_score
                if (initial_score is not None and final_score is not None)
                else 0
            )
            reward_delta = (
                absolute_improvement / abs(initial_score)
                if initial_score and abs(initial_score) > 0.01
                else 0
            )

            # Compute improvements per turn
            improvements_per_turn = []
            for i in range(1, len(turns)):
                prev_score = turns[i - 1].get("reward_score") or turns[i - 1].get(
                    "pointwise", {}
                ).get("score", 0)
                curr_score = turns[i].get("reward_score") or turns[i].get("pointwise", {}).get(
                    "score", 0
                )
                if prev_score is not None and curr_score is not None:
                    improvements_per_turn.append(curr_score - prev_score)

            # Check monotonicity
            is_monotonic = (
                all(imp >= 0 for imp in improvements_per_turn) if improvements_per_turn else True
            )

            result = SampleResult(
                sample_id=gen_result.get("sample_id", f"sample_{sample['sample_index']}"),
                sample_index=sample["sample_index"],
                image_path=image_path,
                original_question=clean_question[:200],
                num_turns=len(turns),
                turns=turns,
                initial_score=initial_score or 0,
                final_score=final_score or 0,
                absolute_improvement=absolute_improvement,
                reward_delta=reward_delta,
                is_monotonic=is_monotonic,
                improvements_per_turn=improvements_per_turn,
                mode="generated",
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample.get('sample_index', '?')}: {e}")
            import traceback

            traceback.print_exc()

    return results


def compute_aggregate_metrics(
    results: list[SampleResult],
    plateau_threshold: float = 0.5,
) -> AggregateMetrics:
    """Compute aggregate metrics across all samples.

    Args:
        results: List of per-sample results
        plateau_threshold: Minimum improvement to not count as plateau

    Returns:
        AggregateMetrics dataclass
    """
    import numpy as np

    if not results:
        return AggregateMetrics(
            num_samples=0,
            avg_initial_score=0,
            avg_final_score=0,
            avg_absolute_improvement=0,
            avg_reward_delta=0,
            monotonic_improvement_rate=0,
            avg_turns_per_sample=0,
            avg_improvement_per_turn=0,
            avg_turns_to_plateau=0,
            plateau_threshold=plateau_threshold,
            score_delta_std=0,
            score_delta_percentiles={},
            improvement_rate=0,
        )

    initial_scores = [r.initial_score for r in results]
    final_scores = [r.final_score for r in results]
    reward_deltas = [r.reward_delta for r in results]
    absolute_improvements = [r.absolute_improvement for r in results]
    is_monotonic = [r.is_monotonic for r in results]
    turns_per_sample = [r.num_turns for r in results]

    # Collect all per-turn improvements
    all_improvements = []
    for r in results:
        all_improvements.extend(r.improvements_per_turn)

    # Compute turns to plateau
    turns_to_plateau = []
    for r in results:
        plateau_turn = len(r.improvements_per_turn) + 1
        for i, impr in enumerate(r.improvements_per_turn):
            if impr < plateau_threshold:
                plateau_turn = i + 1
                break
        turns_to_plateau.append(plateau_turn)

    # Improvement rate (samples that improved at all)
    improved = sum(1 for r in results if r.final_score > r.initial_score)

    return AggregateMetrics(
        num_samples=len(results),
        avg_initial_score=float(np.mean(initial_scores)),
        avg_final_score=float(np.mean(final_scores)),
        avg_absolute_improvement=float(np.mean(absolute_improvements)),
        avg_reward_delta=float(np.mean(reward_deltas)),
        monotonic_improvement_rate=float(np.mean(is_monotonic) * 100),
        avg_turns_per_sample=float(np.mean(turns_per_sample)),
        avg_improvement_per_turn=float(np.mean(all_improvements)) if all_improvements else 0,
        avg_turns_to_plateau=float(np.mean(turns_to_plateau)) if turns_to_plateau else 0,
        plateau_threshold=plateau_threshold,
        score_delta_std=float(np.std(reward_deltas)),
        score_delta_percentiles={
            "p25": float(np.percentile(reward_deltas, 25)),
            "p50": float(np.percentile(reward_deltas, 50)),
            "p75": float(np.percentile(reward_deltas, 75)),
        },
        improvement_rate=float(improved / len(results) * 100),
    )


def generate_report(
    metrics: AggregateMetrics,
    mode: str,
    output_path: Path,
    pairwise_metrics: PairwiseMetrics | None = None,
) -> str:
    """Generate human-readable evaluation report."""
    report = f"""
================================================================================
                    SELF-REFINEMENT EVALUATION REPORT
================================================================================

Evaluation Mode: {mode.upper()}
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DATASET STATISTICS
------------------
  Total Samples Evaluated: {metrics.num_samples}
  Average Turns per Sample: {metrics.avg_turns_per_sample:.2f}

REWARD SCORE ANALYSIS
---------------------
  Initial Score (Turn 0):
    Mean: {metrics.avg_initial_score:.2f}

  Final Score (Last Turn):
    Mean: {metrics.avg_final_score:.2f}

  Absolute Improvement:
    Mean: {metrics.avg_absolute_improvement:.2f}

IMPROVEMENT METRICS
-------------------
  Score Delta (final - initial):
    Mean: {metrics.avg_absolute_improvement:.2f}
    (This is the primary metric - positive means improvement)

  Reward Delta (% change from initial):
    Mean: {metrics.avg_reward_delta * 100:.1f}%
    Std Dev: {metrics.score_delta_std * 100:.1f}%
    25th Percentile: {metrics.score_delta_percentiles["p25"] * 100:.1f}%
    Median: {metrics.score_delta_percentiles["p50"] * 100:.1f}%
    75th Percentile: {metrics.score_delta_percentiles["p75"] * 100:.1f}%
    (Note: Mean % can be skewed by outliers; use median for interpretation)

  Improvement Rate: {metrics.improvement_rate:.1f}%
    (Samples where final score > initial score)

  Monotonic Improvement Rate: {metrics.monotonic_improvement_rate:.1f}%
    (Samples where reward strictly increases each turn)

  Average Improvement per Turn: {metrics.avg_improvement_per_turn:.2f}

CONVERGENCE ANALYSIS
--------------------
  Average Turns to Plateau: {metrics.avg_turns_to_plateau:.2f}
    (Plateau threshold: {metrics.plateau_threshold} reward improvement)

================================================================================
                              INTERPRETATION
================================================================================

Key Findings:
"""

    # Use median for interpretation (more robust to outliers)
    median_delta = metrics.score_delta_percentiles["p50"]

    # Add interpretation based on MEDIAN (more robust) and improvement rate
    if median_delta > 0:
        report += f"  - Responses IMPROVE with refinement (median: +{median_delta * 100:.1f}%)\n"
    else:
        report += f"  - Responses show mixed results (median: {median_delta * 100:.1f}%)\n"

    if metrics.improvement_rate > 50:
        report += (
            f"  - {metrics.improvement_rate:.1f}% of samples show improvement (final > initial)\n"
        )
    else:
        report += f"  - Only {metrics.improvement_rate:.1f}% of samples show improvement\n"

    if metrics.monotonic_improvement_rate > 50:
        report += f"  - {metrics.monotonic_improvement_rate:.0f}% of samples show consistent turn-over-turn improvement\n"
    else:
        report += f"  - Only {metrics.monotonic_improvement_rate:.0f}% of samples show consistent improvement\n"

    if metrics.avg_turns_to_plateau < metrics.avg_turns_per_sample:
        report += (
            f"  - Improvement plateaus after ~{metrics.avg_turns_to_plateau:.1f} turns on average\n"
        )

    # Add pairwise metrics section if available
    if pairwise_metrics is not None:
        report += f"""
================================================================================
                         VLM JUDGE PAIRWISE EVALUATION
================================================================================

PAIRWISE COMPARISON RESULTS
---------------------------
  Win Rate (B > A):         {pairwise_metrics.pairwise_win_rate:.1%}
    (% of turns where revised answer is better than previous)

  Regression Rate (A > B):  {pairwise_metrics.regression_rate:.1%}
    (% of turns where revision made answer worse)

  Tie Rate:                 {pairwise_metrics.tie_rate:.1%}
    (% of turns with no clear winner)

POSITION BIAS ANALYSIS
----------------------
  High Confidence Results:  {pairwise_metrics.high_confidence_rate:.1%}
    (Both orderings agree on winner)

  Position Bias Detected:   {pairwise_metrics.position_bias_rate:.1%}
    (Results differed when answer order was swapped)

TOTAL COMPARISONS
-----------------
  Wins: {pairwise_metrics.total_wins}
  Losses: {pairwise_metrics.total_losses}
  Ties: {pairwise_metrics.total_ties}
  Total: {pairwise_metrics.total_comparisons}

INTERPRETATION
--------------
"""
        # Add interpretation for pairwise metrics
        if pairwise_metrics.pairwise_win_rate > 0.6:
            report += f"  - Strong self-refinement: {pairwise_metrics.pairwise_win_rate:.1%} of revisions improve the answer\n"
        elif pairwise_metrics.pairwise_win_rate > 0.4:
            report += (
                f"  - Moderate self-refinement: {pairwise_metrics.pairwise_win_rate:.1%} win rate\n"
            )
        else:
            report += f"  - Weak self-refinement: Only {pairwise_metrics.pairwise_win_rate:.1%} of revisions improve\n"

        if pairwise_metrics.regression_rate > 0.2:
            report += f"  - WARNING: High regression rate ({pairwise_metrics.regression_rate:.1%}) - model often degrades answers\n"
        elif pairwise_metrics.regression_rate < 0.1:
            report += f"  - Low regression rate ({pairwise_metrics.regression_rate:.1%}) - model rarely degrades answers\n"

        if pairwise_metrics.position_bias_rate > 0.2:
            report += f"  - NOTE: Position bias detected in {pairwise_metrics.position_bias_rate:.1%} of comparisons\n"

    report += "\n================================================================================\n"

    # Save report
    with open(output_path, "w") as f:
        f.write(report)

    return report


def save_results(
    results: list[SampleResult],
    metrics: AggregateMetrics,
    output_dir: Path,
    mode: str,
    pairwise_metrics: PairwiseMetrics | None = None,
):
    """Save all results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-sample results
    results_path = output_dir / "sample_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info(f"Saved sample results to {results_path}")

    # Save aggregate metrics
    metrics_path = output_dir / "aggregate_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)
    logger.info(f"Saved aggregate metrics to {metrics_path}")

    # Save pairwise metrics if available
    if pairwise_metrics is not None:
        pairwise_path = output_dir / "pairwise_metrics.json"
        with open(pairwise_path, "w") as f:
            json.dump(asdict(pairwise_metrics), f, indent=2)
        logger.info(f"Saved pairwise metrics to {pairwise_path}")

    # Generate and save report
    report_path = output_dir / "evaluation_report.txt"
    report = generate_report(metrics, mode, report_path, pairwise_metrics)
    logger.info(f"Saved evaluation report to {report_path}")

    # Print report to console
    print(report)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate VLM self-refinement with reward model scoring"
    )

    # Required arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to test dataset (ShareGPT JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )

    # Evaluation mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ground_truth", "generated"],
        default="ground_truth",
        help="Evaluation mode: ground_truth (score dataset) or generated (generate + score)",
    )

    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (required for generated mode)",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs/fire_images_v2",
        help="Base directory for resolving relative image paths",
    )
    parser.add_argument(
        "--reward_model_id",
        type=str,
        default="Skywork/Skywork-VL-Reward-7B",
        help="Reward model HuggingFace ID",
    )

    # Evaluation configuration
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=6,
        help="Maximum turns per sample (for generated mode)",
    )
    parser.add_argument(
        "--plateau_threshold",
        type=float,
        default=0.5,
        help="Minimum improvement to not count as plateau",
    )
    parser.add_argument(
        "--isolated_scoring",
        action="store_true",
        help="Score each response in isolation (just image + question + response, "
        "ignoring conversation history). Provides cleaner comparison without "
        "context length effects. Default is contextual scoring with history.",
    )

    # Generation configuration (for generated mode)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Generation top-p",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response",
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

    # VLM Judge configuration
    parser.add_argument(
        "--judge",
        type=str,
        choices=["skywork", "llava_critic_r1", "llava_critic", "none"],
        default="skywork",
        help="Judge to use: 'skywork' (reward model), 'llava_critic_r1' (SOTA VLM judge), "
        "'llava_critic' (original VLM judge), 'none' (no scoring)",
    )
    parser.add_argument(
        "--judge_model_id",
        type=str,
        default=None,
        help="Override default model ID for the selected VLM judge (for custom checkpoints)",
    )
    parser.add_argument(
        "--judge_mode",
        type=str,
        choices=["pairwise", "pointwise", "both"],
        default="pairwise",
        help="VLM judge evaluation mode: 'pairwise' (compare A vs B), "
        "'pointwise' (score 0-10), 'both'",
    )
    parser.add_argument(
        "--debias",
        action="store_true",
        help="Run pairwise comparisons in both orderings to mitigate position bias (2x time)",
    )

    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Validate arguments
    if args.mode == "generated" and not args.model_path:
        logger.error("--model_path is required for generated mode")
        sys.exit(1)

    if args.judge == "none":
        logger.warning(
            "No judge selected (--judge none). Will only generate responses without scoring."
        )

    output_dir = Path(args.output_dir)

    # Load dataset
    samples = load_dataset(args.dataset_path, args.max_samples)

    if not samples:
        logger.error("No samples loaded, exiting")
        sys.exit(1)

    # Initialize scorers/judges based on --judge argument
    scorer: SkyworkVLRewardScorer | None = None
    vlm_judge: BaseVLMJudge | None = None

    if args.judge == "skywork":
        # Use Skywork-VL-Reward-7B (scalar reward model)
        logger.info(f"Initializing Skywork reward model: {args.reward_model_id}")
        scorer = SkyworkVLRewardScorer(
            model_id=args.reward_model_id,
            device=args.device,
            use_flash_attn=not args.no_flash_attn,
        )
    elif args.judge in ["llava_critic_r1", "llava_critic"]:
        # Use VLM-as-Judge (LLaVA-Critic)
        logger.info(f"Initializing VLM judge: {args.judge}")
        logger.info(f"Available judges: {list_available_judges()}")

        judge_kwargs = {
            "device": args.device,
            "use_flash_attn": not args.no_flash_attn,
        }
        if args.judge_model_id:
            judge_kwargs["model_id"] = args.judge_model_id

        vlm_judge = create_judge(args.judge, **judge_kwargs)
        logger.info(f"VLM judge initialized: {vlm_judge.model_id}")
        logger.info(f"Judge mode: {args.judge_mode}, Debias: {args.debias}")

    # Run evaluation based on mode
    scoring_mode = "isolated" if args.isolated_scoring else "contextual"
    logger.info(f"Scoring mode: {scoring_mode}")

    if args.mode == "ground_truth":
        logger.info("Running ground truth evaluation...")
        logger.info(f"Image base directory: {args.image_base_dir}")
        results = evaluate_ground_truth(
            samples=samples,
            scorer=scorer,
            vlm_judge=vlm_judge,
            image_base_dir=args.image_base_dir,
            isolated_scoring=args.isolated_scoring,
            judge_mode=args.judge_mode,
            debias=args.debias,
        )
    else:
        logger.info("Running generated evaluation...")
        logger.info(f"Image base directory: {args.image_base_dir}")
        gen_config = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        results = evaluate_generated(
            samples=samples,
            model_path=args.model_path,
            scorer=scorer,
            vlm_judge=vlm_judge,
            max_turns=args.max_turns,
            generation_config=gen_config,
            image_base_dir=args.image_base_dir,
            isolated_scoring=args.isolated_scoring,
            judge_mode=args.judge_mode,
            debias=args.debias,
        )

    if not results:
        logger.error("No results generated, exiting")
        sys.exit(1)

    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(results, args.plateau_threshold)

    # Compute pairwise metrics if VLM judge was used
    pairwise_metrics: PairwiseMetrics | None = None
    if vlm_judge is not None and args.judge_mode in ["pairwise", "both"]:
        pairwise_metrics = compute_pairwise_metrics(results)
        logger.info(
            f"Pairwise metrics computed: win_rate={pairwise_metrics.pairwise_win_rate:.1%}, "
            f"regression_rate={pairwise_metrics.regression_rate:.1%}"
        )

    # Save results
    save_results(results, metrics, output_dir, args.mode, pairwise_metrics)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
