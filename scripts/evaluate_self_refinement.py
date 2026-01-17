#!/usr/bin/env python3
"""
Evaluate VLM self-refinement capability using reward model scoring.

This script is the main evaluation pipeline that:
1. Loads test data (FIRE ShareGPT format)
2. Optionally generates new responses with fine-tuned model
3. Scores each turn with Skywork-VL-Reward-7B
4. Computes improvement metrics across refinement turns
5. Saves detailed results and generates reports

Evaluation Modes:
- ground_truth: Score existing responses from FIRE dataset
- generated: Generate new responses with fine-tuned model, then score

Key Metrics:
- Reward Delta: (final_score - initial_score) / |initial_score|
- Monotonic Improvement Rate: % samples with strictly increasing scores
- Average Improvement per Turn: Mean score gain per refinement
- Turns to Plateau: When improvement becomes negligible

Usage:
    # Ground truth evaluation (validate reward model)
    python scripts/evaluate_self_refinement.py \
        --dataset_path /outputs/fire_sharegpt_test.jsonl \
        --mode ground_truth \
        --output_dir /outputs/eval_results

    # Generated mode (evaluate fine-tuned model)
    python scripts/evaluate_self_refinement.py \
        --dataset_path /outputs/fire_sharegpt_test.jsonl \
        --model_path /outputs/checkpoint-final \
        --mode generated \
        --output_dir /outputs/eval_results

Reference:
    - arXiv 2502.05605: Self-refinement evaluation methodology
    - Skywork-VL-Reward: https://huggingface.co/Skywork/Skywork-VL-Reward-7B
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# Add scripts directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from score_with_reward_model import SkyworkVLRewardScorer

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
    turns: List[Dict]
    initial_score: float
    final_score: float
    absolute_improvement: float
    reward_delta: float
    is_monotonic: bool
    improvements_per_turn: List[float]
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
    score_delta_percentiles: Dict[str, float]
    improvement_rate: float  # % of samples that improved at all


def load_dataset(dataset_path: str, max_samples: int = 0) -> List[Dict]:
    """Load test dataset in ShareGPT format."""
    samples = []

    with open(dataset_path, "r") as f:
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


def resolve_image_paths(sample: Dict, image_base_dir: str) -> Dict:
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
    samples: List[Dict],
    scorer: SkyworkVLRewardScorer,
    image_base_dir: str = "/outputs/fire_images_v2",
    isolated_scoring: bool = False,
) -> List[SampleResult]:
    """Evaluate ground truth responses from the dataset.

    This mode scores the existing responses in the FIRE dataset
    to validate that the reward model captures improvement across turns.

    Args:
        samples: List of ShareGPT format samples
        scorer: Initialized reward model scorer
        image_base_dir: Base directory for resolving relative image paths
        isolated_scoring: If True, score each response in isolation (just image +
            question + response, ignoring conversation history). If False (default),
            use contextual scoring with full conversation history.

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
                logger.warning(f"Skipping sample {sample.get('sample_index', '?')}: missing conversation or images")
                continue

            # Score all turns
            scoring_result = scorer.score_conversation_turns(
                sample, return_details=True, isolated_scoring=isolated_scoring
            )

            turns = scoring_result["turns"]
            metrics = scoring_result["metrics"]

            if not turns:
                continue

            result = SampleResult(
                sample_id=sample.get("id", f"sample_{sample['sample_index']}"),
                sample_index=sample["sample_index"],
                image_path=images[0],
                original_question=conversation[0]["human"][:200],
                num_turns=len(turns),
                turns=turns,
                initial_score=metrics["initial_score"],
                final_score=metrics["final_score"],
                absolute_improvement=metrics["absolute_improvement"],
                reward_delta=metrics["reward_delta"],
                is_monotonic=metrics["is_monotonic"],
                improvements_per_turn=metrics["improvements_per_turn"],
                mode="ground_truth",
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample.get('sample_index', '?')}: {e}")

    return results


def evaluate_generated(
    samples: List[Dict],
    model_path: str,
    scorer: SkyworkVLRewardScorer,
    max_turns: int = 3,
    generation_config: Optional[Dict] = None,
    image_base_dir: str = "/outputs/fire_images_v2",
    isolated_scoring: bool = False,
) -> List[SampleResult]:
    """Evaluate generated responses from fine-tuned model.

    This mode generates new responses using the fine-tuned model,
    uses ground truth feedback to prompt refinements, and scores
    each turn to measure self-refinement capability.

    Args:
        samples: List of ShareGPT format samples
        model_path: Path to fine-tuned model checkpoint
        scorer: Initialized reward model scorer
        max_turns: Maximum refinement turns
        generation_config: Generation parameters
        image_base_dir: Base directory for resolving relative image paths
        isolated_scoring: If True, score each response in isolation (just image +
            question + response, ignoring conversation history). If False (default),
            use contextual scoring with full conversation history.

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

            scoring_sample = {
                "conversation": [
                    {"human": turn["human"], "assistant": turn["assistant"]}
                    for turn in gen_conversation
                ],
                "images": images,
            }

            # Score generated turns
            scoring_result = scorer.score_conversation_turns(
                scoring_sample, return_details=True, isolated_scoring=isolated_scoring
            )

            turns = scoring_result["turns"]
            metrics = scoring_result["metrics"]

            if not turns:
                continue

            result = SampleResult(
                sample_id=gen_result.get("sample_id", f"sample_{sample['sample_index']}"),
                sample_index=sample["sample_index"],
                image_path=images[0],
                original_question=gen_result["original_question"][:200],
                num_turns=len(turns),
                turns=turns,
                initial_score=metrics["initial_score"],
                final_score=metrics["final_score"],
                absolute_improvement=metrics["absolute_improvement"],
                reward_delta=metrics["reward_delta"],
                is_monotonic=metrics["is_monotonic"],
                improvements_per_turn=metrics["improvements_per_turn"],
                mode="generated",
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample.get('sample_index', '?')}: {e}")

    return results


def compute_aggregate_metrics(
    results: List[SampleResult],
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
) -> str:
    """Generate human-readable evaluation report."""
    report = f"""
================================================================================
                    SELF-REFINEMENT EVALUATION REPORT
================================================================================

Evaluation Mode: {mode.upper()}
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
  Reward Delta (% change from initial):
    Mean: {metrics.avg_reward_delta * 100:.1f}%
    Std Dev: {metrics.score_delta_std * 100:.1f}%
    25th Percentile: {metrics.score_delta_percentiles['p25'] * 100:.1f}%
    Median: {metrics.score_delta_percentiles['p50'] * 100:.1f}%
    75th Percentile: {metrics.score_delta_percentiles['p75'] * 100:.1f}%

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

    # Add interpretation
    if metrics.avg_reward_delta > 0:
        report += f"  - Responses IMPROVE with refinement ({metrics.avg_reward_delta*100:.1f}% avg improvement)\n"
    else:
        report += f"  - Responses DO NOT improve with refinement ({metrics.avg_reward_delta*100:.1f}% avg change)\n"

    if metrics.monotonic_improvement_rate > 50:
        report += f"  - Majority of samples ({metrics.monotonic_improvement_rate:.0f}%) show consistent improvement\n"
    else:
        report += f"  - Only {metrics.monotonic_improvement_rate:.0f}% of samples show consistent improvement\n"

    if metrics.avg_turns_to_plateau < metrics.avg_turns_per_sample:
        report += f"  - Improvement plateaus after ~{metrics.avg_turns_to_plateau:.1f} turns on average\n"

    report += "\n================================================================================\n"

    # Save report
    with open(output_path, "w") as f:
        f.write(report)

    return report


def save_results(
    results: List[SampleResult],
    metrics: AggregateMetrics,
    output_dir: Path,
    mode: str,
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

    # Generate and save report
    report_path = output_dir / "evaluation_report.txt"
    report = generate_report(metrics, mode, report_path)
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

    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Validate arguments
    if args.mode == "generated" and not args.model_path:
        logger.error("--model_path is required for generated mode")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Load dataset
    samples = load_dataset(args.dataset_path, args.max_samples)

    if not samples:
        logger.error("No samples loaded, exiting")
        sys.exit(1)

    # Initialize reward model scorer
    logger.info(f"Initializing reward model: {args.reward_model_id}")
    scorer = SkyworkVLRewardScorer(
        model_id=args.reward_model_id,
        device=args.device,
        use_flash_attn=not args.no_flash_attn,
    )

    # Run evaluation based on mode
    scoring_mode = "isolated" if args.isolated_scoring else "contextual"
    logger.info(f"Scoring mode: {scoring_mode}")

    if args.mode == "ground_truth":
        logger.info("Running ground truth evaluation...")
        logger.info(f"Image base directory: {args.image_base_dir}")
        results = evaluate_ground_truth(
            samples, scorer, args.image_base_dir, args.isolated_scoring
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
            samples,
            args.model_path,
            scorer,
            args.max_turns,
            gen_config,
            args.image_base_dir,
            args.isolated_scoring,
        )

    if not results:
        logger.error("No results generated, exiting")
        sys.exit(1)

    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(results, args.plateau_threshold)

    # Save results
    save_results(results, metrics, output_dir, args.mode)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
