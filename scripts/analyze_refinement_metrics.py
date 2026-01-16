#!/usr/bin/env python3
"""
Analyze and visualize self-refinement evaluation results.

This script provides post-hoc analysis of evaluation results including:
- Statistical analysis of reward scores
- Comparison between ground truth and generated modes
- Visualization of improvement trends
- Per-turn analysis and breakdowns
- Identification of best/worst performing samples

Usage:
    # Basic analysis
    python scripts/analyze_refinement_metrics.py \
        --results_path /outputs/eval_results/sample_results.jsonl \
        --output_dir /outputs/eval_results/analysis

    # Compare two evaluation runs
    python scripts/analyze_refinement_metrics.py \
        --results_path /outputs/eval_gt/sample_results.jsonl \
        --compare_path /outputs/eval_gen/sample_results.jsonl \
        --output_dir /outputs/comparison_analysis
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class TurnAnalysis:
    """Analysis for a specific turn index."""
    turn_index: int
    num_samples: int
    mean_score: float
    std_score: float
    mean_improvement: float
    std_improvement: float
    improvement_rate: float  # % of samples that improved at this turn


def load_results(results_path: str) -> List[Dict]:
    """Load evaluation results from JSONL file."""
    results = []

    with open(results_path, "r") as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")

    logger.info(f"Loaded {len(results)} results from {results_path}")
    return results


def analyze_by_turn(results: List[Dict]) -> List[TurnAnalysis]:
    """Analyze metrics broken down by turn index.

    Args:
        results: List of sample results

    Returns:
        List of TurnAnalysis for each turn index
    """
    # Group scores and improvements by turn
    turn_scores = defaultdict(list)
    turn_improvements = defaultdict(list)

    for r in results:
        turns = r.get("turns", [])
        for i, turn in enumerate(turns):
            turn_scores[i].append(turn["reward_score"])
            if i > 0:
                improvement = turn["reward_score"] - turns[i-1]["reward_score"]
                turn_improvements[i].append(improvement)

    # Compute statistics per turn
    analyses = []
    max_turn = max(turn_scores.keys()) if turn_scores else 0

    for turn_idx in range(max_turn + 1):
        scores = turn_scores.get(turn_idx, [])
        improvements = turn_improvements.get(turn_idx, [])

        if not scores:
            continue

        # Improvement rate (% of samples that improved at this turn)
        improvement_rate = 0.0
        if improvements:
            improvement_rate = sum(1 for imp in improvements if imp > 0) / len(improvements) * 100

        analyses.append(TurnAnalysis(
            turn_index=turn_idx,
            num_samples=len(scores),
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            mean_improvement=float(np.mean(improvements)) if improvements else 0.0,
            std_improvement=float(np.std(improvements)) if improvements else 0.0,
            improvement_rate=improvement_rate,
        ))

    return analyses


def analyze_improvement_distribution(results: List[Dict]) -> Dict:
    """Analyze the distribution of improvements.

    Returns:
        Dict with improvement distribution statistics
    """
    reward_deltas = [r["reward_delta"] for r in results]
    absolute_improvements = [r["absolute_improvement"] for r in results]

    # Categorize samples
    improved = [d for d in reward_deltas if d > 0]
    degraded = [d for d in reward_deltas if d < 0]
    unchanged = [d for d in reward_deltas if d == 0]

    return {
        "total_samples": len(results),
        "improved_count": len(improved),
        "degraded_count": len(degraded),
        "unchanged_count": len(unchanged),
        "improved_pct": len(improved) / len(results) * 100 if results else 0,
        "degraded_pct": len(degraded) / len(results) * 100 if results else 0,
        "unchanged_pct": len(unchanged) / len(results) * 100 if results else 0,
        "reward_delta": {
            "mean": float(np.mean(reward_deltas)),
            "std": float(np.std(reward_deltas)),
            "min": float(np.min(reward_deltas)),
            "max": float(np.max(reward_deltas)),
            "percentiles": {
                "p10": float(np.percentile(reward_deltas, 10)),
                "p25": float(np.percentile(reward_deltas, 25)),
                "p50": float(np.percentile(reward_deltas, 50)),
                "p75": float(np.percentile(reward_deltas, 75)),
                "p90": float(np.percentile(reward_deltas, 90)),
            },
        },
        "absolute_improvement": {
            "mean": float(np.mean(absolute_improvements)),
            "std": float(np.std(absolute_improvements)),
            "min": float(np.min(absolute_improvements)),
            "max": float(np.max(absolute_improvements)),
        },
    }


def find_extreme_samples(
    results: List[Dict],
    n: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """Find the best and worst performing samples.

    Args:
        results: List of sample results
        n: Number of samples to return for each category

    Returns:
        Tuple of (best_samples, worst_samples)
    """
    # Sort by reward delta
    sorted_results = sorted(results, key=lambda x: x["reward_delta"], reverse=True)

    best = sorted_results[:n]
    worst = sorted_results[-n:]

    return best, worst


def compare_evaluations(
    results1: List[Dict],
    results2: List[Dict],
    label1: str = "Evaluation 1",
    label2: str = "Evaluation 2",
) -> Dict:
    """Compare two evaluation runs.

    Args:
        results1: First evaluation results
        results2: Second evaluation results
        label1: Label for first evaluation
        label2: Label for second evaluation

    Returns:
        Dict with comparison statistics
    """
    # Compute basic stats for each
    def compute_stats(results):
        return {
            "num_samples": len(results),
            "avg_initial_score": float(np.mean([r["initial_score"] for r in results])),
            "avg_final_score": float(np.mean([r["final_score"] for r in results])),
            "avg_reward_delta": float(np.mean([r["reward_delta"] for r in results])),
            "monotonic_rate": float(np.mean([r["is_monotonic"] for r in results]) * 100),
            "improvement_rate": float(
                sum(1 for r in results if r["final_score"] > r["initial_score"]) / len(results) * 100
            ),
        }

    stats1 = compute_stats(results1)
    stats2 = compute_stats(results2)

    # Compute differences
    differences = {}
    for key in stats1:
        if key != "num_samples":
            diff = stats2[key] - stats1[key]
            pct_diff = diff / abs(stats1[key]) * 100 if stats1[key] != 0 else 0
            differences[key] = {
                "absolute": diff,
                "percentage": pct_diff,
            }

    return {
        label1: stats1,
        label2: stats2,
        "differences": differences,
    }


def create_visualizations(
    results: List[Dict],
    turn_analyses: List[TurnAnalysis],
    output_dir: Path,
):
    """Create visualization plots.

    Args:
        results: Evaluation results
        turn_analyses: Per-turn analysis
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        logger.warning("matplotlib/seaborn not installed, skipping visualizations")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Score progression across turns
    fig, ax = plt.subplots(figsize=(10, 6))
    turns = [ta.turn_index for ta in turn_analyses]
    mean_scores = [ta.mean_score for ta in turn_analyses]
    std_scores = [ta.std_score for ta in turn_analyses]

    ax.errorbar(turns, mean_scores, yerr=std_scores, marker='o', capsize=5, capthick=2)
    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Mean Reward Score")
    ax.set_title("Reward Score Progression Across Turns")
    ax.set_xticks(turns)
    plt.tight_layout()
    plt.savefig(output_dir / "score_progression.png", dpi=150)
    plt.close()

    # 2. Improvement per turn
    fig, ax = plt.subplots(figsize=(10, 6))
    improvements = [ta.mean_improvement for ta in turn_analyses if ta.turn_index > 0]
    improvement_turns = [ta.turn_index for ta in turn_analyses if ta.turn_index > 0]

    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax.bar(improvement_turns, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Mean Improvement (Score Delta)")
    ax.set_title("Average Improvement per Turn")
    ax.set_xticks(improvement_turns)
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_per_turn.png", dpi=150)
    plt.close()

    # 3. Reward delta distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    reward_deltas = [r["reward_delta"] * 100 for r in results]  # Convert to percentage

    ax.hist(reward_deltas, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', label='No improvement')
    ax.axvline(x=np.mean(reward_deltas), color='green', linestyle='--', label=f'Mean: {np.mean(reward_deltas):.1f}%')
    ax.set_xlabel("Reward Delta (%)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of Reward Improvements")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_delta_distribution.png", dpi=150)
    plt.close()

    # 4. Improvement rate per turn
    fig, ax = plt.subplots(figsize=(10, 6))
    improvement_rates = [ta.improvement_rate for ta in turn_analyses if ta.turn_index > 0]
    improvement_turns = [ta.turn_index for ta in turn_analyses if ta.turn_index > 0]

    ax.bar(improvement_turns, improvement_rates, color='steelblue', alpha=0.7)
    ax.axhline(y=50, color='red', linestyle='--', label='50% threshold')
    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Improvement Rate (%)")
    ax.set_title("Percentage of Samples Improving at Each Turn")
    ax.set_xticks(improvement_turns)
    ax.set_ylim(0, 100)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_rate_per_turn.png", dpi=150)
    plt.close()

    # 5. Initial vs Final score scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    initial_scores = [r["initial_score"] for r in results]
    final_scores = [r["final_score"] for r in results]

    ax.scatter(initial_scores, final_scores, alpha=0.5, s=20)
    # Add diagonal line (no improvement)
    min_val = min(min(initial_scores), min(final_scores))
    max_val = max(max(initial_scores), max(final_scores))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='No improvement')
    ax.set_xlabel("Initial Score (Turn 0)")
    ax.set_ylabel("Final Score (Last Turn)")
    ax.set_title("Initial vs Final Reward Scores")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "initial_vs_final.png", dpi=150)
    plt.close()

    logger.info(f"Saved visualizations to {output_dir}")


def generate_analysis_report(
    results: List[Dict],
    turn_analyses: List[TurnAnalysis],
    improvement_dist: Dict,
    best_samples: List[Dict],
    worst_samples: List[Dict],
    output_path: Path,
) -> str:
    """Generate detailed analysis report."""
    report = """
================================================================================
                    DETAILED SELF-REFINEMENT ANALYSIS
================================================================================

1. TURN-BY-TURN ANALYSIS
------------------------
"""

    for ta in turn_analyses:
        report += f"""
Turn {ta.turn_index}:
  Samples: {ta.num_samples}
  Mean Score: {ta.mean_score:.2f} (+/- {ta.std_score:.2f})
"""
        if ta.turn_index > 0:
            report += f"  Mean Improvement: {ta.mean_improvement:.2f} (+/- {ta.std_improvement:.2f})\n"
            report += f"  Improvement Rate: {ta.improvement_rate:.1f}%\n"

    report += """
2. IMPROVEMENT DISTRIBUTION
---------------------------
"""
    report += f"""
Total Samples: {improvement_dist['total_samples']}

Outcome Breakdown:
  - Improved: {improvement_dist['improved_count']} ({improvement_dist['improved_pct']:.1f}%)
  - Degraded: {improvement_dist['degraded_count']} ({improvement_dist['degraded_pct']:.1f}%)
  - Unchanged: {improvement_dist['unchanged_count']} ({improvement_dist['unchanged_pct']:.1f}%)

Reward Delta Statistics:
  Mean: {improvement_dist['reward_delta']['mean']*100:.1f}%
  Std: {improvement_dist['reward_delta']['std']*100:.1f}%
  Min: {improvement_dist['reward_delta']['min']*100:.1f}%
  Max: {improvement_dist['reward_delta']['max']*100:.1f}%

  Percentiles:
    10th: {improvement_dist['reward_delta']['percentiles']['p10']*100:.1f}%
    25th: {improvement_dist['reward_delta']['percentiles']['p25']*100:.1f}%
    50th (Median): {improvement_dist['reward_delta']['percentiles']['p50']*100:.1f}%
    75th: {improvement_dist['reward_delta']['percentiles']['p75']*100:.1f}%
    90th: {improvement_dist['reward_delta']['percentiles']['p90']*100:.1f}%
"""

    report += """
3. BEST PERFORMING SAMPLES
--------------------------
"""
    for i, sample in enumerate(best_samples, 1):
        report += f"""
#{i}: Sample {sample.get('sample_id', sample.get('sample_index', '?'))}
  Initial Score: {sample['initial_score']:.2f}
  Final Score: {sample['final_score']:.2f}
  Reward Delta: {sample['reward_delta']*100:.1f}%
  Monotonic: {sample['is_monotonic']}
"""

    report += """
4. WORST PERFORMING SAMPLES
---------------------------
"""
    for i, sample in enumerate(worst_samples, 1):
        report += f"""
#{i}: Sample {sample.get('sample_id', sample.get('sample_index', '?'))}
  Initial Score: {sample['initial_score']:.2f}
  Final Score: {sample['final_score']:.2f}
  Reward Delta: {sample['reward_delta']*100:.1f}%
  Monotonic: {sample['is_monotonic']}
"""

    report += "\n================================================================================\n"

    with open(output_path, "w") as f:
        f.write(report)

    return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze self-refinement evaluation results"
    )

    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to sample_results.jsonl from evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--compare_path",
        type=str,
        default=None,
        help="Optional path to second results file for comparison",
    )
    parser.add_argument(
        "--n_extreme",
        type=int,
        default=5,
        help="Number of best/worst samples to show",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip generating visualizations",
    )

    return parser.parse_args()


def main():
    """Main analysis function."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(args.results_path)

    if not results:
        logger.error("No results loaded, exiting")
        sys.exit(1)

    # Run analyses
    logger.info("Analyzing turn-by-turn metrics...")
    turn_analyses = analyze_by_turn(results)

    logger.info("Analyzing improvement distribution...")
    improvement_dist = analyze_improvement_distribution(results)

    logger.info("Finding extreme samples...")
    best_samples, worst_samples = find_extreme_samples(results, args.n_extreme)

    # Save analysis results
    analysis_data = {
        "turn_analyses": [
            {
                "turn_index": ta.turn_index,
                "num_samples": ta.num_samples,
                "mean_score": ta.mean_score,
                "std_score": ta.std_score,
                "mean_improvement": ta.mean_improvement,
                "std_improvement": ta.std_improvement,
                "improvement_rate": ta.improvement_rate,
            }
            for ta in turn_analyses
        ],
        "improvement_distribution": improvement_dist,
    }

    with open(output_dir / "detailed_analysis.json", "w") as f:
        json.dump(analysis_data, f, indent=2)

    # Generate report
    report = generate_analysis_report(
        results, turn_analyses, improvement_dist,
        best_samples, worst_samples,
        output_dir / "detailed_analysis_report.txt"
    )
    print(report)

    # Create visualizations
    if not args.no_plots:
        logger.info("Creating visualizations...")
        create_visualizations(results, turn_analyses, output_dir / "plots")

    # Handle comparison if provided
    if args.compare_path:
        logger.info(f"Loading comparison results from {args.compare_path}")
        results2 = load_results(args.compare_path)

        if results2:
            comparison = compare_evaluations(
                results, results2,
                label1=Path(args.results_path).parent.name,
                label2=Path(args.compare_path).parent.name,
            )

            with open(output_dir / "comparison.json", "w") as f:
                json.dump(comparison, f, indent=2)

            print("\n" + "=" * 60)
            print("COMPARISON RESULTS")
            print("=" * 60)
            print(json.dumps(comparison, indent=2))

    logger.info(f"Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
