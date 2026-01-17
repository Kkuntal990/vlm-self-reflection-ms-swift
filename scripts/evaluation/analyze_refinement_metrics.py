#!/usr/bin/env python3
"""
Analyze and visualize self-refinement evaluation results.

This script provides post-hoc analysis of evaluation results including:
- Statistical analysis of reward scores
- Pairwise comparison metrics (VLM judge results)
- Comparison between ground truth and generated modes
- Visualization of improvement trends
- Per-turn analysis and breakdowns
- Identification of best/worst performing samples

Usage:
    # Basic analysis (reward model scores)
    python scripts/evaluation/analyze_refinement_metrics.py \
        --results_path /outputs/eval_results/sample_results.jsonl \
        --output_dir /outputs/eval_results/analysis

    # Analysis with pairwise metrics
    python scripts/evaluation/analyze_refinement_metrics.py \
        --results_path /outputs/eval_results/sample_results.jsonl \
        --pairwise_path /outputs/eval_results/pairwise_metrics.json \
        --output_dir /outputs/eval_results/analysis

    # Compare two evaluation runs
    python scripts/evaluation/analyze_refinement_metrics.py \
        --results_path /outputs/eval_gt/sample_results.jsonl \
        --compare_path /outputs/eval_gen/sample_results.jsonl \
        --output_dir /outputs/comparison_analysis
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# Add scripts directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from vlm_judge import PairwiseMetrics, compute_pairwise_metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class TurnAnalysis:
    """Analysis for a specific turn index (reward model scores)."""

    turn_index: int
    num_samples: int
    mean_score: float
    std_score: float
    mean_improvement: float
    std_improvement: float
    improvement_rate: float  # % of samples that improved at this turn


@dataclass
class PairwiseTurnAnalysis:
    """Analysis for pairwise comparisons at a specific turn index."""

    turn_index: int
    num_comparisons: int
    win_rate: float  # B wins / total
    loss_rate: float  # A wins / total (regression)
    tie_rate: float
    high_confidence_rate: float
    position_bias_rate: float


def load_results(results_path: str) -> list[dict]:
    """Load evaluation results from JSONL file."""
    results = []

    with open(results_path) as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")

    logger.info(f"Loaded {len(results)} results from {results_path}")
    return results


def analyze_by_turn(results: list[dict]) -> list[TurnAnalysis]:
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
                improvement = turn["reward_score"] - turns[i - 1]["reward_score"]
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

        analyses.append(
            TurnAnalysis(
                turn_index=turn_idx,
                num_samples=len(scores),
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores)),
                mean_improvement=float(np.mean(improvements)) if improvements else 0.0,
                std_improvement=float(np.std(improvements)) if improvements else 0.0,
                improvement_rate=improvement_rate,
            )
        )

    return analyses


def analyze_pairwise_by_turn(results: list[dict]) -> list[PairwiseTurnAnalysis]:
    """Analyze pairwise comparison metrics broken down by turn index.

    Args:
        results: List of sample results with pairwise data in turns

    Returns:
        List of PairwiseTurnAnalysis for each turn index (starting from turn 1)
    """
    # Group pairwise results by turn
    turn_data: dict[int, dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "ties": 0, "high_conf": 0, "bias_detected": 0, "total": 0}
    )

    for r in results:
        turns = r.get("turns", [])
        for turn in turns:
            pairwise = turn.get("pairwise")
            if not pairwise:
                continue

            turn_idx = turn.get("turn_index", 0)
            if turn_idx == 0:
                # Skip turn 0 (no pairwise for initial response)
                continue

            data = turn_data[turn_idx]
            data["total"] += 1

            better = pairwise.get("better", "tie")
            if better == "B":
                data["wins"] += 1
            elif better == "A":
                data["losses"] += 1
            else:
                data["ties"] += 1

            if pairwise.get("confidence") == "high":
                data["high_conf"] += 1
            if pairwise.get("position_bias_detected", False):
                data["bias_detected"] += 1

    # Build analysis objects
    analyses = []
    for turn_idx in sorted(turn_data.keys()):
        data = turn_data[turn_idx]
        total = data["total"]
        if total == 0:
            continue

        analyses.append(
            PairwiseTurnAnalysis(
                turn_index=turn_idx,
                num_comparisons=total,
                win_rate=data["wins"] / total,
                loss_rate=data["losses"] / total,
                tie_rate=data["ties"] / total,
                high_confidence_rate=data["high_conf"] / total,
                position_bias_rate=data["bias_detected"] / total,
            )
        )

    return analyses


def has_pairwise_data(results: list[dict]) -> bool:
    """Check if results contain pairwise evaluation data."""
    for r in results:
        turns = r.get("turns", [])
        for turn in turns:
            if turn.get("pairwise"):
                return True
    return False


def analyze_improvement_distribution(results: list[dict]) -> dict:
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
    results: list[dict],
    n: int = 5,
) -> tuple[list[dict], list[dict]]:
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
    results1: list[dict],
    results2: list[dict],
    label1: str = "Evaluation 1",
    label2: str = "Evaluation 2",
) -> dict:
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
                sum(1 for r in results if r["final_score"] > r["initial_score"])
                / len(results)
                * 100
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
    results: list[dict],
    turn_analyses: list[TurnAnalysis],
    output_dir: Path,
    pairwise_turn_analyses: list[PairwiseTurnAnalysis] | None = None,
):
    """Create visualization plots.

    Args:
        results: Evaluation results
        turn_analyses: Per-turn analysis (reward model scores)
        output_dir: Directory to save plots
        pairwise_turn_analyses: Optional per-turn pairwise analysis (VLM judge)
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

    ax.errorbar(turns, mean_scores, yerr=std_scores, marker="o", capsize=5, capthick=2)
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

    colors = ["green" if imp > 0 else "red" for imp in improvements]
    ax.bar(improvement_turns, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
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

    ax.hist(reward_deltas, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", label="No improvement")
    ax.axvline(
        x=np.mean(reward_deltas),
        color="green",
        linestyle="--",
        label=f"Mean: {np.mean(reward_deltas):.1f}%",
    )
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

    ax.bar(improvement_turns, improvement_rates, color="steelblue", alpha=0.7)
    ax.axhline(y=50, color="red", linestyle="--", label="50% threshold")
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
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="No improvement")
    ax.set_xlabel("Initial Score (Turn 0)")
    ax.set_ylabel("Final Score (Last Turn)")
    ax.set_title("Initial vs Final Reward Scores")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "initial_vs_final.png", dpi=150)
    plt.close()

    # Pairwise visualizations (if available)
    if pairwise_turn_analyses:
        # 6. Pairwise win/loss/tie rates by turn
        fig, ax = plt.subplots(figsize=(12, 6))
        turns = [ta.turn_index for ta in pairwise_turn_analyses]
        win_rates = [ta.win_rate * 100 for ta in pairwise_turn_analyses]
        loss_rates = [ta.loss_rate * 100 for ta in pairwise_turn_analyses]
        tie_rates = [ta.tie_rate * 100 for ta in pairwise_turn_analyses]

        x = np.arange(len(turns))
        width = 0.25

        ax.bar(x - width, win_rates, width, label="Win (B > A)", color="green", alpha=0.7)
        ax.bar(x, tie_rates, width, label="Tie", color="gray", alpha=0.7)
        ax.bar(x + width, loss_rates, width, label="Loss (A > B)", color="red", alpha=0.7)

        ax.set_xlabel("Turn Index")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Pairwise Comparison Results by Turn")
        ax.set_xticks(x)
        ax.set_xticklabels(turns)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.axhline(y=50, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "pairwise_by_turn.png", dpi=150)
        plt.close()

        # 7. Position bias detection rate by turn
        fig, ax = plt.subplots(figsize=(10, 6))
        bias_rates = [ta.position_bias_rate * 100 for ta in pairwise_turn_analyses]
        conf_rates = [ta.high_confidence_rate * 100 for ta in pairwise_turn_analyses]

        x = np.arange(len(turns))
        width = 0.35

        ax.bar(
            x - width / 2, conf_rates, width, label="High Confidence", color="steelblue", alpha=0.7
        )
        ax.bar(
            x + width / 2,
            bias_rates,
            width,
            label="Position Bias Detected",
            color="orange",
            alpha=0.7,
        )

        ax.set_xlabel("Turn Index")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Judge Confidence and Position Bias by Turn")
        ax.set_xticks(x)
        ax.set_xticklabels(turns)
        ax.set_ylim(0, 100)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "pairwise_confidence_by_turn.png", dpi=150)
        plt.close()

        # 8. Overall pairwise pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        total_wins = sum(ta.win_rate * ta.num_comparisons for ta in pairwise_turn_analyses)
        total_losses = sum(ta.loss_rate * ta.num_comparisons for ta in pairwise_turn_analyses)
        total_ties = sum(ta.tie_rate * ta.num_comparisons for ta in pairwise_turn_analyses)
        total = total_wins + total_losses + total_ties

        if total > 0:
            sizes = [total_wins / total * 100, total_ties / total * 100, total_losses / total * 100]
            labels = [
                f"Win (B > A)\n{sizes[0]:.1f}%",
                f"Tie\n{sizes[1]:.1f}%",
                f"Loss (A > B)\n{sizes[2]:.1f}%",
            ]
            colors = ["green", "gray", "red"]
            explode = (0.05, 0, 0.05)

            ax.pie(
                sizes,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct="",
                shadow=False,
                startangle=90,
                textprops={"fontsize": 11},
            )
            ax.set_title("Overall Pairwise Comparison Results")
            plt.tight_layout()
            plt.savefig(output_dir / "pairwise_overall_pie.png", dpi=150)
        plt.close()

    logger.info(f"Saved visualizations to {output_dir}")


def generate_analysis_report(
    results: list[dict],
    turn_analyses: list[TurnAnalysis],
    improvement_dist: dict,
    best_samples: list[dict],
    worst_samples: list[dict],
    output_path: Path,
    pairwise_turn_analyses: list[PairwiseTurnAnalysis] | None = None,
    pairwise_metrics: PairwiseMetrics | None = None,
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
            report += (
                f"  Mean Improvement: {ta.mean_improvement:.2f} (+/- {ta.std_improvement:.2f})\n"
            )
            report += f"  Improvement Rate: {ta.improvement_rate:.1f}%\n"

    report += """
2. IMPROVEMENT DISTRIBUTION
---------------------------
"""
    report += f"""
Total Samples: {improvement_dist["total_samples"]}

Outcome Breakdown:
  - Improved: {improvement_dist["improved_count"]} ({improvement_dist["improved_pct"]:.1f}%)
  - Degraded: {improvement_dist["degraded_count"]} ({improvement_dist["degraded_pct"]:.1f}%)
  - Unchanged: {improvement_dist["unchanged_count"]} ({improvement_dist["unchanged_pct"]:.1f}%)

Reward Delta Statistics:
  Mean: {improvement_dist["reward_delta"]["mean"] * 100:.1f}%
  Std: {improvement_dist["reward_delta"]["std"] * 100:.1f}%
  Min: {improvement_dist["reward_delta"]["min"] * 100:.1f}%
  Max: {improvement_dist["reward_delta"]["max"] * 100:.1f}%

  Percentiles:
    10th: {improvement_dist["reward_delta"]["percentiles"]["p10"] * 100:.1f}%
    25th: {improvement_dist["reward_delta"]["percentiles"]["p25"] * 100:.1f}%
    50th (Median): {improvement_dist["reward_delta"]["percentiles"]["p50"] * 100:.1f}%
    75th: {improvement_dist["reward_delta"]["percentiles"]["p75"] * 100:.1f}%
    90th: {improvement_dist["reward_delta"]["percentiles"]["p90"] * 100:.1f}%
"""

    report += """
3. BEST PERFORMING SAMPLES
--------------------------
"""
    for i, sample in enumerate(best_samples, 1):
        report += f"""
#{i}: Sample {sample.get("sample_id", sample.get("sample_index", "?"))}
  Initial Score: {sample["initial_score"]:.2f}
  Final Score: {sample["final_score"]:.2f}
  Reward Delta: {sample["reward_delta"] * 100:.1f}%
  Monotonic: {sample["is_monotonic"]}
"""

    report += """
4. WORST PERFORMING SAMPLES
---------------------------
"""
    for i, sample in enumerate(worst_samples, 1):
        report += f"""
#{i}: Sample {sample.get("sample_id", sample.get("sample_index", "?"))}
  Initial Score: {sample["initial_score"]:.2f}
  Final Score: {sample["final_score"]:.2f}
  Reward Delta: {sample["reward_delta"] * 100:.1f}%
  Monotonic: {sample["is_monotonic"]}
"""

    # Add pairwise analysis if available
    if pairwise_turn_analyses or pairwise_metrics:
        report += """
================================================================================
                    VLM JUDGE PAIRWISE ANALYSIS
================================================================================
"""

    if pairwise_metrics:
        report += f"""
5. OVERALL PAIRWISE METRICS
---------------------------
  Win Rate (B > A):         {pairwise_metrics.pairwise_win_rate:.1%}
  Regression Rate (A > B):  {pairwise_metrics.regression_rate:.1%}
  Tie Rate:                 {pairwise_metrics.tie_rate:.1%}

  High Confidence Rate:     {pairwise_metrics.high_confidence_rate:.1%}
  Position Bias Rate:       {pairwise_metrics.position_bias_rate:.1%}

  Total Comparisons:        {pairwise_metrics.total_comparisons}
    - Wins:   {pairwise_metrics.total_wins}
    - Losses: {pairwise_metrics.total_losses}
    - Ties:   {pairwise_metrics.total_ties}
"""

    if pairwise_turn_analyses:
        report += """
6. PAIRWISE RESULTS BY TURN
---------------------------
"""
        for pta in pairwise_turn_analyses:
            report += f"""
Turn {pta.turn_index}:
  Comparisons: {pta.num_comparisons}
  Win Rate:    {pta.win_rate:.1%}
  Loss Rate:   {pta.loss_rate:.1%}
  Tie Rate:    {pta.tie_rate:.1%}
  High Conf:   {pta.high_confidence_rate:.1%}
  Pos. Bias:   {pta.position_bias_rate:.1%}
"""

    report += "\n================================================================================\n"

    with open(output_path, "w") as f:
        f.write(report)

    return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze self-refinement evaluation results")

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
        "--pairwise_path",
        type=str,
        default=None,
        help="Optional path to pairwise_metrics.json from evaluation",
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

    # Pairwise analysis (if data available)
    pairwise_turn_analyses: list[PairwiseTurnAnalysis] | None = None
    pairwise_metrics: PairwiseMetrics | None = None

    # Check if results contain pairwise data
    if has_pairwise_data(results):
        logger.info("Analyzing pairwise comparison data...")
        pairwise_turn_analyses = analyze_pairwise_by_turn(results)
        pairwise_metrics = compute_pairwise_metrics(results)
        logger.info(f"Pairwise win rate: {pairwise_metrics.pairwise_win_rate:.1%}")

    # Load separate pairwise metrics file if provided
    if args.pairwise_path:
        logger.info(f"Loading pairwise metrics from {args.pairwise_path}")
        try:
            with open(args.pairwise_path) as f:
                pairwise_data = json.load(f)
                pairwise_metrics = PairwiseMetrics(**pairwise_data)
        except Exception as e:
            logger.warning(f"Failed to load pairwise metrics: {e}")

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

    # Add pairwise data if available
    if pairwise_turn_analyses:
        analysis_data["pairwise_turn_analyses"] = [asdict(pta) for pta in pairwise_turn_analyses]
    if pairwise_metrics:
        analysis_data["pairwise_metrics"] = pairwise_metrics.to_dict()

    with open(output_dir / "detailed_analysis.json", "w") as f:
        json.dump(analysis_data, f, indent=2)

    # Generate report
    report = generate_analysis_report(
        results,
        turn_analyses,
        improvement_dist,
        best_samples,
        worst_samples,
        output_dir / "detailed_analysis_report.txt",
        pairwise_turn_analyses=pairwise_turn_analyses,
        pairwise_metrics=pairwise_metrics,
    )
    print(report)

    # Create visualizations
    if not args.no_plots:
        logger.info("Creating visualizations...")
        create_visualizations(
            results,
            turn_analyses,
            output_dir / "plots",
            pairwise_turn_analyses=pairwise_turn_analyses,
        )

    # Handle comparison if provided
    if args.compare_path:
        logger.info(f"Loading comparison results from {args.compare_path}")
        results2 = load_results(args.compare_path)

        if results2:
            comparison = compare_evaluations(
                results,
                results2,
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
