#!/usr/bin/env python3
"""Compare benchmark evaluation results across frameworks and models.

Loads results from VLMEvalKit and lmms-eval output directories, aligns
scores by benchmark, and produces a comparison table.

Usage:
    python scripts/evaluation/compare_benchmark_results.py \
        --vlmevalkit-dir /outputs/benchmark_results/vlmevalkit \
        --lmms-eval-dir /outputs/benchmark_results/lmms_eval \
        --output-dir /outputs/benchmark_results/comparison

    # Compare fine-tuned vs base model (lmms-eval only)
    python scripts/evaluation/compare_benchmark_results.py \
        --lmms-eval-dir /outputs/benchmark_results/lmms_eval \
        --lmms-eval-base-dir /outputs/benchmark_results/lmms_eval_base \
        --output-dir /outputs/benchmark_results/comparison
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Canonical benchmark names mapping
BENCHMARK_ALIASES: dict[str, str] = {
    # VLMEvalKit names -> canonical
    "MMBench_DEV_EN": "MMBench",
    "MMBench_DEV_EN_V11": "MMBench-v1.1",
    "MME": "MME",
    "SEEDBench_IMG": "SEED-Bench",
    "MMMU_DEV_VAL": "MMMU",
    "MMVet": "MM-Vet",
    "AI2D_TEST": "AI2D",
    "OCRBench": "OCRBench",
    "MathVista_MINI": "MathVista",
    # lmms-eval names -> canonical
    "mmbench_en_dev": "MMBench",
    "mme": "MME",
    "seedbench": "SEED-Bench",
    "mmmu_val": "MMMU",
    "mmvet": "MM-Vet",
    "ai2d": "AI2D",
    "ocrbench": "OCRBench",
    "mathvista_testmini": "MathVista",
}


@dataclass
class BenchmarkScore:
    """Score for a single benchmark from one source."""

    benchmark: str
    score: float
    metric_name: str = "accuracy"
    num_samples: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ComparisonRow:
    """A row in the comparison table."""

    benchmark: str
    vlmevalkit_score: float | None = None
    lmms_eval_score: float | None = None
    base_model_score: float | None = None
    delta_vs_base: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ComparisonReport:
    """Full comparison report."""

    rows: list[ComparisonRow] = field(default_factory=list)
    model_path: str = ""
    base_model_path: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_path": self.model_path,
            "base_model_path": self.base_model_path,
            "benchmarks": [r.to_dict() for r in self.rows],
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Compare VLM benchmark results across frameworks and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Result directories
    parser.add_argument(
        "--vlmevalkit-dir",
        type=str,
        default=None,
        help="VLMEvalKit results directory",
    )
    parser.add_argument(
        "--lmms-eval-dir",
        type=str,
        default=None,
        help="lmms-eval results directory (fine-tuned model)",
    )
    parser.add_argument(
        "--lmms-eval-base-dir",
        type=str,
        default=None,
        help="lmms-eval results directory (base model, for delta computation)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_comparison",
        help="Output directory for comparison report",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Model path (for report metadata)",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Base model path (for report metadata)",
    )

    return parser.parse_args()


def load_vlmevalkit_results(results_dir: str) -> dict[str, BenchmarkScore]:
    """Load VLMEvalKit results from CSV files.

    VLMEvalKit saves results as CSV files in the work directory,
    typically named like {model_name}/{benchmark}.csv.

    Args:
        results_dir: Path to VLMEvalKit work directory.

    Returns:
        Dictionary mapping canonical benchmark name to score.
    """
    scores: dict[str, BenchmarkScore] = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.warning(f"VLMEvalKit results directory not found: {results_dir}")
        return scores

    # VLMEvalKit saves results as CSV files
    for csv_file in results_path.rglob("*.csv"):
        try:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # VLMEvalKit CSV format varies by benchmark
                    # Common columns: "split", "Overall", "score"
                    score_val = None
                    for key in ["Overall", "score", "Accuracy", "acc"]:
                        if key in row and row[key]:
                            try:
                                score_val = float(row[key])
                                break
                            except ValueError:
                                continue

                    if score_val is not None:
                        bench_name = csv_file.stem
                        canonical = BENCHMARK_ALIASES.get(bench_name, bench_name)
                        scores[canonical] = BenchmarkScore(
                            benchmark=canonical,
                            score=score_val,
                            metric_name="Overall",
                        )
        except Exception as e:
            logger.warning(f"Failed to parse {csv_file}: {e}")

    logger.info(f"Loaded {len(scores)} VLMEvalKit benchmark scores")
    return scores


def load_lmms_eval_results(results_dir: str) -> dict[str, BenchmarkScore]:
    """Load lmms-eval results from JSON output files.

    lmms-eval saves results as JSON files with structure:
    {task_name: {metric_name: score, ...}, ...}

    Args:
        results_dir: Path to lmms-eval output directory.

    Returns:
        Dictionary mapping canonical benchmark name to score.
    """
    scores: dict[str, BenchmarkScore] = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.warning(f"lmms-eval results directory not found: {results_dir}")
        return scores

    # lmms-eval outputs results.json in the output directory
    for json_file in results_path.rglob("results*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # lmms-eval results format: {"results": {task: {metric: value}}}
            results_data = data.get("results", data)

            for task_name, metrics in results_data.items():
                if not isinstance(metrics, dict):
                    continue

                # Find the primary score metric
                score_val = None
                metric_name = ""
                for key in [
                    "acc,none",
                    "accuracy,none",
                    "exact_match,none",
                    "score,none",
                    "acc",
                    "accuracy",
                    "exact_match",
                    "score",
                ]:
                    if key in metrics:
                        try:
                            score_val = float(metrics[key])
                            metric_name = key.split(",")[0]
                            break
                        except (ValueError, TypeError):
                            continue

                # Fallback: take first numeric value
                if score_val is None:
                    for key, val in metrics.items():
                        if key.startswith("alias"):
                            continue
                        try:
                            score_val = float(val)
                            metric_name = key.split(",")[0]
                            break
                        except (ValueError, TypeError):
                            continue

                if score_val is not None:
                    canonical = BENCHMARK_ALIASES.get(task_name, task_name)
                    scores[canonical] = BenchmarkScore(
                        benchmark=canonical,
                        score=score_val,
                        metric_name=metric_name,
                    )
        except Exception as e:
            logger.warning(f"Failed to parse {json_file}: {e}")

    logger.info(f"Loaded {len(scores)} lmms-eval benchmark scores")
    return scores


def build_comparison(
    vlmevalkit_scores: dict[str, BenchmarkScore],
    lmms_eval_scores: dict[str, BenchmarkScore],
    base_scores: dict[str, BenchmarkScore],
) -> list[ComparisonRow]:
    """Build comparison rows from all score sources.

    Args:
        vlmevalkit_scores: Scores from VLMEvalKit.
        lmms_eval_scores: Scores from lmms-eval (fine-tuned).
        base_scores: Scores from lmms-eval (base model).

    Returns:
        List of comparison rows sorted by benchmark name.
    """
    all_benchmarks = sorted(
        set(vlmevalkit_scores.keys()) | set(lmms_eval_scores.keys()) | set(base_scores.keys())
    )

    rows = []
    for bench in all_benchmarks:
        row = ComparisonRow(benchmark=bench)

        if bench in vlmevalkit_scores:
            row.vlmevalkit_score = vlmevalkit_scores[bench].score
        if bench in lmms_eval_scores:
            row.lmms_eval_score = lmms_eval_scores[bench].score
        if bench in base_scores:
            row.base_model_score = base_scores[bench].score

        # Compute delta vs base (prefer lmms-eval score for delta)
        finetuned_score = row.lmms_eval_score or row.vlmevalkit_score
        if finetuned_score is not None and row.base_model_score is not None:
            row.delta_vs_base = finetuned_score - row.base_model_score

        rows.append(row)

    return rows


def format_markdown_table(report: ComparisonReport) -> str:
    """Format comparison report as a markdown table.

    Args:
        report: The comparison report to format.

    Returns:
        Markdown-formatted table string.
    """
    lines = []
    lines.append("# VLM Benchmark Comparison\n")

    if report.model_path:
        lines.append(f"**Fine-tuned model**: `{report.model_path}`\n")
    if report.base_model_path:
        lines.append(f"**Base model**: `{report.base_model_path}`\n")

    # Build header
    has_vlmevalkit = any(r.vlmevalkit_score is not None for r in report.rows)
    has_lmms_eval = any(r.lmms_eval_score is not None for r in report.rows)
    has_base = any(r.base_model_score is not None for r in report.rows)
    has_delta = any(r.delta_vs_base is not None for r in report.rows)

    header = "| Benchmark"
    separator = "|---"
    if has_vlmevalkit:
        header += " | VLMEvalKit"
        separator += "|---"
    if has_lmms_eval:
        header += " | lmms-eval"
        separator += "|---"
    if has_base:
        header += " | Base Model"
        separator += "|---"
    if has_delta:
        header += " | Delta"
        separator += "|---"
    header += " |"
    separator += "|"

    lines.append(header)
    lines.append(separator)

    for row in report.rows:
        line = f"| {row.benchmark}"
        if has_vlmevalkit:
            val = f"{row.vlmevalkit_score:.2f}" if row.vlmevalkit_score is not None else "-"
            line += f" | {val}"
        if has_lmms_eval:
            val = f"{row.lmms_eval_score:.2f}" if row.lmms_eval_score is not None else "-"
            line += f" | {val}"
        if has_base:
            val = f"{row.base_model_score:.2f}" if row.base_model_score is not None else "-"
            line += f" | {val}"
        if has_delta:
            if row.delta_vs_base is not None:
                sign = "+" if row.delta_vs_base >= 0 else ""
                val = f"{sign}{row.delta_vs_base:.2f}"
            else:
                val = "-"
            line += f" | {val}"
        line += " |"
        lines.append(line)

    return "\n".join(lines)


def save_report(
    report: ComparisonReport,
    output_dir: str,
) -> None:
    """Save comparison report to output directory.

    Args:
        report: The comparison report to save.
        output_dir: Directory to write output files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_path / "benchmark_comparison.json"
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Saved JSON report to {json_path}")

    # Save markdown table
    md_path = output_path / "benchmark_comparison.md"
    md_content = format_markdown_table(report)
    with open(md_path, "w") as f:
        f.write(md_content + "\n")
    logger.info(f"Saved markdown report to {md_path}")

    # Print to console
    print("\n" + md_content + "\n")


def main() -> None:
    """Main function."""
    args = parse_args()

    # Load results from each source
    vlmevalkit_scores: dict[str, BenchmarkScore] = {}
    lmms_eval_scores: dict[str, BenchmarkScore] = {}
    base_scores: dict[str, BenchmarkScore] = {}

    if args.vlmevalkit_dir:
        vlmevalkit_scores = load_vlmevalkit_results(args.vlmevalkit_dir)

    if args.lmms_eval_dir:
        lmms_eval_scores = load_lmms_eval_results(args.lmms_eval_dir)

    if args.lmms_eval_base_dir:
        base_scores = load_lmms_eval_results(args.lmms_eval_base_dir)

    if not vlmevalkit_scores and not lmms_eval_scores:
        logger.error("No results found. Provide at least one results directory.")
        sys.exit(1)

    # Build comparison
    rows = build_comparison(vlmevalkit_scores, lmms_eval_scores, base_scores)

    report = ComparisonReport(
        rows=rows,
        model_path=args.model_path,
        base_model_path=args.base_model_path,
    )

    # Save report
    save_report(report, args.output_dir)


if __name__ == "__main__":
    main()
