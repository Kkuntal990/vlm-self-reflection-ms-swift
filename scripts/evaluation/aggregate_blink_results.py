#!/usr/bin/env python3
"""Aggregate BLINK benchmark results from VLMEvalKit per-subtask CSVs.

Reads per-subtask CSV outputs from VLMEvalKit, computes per-subtask accuracy
and overall average, and prints a results table matching the BLINK paper format.

Usage:
    python scripts/evaluation/aggregate_blink_results.py \
        --results_dir /outputs/benchmark_results/blink_self_reflective

    # Compare with base model results
    python scripts/evaluation/aggregate_blink_results.py \
        --results_dir /outputs/benchmark_results/blink_self_reflective \
        --base_results_dir /outputs/benchmark_results/blink_base
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Known baselines from BLINK paper (ECCV 2024)
BLINK_BASELINES: dict[str, float] = {
    "Human": 95.70,
    "GPT-4V": 51.26,
    "Gemini Pro": 45.72,
    "LLaVA-1.5-13B": 38.74,
}

# All BLINK subtasks
ALL_SUBTASKS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]


@dataclass
class SubtaskResult:
    """Result for a single BLINK subtask."""

    subtask: str
    accuracy: float
    num_samples: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def load_vlmevalkit_csv(csv_path: str) -> float | None:
    """Extract accuracy score from a VLMEvalKit result CSV.

    VLMEvalKit outputs vary in column naming. This tries common
    column names in priority order.

    Args:
        csv_path: Path to VLMEvalKit CSV output file.

    Returns:
        Accuracy score as float, or None if not found.
    """
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in ["Overall", "score", "Accuracy", "acc"]:
                    if key in row and row[key]:
                        try:
                            return float(row[key])
                        except ValueError:
                            continue
    except Exception as e:
        logger.warning(f"Failed to parse {csv_path}: {e}")
    return None


def find_subtask_results(results_dir: str) -> dict[str, SubtaskResult]:
    """Find and load all BLINK subtask results from a VLMEvalKit output dir.

    Searches for CSV files matching the pattern BLINK_{SubTask}*.csv.

    Args:
        results_dir: VLMEvalKit work directory path.

    Returns:
        Dictionary mapping subtask name to SubtaskResult.
    """
    results: dict[str, SubtaskResult] = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return results

    # Search for BLINK CSV files (may be in model-name subdirectory)
    for csv_file in results_path.rglob("BLINK_*.csv"):
        # Extract subtask name from filename: BLINK_Art_Style.csv -> Art_Style
        stem = csv_file.stem
        if not stem.startswith("BLINK_"):
            continue
        subtask = stem[len("BLINK_") :]

        # Strip any trailing suffixes VLMEvalKit might add
        for suffix in ["_result", "_score", "_acc"]:
            if subtask.endswith(suffix):
                subtask = subtask[: -len(suffix)]

        accuracy = load_vlmevalkit_csv(str(csv_file))
        if accuracy is not None:
            results[subtask] = SubtaskResult(
                subtask=subtask,
                accuracy=accuracy,
            )
            logger.info(f"  {subtask}: {accuracy:.2f}%")
        else:
            logger.warning(f"  {subtask}: no score found in {csv_file}")

    return results


def print_results_table(
    results: dict[str, SubtaskResult],
    base_results: dict[str, SubtaskResult] | None = None,
    model_name: str = "Model",
) -> None:
    """Print a formatted results table.

    Args:
        results: Per-subtask results.
        base_results: Optional base model results for delta computation.
        model_name: Display name for the model.
    """
    has_base = base_results is not None and len(base_results) > 0

    # Header
    print()
    print("=" * 70)
    print("BLINK Benchmark Results")
    print("=" * 70)
    print()

    if has_base:
        header = f"{'Subtask':<30} {model_name:>10} {'Base':>10} {'Delta':>10}"
    else:
        header = f"{'Subtask':<30} {model_name:>10}"
    print(header)
    print("-" * len(header))

    # Per-subtask rows
    found_subtasks = []
    for subtask in ALL_SUBTASKS:
        if subtask in results:
            acc = results[subtask].accuracy
            found_subtasks.append(subtask)
            if has_base and subtask in base_results:
                base_acc = base_results[subtask].accuracy
                delta = acc - base_acc
                sign = "+" if delta >= 0 else ""
                print(f"  {subtask:<28} {acc:>9.2f}% {base_acc:>9.2f}% {sign}{delta:>8.2f}")
            else:
                print(f"  {subtask:<28} {acc:>9.2f}%")
        else:
            print(f"  {subtask:<28} {'N/A':>10}")

    # Overall average
    if found_subtasks:
        avg = sum(results[s].accuracy for s in found_subtasks) / len(found_subtasks)
        print("-" * len(header))
        if has_base:
            base_found = [s for s in found_subtasks if s in base_results]
            if base_found:
                base_avg = sum(base_results[s].accuracy for s in base_found) / len(base_found)
                delta = avg - base_avg
                sign = "+" if delta >= 0 else ""
                print(
                    f"  {'Overall Average':<28} {avg:>9.2f}% {base_avg:>9.2f}% {sign}{delta:>8.2f}"
                )
            else:
                print(f"  {'Overall Average':<28} {avg:>9.2f}%")
        else:
            print(f"  {'Overall Average':<28} {avg:>9.2f}%")

        # Known baselines
        print()
        print("Known baselines (BLINK paper, ECCV 2024):")
        for name, score in BLINK_BASELINES.items():
            print(f"  {name:<28} {score:>9.2f}%")

    print()


def save_results_json(
    results: dict[str, SubtaskResult],
    output_path: str,
    model_name: str = "",
) -> None:
    """Save results to JSON file.

    Args:
        results: Per-subtask results.
        output_path: Path to output JSON file.
        model_name: Model identifier.
    """
    found = [s for s in ALL_SUBTASKS if s in results]
    avg = sum(results[s].accuracy for s in found) / len(found) if found else 0.0

    output = {
        "model_name": model_name,
        "overall_accuracy": round(avg, 2),
        "num_subtasks_found": len(found),
        "num_subtasks_total": len(ALL_SUBTASKS),
        "per_subtask": {s: results[s].to_dict() for s in found},
        "baselines": BLINK_BASELINES,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved results to {output_path}")


def main() -> None:
    """Aggregate BLINK benchmark results and print summary table."""
    parser = argparse.ArgumentParser(
        description="Aggregate BLINK benchmark results from VLMEvalKit."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="VLMEvalKit work directory with BLINK results.",
    )
    parser.add_argument(
        "--base_results_dir",
        type=str,
        default=None,
        help="Optional base model results directory for comparison.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Model",
        help="Display name for the model.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save results as JSON. Default: {results_dir}/blink_summary.json",
    )
    args = parser.parse_args()

    # Load results
    logger.info(f"Loading results from: {args.results_dir}")
    results = find_subtask_results(args.results_dir)

    base_results = None
    if args.base_results_dir:
        logger.info(f"Loading base results from: {args.base_results_dir}")
        base_results = find_subtask_results(args.base_results_dir)

    if not results:
        logger.error("No BLINK results found. Check the results directory.")
        sys.exit(1)

    # Print table
    print_results_table(results, base_results, args.model_name)

    # Save JSON
    output_json = args.output_json or os.path.join(args.results_dir, "blink_summary.json")
    save_results_json(results, output_json, args.model_name)


if __name__ == "__main__":
    main()
