#!/usr/bin/env python3
"""Analyze per-turn accuracy from BLINK self-reflective evaluation logs.

Reads the conversation_log.jsonl produced by the self-reflective wrapper
and the BLINK TSV files (for ground truth), then computes:
- A1 accuracy (before self-reflection)
- A2 accuracy (after self-reflection)
- Transition matrix: RR, RW, WR, WW counts and percentages
- Per-subtask breakdown

Usage:
    python scripts/evaluation/analyze_blink_turns.py \
        --conversation_log /outputs/benchmark_results/blink_self_reflective_v1/conversation_log.jsonl \
        --blink_tsv_dir /outputs/benchmark_data/blink/tsv

    # Compare two runs
    python scripts/evaluation/analyze_blink_turns.py \
        --conversation_log /outputs/benchmark_results/blink_self_reflective_v1/conversation_log.jsonl \
        --conversation_log_2 /outputs/benchmark_results/blink_self_reflective_v3/conversation_log.jsonl \
        --blink_tsv_dir /outputs/benchmark_data/blink/tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MCQ_PATTERNS = [
    re.compile(r"<answer>\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"[Aa]nswer\s*(?:is|:)\s*\(?([A-D])\)?"),
    re.compile(r"[Tt]herefore[,\s]+.*\(?([A-D])\)?"),
    re.compile(r"\(([A-D])\)"),
    re.compile(r"(?:^|\s)([A-D])\.\s"),
    re.compile(r"(?:^|\s)([A-D])(?:\s|$|\.|,)"),
]


def extract_mcq_letter(response: str) -> str:
    """Extract MCQ letter from a model response."""
    for pat in MCQ_PATTERNS:
        m = pat.search(response)
        if m:
            return m.group(1).upper()
    return response


@dataclass
class SubtaskTurnStats:
    """Per-subtask turn statistics."""

    subtask: str
    total: int = 0
    a1_correct: int = 0
    a2_correct: int = 0
    rr: int = 0  # right -> right
    rw: int = 0  # right -> wrong
    wr: int = 0  # wrong -> right
    ww: int = 0  # wrong -> wrong

    @property
    def a1_acc(self) -> float:
        """A1 accuracy percentage."""
        return (self.a1_correct / self.total * 100) if self.total > 0 else 0.0

    @property
    def a2_acc(self) -> float:
        """A2 accuracy percentage."""
        return (self.a2_correct / self.total * 100) if self.total > 0 else 0.0

    @property
    def delta(self) -> float:
        """A2 - A1 accuracy delta."""
        return self.a2_acc - self.a1_acc

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "subtask": self.subtask,
            "total": self.total,
            "a1_acc": round(self.a1_acc, 2),
            "a2_acc": round(self.a2_acc, 2),
            "delta": round(self.delta, 2),
            "RR": self.rr,
            "RW": self.rw,
            "WR": self.wr,
            "WW": self.ww,
        }


def load_ground_truth(blink_tsv_dir: str) -> dict[str, dict[int, str]]:
    """Load ground truth answers from BLINK TSV files.

    Args:
        blink_tsv_dir: Directory containing BLINK_*.tsv files.

    Returns:
        Dict mapping dataset name -> {sample_index: answer_letter}.
    """
    csv.field_size_limit(sys.maxsize)
    ground_truth: dict[str, dict[int, str]] = {}
    tsv_dir = Path(blink_tsv_dir)

    for tsv_path in sorted(tsv_dir.glob("BLINK_*.tsv")):
        dataset = tsv_path.stem  # e.g., BLINK_Art_Style
        gt: dict[int, str] = {}
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                idx = int(row["index"])
                answer = row.get("answer", "").strip()
                # Normalize: strip parentheses if present
                answer = re.sub(r"^\(([A-D])\)$", r"\1", answer)
                gt[idx] = answer
        ground_truth[dataset] = gt
        logger.info(f"  Loaded {len(gt)} ground truth answers for {dataset}")

    return ground_truth


def analyze_conversation_log(
    log_path: str,
    ground_truth: dict[str, dict[int, str]],
) -> dict[str, SubtaskTurnStats]:
    """Analyze per-turn accuracy from conversation log.

    Args:
        log_path: Path to conversation_log.jsonl.
        ground_truth: Ground truth from load_ground_truth().

    Returns:
        Dict mapping subtask name -> SubtaskTurnStats.
    """
    stats: dict[str, SubtaskTurnStats] = {}
    # Track per-dataset sequential index (sample_id is global, GT uses per-dataset)
    dataset_counters: dict[str, int] = defaultdict(int)

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            conv = json.loads(line)
            dataset = conv.get("dataset", "")

            if dataset not in ground_truth:
                continue

            # Use sequential per-dataset counter to match GT index
            local_idx = dataset_counters[dataset]
            dataset_counters[dataset] += 1

            if local_idx not in ground_truth[dataset]:
                continue

            gt_answer = ground_truth[dataset][local_idx]

            if dataset not in stats:
                stats[dataset] = SubtaskTurnStats(subtask=dataset)
            st = stats[dataset]
            st.total += 1

            # Extract A1 answer
            a1_extracted = conv.get("a1_extracted")
            if a1_extracted is None:
                # Fallback for logs without a1_extracted field
                a1_extracted = extract_mcq_letter(conv.get("initial_answer", ""))

            # Extract A2 answer (final after all turns)
            final_extracted = conv.get("final_extracted")
            if final_extracted is None:
                turns = conv.get("turns", [])
                if turns:
                    last_turn = turns[-1]
                    a2_raw = last_turn.get("answer_after", "")
                    final_extracted = last_turn.get(
                        "answer_after_extracted", extract_mcq_letter(a2_raw)
                    )
                else:
                    final_extracted = a1_extracted

            a1_correct = a1_extracted == gt_answer
            a2_correct = final_extracted == gt_answer

            if a1_correct:
                st.a1_correct += 1
            if a2_correct:
                st.a2_correct += 1

            # Transition
            if a1_correct and a2_correct:
                st.rr += 1
            elif a1_correct and not a2_correct:
                st.rw += 1
            elif not a1_correct and a2_correct:
                st.wr += 1
            else:
                st.ww += 1

    return stats


def print_results(
    stats: dict[str, SubtaskTurnStats],
    label: str = "Model",
) -> None:
    """Print formatted per-turn accuracy results.

    Args:
        stats: Per-subtask stats from analyze_conversation_log().
        label: Display label for the model.
    """
    print()
    print("=" * 90)
    print(f"Per-Turn Accuracy Analysis: {label}")
    print("=" * 90)
    print()

    header = (
        f"{'Subtask':<30} {'N':>5} {'A1%':>7} {'A2%':>7} {'Delta':>7}"
        f" {'RR':>5} {'RW':>5} {'WR':>5} {'WW':>5}"
    )
    print(header)
    print("-" * len(header))

    total_n = 0
    total_a1 = 0
    total_a2 = 0
    total_rr = 0
    total_rw = 0
    total_wr = 0
    total_ww = 0

    for dataset in sorted(stats.keys()):
        st = stats[dataset]
        short = dataset.replace("BLINK_", "")
        delta_str = f"{st.delta:+.1f}"
        print(
            f"  {short:<28} {st.total:>5} {st.a1_acc:>6.1f} {st.a2_acc:>6.1f}"
            f" {delta_str:>7} {st.rr:>5} {st.rw:>5} {st.wr:>5} {st.ww:>5}"
        )
        total_n += st.total
        total_a1 += st.a1_correct
        total_a2 += st.a2_correct
        total_rr += st.rr
        total_rw += st.rw
        total_wr += st.wr
        total_ww += st.ww

    print("-" * len(header))
    if total_n > 0:
        a1_pct = total_a1 / total_n * 100
        a2_pct = total_a2 / total_n * 100
        delta = a2_pct - a1_pct
        delta_str = f"{delta:+.1f}"
        print(
            f"  {'OVERALL':<28} {total_n:>5} {a1_pct:>6.1f} {a2_pct:>6.1f}"
            f" {delta_str:>7} {total_rr:>5} {total_rw:>5} {total_wr:>5} {total_ww:>5}"
        )

    print()
    if total_n > 0:
        print(f"  RR (maintained correct): {total_rr:>5} ({total_rr / total_n * 100:.1f}%)")
        print(f"  RW (regressed):          {total_rw:>5} ({total_rw / total_n * 100:.1f}%)")
        print(f"  WR (corrected):          {total_wr:>5} ({total_wr / total_n * 100:.1f}%)")
        print(f"  WW (remained wrong):     {total_ww:>5} ({total_ww / total_n * 100:.1f}%)")
    print()


def save_results_json(
    stats: dict[str, SubtaskTurnStats],
    output_path: str,
    label: str = "",
) -> None:
    """Save per-turn results to JSON.

    Args:
        stats: Per-subtask stats.
        output_path: Output JSON path.
        label: Model label.
    """
    output = {
        "model": label,
        "per_subtask": {k: v.to_dict() for k, v in sorted(stats.items())},
    }

    total_n = sum(s.total for s in stats.values())
    total_a1 = sum(s.a1_correct for s in stats.values())
    total_a2 = sum(s.a2_correct for s in stats.values())
    output["overall"] = {
        "total": total_n,
        "a1_acc": round(total_a1 / total_n * 100, 2) if total_n else 0,
        "a2_acc": round(total_a2 / total_n * 100, 2) if total_n else 0,
        "RR": sum(s.rr for s in stats.values()),
        "RW": sum(s.rw for s in stats.values()),
        "WR": sum(s.wr for s in stats.values()),
        "WW": sum(s.ww for s in stats.values()),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved turn analysis to {output_path}")


def main() -> None:
    """Analyze per-turn accuracy from BLINK self-reflective eval logs."""
    parser = argparse.ArgumentParser(
        description="Analyze per-turn accuracy from BLINK self-reflective evaluation."
    )
    parser.add_argument(
        "--conversation_log",
        type=str,
        required=True,
        help="Path to conversation_log.jsonl from self-reflective eval.",
    )
    parser.add_argument(
        "--conversation_log_2",
        type=str,
        default=None,
        help="Optional second log for side-by-side comparison.",
    )
    parser.add_argument(
        "--blink_tsv_dir",
        type=str,
        default="/outputs/benchmark_data/blink/tsv",
        help="Directory containing BLINK TSV files with ground truth.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Model",
        help="Display label for first model.",
    )
    parser.add_argument(
        "--label_2",
        type=str,
        default="Model 2",
        help="Display label for second model.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save JSON results.",
    )
    args = parser.parse_args()

    logger.info(f"Loading ground truth from: {args.blink_tsv_dir}")
    ground_truth = load_ground_truth(args.blink_tsv_dir)

    logger.info(f"Analyzing: {args.conversation_log}")
    stats = analyze_conversation_log(args.conversation_log, ground_truth)
    print_results(stats, args.label)

    if args.output_json:
        save_results_json(stats, args.output_json, args.label)

    if args.conversation_log_2:
        logger.info(f"Analyzing: {args.conversation_log_2}")
        stats_2 = analyze_conversation_log(args.conversation_log_2, ground_truth)
        print_results(stats_2, args.label_2)


if __name__ == "__main__":
    main()
