#!/usr/bin/env python3
"""
Combine inference results from multiple files.

This script combines inference results from multiple JSONL files, matching
samples by sample_id and combining their generated responses into a single
output file.

Usage:
    python scripts/data_prep/combine_inference_results.py \
        --input_dir data/inference_results_v1 \
        --output_path data/inference_results_v1/combined_results.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_results(file_path: Path) -> dict[str, dict]:
    """Load results from JSONL file into a dict keyed by sample_id."""
    results = {}
    with open(file_path) as f:
        for line in f:
            sample = json.loads(line.strip())
            sample_id = sample.get("sample_id")
            if sample_id:
                results[sample_id] = sample
    logger.info(f"Loaded {len(results)} samples from {file_path.name}")
    return results


def combine_results(
    file_paths: list[Path],
    output_path: Path,
) -> None:
    """Combine results from multiple files.

    For each sample, combines the conversation history and adds generated
    responses from each file with a 'file' key indicating the source.

    Args:
        file_paths: List of input JSONL file paths
        output_path: Output path for combined results
    """
    # Load all results
    all_results = {}
    for file_path in file_paths:
        file_name = file_path.stem  # Get filename without extension
        results = load_results(file_path)
        all_results[file_name] = results

    # Find common sample_ids (intersection of all files)
    sample_id_sets = [set(results.keys()) for results in all_results.values()]
    common_sample_ids = set.intersection(*sample_id_sets)
    logger.info(f"Found {len(common_sample_ids)} common samples across all files")

    # Combine results
    combined = []
    for sample_id in sorted(common_sample_ids, key=lambda x: int(x.split("_")[1]) if "_" in x else 0):
        # Get the base sample from the first file
        first_file = list(all_results.keys())[0]
        base_sample = all_results[first_file][sample_id]

        # Build conversation history (without the generated response)
        conversation_history = []
        for msg in base_sample.get("conversation_history", []):
            if not msg.get("is_generated", False):
                conversation_history.append(msg)

        # Add generated responses from each file
        for file_name, results in all_results.items():
            sample = results[sample_id]
            generated_response = sample.get("generated_response", "")
            conversation_history.append({
                "role": "assistant",
                "content": generated_response,
                "is_generated": True,
                "file": file_name,
            })

        # Create combined sample
        combined_sample = {
            "sample_id": sample_id,
            "mode": base_sample.get("mode", "continuation"),
            "image_path": base_sample.get("image_path", ""),
            "conversation_history": conversation_history,
            "ground_truth_response": base_sample.get("ground_truth_response", ""),
            "context_turns": base_sample.get("context_turns", 0),
            "total_turns": base_sample.get("total_turns", 0),
        }
        combined.append(combined_sample)

    # Save combined results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in combined:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Saved {len(combined)} combined samples to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine inference results from multiple files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing inference result files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for combined results",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.jsonl*",
        help="Glob pattern to match input files (default: *.jsonl*)",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    # Find input files
    file_paths = sorted(input_dir.glob(args.file_pattern))

    # Exclude the output file if it exists in the same directory
    file_paths = [p for p in file_paths if p.name != output_path.name]

    if not file_paths:
        logger.error(f"No files matching pattern '{args.file_pattern}' in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(file_paths)} input files:")
    for p in file_paths:
        logger.info(f"  - {p.name}")

    combine_results(file_paths, output_path)


if __name__ == "__main__":
    main()
