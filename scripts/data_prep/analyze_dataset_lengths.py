#!/usr/bin/env python3
"""
Analyze token lengths in FIRE ShareGPT dataset to determine optimal max_length.

This script:
1. Loads the ShareGPT JSONL dataset
2. Tokenizes each conversation using the target model's tokenizer
3. Computes statistics on sequence lengths (text + image tokens)
4. Recommends optimal max_length values
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def count_images_in_conversation(messages: list[dict]) -> int:
    """Count number of images in a conversation."""
    image_count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-modal content format
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_count += 1
        elif isinstance(content, str):
            # Count <image> placeholders
            image_count += content.count("<image>")
    return image_count


def estimate_conversation_tokens(
    messages: list[dict], tokenizer, image_token_count: int = 2048
) -> dict:
    """
    Estimate total tokens for a conversation.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer
        image_token_count: Tokens per image (for Qwen2.5-VL: ~2048)

    Returns:
        Dict with text_tokens, image_tokens, total_tokens
    """
    text_parts = []
    image_count = 0

    for msg in messages:
        # Add role tokens
        role = msg.get("role", "")
        text_parts.append(f"<|im_start|>{role}\n")

        # Process content
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-modal content
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        image_count += 1
        elif isinstance(content, str):
            # Count image placeholders
            image_count += content.count("<image>")
            # Remove image placeholders for text tokenization
            text_parts.append(content.replace("<image>", ""))

        text_parts.append("<|im_end|>\n")

    # Tokenize text
    full_text = "".join(text_parts)
    text_tokens = len(tokenizer.encode(full_text, add_special_tokens=True))

    # Estimate image tokens
    image_tokens = image_count * image_token_count

    return {
        "text_tokens": text_tokens,
        "image_tokens": image_tokens,
        "image_count": image_count,
        "total_tokens": text_tokens + image_tokens,
    }


def analyze_dataset(
    dataset_path: Path,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    image_token_count: int = 2048,
    max_samples: int = 0,
):
    """Analyze token distribution in dataset."""

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Analyzing dataset: {dataset_path}")
    print("=" * 60)

    token_stats = []
    text_token_stats = []
    image_count_stats = []

    with open(dataset_path) as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break

            data = json.loads(line)
            messages = data.get("messages", data.get("conversations", []))

            stats = estimate_conversation_tokens(messages, tokenizer, image_token_count)

            token_stats.append(stats["total_tokens"])
            text_token_stats.append(stats["text_tokens"])
            image_count_stats.append(stats["image_count"])

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} samples...", end="\r")

    print(f"\nProcessed {len(token_stats)} samples total")
    print("=" * 60)

    # Compute statistics
    token_array = np.array(token_stats)
    text_array = np.array(text_token_stats)
    image_array = np.array(image_count_stats)

    print("\n📊 TOKEN LENGTH DISTRIBUTION")
    print("=" * 60)
    print("Total Tokens (text + images):")
    print(f"  Mean:       {token_array.mean():.1f}")
    print(f"  Median:     {np.median(token_array):.1f}")
    print(f"  Std Dev:    {token_array.std():.1f}")
    print(f"  Min:        {token_array.min():.1f}")
    print(f"  Max:        {token_array.max():.1f}")
    print()
    print("Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(token_array, p)
        pct_kept = (token_array <= val).sum() / len(token_array) * 100
        print(f"  {p}th:       {val:.1f} tokens ({pct_kept:.1f}% of samples fit)")

    print("\n📝 TEXT TOKENS (excluding images):")
    print(f"  Mean:       {text_array.mean():.1f}")
    print(f"  Median:     {np.median(text_array):.1f}")
    print(f"  Max:        {text_array.max():.1f}")

    print("\n🖼️  IMAGE STATISTICS:")
    print(f"  Images per sample (mean):  {image_array.mean():.2f}")
    print(f"  Images per sample (max):   {image_array.max():.0f}")
    print(
        f"  Samples with images:       {(image_array > 0).sum()} ({(image_array > 0).sum() / len(image_array) * 100:.1f}%)"
    )
    print(f"  Image token count used:    {image_token_count}")

    # Recommendations
    print("\n💡 RECOMMENDED MAX_LENGTH VALUES")
    print("=" * 60)

    recommendations = [
        (4096, "Conservative (fits ~75-85% of data)"),
        (8192, "Balanced (fits ~90-95% of data)"),
        (16384, "Comprehensive (fits ~99% of data)"),
    ]

    for max_len, desc in recommendations:
        fit_count = (token_array <= max_len).sum()
        fit_pct = fit_count / len(token_array) * 100
        truncated = len(token_array) - fit_count

        # Estimate memory impact
        memory_factor = max_len / 8192  # relative to 8192 baseline

        print(f"\nmax_length = {max_len}")
        print(f"  {desc}")
        print(f"  Samples that fit: {fit_count}/{len(token_array)} ({fit_pct:.1f}%)")
        print(f"  Samples truncated: {truncated} ({100 - fit_pct:.1f}%)")
        print(f"  Memory usage: ~{memory_factor:.1f}x compared to 8192")

    # Data loss analysis
    print("\n⚠️  TRUNCATION IMPACT")
    print("=" * 60)
    for max_len in [4096, 8192, 16384]:
        truncated_samples = token_array > max_len
        if truncated_samples.sum() > 0:
            avg_loss = token_array[truncated_samples].mean() - max_len
            print(f"max_length={max_len}: Avg {avg_loss:.0f} tokens lost per truncated sample")

    print("\n✅ RECOMMENDATION")
    print("=" * 60)
    p95 = np.percentile(token_array, 95)
    if p95 <= 4096:
        recommended = 4096
    elif p95 <= 8192:
        recommended = 8192
    else:
        recommended = 16384

    fit_pct = (token_array <= recommended).sum() / len(token_array) * 100
    print(f"Start with max_length={recommended}")
    print(f"  - Fits {fit_pct:.1f}% of your data")
    print("  - Good balance of coverage and memory efficiency")
    print("  - Monitor training logs for truncation warnings")
    print()
    print("To increase coverage, raise max_length in the job YAML:")
    print("  k8s/job-full-sft-qwen3vl-fire-8gpu.yaml → MAX_LEN")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze token lengths in FIRE ShareGPT dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/outputs/fire_sharegpt/fire_sharegpt_test.jsonl"),
        help="Path to ShareGPT JSONL file",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model ID for tokenizer"
    )
    parser.add_argument(
        "--image-tokens",
        type=int,
        default=2048,
        help="Number of tokens per image (Qwen2.5-VL default: 2048)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit analysis to N samples (for quick testing)",
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        print("\nMake sure the dataset exists. You can test locally with:")
        print("  python scripts/prepare_fire_sharegpt.py --max_samples 100 --skip-images")
        return

    analyze_dataset(args.dataset, args.model, args.image_tokens, args.max_samples)


if __name__ == "__main__":
    main()
