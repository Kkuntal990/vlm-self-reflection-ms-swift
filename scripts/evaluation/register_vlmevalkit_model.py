#!/usr/bin/env python3
"""Register a custom fine-tuned model in VLMEvalKit's config.

This script patches VLMEvalKit's supported_VLM dictionary to add a custom
model entry pointing to a local checkpoint. This avoids manually editing
VLMEvalKit source code.

Usage:
    python scripts/evaluation/register_vlmevalkit_model.py \
        --model-path /outputs/checkpoint-3752 \
        --model-name FireSFT-Qwen2-5-VL-7B \
        --vlmevalkit-dir /tmp/VLMEvalKit
"""

import argparse
import logging
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Register custom model in VLMEvalKit config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="FireSFT-Qwen2-5-VL-7B",
        help="Name to register the model under (default: FireSFT-Qwen2-5-VL-7B)",
    )
    parser.add_argument(
        "--vlmevalkit-dir",
        type=str,
        default="/tmp/VLMEvalKit",
        help="Path to VLMEvalKit installation directory",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=1003520,
        help="Minimum pixel count for image processing (default: 1280*28*28)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=12845056,
        help="Maximum pixel count for image processing (default: 16384*28*28)",
    )
    return parser.parse_args()


def register_model(
    model_path: str,
    model_name: str,
    vlmevalkit_dir: str,
    min_pixels: int,
    max_pixels: int,
) -> None:
    """Register a custom model in VLMEvalKit's supported_VLM dictionary.

    This patches the config.py file to add a new model entry that points
    to a local checkpoint path.

    Args:
        model_path: Path to the fine-tuned model checkpoint.
        model_name: Name to register the model under.
        vlmevalkit_dir: Path to VLMEvalKit installation directory.
        min_pixels: Minimum pixel count for image processing.
        max_pixels: Maximum pixel count for image processing.
    """
    config_path = Path(vlmevalkit_dir) / "vlmeval" / "config.py"
    if not config_path.exists():
        logger.error(f"VLMEvalKit config not found at {config_path}")
        sys.exit(1)

    config_content = config_path.read_text()

    # Check if model is already registered
    if model_name in config_content:
        logger.info(f"Model '{model_name}' already registered in config")
        return

    # Build the entry to insert
    entry = (
        f'\n    "{model_name}": partial(\n'
        f"        Qwen2VLChat,\n"
        f'        model_path="{model_path}",\n'
        f"        min_pixels={min_pixels},\n"
        f"        max_pixels={max_pixels},\n"
        f"        use_custom_prompt=False,\n"
        f"    ),\n"
    )

    # Find the Qwen2-VL section and insert after an existing entry
    # Look for 'Qwen2.5-VL' entries to insert nearby
    marker = "Qwen2.5-VL-7B-Instruct"
    if marker not in config_content:
        # Fallback: look for any Qwen2VLChat entry
        marker = "Qwen2VLChat"

    if marker not in config_content:
        logger.error(
            "Could not find Qwen2VLChat section in config.py. "
            "VLMEvalKit version may be incompatible."
        )
        sys.exit(1)

    # Find the closing paren of the first matching entry after marker
    marker_pos = config_content.index(marker)
    # Find the next "),\n" after the marker to insert after a complete entry
    insert_pos = config_content.find("),\n", marker_pos)
    if insert_pos == -1:
        logger.error("Could not find insertion point in config.py")
        sys.exit(1)

    # Insert after the "),\n"
    insert_pos += len("),\n")
    new_content = config_content[:insert_pos] + entry + config_content[insert_pos:]

    config_path.write_text(new_content)
    logger.info(f"Registered model '{model_name}' in {config_path}")
    logger.info(f"  model_path: {model_path}")
    logger.info(f"  min_pixels: {min_pixels}")
    logger.info(f"  max_pixels: {max_pixels}")


def main() -> None:
    """Main function."""
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.warning(f"Model path does not exist locally: {model_path}")
        logger.warning("Proceeding anyway (path may be valid inside container)")

    register_model(
        model_path=args.model_path,
        model_name=args.model_name,
        vlmevalkit_dir=args.vlmevalkit_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )


if __name__ == "__main__":
    main()
