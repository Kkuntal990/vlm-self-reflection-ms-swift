#!/usr/bin/env python3
"""Register a custom fine-tuned model in VLMEvalKit's config.

This script patches VLMEvalKit's supported_VLM dictionary to add a custom
model entry pointing to a local checkpoint. This avoids manually editing
VLMEvalKit source code.

Usage:
    # Qwen2.5-VL model (default)
    python scripts/evaluation/register_vlmevalkit_model.py \
        --model-path /outputs/checkpoint-3752 \
        --model-name FireSFT-Qwen2-5-VL-7B \
        --vlmevalkit-dir /tmp/VLMEvalKit

    # LLaVA model
    python scripts/evaluation/register_vlmevalkit_model.py \
        --model-path /outputs/llava-checkpoint \
        --model-name FireSFT-LLaVA-7B \
        --model-class llava \
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

# Mapping from user-friendly model class names to VLMEvalKit class names
# and their config section markers
MODEL_CLASS_CONFIG = {
    "qwen2vl": {
        "class_name": "Qwen2VLChat",
        "markers": ["Qwen2.5-VL-7B-Instruct", "Qwen2VLChat"],
    },
    "llava": {
        "class_name": "LLaVA",
        "markers": ["llava_v1.5_7b", "LLaVA, model_path"],
    },
    "llava_next": {
        "class_name": "LLaVA_Next",
        "markers": ["llava_next_vicuna_7b", "LLaVA_Next, model_path"],
    },
}


def detect_model_class(model_path: str) -> str:
    """Auto-detect model class from the model path.

    Args:
        model_path: Path to the model checkpoint.

    Returns:
        Detected model class key (e.g., 'qwen2vl', 'llava').
    """
    path_lower = model_path.lower()
    if "llava" in path_lower and "next" in path_lower:
        return "llava_next"
    if "llava" in path_lower:
        return "llava"
    if "qwen" in path_lower:
        return "qwen2vl"
    return ""


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
        "--model-class",
        type=str,
        default="",
        choices=["", "qwen2vl", "llava", "llava_next"],
        help="Model class for VLMEvalKit (default: auto-detect from model path)",
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
        help="Minimum pixel count for Qwen2VL image processing (default: 1280*28*28)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=12845056,
        help="Maximum pixel count for Qwen2VL image processing (default: 16384*28*28)",
    )
    return parser.parse_args()


def build_entry(
    model_name: str,
    model_path: str,
    model_class: str,
    min_pixels: int,
    max_pixels: int,
) -> str:
    """Build the config entry string for the model.

    Args:
        model_name: Name to register the model under.
        model_path: Path to the model checkpoint.
        model_class: Model class key (e.g., 'qwen2vl', 'llava').
        min_pixels: Minimum pixel count (Qwen2VL only).
        max_pixels: Maximum pixel count (Qwen2VL only).

    Returns:
        Config entry string to insert into VLMEvalKit config.py.
    """
    class_name = MODEL_CLASS_CONFIG[model_class]["class_name"]

    if model_class == "qwen2vl":
        return (
            f'\n    "{model_name}": partial(\n'
            f"        {class_name},\n"
            f'        model_path="{model_path}",\n'
            f"        min_pixels={min_pixels},\n"
            f"        max_pixels={max_pixels},\n"
            f"        use_custom_prompt=False,\n"
            f"    ),\n"
        )
    else:
        return (
            f'\n    "{model_name}": partial(\n'
            f"        {class_name},\n"
            f'        model_path="{model_path}",\n'
            f"    ),\n"
        )


def register_model(
    model_path: str,
    model_name: str,
    model_class: str,
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
        model_class: Model class key (e.g., 'qwen2vl', 'llava').
        vlmevalkit_dir: Path to VLMEvalKit installation directory.
        min_pixels: Minimum pixel count for image processing (Qwen2VL only).
        max_pixels: Maximum pixel count for image processing (Qwen2VL only).
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

    # Build the entry
    entry = build_entry(model_name, model_path, model_class, min_pixels, max_pixels)

    # Find the right section to insert into
    class_config = MODEL_CLASS_CONFIG[model_class]
    marker = None
    for candidate in class_config["markers"]:
        if candidate in config_content:
            marker = candidate
            break

    if marker is None:
        logger.error(
            f"Could not find {class_config['class_name']} section in config.py. "
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
    logger.info(f"  model_class: {class_config['class_name']}")


def main() -> None:
    """Main function."""
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.warning(f"Model path does not exist locally: {model_path}")
        logger.warning("Proceeding anyway (path may be valid inside container)")

    # Auto-detect model class if not specified
    model_class = args.model_class
    if not model_class:
        model_class = detect_model_class(args.model_path)
        if not model_class:
            logger.error(
                "Could not auto-detect model class from path. "
                "Please specify --model-class (qwen2vl, llava, llava_next)"
            )
            sys.exit(1)
        logger.info(f"Auto-detected model class: {model_class}")

    register_model(
        model_path=args.model_path,
        model_name=args.model_name,
        model_class=model_class,
        vlmevalkit_dir=args.vlmevalkit_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )


if __name__ == "__main__":
    main()
