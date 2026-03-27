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

    # LLaVA model (uses custom HF wrapper)
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
        "--base-model-path",
        type=str,
        default="",
        help="Base model path for LoRA adapters (empty = full model checkpoint)",
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
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="System prompt to inject during evaluation (empty = no system prompt)",
    )
    return parser.parse_args()


def register_qwen_model(
    config_path: Path,
    model_path: str,
    model_name: str,
    min_pixels: int,
    max_pixels: int,
    system_prompt: str = "",
) -> None:
    """Register a Qwen2VL model by inserting into existing config section.

    Args:
        config_path: Path to VLMEvalKit config.py.
        model_path: Path to the model checkpoint.
        model_name: Name to register the model under.
        min_pixels: Minimum pixel count for image processing.
        max_pixels: Maximum pixel count for image processing.
        system_prompt: System prompt for evaluation (empty = no system prompt).
    """
    config_content = config_path.read_text()

    if model_name in config_content:
        logger.info(f"Model '{model_name}' already registered in config")
        return

    system_prompt_line = ""
    if system_prompt:
        escaped = system_prompt.replace('"', '\\"')
        system_prompt_line = f'        system_prompt="{escaped}",\n'

    entry = (
        f'\n    "{model_name}": partial(\n'
        f"        Qwen2VLChat,\n"
        f'        model_path="{model_path}",\n'
        f"        min_pixels={min_pixels},\n"
        f"        max_pixels={max_pixels},\n"
        f"        use_custom_prompt=True,\n"
        f"{system_prompt_line}"
        f"    ),\n"
    )

    # Find the Qwen2-VL section and insert after an existing entry
    for marker in ["Qwen2.5-VL-7B-Instruct", "Qwen2VLChat"]:
        if marker in config_content:
            marker_pos = config_content.index(marker)
            insert_pos = config_content.find("),\n", marker_pos)
            if insert_pos != -1:
                insert_pos += len("),\n")
                new_content = config_content[:insert_pos] + entry + config_content[insert_pos:]
                config_path.write_text(new_content)
                logger.info(f"Registered Qwen2VL model '{model_name}'")
                return

    logger.error("Could not find Qwen2VLChat section in config.py")
    sys.exit(1)


def register_llava_model(
    config_path: Path,
    vlmevalkit_dir: str,
    model_path: str,
    model_name: str,
    base_model_path: str = "",
    system_prompt: str = "",
) -> None:
    """Register a LLaVA model using the custom HF wrapper.

    Appends to the end of config.py to avoid complex insertion logic.

    Args:
        config_path: Path to VLMEvalKit config.py.
        vlmevalkit_dir: Path to VLMEvalKit installation directory.
        model_path: Path to the model checkpoint or LoRA adapter.
        model_name: Name to register the model under.
        base_model_path: Base model path for LoRA adapters.
        system_prompt: System prompt for evaluation (empty = no system prompt).
    """
    # Always copy the wrapper module (may have been updated since last registration)
    wrapper_src = Path(__file__).parent / "llava_hf_wrapper.py"
    wrapper_dst = Path(vlmevalkit_dir) / "vlmeval" / "vlm" / "llava_hf_wrapper.py"
    if wrapper_src.exists():
        wrapper_dst.write_text(wrapper_src.read_text())
        logger.info(f"Copied LLaVA_HF wrapper to {wrapper_dst}")
    else:
        logger.error(f"LLaVA_HF wrapper not found at {wrapper_src}")
        sys.exit(1)

    config_content = config_path.read_text()

    if model_name in config_content:
        logger.info(f"Model '{model_name}' already registered in config")
        return

    # Append import + registration to the end of config.py
    partial_parts = [f"model_path='{model_path}'"]
    if base_model_path:
        partial_parts.append(f"base_model_path='{base_model_path}'")
    if system_prompt:
        escaped = system_prompt.replace("'", "\\'")
        partial_parts.append(f"system_prompt='{escaped}'")
    partial_args = ", ".join(partial_parts)
    append_block = (
        f"\n# Custom LLaVA-HF model registration\n"
        f"from vlmeval.vlm.llava_hf_wrapper import LLaVA_HF\n"
        f"supported_VLM['{model_name}'] = partial(LLaVA_HF, {partial_args})\n"
    )

    config_content += append_block
    config_path.write_text(config_content)
    logger.info(f"Registered LLaVA_HF model '{model_name}'")


def register_llava_next_model(
    config_path: Path,
    model_path: str,
    model_name: str,
) -> None:
    """Register a LLaVA-Next model in existing config section.

    Args:
        config_path: Path to VLMEvalKit config.py.
        model_path: Path to the model checkpoint.
        model_name: Name to register the model under.
    """
    config_content = config_path.read_text()

    if model_name in config_content:
        logger.info(f"Model '{model_name}' already registered in config")
        return

    entry = (
        f'\n    "{model_name}": partial(\n'
        f"        LLaVA_Next,\n"
        f'        model_path="{model_path}",\n'
        f"    ),\n"
    )

    for marker in ["llava_next_vicuna_7b", "LLaVA_Next, model_path"]:
        if marker in config_content:
            marker_pos = config_content.index(marker)
            insert_pos = config_content.find("),\n", marker_pos)
            if insert_pos != -1:
                insert_pos += len("),\n")
                new_content = config_content[:insert_pos] + entry + config_content[insert_pos:]
                config_path.write_text(new_content)
                logger.info(f"Registered LLaVA_Next model '{model_name}'")
                return

    logger.error("Could not find LLaVA_Next section in config.py")
    sys.exit(1)


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
        # Try model_path first, then fall back to base_model_path (for LoRA adapters)
        model_class = detect_model_class(args.model_path)
        if not model_class and args.base_model_path:
            model_class = detect_model_class(args.base_model_path)
        if not model_class:
            logger.error(
                "Could not auto-detect model class from path. "
                "Please specify --model-class (qwen2vl, llava, llava_next)"
            )
            sys.exit(1)
        logger.info(f"Auto-detected model class: {model_class}")

    config_path = Path(args.vlmevalkit_dir) / "vlmeval" / "config.py"
    if not config_path.exists():
        logger.error(f"VLMEvalKit config not found at {config_path}")
        sys.exit(1)

    if model_class == "qwen2vl":
        register_qwen_model(
            config_path, args.model_path, args.model_name,
            args.min_pixels, args.max_pixels, args.system_prompt,
        )
    elif model_class == "llava":
        register_llava_model(
            config_path, args.vlmevalkit_dir, args.model_path, args.model_name,
            args.base_model_path, args.system_prompt,
        )
    elif model_class == "llava_next":
        register_llava_next_model(
            config_path, args.model_path, args.model_name,
        )


if __name__ == "__main__":
    main()
