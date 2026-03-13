#!/usr/bin/env python3
"""
Self-reflective inference with natural role assignments.

This script uses a natural conversation flow where:
1. Model generates an answer (as assistant)
2. For feedback: answer appears as assistant's own output, user asks for critique
3. For refinement: feedback appears as user message, model revises

Key insight: The model should "own" both the answer it critiques AND
receive feedback as user input when refining. This matches natural
conversation patterns.

Conversation Flow:

    Turn 0 - Initial Answer:
        User: [image] + question
        Assistant: initial_answer

    Turn 1+ - Feedback Generation (ACCUMULATED history, critic mode, FLIPPED roles):
        Assistant: {question} + [image]
        User: answer_0
        Assistant: feedback_0
        User: answer_1
        Assistant: feedback_1
        ...
        User: current_answer
        Assistant: <generates feedback>

    Turn 1+ - Refinement (VL assistant mode, ACCUMULATED history):
        User: [image] + question
        Assistant: answer_0
        User: feedback_0
        Assistant: answer_1
        User: feedback_1
        ...
        Assistant: <refined answer>

Usage:
    python scripts/evaluation/self_reflective_inference_v2.py \
        --model_path /outputs/checkpoint-final \
        --dataset_path data/fire_preprocessed_v2/fire_messages_test.jsonl \
        --output_path outputs/self_reflective_v2_results.jsonl \
        --image_base_dir /outputs \
        --max_samples 10

Reference:
    - FIRE dataset: https://huggingface.co/datasets/PengxiangLi/FIRE
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================
# System Prompts (matching training data)
# ============================================

# System prompt for VL Assistant (from fire_messages training)
_DEFAULT_VL_ASSISTANT_PROMPT = "You are a helpful vision-language assistant. You should produce accurate, detailed, and grounded answers based on the image and the user's instructions. When given feedback, critique, or scores, revise your response to improve correctness, specificity, and completeness."
VL_ASSISTANT_SYSTEM_PROMPT = os.environ.get(
    "VL_ASSISTANT_SYSTEM_PROMPT", _DEFAULT_VL_ASSISTANT_PROMPT
)

# System prompt for Feedback Critic (from fire_feedback training)
# _DEFAULT_FEEDBACK_CRITIC_PROMPT = """You are a helpful assistant that provides constructive feedback on answers to visual questions.

# Given an image, a question, an answer, and the conversation history:
# 1. Identify what is correct and what is incorrect in the answer.
# 2. Base your critique ONLY on visual evidence from the image.

# IMPORTANT:
# - First write a line starting with "EVIDENCE:" and briefly state the specific visual evidence you are using
#   (e.g., an object count, a color, a label, a number, a position, or a visible text).
# - Then write a line starting with "FIX:" and state exactly what should be changed or corrected in the answer.
# - If the answer is fully correct and supported by the image, write:
#   "EVIDENCE: The answer matches the visible evidence."
#   "FIX: No change needed."

# Do NOT:
# - Introduce new facts not visible in the image.
# - Reinterpret or question the intent of the question.
# - Give generic advice like "look again" without stating evidence.

# """

_DEFAULT_FEEDBACK_CRITIC_PROMPT = """
You are a vision-language critic that evaluates answers to visual questions and helps improve them. Use the image, question, and dialogue history to judge the latest answer by: - correctness and visual grounding (matches what's visible / implied), - compliance with the requested format (option letter, units, etc.), - completeness. Be conservative: confirm the answer as correct if it is consistent with the image/question and follows the required format. Only say "incorrect" when you can name a specific contradiction or missing requirement. If you are uncertain, do not guess—ask to re-check one concrete detail. Write a brief natural paragraph: start with a clear verdict, give 1–2 grounded reasons, and (if needed) one practical next step. Keep the tone polite and encouraging.
"""


FEEDBACK_CRITIC_SYSTEM_PROMPT = os.environ.get(
    "FEEDBACK_CRITIC_SYSTEM_PROMPT", _DEFAULT_FEEDBACK_CRITIC_PROMPT
)


# ============================================
# Model Type Detection
# ============================================

MODEL_TYPE_QWEN2_5_VL = "qwen2_5_vl"
MODEL_TYPE_LLAVA_ONEVISION = "llava_onevision"
MODEL_TYPE_LLAVA = "llava"
SUPPORTED_MODEL_TYPES = [MODEL_TYPE_QWEN2_5_VL, MODEL_TYPE_LLAVA_ONEVISION, MODEL_TYPE_LLAVA]

# LLaVA family types share the same generation pipeline (PIL images + apply_chat_template)
_LLAVA_FAMILY_TYPES = {MODEL_TYPE_LLAVA_ONEVISION, MODEL_TYPE_LLAVA}


def detect_model_type(model_path: str, model_type_override: str | None = None) -> str:
    """Detect model type from config, path heuristics, or explicit override.

    Detection priority:
        1. Explicit override via --model_type CLI argument
        2. config.json architectures field (reliable for local checkpoints)
        3. Path-based heuristics (fallback for HuggingFace IDs)

    Args:
        model_path: Path to model checkpoint or HuggingFace model ID
        model_type_override: Explicit model type from CLI

    Returns:
        Detected model type string

    Raises:
        ValueError: If model_type_override is not a supported type
    """
    # Priority 1: Explicit override
    if model_type_override:
        if model_type_override not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model type: '{model_type_override}'. "
                f"Supported: {SUPPORTED_MODEL_TYPES}"
            )
        logger.info(f"Using explicit model type: {model_type_override}")
        return model_type_override

    # Priority 2: config.json architectures field
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            architectures = config.get("architectures", [])
            for arch in architectures:
                if "LlavaOnevision" in arch:
                    logger.info(
                        f"Detected model type from config.json: {MODEL_TYPE_LLAVA_ONEVISION}"
                    )
                    return MODEL_TYPE_LLAVA_ONEVISION
                if "Llava" in arch:
                    # Generic LLaVA (e.g. LlavaForConditionalGeneration) — checked
                    # after LlavaOnevision so OV models match the more specific type
                    logger.info(f"Detected model type from config.json: {MODEL_TYPE_LLAVA}")
                    return MODEL_TYPE_LLAVA
                if "Qwen2_5_VL" in arch or "Qwen2_5VL" in arch:
                    logger.info(f"Detected model type from config.json: {MODEL_TYPE_QWEN2_5_VL}")
                    return MODEL_TYPE_QWEN2_5_VL
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read config.json: {e}")

    # Priority 3: Path-based heuristics
    path_lower = model_path.lower()
    if re.search(r"llava.*one.*vision|llava[-_]ov", path_lower):
        logger.info(f"Detected model type from path: {MODEL_TYPE_LLAVA_ONEVISION}")
        return MODEL_TYPE_LLAVA_ONEVISION
    if re.search(r"llava", path_lower):
        logger.info(f"Detected model type from path: {MODEL_TYPE_LLAVA}")
        return MODEL_TYPE_LLAVA
    if re.search(r"qwen.*2.*5.*vl", path_lower):
        logger.info(f"Detected model type from path: {MODEL_TYPE_QWEN2_5_VL}")
        return MODEL_TYPE_QWEN2_5_VL

    # Default fallback
    logger.warning(
        f"Could not auto-detect model type for '{model_path}'. "
        f"Defaulting to '{MODEL_TYPE_QWEN2_5_VL}'. Use --model_type to specify."
    )
    return MODEL_TYPE_QWEN2_5_VL


# ============================================
# Data Classes
# ============================================


@dataclass
class TurnResult:
    """Result for a single turn in the refinement loop."""

    answer: str
    feedback: str  # Empty for final turn

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SampleResult:
    """Result for a single sample's self-reflective dialogue."""

    sample_index: int
    image_path: str
    question: str
    generated_turns: list[dict]
    final_answer: str
    gt_final_answer: str
    num_turns: int
    messages_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ============================================
# Model Wrapper
# ============================================


class SelfReflectionEngine:
    """Inference engine for self-reflective generation with role-flipped feedback.

    Supports multiple model architectures:
        - Qwen2.5-VL: Uses qwen_vl_utils for image processing
        - LLaVA-OneVision: Uses PIL for image loading

    Attributes:
        model_path: Path to the fine-tuned model checkpoint
        model: The loaded model
        processor: The model's processor
        device: Device for inference
        model_type: Detected model architecture type
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = MODEL_TYPE_QWEN2_5_VL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attn: bool = True,
        device_map_strategy: str = "auto",
        base_model_path: str = "",
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to fine-tuned model checkpoint or HuggingFace ID
            model_type: Model architecture type ("qwen2_5_vl", "llava_onevision", or "llava")
            device: Device to run inference on (e.g. "cuda", "cuda:0", "cpu")
            dtype: Model data type
            use_flash_attn: Whether to use flash attention
            device_map_strategy: Model placement strategy:
                "auto" - use device_map="auto" (single GPU, spreads across devices)
                "per_gpu" - no device_map, explicitly place on `device` (multi-GPU)
            base_model_path: Base model for LoRA adapters (empty = full model)
        """
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: '{model_type}'. Supported: {SUPPORTED_MODEL_TYPES}"
            )

        self.base_model_path = base_model_path
        # For LoRA: load from base model path, apply adapter from model_path
        load_path = base_model_path if base_model_path else model_path
        self.model_path = load_path
        self.model_type = model_type
        self.device = device
        self.dtype = dtype
        self.device_map_strategy = device_map_strategy

        if base_model_path:
            logger.info(f"Loading base model from {base_model_path} (type: {model_type})")
            logger.info(f"LoRA adapter: {model_path}")
        else:
            logger.info(f"Loading model from {model_path} (type: {model_type}, device: {device})")

        # Lazy import for processor (works for both model types)
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(load_path)

        # Left-pad for batched generation (decoder-only models need left-padding
        # so all sequences align at the right/generation end)
        self.processor.tokenizer.padding_side = "left"

        # Load model class based on type
        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        if model_type == MODEL_TYPE_LLAVA_ONEVISION:
            self._load_llava_onevision(attn_impl, device, dtype)
        elif model_type == MODEL_TYPE_LLAVA:
            self._load_llava(attn_impl, device, dtype)
        else:
            self._load_qwen2_5_vl(attn_impl, device, dtype)

        # Apply LoRA adapter if base_model_path was specified
        if base_model_path:
            from peft import PeftModel

            logger.info(f"Applying LoRA adapter from {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.model = self.model.merge_and_unload()
            logger.info("LoRA adapter merged")

        # For per_gpu strategy, move model to the specific device
        if device_map_strategy == "per_gpu":
            self.model = self.model.to(device)

        self.model.eval()
        logger.info(f"Model loaded successfully (type: {model_type})")

    def _get_device_map(self, device: str) -> str | None:
        """Get the device_map value based on strategy.

        Args:
            device: Target device string

        Returns:
            device_map argument for from_pretrained()
        """
        if self.device_map_strategy == "per_gpu":
            # Multi-GPU: no device_map, model placed via .to(device) after loading
            return None
        # Single-GPU: use device_map="auto" for CUDA
        return "auto" if device.startswith("cuda") else None

    def _load_qwen2_5_vl(self, attn_impl: str, device: str, dtype: torch.dtype) -> None:
        """Load Qwen2.5-VL model.

        Args:
            attn_impl: Attention implementation ("flash_attention_2" or "eager")
            device: Device to load model on
            dtype: Model data type
        """
        from transformers import Qwen2_5_VLForConditionalGeneration

        dm = self._get_device_map(device)
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=dm,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(f"Flash attention failed ({e}), falling back to eager")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=dm,
                torch_dtype=dtype,
                attn_implementation="eager",
            )

    def _load_llava_onevision(self, attn_impl: str, device: str, dtype: torch.dtype) -> None:
        """Load LLaVA-OneVision model.

        Args:
            attn_impl: Attention implementation ("flash_attention_2" or "eager")
            device: Device to load model on
            dtype: Model data type
        """
        from transformers import LlavaOnevisionForConditionalGeneration

        dm = self._get_device_map(device)
        try:
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=dm,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(f"Flash attention failed ({e}), falling back to eager")
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=dm,
                torch_dtype=dtype,
                attn_implementation="eager",
            )

    def _load_llava(self, attn_impl: str, device: str, dtype: torch.dtype) -> None:
        """Load plain LLaVA model (e.g. LLaVA-1.5, LLaVA-v1.6).

        Uses LlavaForConditionalGeneration. Generation pipeline is identical
        to LLaVA-OneVision (PIL images + apply_chat_template).

        Args:
            attn_impl: Attention implementation ("flash_attention_2" or "eager")
            device: Device to load model on
            dtype: Model data type
        """
        from transformers import LlavaForConditionalGeneration

        dm = self._get_device_map(device)
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=dm,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(f"Flash attention failed ({e}), falling back to eager")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=dm,
                torch_dtype=dtype,
                attn_implementation="eager",
            )

    def generate(
        self,
        messages: list[dict],
        system_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a response given the message history.

        Handles image processing differently based on model type:
        - Qwen2.5-VL: Uses qwen_vl_utils.process_vision_info()
        - LLaVA-OneVision: Uses PIL to load images, passes them to processor

        Args:
            messages: List of messages in chat format (without system prompt).
                Images referenced as {"type": "image", "image": "/path"} in
                content lists. Translated to model-specific format internally.
            system_prompt: System prompt to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling

        Returns:
            Generated response text
        """
        if self.model_type in _LLAVA_FAMILY_TYPES:
            return self._generate_llava_onevision(
                messages, system_prompt, max_new_tokens, temperature, top_p, do_sample
            )
        return self._generate_qwen2_5_vl(
            messages, system_prompt, max_new_tokens, temperature, top_p, do_sample
        )

    def generate_batch(
        self,
        messages_list: list[list[dict]],
        system_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> list[str]:
        """Generate responses for multiple prompts in a single forward pass.

        Args:
            messages_list: List of message histories, one per sample
            system_prompt: System prompt (shared across batch)
            max_new_tokens: Maximum tokens to generate per response
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling

        Returns:
            List of generated response texts, one per input
        """
        if self.model_type in _LLAVA_FAMILY_TYPES:
            return self._generate_batch_llava_onevision(
                messages_list, system_prompt, max_new_tokens, temperature, top_p, do_sample
            )
        return self._generate_batch_qwen2_5_vl(
            messages_list, system_prompt, max_new_tokens, temperature, top_p, do_sample
        )

    def _generate_batch_qwen2_5_vl(
        self,
        messages_list: list[list[dict]],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> list[str]:
        """Batch generate responses using Qwen2.5-VL pipeline.

        Args:
            messages_list: List of message histories, one per sample
            system_prompt: System prompt (shared across batch)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling

        Returns:
            List of generated response texts
        """
        from qwen_vl_utils import process_vision_info

        texts = []
        all_image_inputs = []
        all_video_inputs = []

        for messages in messages_list:
            full_messages = [{"role": "system", "content": system_prompt}]
            full_messages.extend(messages)

            text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

            image_inputs, video_inputs = process_vision_info(full_messages)
            if image_inputs:
                all_image_inputs.extend(image_inputs)
            if video_inputs:
                all_video_inputs.extend(video_inputs)

        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        return self._run_generation_batch(inputs, max_new_tokens, temperature, top_p, do_sample)

    def _generate_batch_llava_onevision(
        self,
        messages_list: list[list[dict]],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> list[str]:
        """Batch generate responses using LLaVA-OneVision pipeline.

        Applies the same message translation as _generate_llava_onevision()
        (image hoisting, content wrapping) to each sample, then processes
        all samples in a single forward pass.

        Args:
            messages_list: List of message histories, one per sample
            system_prompt: System prompt (shared across batch)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling

        Returns:
            List of generated response texts
        """
        from PIL import Image

        texts = []
        all_pil_images = []

        for messages in messages_list:
            pil_images = []

            # First pass: hoist images from non-user messages to system message
            hoisted_image_items = []
            cleaned_messages = []
            for msg in messages:
                content = msg.get("content")
                role = msg.get("role", "user")

                if isinstance(content, list) and role != "user":
                    non_image_items = []
                    for item in content:
                        if item.get("type") == "image" and "image" in item:
                            hoisted_image_items.append(item)
                        else:
                            non_image_items.append(item)
                    cleaned_messages.append({"role": role, "content": non_image_items or content})
                else:
                    cleaned_messages.append(msg)

            # Build system message with hoisted images
            system_content = []
            for item in hoisted_image_items:
                image_path = item["image"]
                try:
                    pil_image = Image.open(image_path)
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    pil_images.append(pil_image)
                    system_content.append({"type": "image"})
                except Exception as e:
                    logger.warning(f"Failed to load hoisted image {image_path}: {e}")
            system_content.append({"type": "text", "text": system_prompt})
            llava_messages = [{"role": "system", "content": system_content}]

            # Second pass: process remaining messages
            for msg in cleaned_messages:
                content = msg.get("content")
                role = msg.get("role", "user")

                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        if item.get("type") == "image" and "image" in item:
                            image_path = item["image"]
                            try:
                                pil_image = Image.open(image_path)
                                if pil_image.mode != "RGB":
                                    pil_image = pil_image.convert("RGB")
                                pil_images.append(pil_image)
                                new_content.append({"type": "image"})
                            except Exception as e:
                                logger.warning(f"Failed to load image {image_path}: {e}")
                        else:
                            new_content.append(item)
                    llava_messages.append({"role": role, "content": new_content})
                else:
                    llava_messages.append(
                        {"role": role, "content": [{"type": "text", "text": content}]}
                    )

            text = self.processor.apply_chat_template(
                llava_messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            all_pil_images.extend(pil_images)

        if all_pil_images:
            inputs = self.processor(
                text=texts,
                images=all_pil_images,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.device)

        return self._run_generation_batch(inputs, max_new_tokens, temperature, top_p, do_sample)

    def _run_generation_batch(
        self,
        inputs: dict,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> list[str]:
        """Run batched model generation and decode all outputs.

        Args:
            inputs: Tokenized and processed model inputs (batched)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling

        Returns:
            List of decoded response texts
        """
        use_sampling = do_sample and temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": use_sampling,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if use_sampling:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # With left-padding, all inputs are padded to the same length,
        # so input_len is uniform across the batch
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return [r.strip() for r in responses]

    def _generate_qwen2_5_vl(
        self,
        messages: list[dict],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        """Generate response using Qwen2.5-VL pipeline.

        Args:
            messages: Messages with image paths in content dicts
            system_prompt: System prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling

        Returns:
            Generated response text
        """
        from qwen_vl_utils import process_vision_info

        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(full_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        return self._run_generation(inputs, max_new_tokens, temperature, top_p, do_sample)

    def _generate_llava_onevision(
        self,
        messages: list[dict],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        """Generate response using LLaVA-OneVision pipeline.

        Translates Qwen-format messages to LLaVA format:
        - {"type": "image", "image": "/path"} -> {"type": "image"} placeholder
        - PIL images collected and passed to processor separately

        Images in non-user messages (e.g. assistant role from role-flipped critic
        history) are hoisted into the system message. LLaVA base models only
        reliably process images in user/system turns.

        Args:
            messages: Messages in Qwen format (translated internally)
            system_prompt: System prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling

        Returns:
            Generated response text
        """
        from PIL import Image

        # LLaVA-OneVision's chat template silently drops plain string content.
        # All content must be wrapped as [{"type": "text", "text": "..."}] lists.
        pil_images = []

        # First pass: extract images from non-user messages (e.g. assistant role
        # in role-flipped critic history). LLaVA base models don't reliably
        # process images in assistant turns, so hoist them to the system message.
        hoisted_image_items = []
        cleaned_messages = []
        for msg in messages:
            content = msg.get("content")
            role = msg.get("role", "user")

            if isinstance(content, list) and role != "user":
                non_image_items = []
                for item in content:
                    if item.get("type") == "image" and "image" in item:
                        hoisted_image_items.append(item)
                    else:
                        non_image_items.append(item)
                cleaned_messages.append({"role": role, "content": non_image_items or content})
            else:
                cleaned_messages.append(msg)

        # Build system message: hoisted images first, then system prompt text
        system_content = []
        for item in hoisted_image_items:
            image_path = item["image"]
            try:
                pil_image = Image.open(image_path)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                pil_images.append(pil_image)
                system_content.append({"type": "image"})
            except Exception as e:
                logger.warning(f"Failed to load hoisted image {image_path}: {e}")
        system_content.append({"type": "text", "text": system_prompt})
        llava_messages = [{"role": "system", "content": system_content}]

        # Second pass: process remaining messages, loading user-role images normally
        for msg in cleaned_messages:
            content = msg.get("content")
            role = msg.get("role", "user")

            if isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "image" and "image" in item:
                        image_path = item["image"]
                        try:
                            pil_image = Image.open(image_path)
                            if pil_image.mode != "RGB":
                                pil_image = pil_image.convert("RGB")
                            pil_images.append(pil_image)
                            new_content.append({"type": "image"})
                        except Exception as e:
                            logger.warning(f"Failed to load image {image_path}: {e}")
                    else:
                        new_content.append(item)
                llava_messages.append({"role": role, "content": new_content})
            else:
                # Wrap plain text as structured content for LLaVA template
                llava_messages.append(
                    {"role": role, "content": [{"type": "text", "text": content}]}
                )

        text = self.processor.apply_chat_template(
            llava_messages, tokenize=False, add_generation_prompt=True
        )

        if pil_images:
            inputs = self.processor(
                text=[text],
                images=pil_images,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.device)

        return self._run_generation(inputs, max_new_tokens, temperature, top_p, do_sample)

    def _run_generation(
        self,
        inputs: dict,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        """Run model generation and decode output.

        Args:
            inputs: Tokenized and processed model inputs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to use sampling

        Returns:
            Decoded response text
        """
        use_sampling = do_sample and temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": use_sampling,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if use_sampling:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()


# ============================================
# Dataset Loading
# ============================================


def load_dataset(dataset_path: str, max_samples: int = 0, start_index: int = 0) -> list[dict]:
    """Load dataset in Messages format.

    Args:
        dataset_path: Path to JSONL file
        max_samples: Maximum samples to load (0 = all)
        start_index: Index of first sample to include (skip earlier samples)

    Returns:
        List of sample dictionaries
    """
    samples = []

    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if i < start_index:
                continue
            if max_samples > 0 and len(samples) >= max_samples:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i}: {e}")

    logger.info(f"Loaded {len(samples)} samples from {dataset_path} (start_index={start_index})")
    return samples


def resolve_image_path(image_path: str | dict, image_base_dir: str) -> str | None:
    """Resolve relative image path to absolute path.

    Args:
        image_path: Relative or absolute image path (str or dict with 'path' key)
        image_base_dir: Base directory for relative paths

    Returns:
        Absolute path if exists, None otherwise
    """
    # Handle dict format: {'bytes': None, 'path': '/path/to/image.jpg'}
    if isinstance(image_path, dict):
        image_path = image_path.get("path", "")
        if not image_path:
            logger.warning("Image dict has no 'path' key")
            return None

    if os.path.isabs(image_path):
        full_path = image_path
    else:
        full_path = os.path.join(image_base_dir, image_path)

    if not os.path.exists(full_path):
        logger.warning(f"Image not found: {full_path}")
        return None

    return full_path


def parse_sample(sample: dict) -> tuple[str, list[str], list[str], str | None]:
    """Parse a sample in Messages format.

    Args:
        sample: Sample dictionary with 'messages' and 'images' keys

    Returns:
        Tuple of (question, gt_assistant_responses, images, system_prompt)
    """
    messages = sample.get("messages", [])
    images = sample.get("images", [])

    question = None
    gt_responses = []
    system_prompt = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_prompt = content
        elif role == "user" and question is None:
            # First user message is the question
            question = content
        elif role == "assistant":
            gt_responses.append(content)

    return question, gt_responses, images, system_prompt


# ============================================
# Self-Reflective Generation with Role Flip
# ============================================


def generate_self_reflective_dialogue(
    engine: SelfReflectionEngine,
    sample: dict,
    sample_index: int,
    image_base_dir: str,
    generation_config: dict,
) -> SampleResult | None:
    """Generate a self-reflective dialogue with role-flipped feedback.

    Conversation Flow:
        Turn 0:
            User: [image] + question
            Assistant: initial_answer

        Turn 1+:
            User: [image] + question
            Assistant: previous_answer
            User: <feedback>  ← Role-flipped (feedback as user message)
            Assistant: <refined answer>

    Args:
        engine: Self-reflection inference engine
        sample: Input sample with messages and images
        sample_index: Index of sample in dataset
        image_base_dir: Base directory for image paths
        generation_config: Generation parameters

    Returns:
        SampleResult with generated dialogue, or None if failed
    """
    max_new_tokens = generation_config.get("max_new_tokens", 512)
    answer_temperature = generation_config.get("answer_temperature", 0.7)
    feedback_temperature = generation_config.get("feedback_temperature", 0.7)
    top_p = generation_config.get("top_p", 0.9)
    requested_num_turns = generation_config.get("num_turns", 0)

    # Parse sample
    question, gt_responses, images, _ = parse_sample(sample)

    if not question:
        logger.warning(f"Sample {sample_index}: No question found")
        return None

    if not images:
        logger.warning(f"Sample {sample_index}: No images found")
        return None

    if not gt_responses:
        logger.warning(f"Sample {sample_index}: No ground truth responses found")
        return None

    # Resolve image path
    image_path = resolve_image_path(images[0], image_base_dir)
    if not image_path:
        return None

    # Use requested num_turns if specified, otherwise fall back to ground truth count
    num_turns = requested_num_turns if requested_num_turns > 0 else len(gt_responses)
    gt_final_answer = gt_responses[-1]

    # Track generated turns and message history
    generated_turns = []
    full_history = []  # For logging

    # Clean question (remove any <image> tag if present)
    clean_question = question.replace("<image>", "").strip()

    # Build initial user message with image (matches fire_messages format)
    # Format: "Question text\n<image>" - the <image> tag tells the model where to look
    initial_user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": clean_question},
            {"type": "image", "image": image_path},
        ],
    }

    # Running conversation history for refinement (accumulates across turns)
    # Format: [user_question, assistant_answer_0, user_feedback_0, assistant_answer_1, ...]
    refinement_history = [initial_user_message]

    # Running conversation history for feedback (accumulates with FLIPPED roles)
    # Format: [assistant_question, user_answer_0, assistant_feedback_0, user_answer_1, ...]
    # Matches fire_feedback training format
    critic_history = [
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": clean_question},
            ],
        }
    ]

    for turn_idx in range(num_turns):
        if turn_idx == 0:
            # =====================================================
            # Turn 0: Generate initial answer
            # =====================================================
            answer = engine.generate(
                messages=refinement_history,
                system_prompt=VL_ASSISTANT_SYSTEM_PROMPT,
                max_new_tokens=max_new_tokens,
                temperature=answer_temperature,
                top_p=top_p,
            )

            # Add to refinement history
            refinement_history.append({"role": "assistant", "content": answer})

            # Add to critic history (with flipped role - answer is USER)
            critic_history.append({"role": "user", "content": answer})

            # Log history
            full_history.append({"role": "user", "content": f"[IMAGE]\n{clean_question}"})
            full_history.append({"role": "assistant", "content": answer})

            # Store turn
            generated_turns.append({"answer": answer, "feedback": ""})

        else:
            # =====================================================
            # Turn 1+: Generate feedback, then refine with role flip
            # =====================================================

            # -------------------------------------------------
            # Step 1: Generate feedback using CRITIC system prompt
            #
            # IMPORTANT: Matches fire_feedback training format with FLIPPED roles
            # and ACCUMULATED history:
            #   Assistant: question + <image>
            #   User: answer_0
            #   Assistant: feedback_0
            #   User: answer_1
            #   Assistant: feedback_1
            #   ...
            #   User: current_answer (already in critic_history)
            #   Assistant: <generates next feedback>
            #
            # This is the exact format the model was trained on for feedback.
            # -------------------------------------------------
            feedback = engine.generate(
                messages=critic_history,  # Uses accumulated history
                system_prompt=FEEDBACK_CRITIC_SYSTEM_PROMPT,
                max_new_tokens=max_new_tokens,
                temperature=feedback_temperature,
                top_p=top_p,
            )

            # Add feedback to critic history for next iteration
            critic_history.append({"role": "assistant", "content": feedback})

            # Update previous turn with feedback
            generated_turns[-1]["feedback"] = feedback

            # Log the feedback generation
            full_history.append({"role": "user", "content": f"[FEEDBACK]: {feedback}"})

            # -------------------------------------------------
            # Step 2: Generate refined answer using ACCUMULATED history
            #
            # The model sees the FULL conversation so far:
            #   User: [image] + question
            #   Assistant: answer_0
            #   User: feedback_0
            #   Assistant: answer_1
            #   User: feedback_1
            #   ...
            #   Assistant: <generate next refined answer>
            # -------------------------------------------------

            # Add feedback to refinement history as user message
            refinement_history.append({"role": "user", "content": feedback})

            refined_answer = engine.generate(
                messages=refinement_history,
                system_prompt=VL_ASSISTANT_SYSTEM_PROMPT,
                max_new_tokens=max_new_tokens,
                temperature=answer_temperature,
                top_p=top_p,
            )

            # Add refined answer to history for next iteration
            refinement_history.append({"role": "assistant", "content": refined_answer})

            # Add refined answer to critic history (with flipped role - answer is USER)
            critic_history.append({"role": "user", "content": refined_answer})

            # Log the refinement
            full_history.append({"role": "assistant", "content": refined_answer})

            # Store this turn
            generated_turns.append({"answer": refined_answer, "feedback": ""})

    # Get final generated answer
    final_answer = generated_turns[-1]["answer"] if generated_turns else ""

    return SampleResult(
        sample_index=sample_index,
        image_path=image_path,
        question=clean_question,
        generated_turns=generated_turns,
        final_answer=final_answer,
        gt_final_answer=gt_final_answer,
        num_turns=num_turns,
        messages_history=full_history,
    )


# ============================================
# Batched Multi-Turn Orchestrator
# ============================================


def generate_self_reflective_dialogue_batch(
    engine: SelfReflectionEngine,
    samples: list[dict],
    sample_indices: list[int],
    image_base_dir: str,
    generation_config: dict,
    batch_size: int = 4,
) -> list[SampleResult | None]:
    """Generate self-reflective dialogues for multiple samples using batched inference.

    Processes all samples through each turn in lockstep, batching the forward
    passes across samples at the same turn level. Within a sample turns are
    sequential, but across samples at the same turn level all inputs are
    independent and batchable.

    Args:
        engine: Self-reflection inference engine
        samples: List of input samples with messages and images
        sample_indices: List of dataset indices corresponding to each sample
        image_base_dir: Base directory for image paths
        generation_config: Generation parameters
        batch_size: Number of samples per forward pass

    Returns:
        List of SampleResult (or None for failed samples), one per input sample
    """
    max_new_tokens = generation_config.get("max_new_tokens", 512)
    answer_temperature = generation_config.get("answer_temperature", 0.7)
    feedback_temperature = generation_config.get("feedback_temperature", 0.7)
    top_p = generation_config.get("top_p", 0.9)
    requested_num_turns = generation_config.get("num_turns", 0)

    n = len(samples)

    # ---- Step 1: Pre-validate all samples ----
    # Maps sample index -> validated data (only valid samples present)
    valid_data: dict[int, dict] = {}
    for idx, sample in enumerate(samples):
        question, gt_responses, images, _ = parse_sample(sample)
        if not question or not images or not gt_responses:
            logger.warning(f"Sample {sample_indices[idx]}: missing question/images/gt_responses")
            continue

        image_path = resolve_image_path(images[0], image_base_dir)
        if not image_path:
            continue

        num_turns = requested_num_turns if requested_num_turns > 0 else len(gt_responses)
        clean_question = question.replace("<image>", "").strip()

        valid_data[idx] = {
            "question": clean_question,
            "image_path": image_path,
            "gt_responses": gt_responses,
            "num_turns": num_turns,
        }

    # ---- Step 2: Initialize per-sample state ----
    refinement_histories: list[list[dict]] = [[] for _ in range(n)]
    critic_histories: list[list[dict]] = [[] for _ in range(n)]
    generated_turns: list[list[dict]] = [[] for _ in range(n)]
    full_histories: list[list[dict]] = [[] for _ in range(n)]
    active_mask: list[bool] = [False] * n

    max_turns = 0
    for i, p in valid_data.items():
        active_mask[i] = True
        max_turns = max(max_turns, p["num_turns"])

        # Build initial user message with image
        initial_user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": p["question"]},
                {"type": "image", "image": p["image_path"]},
            ],
        }
        refinement_histories[i] = [initial_user_message]

        # Critic history with flipped roles
        critic_histories[i] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "image", "image": p["image_path"]},
                    {"type": "text", "text": p["question"]},
                ],
            }
        ]

    if max_turns == 0:
        return [None] * n

    # ---- Step 3: Turn-by-turn batched generation ----
    for turn_idx in range(max_turns):
        active_indices = [i for i in range(n) if active_mask[i]]
        if not active_indices:
            break

        if turn_idx == 0:
            # Batch-generate initial answers
            for chunk_start in range(0, len(active_indices), batch_size):
                chunk = active_indices[chunk_start : chunk_start + batch_size]
                messages_batch = [refinement_histories[i] for i in chunk]

                try:
                    answers = engine.generate_batch(
                        messages_list=messages_batch,
                        system_prompt=VL_ASSISTANT_SYSTEM_PROMPT,
                        max_new_tokens=max_new_tokens,
                        temperature=answer_temperature,
                        top_p=top_p,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM in batch generation, falling back to sequential")
                    torch.cuda.empty_cache()
                    answers = []
                    for msgs in messages_batch:
                        try:
                            ans = engine.generate(
                                messages=msgs,
                                system_prompt=VL_ASSISTANT_SYSTEM_PROMPT,
                                max_new_tokens=max_new_tokens,
                                temperature=answer_temperature,
                                top_p=top_p,
                            )
                            answers.append(ans)
                        except Exception as e:
                            logger.error(f"Sequential fallback failed: {e}")
                            answers.append("")

                for j, i in enumerate(chunk):
                    answer = answers[j]
                    refinement_histories[i].append({"role": "assistant", "content": answer})
                    critic_histories[i].append({"role": "user", "content": answer})
                    full_histories[i].append(
                        {"role": "user", "content": f"[IMAGE]\n{valid_data[i]['question']}"}
                    )
                    full_histories[i].append({"role": "assistant", "content": answer})
                    generated_turns[i].append({"answer": answer, "feedback": ""})

        else:
            # Step 3a: Batch-generate feedback (critic mode, flipped roles)
            for chunk_start in range(0, len(active_indices), batch_size):
                chunk = active_indices[chunk_start : chunk_start + batch_size]
                messages_batch = [critic_histories[i] for i in chunk]

                try:
                    feedbacks = engine.generate_batch(
                        messages_list=messages_batch,
                        system_prompt=FEEDBACK_CRITIC_SYSTEM_PROMPT,
                        max_new_tokens=max_new_tokens,
                        temperature=feedback_temperature,
                        top_p=top_p,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM in batch feedback, falling back to sequential")
                    torch.cuda.empty_cache()
                    feedbacks = []
                    for msgs in messages_batch:
                        try:
                            fb = engine.generate(
                                messages=msgs,
                                system_prompt=FEEDBACK_CRITIC_SYSTEM_PROMPT,
                                max_new_tokens=max_new_tokens,
                                temperature=feedback_temperature,
                                top_p=top_p,
                            )
                            feedbacks.append(fb)
                        except Exception as e:
                            logger.error(f"Sequential fallback failed: {e}")
                            feedbacks.append("")

                for j, i in enumerate(chunk):
                    feedback = feedbacks[j]
                    critic_histories[i].append({"role": "assistant", "content": feedback})
                    generated_turns[i][-1]["feedback"] = feedback
                    full_histories[i].append({"role": "user", "content": f"[FEEDBACK]: {feedback}"})
                    refinement_histories[i].append({"role": "user", "content": feedback})

            # Step 3b: Batch-generate refined answers
            # Re-collect active indices (some may have been deactivated by OOM)
            active_for_refine = [i for i in active_indices if active_mask[i]]
            for chunk_start in range(0, len(active_for_refine), batch_size):
                chunk = active_for_refine[chunk_start : chunk_start + batch_size]
                messages_batch = [refinement_histories[i] for i in chunk]

                try:
                    refined_answers = engine.generate_batch(
                        messages_list=messages_batch,
                        system_prompt=VL_ASSISTANT_SYSTEM_PROMPT,
                        max_new_tokens=max_new_tokens,
                        temperature=answer_temperature,
                        top_p=top_p,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM in batch refinement, falling back to sequential")
                    torch.cuda.empty_cache()
                    refined_answers = []
                    for msgs in messages_batch:
                        try:
                            ra = engine.generate(
                                messages=msgs,
                                system_prompt=VL_ASSISTANT_SYSTEM_PROMPT,
                                max_new_tokens=max_new_tokens,
                                temperature=answer_temperature,
                                top_p=top_p,
                            )
                            refined_answers.append(ra)
                        except Exception as e:
                            logger.error(f"Sequential fallback failed: {e}")
                            refined_answers.append("")

                for j, i in enumerate(chunk):
                    refined_answer = refined_answers[j]
                    refinement_histories[i].append({"role": "assistant", "content": refined_answer})
                    critic_histories[i].append({"role": "user", "content": refined_answer})
                    full_histories[i].append({"role": "assistant", "content": refined_answer})
                    generated_turns[i].append({"answer": refined_answer, "feedback": ""})

        # Deactivate samples that reached their turn limit
        for i in active_indices:
            if len(generated_turns[i]) >= valid_data[i]["num_turns"]:
                active_mask[i] = False

    # ---- Step 4: Build results ----
    results: list[SampleResult | None] = []
    for i in range(n):
        if i not in valid_data or not generated_turns[i]:
            results.append(None)
            continue

        p = valid_data[i]
        final_answer = generated_turns[i][-1]["answer"] if generated_turns[i] else ""

        results.append(
            SampleResult(
                sample_index=sample_indices[i],
                image_path=p["image_path"],
                question=p["question"],
                generated_turns=generated_turns[i],
                final_answer=final_answer,
                gt_final_answer=p["gt_responses"][-1],
                num_turns=len(generated_turns[i]),
                messages_history=full_histories[i],
            )
        )

    return results


# ============================================
# CLI and Main
# ============================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Self-reflective inference with role-flipped feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint or HuggingFace ID",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save generated results (JSONL)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/fire_preprocessed_v2/fire_messages_test.jsonl",
        help="Path to test dataset (Messages JSONL format)",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs",
        help="Base directory for resolving relative image paths",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Index of first sample to process (skip earlier samples)",
    )
    parser.add_argument(
        "--num_turns",
        type=int,
        default=0,
        help="Number of turns to generate per sample (0 = use ground truth turn count). "
        "Turn 1 = question+answer, Turn 2+ = feedback+refined answer.",
    )

    # Generation configuration
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for answer generation (lower = less hallucination)",
    )
    parser.add_argument(
        "--feedback_temperature",
        type=float,
        default=None,
        help="Sampling temperature for feedback generation (defaults to --temperature)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability",
    )

    # LoRA adapter support
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Base model path for LoRA adapter checkpoints (empty = full model)",
    )

    # Model configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=SUPPORTED_MODEL_TYPES,
        help="Model architecture type. Auto-detected from config.json or path if not specified. "
        f"Options: {', '.join(SUPPORTED_MODEL_TYPES)}",
    )

    # Batch and parallelism configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to process per forward pass. "
        "A6000 (48GB): 2-4, A100 (80GB): 4-8. Default=1 (sequential).",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Enable multi-GPU data parallelism. Each GPU loads a model copy and "
        "processes a shard of the dataset. Launch with: "
        "accelerate launch --num_processes N script.py --multi_gpu",
    )

    return parser.parse_args()


def main():
    """Main function for self-reflective inference with role-flipped feedback."""
    args = parse_args()

    # ---- Multi-GPU setup ----
    if args.multi_gpu:
        from accelerate import Accelerator

        accelerator = Accelerator()
        local_rank = accelerator.local_process_index
        world_size = accelerator.num_processes
        device = f"cuda:{local_rank}"
        is_main = accelerator.is_main_process
        device_map_strategy = "per_gpu"

        logger.info(f"Multi-GPU mode: rank {local_rank}/{world_size}, device={device}")
    else:
        local_rank = 0
        world_size = 1
        device = args.device
        is_main = True
        device_map_strategy = "auto"
        accelerator = None

    # Detect model type (use base model path for detection if LoRA adapter)
    detect_path = args.base_model_path if args.base_model_path else args.model_path
    model_type = detect_model_type(detect_path, args.model_type)

    # Initialize engine
    engine = SelfReflectionEngine(
        model_path=args.model_path,
        model_type=model_type,
        device=device,
        use_flash_attn=not args.no_flash_attn,
        device_map_strategy=device_map_strategy,
        base_model_path=args.base_model_path,
    )

    # Load dataset
    all_samples = load_dataset(args.dataset_path, args.max_samples, args.start_index)

    # ---- Shard dataset across GPUs ----
    if world_size > 1:
        shard_size = len(all_samples) // world_size
        shard_start = local_rank * shard_size
        shard_end = shard_start + shard_size if local_rank < world_size - 1 else len(all_samples)
        samples = all_samples[shard_start:shard_end]
        index_offset = args.start_index + shard_start
        logger.info(
            f"Rank {local_rank}: processing samples {shard_start}-{shard_end} "
            f"({len(samples)} samples)"
        )
    else:
        samples = all_samples
        index_offset = args.start_index

    # Generation config
    feedback_temp = (
        args.feedback_temperature if args.feedback_temperature is not None else args.temperature
    )
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "answer_temperature": args.temperature,
        "feedback_temperature": feedback_temp,
        "top_p": args.top_p,
        "num_turns": args.num_turns,
    }

    # ---- Determine output path (per-rank for multi-GPU) ----
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        rank_output_path = (
            output_path.parent / f"{output_path.stem}_rank{local_rank}{output_path.suffix}"
        )
    else:
        rank_output_path = output_path

    # ---- Process samples ----
    results = []
    failed = 0
    inference_start = time.time()

    if args.batch_size > 1:
        # Batched processing — write results incrementally to avoid data loss on crash
        logger.info(f"Batch mode: batch_size={args.batch_size}")
        pbar = tqdm(total=len(samples), desc=f"Batched inference (bs={args.batch_size})")

        with open(rank_output_path, "w") as f:
            for chunk_start in range(0, len(samples), args.batch_size):
                chunk_end = min(chunk_start + args.batch_size, len(samples))
                chunk_samples = samples[chunk_start:chunk_end]
                chunk_indices = list(
                    range(
                        index_offset + chunk_start,
                        index_offset + chunk_end,
                    )
                )

                try:
                    batch_results = generate_self_reflective_dialogue_batch(
                        engine=engine,
                        samples=chunk_samples,
                        sample_indices=chunk_indices,
                        image_base_dir=args.image_base_dir,
                        generation_config=gen_config,
                        batch_size=args.batch_size,
                    )
                    for result in batch_results:
                        if result:
                            f.write(json.dumps(result.to_dict()) + "\n")
                            f.flush()
                            results.append(result)
                        else:
                            failed += 1
                except Exception as e:
                    logger.error(f"Failed batch starting at index {chunk_start}: {e}")
                    failed += len(chunk_samples)

                pbar.update(len(chunk_samples))
                # Show running per-sample rate in progress bar
                elapsed = time.time() - inference_start
                processed = len(results) + failed
                if processed > 0:
                    pbar.set_postfix(
                        ok=len(results),
                        fail=failed,
                        s_per_sample=f"{elapsed / processed:.1f}",
                    )

        pbar.close()
    else:
        # Original sequential processing (batch_size=1)
        pbar = tqdm(samples, desc="Sequential inference (bs=1)")
        with open(rank_output_path, "w") as f:
            for i, sample in enumerate(pbar, start=index_offset):
                try:
                    result = generate_self_reflective_dialogue(
                        engine=engine,
                        sample=sample,
                        sample_index=i,
                        image_base_dir=args.image_base_dir,
                        generation_config=gen_config,
                    )

                    if result:
                        f.write(json.dumps(result.to_dict()) + "\n")
                        f.flush()
                        results.append(result)
                    else:
                        failed += 1

                except Exception as e:
                    logger.error(f"Failed to process sample {i}: {e}")
                    failed += 1

                # Show running per-sample rate
                elapsed = time.time() - inference_start
                processed = len(results) + failed
                if processed > 0:
                    pbar.set_postfix(
                        ok=len(results),
                        fail=failed,
                        s_per_sample=f"{elapsed / processed:.1f}",
                    )

    inference_elapsed = time.time() - inference_start

    # ---- Merge results from all ranks (file-based sync, no NCCL barrier) ----
    if world_size > 1 and accelerator is not None:
        # Each rank writes a .done marker file when finished
        done_marker = output_path.parent / f"{output_path.stem}_rank{local_rank}.done"
        done_marker.touch()
        logger.info(f"Rank {local_rank} finished, wrote marker: {done_marker}")

        if is_main:
            # Poll for all rank marker files instead of using NCCL barrier
            # This avoids timeout issues when ranks finish at very different times
            all_done = False
            poll_interval = 30  # seconds
            max_wait = 6 * 3600  # 6 hours max wait
            waited = 0

            while not all_done and waited < max_wait:
                missing = []
                for rank in range(world_size):
                    marker = output_path.parent / f"{output_path.stem}_rank{rank}.done"
                    if not marker.exists():
                        missing.append(rank)

                if not missing:
                    all_done = True
                else:
                    logger.info(
                        f"Waiting for ranks {missing} to finish... "
                        f"({waited}s elapsed, polling every {poll_interval}s)"
                    )
                    time.sleep(poll_interval)
                    waited += poll_interval

            if not all_done:
                logger.error(
                    f"Timed out waiting for all ranks after {max_wait}s. "
                    "Merging available results."
                )

            # Merge results
            logger.info("Merging results from all ranks...")
            merged = []
            for rank in range(world_size):
                rank_file = (
                    output_path.parent / f"{output_path.stem}_rank{rank}{output_path.suffix}"
                )
                if rank_file.exists():
                    with open(rank_file) as f_in:
                        for line in f_in:
                            merged.append(json.loads(line))
                    rank_file.unlink()

                # Clean up marker file
                marker = output_path.parent / f"{output_path.stem}_rank{rank}.done"
                if marker.exists():
                    marker.unlink()

            # Sort by sample_index for deterministic output
            merged.sort(key=lambda x: x["sample_index"])
            with open(output_path, "w") as f_out:
                for r in merged:
                    f_out.write(json.dumps(r) + "\n")

            logger.info(f"Merged {len(merged)} results to {output_path}")

    # ---- Print summary (main process only) ----
    if is_main:
        final_output = output_path if world_size > 1 else rank_output_path
        total_processed = len(results) + failed

        print("\n" + "=" * 60)
        print("SELF-REFLECTIVE INFERENCE V2 (Role-Flipped) SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(all_samples if world_size > 1 else samples)}")
        print(f"Successfully processed: {len(results)}")
        print(f"Failed: {failed}")
        if world_size > 1:
            print(f"GPUs used: {world_size}")
        if args.batch_size > 1:
            print(f"Batch size: {args.batch_size}")
        print(f"Output saved to: {final_output}")

        # Timing stats
        print("\n--- Timing ---")
        minutes, seconds = divmod(inference_elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            print(f"Total inference time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        elif minutes > 0:
            print(f"Total inference time: {int(minutes)}m {seconds:.1f}s")
        else:
            print(f"Total inference time: {seconds:.1f}s")
        if total_processed > 0:
            per_sample = inference_elapsed / total_processed
            samples_per_min = 60.0 / per_sample if per_sample > 0 else 0
            print(f"Per-sample time: {per_sample:.2f}s")
            print(f"Throughput: {samples_per_min:.1f} samples/min")

        if results:
            avg_turns = sum(r.num_turns for r in results) / len(results)
            print(f"Average turns per sample: {avg_turns:.2f}")

            # Show a preview of first result
            print("\n--- First Result Preview ---")
            first = results[0]
            print(f"Question: {first.question[:100]}...")
            print(f"Num turns: {first.num_turns}")
            for i, turn in enumerate(first.generated_turns):
                print(f"  Turn {i}: {turn['answer'][:80]}...")
                if turn["feedback"]:
                    print(f"    Feedback: {turn['feedback'][:80]}...")
            print(f"GT final: {first.gt_final_answer[:100]}...")

        print("=" * 60)


if __name__ == "__main__":
    main()
