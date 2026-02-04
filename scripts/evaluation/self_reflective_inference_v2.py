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
_DEFAULT_FEEDBACK_CRITIC_PROMPT = """You are a helpful assistant that provides constructive feedback on answers to visual questions.

Given an image, a question, an answer, and the conversation history:
1. Identify what is correct and what is incorrect in the answer.
2. Base your critique ONLY on visual evidence from the image.

IMPORTANT:
- First write a line starting with "EVIDENCE:" and briefly state the specific visual evidence you are using
  (e.g., an object count, a color, a label, a number, a position, or a visible text).
- Then write a line starting with "FIX:" and state exactly what should be changed or corrected in the answer.
- If the answer is fully correct and supported by the image, write:
  "EVIDENCE: The answer matches the visible evidence."
  "FIX: No change needed."

Do NOT:
- Introduce new facts not visible in the image.
- Reinterpret or question the intent of the question.
- Give generic advice like "look again" without stating evidence.

"""
FEEDBACK_CRITIC_SYSTEM_PROMPT = os.environ.get(
    "FEEDBACK_CRITIC_SYSTEM_PROMPT", _DEFAULT_FEEDBACK_CRITIC_PROMPT
)


# ============================================
# Model Type Detection
# ============================================

MODEL_TYPE_QWEN2_5_VL = "qwen2_5_vl"
MODEL_TYPE_LLAVA_ONEVISION = "llava_onevision"
SUPPORTED_MODEL_TYPES = [MODEL_TYPE_QWEN2_5_VL, MODEL_TYPE_LLAVA_ONEVISION]


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
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to fine-tuned model checkpoint or HuggingFace ID
            model_type: Model architecture type ("qwen2_5_vl" or "llava_onevision")
            device: Device to run inference on
            dtype: Model data type
            use_flash_attn: Whether to use flash attention
        """
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: '{model_type}'. Supported: {SUPPORTED_MODEL_TYPES}"
            )

        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.dtype = dtype

        logger.info(f"Loading model from {model_path} (type: {model_type})")

        # Lazy import for processor (works for both model types)
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)

        # Load model class based on type
        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        if model_type == MODEL_TYPE_LLAVA_ONEVISION:
            self._load_llava_onevision(attn_impl, device, dtype)
        else:
            self._load_qwen2_5_vl(attn_impl, device, dtype)

        self.model.eval()
        logger.info(f"Model loaded successfully (type: {model_type})")

    def _load_qwen2_5_vl(self, attn_impl: str, device: str, dtype: torch.dtype) -> None:
        """Load Qwen2.5-VL model.

        Args:
            attn_impl: Attention implementation ("flash_attention_2" or "eager")
            device: Device to load model on
            dtype: Model data type
        """
        from transformers import Qwen2_5_VLForConditionalGeneration

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(f"Flash attention failed ({e}), falling back to eager")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto" if device == "cuda" else None,
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

        try:
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(f"Flash attention failed ({e}), falling back to eager")
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto" if device == "cuda" else None,
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
        if self.model_type == MODEL_TYPE_LLAVA_ONEVISION:
            return self._generate_llava_onevision(
                messages, system_prompt, max_new_tokens, temperature, top_p, do_sample
            )
        return self._generate_qwen2_5_vl(
            messages, system_prompt, max_new_tokens, temperature, top_p, do_sample
        )

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
        llava_messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        pil_images = []

        for msg in messages:
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

    return parser.parse_args()


def main():
    """Main function for self-reflective inference with role-flipped feedback."""
    args = parse_args()

    # Detect model type
    model_type = detect_model_type(args.model_path, args.model_type)

    # Initialize engine
    engine = SelfReflectionEngine(
        model_path=args.model_path,
        model_type=model_type,
        device=args.device,
        use_flash_attn=not args.no_flash_attn,
    )

    # Load dataset
    samples = load_dataset(args.dataset_path, args.max_samples, args.start_index)

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

    # Process samples
    results = []
    failed = 0

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, sample in enumerate(
            tqdm(samples, desc="Self-reflective inference (v2)"), start=args.start_index
        ):
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
                    results.append(result)
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                failed += 1

    # Print summary
    print("\n" + "=" * 60)
    print("SELF-REFLECTIVE INFERENCE V2 (Role-Flipped) SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_path}")

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
