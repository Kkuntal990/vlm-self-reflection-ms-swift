"""
Self-Reflective VLMEvalKit Wrapper for Qwen2.5-VL.

Extends Qwen2VLChat to perform multi-turn self-reflection during benchmark
evaluation. Instead of a single-pass answer, the model:

1. Generates an initial answer (Turn 0)
2. Generates feedback in critic mode with flipped roles (Turn 1+)
3. Generates a refined answer incorporating feedback (Turn 1+)
4. Returns the final refined answer to VLMEvalKit's scoring pipeline

The conversation patterns match the FIRE training data format exactly:
- Feedback uses critic system prompt with role-flipped history
- Refinement uses VL assistant system prompt with accumulated history

Usage:
    Copy this file to VLMEvalKit's vlmeval/vlm/ directory and register
    the model class in vlmeval/config.py via register_vlmevalkit_model.py.
"""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)

# System prompts (matching training data)
_DEFAULT_VL_ASSISTANT_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate, "
    "detailed, and grounded answers based on the image and the user's "
    "instructions. When given feedback, critique, or scores, revise your "
    "response to improve correctness, specificity, and completeness."
)

_DEFAULT_FEEDBACK_CRITIC_PROMPT = (
    "You are a vision-language critic that evaluates answers to visual "
    "questions and helps improve them. Use the image, question, and dialogue "
    "history to judge the latest answer by: - correctness and visual grounding "
    "(matches what's visible / implied), - compliance with the requested format "
    "(option letter, units, etc.), - completeness. Be conservative: confirm "
    "the answer as correct if it is consistent with the image/question and "
    'follows the required format. Only say "incorrect" when you can name a '
    "specific contradiction or missing requirement. If you are uncertain, do "
    "not guess—ask to re-check one concrete detail. Write a brief natural "
    "paragraph: start with a clear verdict, give 1–2 grounded reasons, and "
    "(if needed) one practical next step. Keep the tone polite and encouraging."
)


def Qwen2VLSelfReflectiveChat(*args, **kwargs):
    """Factory function that creates a self-reflective Qwen2VL model.

    This is a factory because VLMEvalKit's config.py uses partial() to
    register models, and the base class Qwen2VLChat may not be importable
    at config parse time. This defers the import.

    All arguments are forwarded to _Qwen2VLSelfReflectiveChat.__init__.
    """
    return _Qwen2VLSelfReflectiveChat(*args, **kwargs)


class _Qwen2VLSelfReflectiveChat:
    """Self-reflective wrapper around Qwen2VLChat.

    Inherits from Qwen2VLChat and overrides generate_inner to perform
    multi-turn self-reflection before returning the final answer.

    Additional Args (beyond Qwen2VLChat):
        num_turns: Number of feedback-refinement cycles (default: 1)
        feedback_max_tokens: Max tokens for feedback generation (default: 512)
        vl_system_prompt: System prompt for VL assistant mode
        critic_system_prompt: System prompt for critic/feedback mode
        feedback_temperature: Temperature for feedback generation (default: 0.7)
        conversation_log_path: Path to save full conversations as JSONL (default: None)
    """

    def __init__(
        self,
        model_path: str,
        num_turns: int = 1,
        feedback_max_tokens: int = 512,
        vl_system_prompt: str | None = None,
        critic_system_prompt: str | None = None,
        feedback_temperature: float = 0.7,
        conversation_log_path: str | None = None,
        **kwargs,
    ):
        # Import the base class at instantiation time
        from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

        # Store self-reflection params before super init
        self.num_turns = num_turns
        self.feedback_max_tokens = feedback_max_tokens
        self.feedback_temperature = feedback_temperature
        self.vl_system_prompt = vl_system_prompt or os.environ.get(
            "VL_ASSISTANT_SYSTEM_PROMPT", _DEFAULT_VL_ASSISTANT_PROMPT
        )
        self.critic_system_prompt = critic_system_prompt or os.environ.get(
            "FEEDBACK_CRITIC_SYSTEM_PROMPT", _DEFAULT_FEEDBACK_CRITIC_PROMPT
        )

        # Conversation log for debugging (saves full question→answer→feedback→refined)
        self.conversation_log_path = conversation_log_path or os.environ.get(
            "CONVERSATION_LOG_PATH", None
        )
        self._sample_counter = 0

        # Initialize the base Qwen2VLChat model with VL assistant system prompt
        # so the initial answer also uses it (matches FT training format)
        kwargs.setdefault("system_prompt", self.vl_system_prompt)
        self._base = Qwen2VLChat(model_path=model_path, **kwargs)

        # Copy attributes that VLMEvalKit expects on the model object
        self.model_path = self._base.model_path
        self.INTERLEAVE = self._base.INTERLEAVE
        self.VIDEO_LLM = self._base.VIDEO_LLM
        self.INSTALL_REQ = self._base.INSTALL_REQ

        logger.info(
            f"Self-reflective wrapper initialized: "
            f"num_turns={self.num_turns}, "
            f"feedback_max_tokens={self.feedback_max_tokens}"
        )

    def __getattr__(self, name):
        """Delegate attribute access to the base Qwen2VLChat instance."""
        return getattr(self._base, name)

    def generate_inner(self, message, dataset=None):
        """Override generate_inner to perform multi-turn self-reflection.

        Flow:
            1. Initial answer via standard single-turn generation
            2. For each refinement turn:
               a. Build critic conversation (flipped roles) → generate feedback
               b. Build refinement conversation (accumulated) → generate refined answer
            3. Return final answer

        Args:
            message: List of dicts with keys ['type', 'value'] from VLMEvalKit
            dataset: Dataset name (e.g., 'MMBench_DEV_EN')

        Returns:
            Final (refined) answer string
        """
        # Step 1: Generate initial answer using the base class
        initial_answer = self._base.generate_inner(message, dataset=dataset)

        if self.num_turns <= 0:
            return initial_answer

        # Extract image and text content from the VLMEvalKit message format
        image_items, question_text = self._extract_content(message, dataset)

        # Track conversation for logging
        conversation = {
            "sample_id": self._sample_counter,
            "dataset": dataset,
            "question": question_text,
            "initial_answer": initial_answer,
            "turns": [],
        }
        self._sample_counter += 1

        current_answer = initial_answer

        for turn_idx in range(self.num_turns):
            # Step 2a: Generate feedback (critic mode, flipped roles)
            feedback = self._generate_feedback(
                image_items, question_text, current_answer, turn_idx, dataset
            )

            logger.info(f"Turn {turn_idx + 1} feedback: {feedback[:120]}")

            # Step 2b: Generate refined answer
            refined_answer = self._generate_refinement(
                image_items, question_text, current_answer, feedback, dataset
            )

            logger.info(f"Turn {turn_idx + 1} refined: {refined_answer[:120]}")

            conversation["turns"].append(
                {
                    "turn": turn_idx + 1,
                    "answer_before": current_answer,
                    "feedback": feedback,
                    "answer_after": refined_answer,
                }
            )

            current_answer = refined_answer

        conversation["final_answer"] = current_answer

        # Save conversation log
        self._log_conversation(conversation)

        return current_answer

    def _log_conversation(self, conversation: dict) -> None:
        """Append a conversation record to the JSONL log file.

        Args:
            conversation: Dict with question, initial_answer, turns, final_answer
        """
        if not self.conversation_log_path:
            return
        try:
            import json
            import os

            os.makedirs(os.path.dirname(self.conversation_log_path), exist_ok=True)
            with open(self.conversation_log_path, "a") as f:
                f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write conversation log: {e}")

    def _extract_content(self, message, dataset=None):
        """Extract image content items and question text from VLMEvalKit message.

        Args:
            message: VLMEvalKit format list of dicts with type/value keys
            dataset: Dataset name for custom processing

        Returns:
            Tuple of (image_content_items, question_text)
        """
        image_items = []
        text_parts = []

        for item in message:
            if item["type"] == "image":
                image_items.append(self._base._prepare_content([item], dataset=dataset))
            elif item["type"] == "text":
                text_parts.append(item["value"])

        question_text = "\n".join(text_parts)
        # Flatten image items (each _prepare_content returns a list)
        flat_images = []
        for img_list in image_items:
            flat_images.extend(img_list)

        return flat_images, question_text

    def _generate_multi_turn(self, messages, max_new_tokens=None):
        """Generate a response from a multi-turn conversation.

        Uses the base model's processor and model directly to handle
        multi-turn conversations that generate_inner_transformers cannot.

        Args:
            messages: Full conversation in chat format
                [{"role": "system", "content": "..."}, {"role": "user", ...}]
            max_new_tokens: Override max tokens (uses base default if None)

        Returns:
            Generated response string
        """
        from qwen_vl_utils import process_vision_info

        text = self._base.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info([messages])
        inputs = self._base.processor(
            text=text, images=images, videos=videos, padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        gen_kwargs = dict(self._base.generate_kwargs)
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens

        generated_ids = self._base.model.generate(**inputs, **gen_kwargs)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self._base.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    def _generate_feedback(self, image_items, question_text, current_answer, turn_idx, dataset):
        """Generate feedback using critic system prompt with flipped roles.

        Matches the FIRE fire_feedback training format:
            System: critic prompt
            Assistant: [image] + question  (flipped - assistant asks)
            User: current_answer           (flipped - user provides answer)
            Assistant: <generates feedback>

        Args:
            image_items: Prepared image content items
            question_text: The question text
            current_answer: The answer to critique
            turn_idx: Current turn index
            dataset: Dataset name

        Returns:
            Generated feedback string
        """
        messages = [
            {"role": "system", "content": self.critic_system_prompt},
            {
                "role": "assistant",
                "content": image_items + [{"type": "text", "text": question_text}],
            },
            {
                "role": "user",
                "content": current_answer,
            },
        ]

        # Use lower temperature for feedback to get focused critique
        orig_temp = self._base.generate_kwargs.get("temperature", 0.01)
        self._base.generate_kwargs["temperature"] = self.feedback_temperature
        try:
            feedback = self._generate_multi_turn(messages, max_new_tokens=self.feedback_max_tokens)
        finally:
            self._base.generate_kwargs["temperature"] = orig_temp

        return feedback

    def _generate_refinement(self, image_items, question_text, current_answer, feedback, dataset):
        """Generate refined answer using VL assistant prompt with feedback.

        Matches the FIRE fire_messages training format:
            System: VL assistant prompt
            User: [image] + question
            Assistant: current_answer
            User: feedback
            Assistant: <generates refined answer>

        Args:
            image_items: Prepared image content items
            question_text: The question text
            current_answer: The current answer to refine
            feedback: The feedback to incorporate
            dataset: Dataset name

        Returns:
            Generated refined answer string
        """
        messages = [
            {"role": "system", "content": self.vl_system_prompt},
            {
                "role": "user",
                "content": image_items + [{"type": "text", "text": question_text}],
            },
            {
                "role": "assistant",
                "content": current_answer,
            },
            {
                "role": "user",
                "content": feedback,
            },
        ]

        return self._generate_multi_turn(messages)

    # --- VLMEvalKit interface methods delegated to base ---

    def generate(self, message, dataset=None):
        """VLMEvalKit calls this; it wraps generate_inner with retries."""
        return self._base.generate.__func__(self, message, dataset)

    def use_custom_prompt(self, dataset):
        """Delegate to base."""
        return self._base.use_custom_prompt(dataset)

    def build_prompt(self, line, dataset=None):
        """Delegate to base."""
        return self._base.build_prompt(line, dataset=dataset)

    def dump_image(self, line, dataset=None):
        """Delegate to base."""
        return self._base.dump_image(line, dataset=dataset)
