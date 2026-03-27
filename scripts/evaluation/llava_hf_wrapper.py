"""Custom VLMEvalKit wrapper for HuggingFace-format LLaVA-1.5 models.

VLMEvalKit's built-in LLaVA class requires the original llava package.
This wrapper uses HuggingFace transformers directly, supporting checkpoints
with architecture 'LlavaForConditionalGeneration'.

Usage:
    This file is copied into VLMEvalKit's vlm directory at runtime by
    run_vlm_benchmarks.sh, and the model is registered in config.py by
    register_vlmevalkit_model.py.
"""

import torch
from vlmeval.vlm.base import BaseModel
from vlmeval.smp import get_logger

logger = get_logger("LLaVA_HF")


class LLaVA_HF(BaseModel):
    """HuggingFace-native LLaVA-1.5 model wrapper for VLMEvalKit."""

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path: str, base_model_path: str = "", system_prompt: str = "", **kwargs) -> None:
        """Initialize the LLaVA-HF model.

        Args:
            model_path: Path to HuggingFace-format LLaVA checkpoint or LoRA adapter.
            base_model_path: Base model path for LoRA adapters (empty = full model).
            system_prompt: System prompt to prepend to conversations (empty = none).
            **kwargs: Additional arguments passed to BaseModel.
        """
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.model_path = model_path
        self.system_prompt = system_prompt

        # For LoRA adapters, load processor from base model
        processor_path = base_model_path if base_model_path else model_path
        self.processor = AutoProcessor.from_pretrained(processor_path)

        # Determine flash attention support
        attn_kwargs = {}
        try:
            import flash_attn  # noqa: F401
            attn_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

        if base_model_path:
            # LoRA adapter: load base model, apply adapter, merge
            from peft import PeftModel

            logger.info(f"Loading base model from {base_model_path}")
            base_model = LlavaForConditionalGeneration.from_pretrained(
                base_model_path,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                **attn_kwargs,
            )
            logger.info(f"Loading LoRA adapter from {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
            logger.info("LoRA adapter merged")
        else:
            # Full model checkpoint
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                **attn_kwargs,
            )

        model = model.eval()
        self.model = model.cuda()

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        logger.info(f"Loaded LLaVA-HF model from {model_path}")

    def generate_inner(self, message, dataset=None) -> str:
        """Generate a response for the given message.

        Args:
            message: List of dicts with 'type' and 'value' keys.
            dataset: Optional dataset name for dataset-specific behavior.

        Returns:
            Generated text response.
        """
        from PIL import Image

        images = []
        prompt_parts = []

        for item in message:
            if item["type"] == "image":
                images.append(Image.open(item["value"]).convert("RGB"))
                prompt_parts.append("<image>")
            elif item["type"] == "text":
                prompt_parts.append(item["value"])

        prompt = "\n".join(prompt_parts)

        # Format as conversation for LLaVA-1.5
        conversation = []
        if self.system_prompt:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })
        conversation.append({
            "role": "user",
            "content": [
                {"type": "image"} for _ in images
            ]
            + [{"type": "text", "text": prompt.replace("<image>", "").strip()}],
        })

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        if images:
            inputs = self.processor(
                text=text_prompt,
                images=images,
                return_tensors="pt",
            ).to(self.model.device, torch.float16)
        else:
            inputs = self.processor(
                text=text_prompt,
                return_tensors="pt",
            ).to(self.model.device, torch.float16)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **self.kwargs,
            )

        # Decode only the generated tokens (skip the input)
        generated = output[0][inputs["input_ids"].shape[1] :]
        response = self.processor.decode(generated, skip_special_tokens=True)
        return response.strip()
