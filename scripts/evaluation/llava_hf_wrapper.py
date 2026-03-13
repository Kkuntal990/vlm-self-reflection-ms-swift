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

    def __init__(self, model_path: str, **kwargs) -> None:
        """Initialize the LLaVA-HF model.

        Args:
            model_path: Path to HuggingFace-format LLaVA checkpoint.
            **kwargs: Additional arguments passed to BaseModel.
        """
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.model_path = model_path

        self.processor = AutoProcessor.from_pretrained(model_path)

        try:
            import flash_attn  # noqa: F401

            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_flash_attention_2=True,
            )
        except ImportError:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
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
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"} for _ in images
                ]
                + [{"type": "text", "text": prompt.replace("<image>", "").strip()}],
            }
        ]

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
