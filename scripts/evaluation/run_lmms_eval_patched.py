#!/usr/bin/env python3
"""
Wrapper to run lmms-eval with LLaVA-NeXT compatibility patches.

LLaVA-NeXT imports several functions from `transformers.modeling_utils` that newer
transformers versions moved to `transformers.pytorch_utils`. This wrapper patches
them back before lmms-eval loads the llava module.

Patched functions:
    - apply_chunking_to_forward
    - find_pruneable_heads_and_indices
    - prune_linear_layer

Usage:
    python scripts/evaluation/run_lmms_eval_patched.py [lmms-eval args...]

    # Example:
    python scripts/evaluation/run_lmms_eval_patched.py \
        --model llava_onevision \
        --model_args pretrained=/path/to/model,model_name=llava_qwen \
        --tasks mme --batch_size 1 --limit 10
"""

# Functions that LLaVA-NeXT expects in transformers.modeling_utils
# but newer transformers moved to transformers.pytorch_utils
_RELOCATED_FUNCTIONS = [
    "apply_chunking_to_forward",
    "find_pruneable_heads_and_indices",
    "prune_linear_layer",
]


def _make_fallback_stubs() -> dict:
    """Create fallback implementations for functions removed from transformers.

    These functions are only used for model pruning / chunked forward passes,
    which are never invoked during inference.  Providing stubs lets the LLaVA
    module-level imports succeed without affecting evaluation behaviour.
    """
    import torch

    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        if chunk_size > 0:
            num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
            input_chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
            output_chunks = tuple(forward_fn(*chunk) for chunk in zip(*input_chunks))
            return torch.cat(output_chunks, dim=chunk_dim)
        return forward_fn(*input_tensors)

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    def prune_linear_layer(layer, index, dim=0):
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
            layer.weight.device
        )
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    return {
        "apply_chunking_to_forward": apply_chunking_to_forward,
        "find_pruneable_heads_and_indices": find_pruneable_heads_and_indices,
        "prune_linear_layer": prune_linear_layer,
    }


def patch_transformers_for_llava() -> None:
    """Restore functions moved/removed from transformers.modeling_utils for LLaVA-NeXT.

    First tries to copy from transformers.pytorch_utils.  For any function still
    missing, injects a fallback implementation so LLaVA-NeXT imports succeed.
    """
    import transformers.modeling_utils

    # Try to pull from pytorch_utils first (where transformers relocated them)
    try:
        import transformers.pytorch_utils as pytorch_utils
    except ImportError:
        pytorch_utils = None

    fallbacks = _make_fallback_stubs()

    for func_name in _RELOCATED_FUNCTIONS:
        if hasattr(transformers.modeling_utils, func_name):
            continue
        # Prefer the real implementation from pytorch_utils
        func = getattr(pytorch_utils, func_name, None) if pytorch_utils else None
        if func is None:
            # Use our fallback stub
            func = fallbacks.get(func_name)
        if func is not None:
            setattr(transformers.modeling_utils, func_name, func)


class _LlavaOVGenerateWrapper:
    """Wraps HF-native LlavaOnevisionForConditionalGeneration so that the
    lmms-eval llava_onevision wrapper (which passes ``images=``) works
    transparently.  HF-native models expect ``pixel_values=`` instead.
    """

    def __init__(self, model):
        self._model = model

    def generate(self, *args, **kwargs):
        # Remap images -> pixel_values
        if "images" in kwargs:
            images = kwargs.pop("images")
            if images is not None:
                kwargs["pixel_values"] = images
        return self._model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # Remap images -> pixel_values for forward pass (loglikelihood)
        if "images" in kwargs:
            images = kwargs.pop("images")
            if images is not None:
                kwargs["pixel_values"] = images
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def patch_llava_builder() -> None:
    """Replace LLaVA-NeXT's load_pretrained_model with a version that handles
    HuggingFace-native llava_onevision checkpoints (e.g. trained by ms-swift).

    The original builder tries to use LlavaQwenForCausalLM / LlavaLlamaForCausalLM
    based on model_name, which fails for native llava_onevision checkpoints.
    This patch detects the model_type from config.json and uses the correct
    transformers class.
    """
    import llava.model.builder as builder

    _original_load = builder.load_pretrained_model

    def _patched_load_pretrained_model(
        model_path, model_base, model_name, device_map="auto", **kwargs
    ):
        import logging

        import torch
        from transformers import AutoConfig

        logger = logging.getLogger("lmms-eval")

        # Check the model_type in config.json
        config = AutoConfig.from_pretrained(model_path)
        model_type = getattr(config, "model_type", "")

        if model_type == "llava_onevision":
            logger.info(
                "Detected HF-native llava_onevision checkpoint, "
                "loading with LlavaOnevisionForConditionalGeneration"
            )
            from transformers import (
                AutoProcessor,
                AutoTokenizer,
                LlavaOnevisionForConditionalGeneration,
            )

            attn_impl = kwargs.pop("attn_implementation", "flash_attention_2")
            # Remove kwargs the builder passes that from_pretrained doesn't expect
            kwargs.pop("multimodal", None)
            kwargs.pop("overwrite_config", None)
            kwargs.pop("customized_config", None)

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            raw_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.float16,
                attn_implementation=attn_impl,
            )
            raw_model.eval()

            # Wrap model so that lmms-eval's `images=` kwarg is translated to
            # `pixel_values=` which HF-native LlavaOnevision expects.
            model = _LlavaOVGenerateWrapper(raw_model)

            processor = AutoProcessor.from_pretrained(model_path)
            image_processor = processor.image_processor

            max_length = getattr(config, "max_position_embeddings", 4096)

            return tokenizer, model, image_processor, max_length
        else:
            # Fall back to original builder for non-onevision models
            return _original_load(
                model_path, model_base, model_name, device_map=device_map, **kwargs
            )

    builder.load_pretrained_model = _patched_load_pretrained_model


if __name__ == "__main__":
    patch_transformers_for_llava()
    patch_llava_builder()

    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()
