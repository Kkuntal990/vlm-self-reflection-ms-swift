#!/usr/bin/env python3
"""
LLaVA-OneVision evaluation via lmms-eval using HuggingFace transformers directly.

This script registers a clean LLaVA-OneVision model wrapper with lmms-eval that uses
the native LlavaOnevisionForConditionalGeneration from transformers. It does NOT depend
on the LLaVA-NeXT package, avoiding all compatibility issues with that package.

Usage:
    # Single GPU
    python scripts/evaluation/eval_llava_ov_lmms.py \
        --model llava_ov_hf \
        --model_args pretrained=/outputs/checkpoint-2735 \
        --tasks mme \
        --batch_size 1 \
        --output_path ./outputs/eval_results

    # Multi-GPU with accelerate
    accelerate launch --num_processes=4 --main_process_port 29500 \
        scripts/evaluation/eval_llava_ov_lmms.py \
        --model llava_ov_hf \
        --model_args pretrained=/outputs/checkpoint-2735 \
        --tasks mme,mmbench_en_dev \
        --batch_size 1 \
        --output_path ./outputs/eval_results

Reference:
    - lmms-eval: https://github.com/EvolvingLMMs-Lab/lmms-eval
    - LlavaOnevision: https://huggingface.co/docs/transformers/model_doc/llava_onevision
"""

import warnings

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


warnings.filterwarnings("ignore")

from loguru import logger as eval_logger


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"


@register_model("llava_ov_hf")
class LlavaOVHF(lmms):
    """LLaVA-OneVision model wrapper using HuggingFace transformers directly.

    Uses LlavaOnevisionForConditionalGeneration and AutoProcessor from transformers.
    No dependency on the LLaVA-NeXT package.

    Args:
        pretrained: Path to model checkpoint or HuggingFace model ID
        revision: Model revision
        device: Device to load model on
        dtype: Data type for model weights (auto, float16, bfloat16, float32)
        batch_size: Batch size per GPU
        trust_remote_code: Whether to trust remote code
        attn_implementation: Attention implementation (flash_attention_2, eager, sdpa)
        device_map: Device map for model parallelism (auto, or empty for single GPU)
        chat_template: Override chat template for the tokenizer
        use_cache: Whether to use KV cache during generation
        max_frames_num: Maximum number of video frames to sample
    """

    is_simple = True

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        revision: str = "main",
        device: str = "cuda",
        dtype: str | torch.dtype | None = "auto",
        batch_size: int = 1,
        trust_remote_code: bool | None = False,
        attn_implementation: str | None = None,
        device_map: str = "",
        chat_template: str | None = None,
        use_cache: bool = True,
        max_frames_num: int | None = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self.max_frames_num = max_frames_num
        self.pretrained = pretrained

        eval_logger.info(f"Loading LlavaOnevisionForConditionalGeneration from {pretrained}")
        self._model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=dtype,
            device_map=self.device_map if self.device_map else None,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

        self._image_processor = AutoProcessor.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        # Pad from left for batched generation
        self._image_processor.tokenizer.padding_side = "left"
        self._tokenizer = self._image_processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.use_cache = use_cache

        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type. Only DDP, FSDP, and DEEPSPEED are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )
                eval_logger.info(
                    "Detected DistributedType.DEEPSPEED. "
                    "Make sure you run `accelerate config` and set zero stage to 0"
                )
            if accelerator.distributed_type in (DistributedType.FSDP, DistributedType.DEEPSPEED):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.accelerator = accelerator
        eval_logger.info("LLaVA-OneVision model loaded successfully")

    @property
    def config(self) -> object:
        return self._config

    @property
    def tokenizer(self) -> object:
        return self._tokenizer

    @property
    def model(self) -> object:
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return getattr(self._config, "max_position_embeddings", 4096)

    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def input_device(self) -> torch.device:
        """Device to send input tensors to.

        When using device_map='auto' (pipeline parallelism), inputs must go
        to the device of the model's first layer, not necessarily cuda:0.
        """
        if self.device_map == "auto":
            return self.model.device
        return self._device

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def tok_encode(
        self, string: str, left_truncate_len: int | None = None, add_special_tokens: bool = False
    ) -> list[int]:
        """Encode string to token IDs.

        Args:
            string: Text to encode
            left_truncate_len: Truncate from left to this length
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens: list[int]) -> str:
        """Decode token IDs to string.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded string
        """
        return self.tokenizer.decode(tokens)

    def flatten(self, input_list: list) -> list:
        """Flatten a nested list by one level.

        Args:
            input_list: Nested list to flatten

        Returns:
            Flattened list
        """
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_prompt(self, context: str, num_visuals: int, task_type: str) -> str:
        """Build a properly formatted prompt with image/video placeholders.

        LlavaOnevision requires structured content with {"type": "image"} entries
        passed through processor.apply_chat_template() so that <image> placeholder
        tokens are correctly inserted into the prompt.

        Args:
            context: Raw text context (may contain <image>/<video> tokens)
            num_visuals: Number of images or video clips
            task_type: One of "image", "video", "text"

        Returns:
            Formatted prompt string with proper image placeholder tokens
        """
        # Strip any existing <image>/<video> tokens from context
        clean_context = (
            context.replace(DEFAULT_IMAGE_TOKEN, "").replace(DEFAULT_VIDEO_TOKEN, "").strip()
        )

        # Build structured content list for processor
        content = []
        if task_type == "image" and num_visuals > 0:
            for _ in range(num_visuals):
                content.append({"type": "image"})
        elif task_type == "video" and num_visuals > 0:
            for _ in range(num_visuals):
                content.append({"type": "video"})
        content.append({"type": "text", "text": clean_context})

        messages = [{"role": "user", "content": content}]
        return self._image_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_prompt_with_continuation(
        self, context: str, continuation: str, num_visuals: int, task_type: str
    ) -> tuple[str, str]:
        """Build prompt and prompt+continuation for loglikelihood evaluation.

        Args:
            context: Raw text context
            continuation: Target continuation text
            num_visuals: Number of images
            task_type: One of "image", "video", "text"

        Returns:
            Tuple of (prompt, prompt_and_continuation) strings
        """
        clean_context = (
            context.replace(DEFAULT_IMAGE_TOKEN, "").replace(DEFAULT_VIDEO_TOKEN, "").strip()
        )

        content = []
        if task_type == "image" and num_visuals > 0:
            for _ in range(num_visuals):
                content.append({"type": "image"})
        content.append({"type": "text", "text": clean_context})

        messages_prompt = [{"role": "user", "content": content}]
        messages_full = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": continuation}]},
        ]

        prompt = self._image_processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_and_continuation = self._image_processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        return prompt, prompt_and_continuation

    def load_video(self, video_path: str | list, max_frames_num: int) -> np.ndarray:
        """Load video frames with uniform sampling.

        Args:
            video_path: Path to video file
            max_frames_num: Maximum number of frames to sample

        Returns:
            Numpy array of frames (num_frames, height, width, channels)
        """
        if isinstance(video_path, list):
            video_path = video_path[0]
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of continuations given contexts.

        Args:
            requests: List of evaluation instances

        Returns:
            List of (log_probability, is_greedy) tuples
        """
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for context, doc_to_target, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            if isinstance(doc_to_target, str):
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            # Build properly structured prompts with image placeholders
            prompt, prompt_and_continuation = self._build_prompt_with_continuation(
                context, continuation, num_visuals=len(visuals), task_type="image"
            )

            formatted_contexts = [prompt]
            formatted_continuation = [prompt_and_continuation]

            model_inputs = self._image_processor(
                text=formatted_continuation, images=visuals, return_tensors="pt"
            ).to(self.input_device, self.model.dtype)
            labels = model_inputs["input_ids"].clone()
            contxt_id = self._image_processor(text=formatted_contexts, return_tensors="pt")[
                "input_ids"
            ]
            labels[:, : contxt_id.shape[1]] = -100

            if self.accelerator.is_main_process and doc_id % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id}:\n\n{formatted_contexts[0]}\n")

            with torch.inference_mode():
                outputs = self.model(**model_inputs, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = model_inputs["input_ids"][:, contxt_id.shape[1] :]
            greedy_tokens = greedy_tokens[
                :, contxt_id.shape[1] : model_inputs["input_ids"].shape[1]
            ]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text until stopping condition is met.

        Args:
            requests: List of evaluation instances

        Returns:
            List of generated text strings
        """
        res = []

        def _collate(x: tuple) -> tuple:
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size + (
            1 if len(requests) % self.batch_size != 0 else 0
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            if len(visuals) == 0:
                task_type = "text"
            elif isinstance(visuals[0], PIL.Image.Image):
                task_type = "image"
            elif isinstance(visuals[0], str):
                task_type = "video"
            else:
                task_type = "image"

            gen_kwargs = all_gen_kwargs[0]

            until = [self.tok_decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected gen_kwargs['until'] to be Union[str, list] but got {type(until)}"
                    )

            assert self.batch_size_per_gpu == 1, "batch_size_per_gpu > 1 not supported"
            context = contexts[0]

            # Build properly structured prompt with image/video placeholders
            text = self._build_prompt(context, num_visuals=len(visuals), task_type=task_type)

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            # Load video frames if needed
            if task_type == "video":
                try:
                    visuals = [self.load_video(visuals, self.max_frames_num)]
                except Exception as e:
                    res.append("")
                    eval_logger.info(f"Error {e} when loading video: {visuals}")
                    pbar.update(1)
                    continue

            # Process inputs through AutoProcessor
            if task_type == "image":
                inputs = self._image_processor(images=visuals, text=text, return_tensors="pt").to(
                    self._device, self.model.dtype
                )
            elif task_type == "video":
                inputs = self._image_processor(videos=visuals, text=text, return_tensors="pt").to(
                    self._device, self.model.dtype
                )
            else:
                inputs = self._image_processor(text=text, return_tensors="pt").to(
                    self._device, self.model.dtype
                )

            # Generation parameters
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            do_sample = gen_kwargs["temperature"] > 0

            try:
                cont = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=gen_kwargs["temperature"] if do_sample else None,
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    pad_token_id=self.eot_token_id,
                    eos_token_id=self.eot_token_id,
                )
                cont = cont[:, inputs["input_ids"].shape[-1] :]
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""

            if isinstance(cont, str):
                text_outputs = cont
            else:
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        """Multi-round generation (not implemented).

        Args:
            requests: List of evaluation instances

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError("Multi-round generation not implemented for LLaVA-OV")


def register_model_with_lmms_eval() -> None:
    """Register llava_ov_hf model with lmms-eval's model registry.

    This ensures lmms-eval can discover our model when --model llava_ov_hf is passed.
    The @register_model decorator above handles the registry registration, but we also
    need to add it to AVAILABLE_SIMPLE_MODELS so get_model() can find it.
    """
    from lmms_eval.models import AVAILABLE_SIMPLE_MODELS

    AVAILABLE_SIMPLE_MODELS["llava_ov_hf"] = (
        f"{__name__}.LlavaOVHF" if __name__ != "__main__" else "__main__.LlavaOVHF"
    )


if __name__ == "__main__":
    register_model_with_lmms_eval()
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()
