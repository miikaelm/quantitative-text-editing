"""Qwen-Image (diffusion-based) wrapper.

Wraps the DiffSynth QwenImagePipeline so the rest of the codebase
never touches model-specific loading or inference details.
"""

import os
from pathlib import Path

import torch
from PIL import Image

from .base import BaseModel


class QwenImageModel(BaseModel):
    """Qwen-Image-Edit-2511 via DiffSynth."""

    # Default generation params — can be overridden per-call via kwargs
    DEFAULT_STEPS = 40
    DEFAULT_HEIGHT = 1024
    DEFAULT_WIDTH = 1024

    def __init__(
        self,
        model_dir: str | Path = "/workspace/models",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(model_dir, device)
        self.dtype = dtype
        self.pipe = None

    def load(self) -> None:
        # Point every cache env var at our persistent dir *before* any
        # modelscope / huggingface code resolves paths.
        cache = str(self.model_dir)
        for var in [
            "MODELSCOPE_CACHE",
            "MS_CACHE_HOME",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HUB_CACHE",
        ]:
            os.environ[var] = cache

        from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=self.dtype,
            device=self.device,
            model_configs=[
                ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2511",
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="text_encoder/model*.safetensors",
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                ),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(
                model_id="Qwen/Qwen-Image-Edit",
                origin_file_pattern="processor/",
            ),
        )
        self._loaded = True

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        seed: int = 0,
        **kwargs,
    ) -> Image.Image:
        self.ensure_loaded()

        result = self.pipe(
            prompt=prompt,
            edit_image=[image],
            seed=seed,
            num_inference_steps=kwargs.get("num_inference_steps", self.DEFAULT_STEPS),
            height=kwargs.get("height", self.DEFAULT_HEIGHT),
            width=kwargs.get("width", self.DEFAULT_WIDTH),
            edit_image_auto_resize=kwargs.get("edit_image_auto_resize", True),
            zero_cond_t=kwargs.get("zero_cond_t", True),
        )
        return result