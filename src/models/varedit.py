"""VAREdit (Visual Autoregressive) wrapper.

Wraps HiDream-ai/VAREdit so the rest of the codebase never touches
model-specific loading or inference details.
"""

import os
import sys
from pathlib import Path

from PIL import Image

from .base import BaseModel


class VAREditModel(BaseModel):
    """VAREdit 8B-1024 via HiDream-ai inference code."""

    DEFAULT_CFG = 3.0
    DEFAULT_TAU = 0.1

    def __init__(
        self,
        model_dir: str | Path = "/workspace/models",
        device: str = "cuda",
        repo_dir: str | Path = "/workspace/VAREdit",
        model_size: str = "8B",
        image_size: int = 1024,
    ):
        super().__init__(model_dir, device)
        self.repo_dir = Path(repo_dir)
        self.model_size = model_size
        self.image_size = image_size
        self._model_components = None

    def load(self) -> None:
        os.environ["HF_HOME"] = str(self.model_dir)

        # VAREdit's own code expects to be run from its repo root
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        from huggingface_hub import snapshot_download

        local_dir = self.repo_dir / "HiDream-ai" / "VAREdit"
        snapshot_download(
            "HiDream-ai/VAREdit",
            cache_dir=str(self.model_dir),
            local_dir=str(local_dir),
        )

        from infer import load_model

        self._model_components = load_model(
            pretrain_root="HiDream-ai/VAREdit",
            model_path=str(local_dir / f"{self.model_size}-{self.image_size}.pth"),
            model_size=self.model_size,
            image_size=self.image_size,
        )
        self._loaded = True

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        seed: int = 42,
        **kwargs,
    ) -> Image.Image:
        self.ensure_loaded()

        # VAREdit's generate_image expects a file path, so we save to a tmp
        # file if given a PIL image. If this becomes a bottleneck, we can
        # patch their code to accept PIL directly.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
            image.save(tmp_path)

        from infer import generate_image

        result = generate_image(
            self._model_components,
            src_img_path=tmp_path,
            instruction=prompt,
            cfg=kwargs.get("cfg", self.DEFAULT_CFG),
            tau=kwargs.get("tau", self.DEFAULT_TAU),
            seed=seed,
        )

        # Clean up
        Path(tmp_path).unlink(missing_ok=True)
        return result