from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image


class BaseModel(ABC):
    """Interface that every model wrapper must implement.

    Lifecycle:
        model = SomeModel(model_dir="/workspace/models", device="cuda")
        model.load()                    # heavy weights load happens here
        out = model.generate(image, prompt)
        # ... later ...
        model.train(dataset, config)    # placeholder for fine-tuning
    """

    def __init__(self, model_dir: str | Path, device: str = "cuda"):
        self.model_dir = Path(model_dir)
        self.device = device
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Must set self._loaded = True on success."""
        ...

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        seed: int = 0,
        **kwargs,
    ) -> Image.Image:
        """Run a single edit and return the result."""
        ...

    def ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} not loaded. Call .load() first."
            )

    # ----- training stubs (to be filled per-model later) -----

    def train(self, dataset, config: dict | None = None):
        raise NotImplementedError(
            f"Training not yet implemented for {self.__class__.__name__}"
        )

    def load_checkpoint(self, path: str | Path):
        raise NotImplementedError(
            f"Checkpoint loading not yet implemented for {self.__class__.__name__}"
        )