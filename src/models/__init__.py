from .base import BaseModel
from .qwen_image import QwenImageModel
from .varedit import VAREditModel

MODEL_REGISTRY = {
    "qwen": QwenImageModel,
    "varedit": VAREditModel,
}


def get_model(name: str) -> type[BaseModel]:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]