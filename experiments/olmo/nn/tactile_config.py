from dataclasses import dataclass

from olmo.config import BaseConfig
from olmo.data.tactile import tactile_keys_for_layout

BACKENDS = ("as_image", "anytouch2", "sparsh_vjepa", "tactile_mae")
DEFAULT_ENCODER_PATHS = {
    "anytouch2": "/root/RXJ/pretrained_models/huggingface/AnyTouch2-Model",
    "sparsh_vjepa": "/root/RXJ/pretrained_models/huggingface/Sparsh-VJEPA-Small",
    "tactile_mae": "/root/RXJ/pretrained_models/huggingface/AnyTouch-ViT-L-16",
}


@dataclass
class TactileConfig(BaseConfig):
    backend: str = "as_image"
    num_tokens: int = 32
    history_stride: int = 2
    sensor_id: int = -1
    layout: str = "two"
    freeze_encoder: bool = False
    encoder_path: str | None = None

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown tactile backend {self.backend!r}.")
        tactile_keys_for_layout(self.layout)
        if self.num_tokens < 1 or self.history_stride < 1:
            raise ValueError("Tactile num_tokens and history_stride must be positive.")
        if not -10 <= self.sensor_id < 10:
            raise ValueError("Tactile sensor_id must lie in [-10,10).")

    @property
    def independent(self):
        return self.backend != "as_image"

    @property
    def frames(self):
        return 4 if self.backend in ("anytouch2", "sparsh_vjepa") else 1

    @property
    def sensor_indices(self):
        return (0, 3) if self.layout == "two" else (0, 1, 2, 3)

    @property
    def keys(self):
        return tactile_keys_for_layout(self.layout)

    @property
    def mean_std(self):
        return {
            "anytouch2": (
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
            "sparsh_vjepa": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "tactile_mae": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        }[self.backend]
