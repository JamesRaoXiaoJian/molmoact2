"""Shared CPU normalization and prefix-token assembly for train/serve."""

import numpy as np
import torch
from torch.nn import functional as F

from olmo.nn.tactile_config import TactileConfig
from olmo.preprocessing.preprocessor_utils import TensorSpec


def preprocess_tactile(images, config: TactileConfig):
    # Raw [T,S,F,H,W,3] uint8, oldest-to-current for both temporal axes.
    images = np.asarray(images)
    if (
        images.ndim != 6
        or images.shape[1:3] != (len(config.keys), config.frames)
        or images.shape[-1] != 3
    ):
        raise ValueError(f"Invalid raw tactile windows {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError("Raw tactile windows must be uint8 RGB.")
    leading = images.shape[:3]
    x = (
        torch.from_numpy(np.array(images, copy=True))
        .reshape(-1, *images.shape[-3:])
        .permute(0, 3, 1, 2)
        .float()
        / 255
    )
    h, w = x.shape[-2:]
    if config.backend == "sparsh_vjepa" and h < w:
        x = torch.rot90(x, -1, (-2, -1))
    elif config.backend == "tactile_mae":
        side = max(h, w)
        x = F.pad(x, (0, side - w, 0, side - h))
    x = F.interpolate(
        x, (224, 224), mode="bilinear", align_corners=False, antialias=True
    )
    mean, std = config.mean_std
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    return x.numpy().reshape(*leading, 3, 224, 224)


class TactileExamplePreprocessor:
    def __init__(self, base, config: TactileConfig, n_obs_steps):
        self.base, self.config, self.n_obs_steps = base, config, n_obs_steps

    def __getattr__(self, name):
        # Unpickling probes special methods before restoring instance state.
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    @property
    def prefix_length(self):
        return self.n_obs_steps * len(self.config.keys) * self.config.num_tokens

    def get_output_shapes(self):
        shapes = dict(self.base.get_output_shapes())
        shapes["tactile_images"] = TensorSpec(
            (self.n_obs_steps, len(self.config.keys), self.config.frames, 3, 224, 224),
            np.float32,
        )
        shapes["tactile_token_mask"] = TensorSpec(shapes["tokens"].shape, np.bool_)
        return shapes

    def __call__(self, example, rng=np.random, **kwargs):
        if example.get("tactile_windows") is None:
            raise ValueError(
                "Independent tactile checkpoints require real tactile_windows."
            )
        pixels = preprocess_tactile(example["tactile_windows"], self.config)
        if pixels.shape[0] != self.n_obs_steps:
            raise ValueError("Tactile outer history must match checkpoint n_obs_steps.")
        out = self.base(example, rng=rng, **kwargs)
        length = self.prefix_length
        for key in (
            "input_tokens",
            "target_tokens",
            "loss_masks",
            "subsegment_ids",
            "position_ids",
        ):
            if key not in out:
                continue
            original = out[key]
            if key == "position_ids":
                out[key] = np.arange(len(original) + length, dtype=original.dtype)
            else:
                fill = original[0] if key == "subsegment_ids" else 0
                out[key] = np.concatenate(
                    (
                        original[:1],
                        np.full(length, fill, dtype=original.dtype),
                        original[1:],
                    )
                )
        mask = np.zeros(len(out["input_tokens"]), dtype=np.bool_)
        mask[1 : 1 + length] = True
        limit = self.base.preprocessor.text_preprocessor.max_sequence_length
        if limit is not None and len(mask) > limit:
            raise ValueError(
                "RGB/text plus tactile prefix exceeds max_sequence_length; increase seq_len."
            )
        out["tactile_token_mask"] = mask
        out["tactile_images"] = pixels
        return out
