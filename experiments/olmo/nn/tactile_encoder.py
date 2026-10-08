"""Encoder-only PyTorch paths matching openpi-arx and its LeRobot references.

AnyTouch2: GeWu-Lab/AnyTouch2 (MIT). Sparsh: facebookresearch/sparsh
(CC-BY-NC-4.0, with Apache-2.0 layer notices). See tactile_licenses/.
Reconstruction heads are excluded; every used backbone tensor loads strictly.
"""

import argparse
import math
from pathlib import Path

import torch
from olmo.nn.tactile_config import TactileConfig
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPVisionEmbeddings


def backbone_spec(backend):
    return {
        "anytouch2": (768, 12, 12, 16),
        "sparsh_vjepa": (384, 12, 6, 16),
        "tactile_mae": (1024, 24, 16, 14),
    }[backend]


def read_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        path = next(
            (
                path / n
                for n in (
                    "model.safetensors",
                    "checkpoint-4frames.pth",
                    "vjepa_vitsmall.safetensors",
                    "checkpoint.pth",
                )
                if (path / n).is_file()
            ),
            path,
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        state = load_file(str(path))
    else:
        with torch.serialization.safe_globals([argparse.Namespace]):
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        state = payload.get("model", payload.get("state_dict", payload))
    state = {
        k.removeprefix("module."): v
        for k, v in state.items()
        if isinstance(v, torch.Tensor)
    }
    for prefix in ("touch_mae_model.", "tactile_model."):
        if prefix + "sensor_token" in state:
            return {
                k.removeprefix(prefix): v
                for k, v in state.items()
                if k.startswith(prefix)
            }
    return state


class _Scale(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * self.gamma


class _Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(b, n, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        return self.proj(
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        )


class _Mlp(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, width * 4), nn.Linear(width * 4, width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _SparshBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2 = (
            nn.LayerNorm(width, eps=1e-6),
            nn.LayerNorm(width, eps=1e-6),
        )
        self.attn, self.mlp = _Attention(width, heads), _Mlp(width)
        self.ls1, self.ls2 = _Scale(width), _Scale(width)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class _SinePosition(nn.Module):
    def __init__(self, width):
        super().__init__()
        bands = math.ceil(width / 6)
        self.register_buffer(
            "frequency_bands",
            (10000 ** -torch.linspace(0, 1, bands + 1)[:-1]).repeat(3, 1),
        )
        self.width = width

    def forward(self, grid, device):
        axes = [torch.arange(n, device=device, dtype=torch.float32) for n in grid]
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        phase = coords[..., None] * self.frequency_bands.float()
        return torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)[..., : self.width]


class _CLIPBackbone(nn.Module):
    """The stable CLIP encoder submodules, with the AnyTouch checkpoint names."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )


class TactileBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width, depth, heads, patch = backbone_spec(config.backend)
        self.width = width
        if config.backend == "sparsh_vjepa":
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv3d(
                3, width, (2, patch, patch), (2, patch, patch)
            )
            self.register_tokens = nn.Parameter(torch.zeros(1, 1, width))
            self.pos_embed = _SinePosition(width)
            self.blocks = nn.ModuleList(
                [_SparshBlock(width, heads) for _ in range(depth)]
            )
            self.norm = nn.LayerNorm(width, eps=1e-6)
        else:
            cfg = CLIPVisionConfig(
                hidden_size=width,
                intermediate_size=width * 4,
                num_hidden_layers=depth,
                num_attention_heads=heads,
                image_size=224,
                patch_size=patch,
                hidden_act="gelu",
                layer_norm_eps=1e-5,
            )
            cfg._attn_implementation = "eager"
            self.touch_model = _CLIPBackbone(cfg)
            emb = self.touch_model.embeddings
            if config.backend == "anytouch2":
                emb.patch_embedding = nn.Conv3d(
                    3, width, (2, patch, patch), (2, patch, patch), bias=False
                )
                emb.position_embedding = nn.Embedding(
                    2 * (224 // patch) ** 2 + 1, width
                )
                self.sensor_token = nn.Parameter(torch.zeros(20, 5, width))
                self.touch_model.post_layernorm.requires_grad_(False)
            else:
                self.video_patch_embedding = nn.Conv3d(
                    3, width, (3, patch, patch), (3, patch, patch), bias=False
                )
                self.sensor_token = nn.Parameter(torch.zeros(10, 5, width))
                emb.patch_embedding.requires_grad_(False)

    def load_pretrained(self, path):
        state = read_checkpoint(path)
        required = self.state_dict()
        missing = set(required) - set(state)
        wrong = [
            k for k in set(required) & set(state) if required[k].shape != state[k].shape
        ]
        if missing or wrong:
            raise ValueError(
                f"Tactile encoder mismatch: missing={sorted(missing)[:8]}, shape={wrong[:8]}"
            )
        self.load_state_dict({k: state[k] for k in required}, strict=True)
        return {
            "loaded_tensors": len(required),
            "excluded_auxiliary_tensors": len(set(state) - set(required)),
        }

    def forward(self, images, queries=None):
        if self.config.backend == "sparsh_vjepa":
            patches = self.patch_embed.proj(images.flip(1).transpose(1, 2))
            x = patches.flatten(2).transpose(1, 2) + self.pos_embed(
                patches.shape[2:], patches.device
            ).to(dtype=patches.dtype)
            x = torch.cat((self.register_tokens.expand(len(x), -1, -1), x), 1)
            for block in self.blocks:
                x = (
                    checkpoint(block, x, use_reentrant=False)
                    if self.training
                    else block(x)
                )
            return self.norm(x)[:, 1:]
        model = self.touch_model
        emb = model.embeddings
        if self.config.backend == "tactile_mae":
            # Match AnyTouch1's single-image Conv3d channel/time path exactly.
            patches = self.video_patch_embedding(
                images[:, 0].unsqueeze(1).repeat(1, 3, 1, 1, 1)
            )
        else:
            patches = emb.patch_embedding(images.transpose(1, 2))
        patches = patches.flatten(2).transpose(1, 2)
        position = emb.position_embedding.weight
        cls = (emb.class_embedding + position[0]).expand(len(images), 1, -1)
        sensors = self.sensor_token[self.config.sensor_id].expand(len(images), -1, -1)
        parts = [cls, sensors]
        if queries is not None:
            parts.append(queries.expand(len(images), -1, -1))
        parts.append(patches + position[None, 1:])
        x = model.pre_layrnorm(torch.cat(parts, 1))
        for layer in model.encoder.layers:

            def run(value, layer=layer):
                result = layer(value, attention_mask=None)
                return result[0] if isinstance(result, tuple) else result

            x = checkpoint(run, x, use_reentrant=False) if self.training else run(x)
        return (
            model.post_layernorm(x[:, 6 : 6 + self.config.num_tokens])
            if queries is not None
            else x
        )


class TactileEncoder(nn.Module):
    def __init__(self, config: TactileConfig, output_dim: int):
        super().__init__()
        self.config = config
        self.backbone = TactileBackbone(config)
        width = self.backbone.width
        if config.backend == "tactile_mae":
            self.queries = nn.Parameter(torch.randn(1, config.num_tokens, width) * 0.02)
        else:
            self.queries = nn.Parameter(torch.randn(1, config.num_tokens, width) * 0.02)
            self.pool_norm = nn.LayerNorm(width)
            self.pool_attention = nn.MultiheadAttention(
                width, max(1, width // 64), batch_first=True
            )
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, output_dim)
        self.sensor_embedding = nn.Parameter(torch.zeros(4, 1, output_dim))
        if config.freeze_encoder:
            self.backbone.requires_grad_(False)

    def initialize_pretrained(self):
        if not self.config.encoder_path:
            raise ValueError(
                "Initial independent tactile training requires tactile.encoder_path."
            )
        report = self.backbone.load_pretrained(self.config.encoder_path)
        if self.config.backend == "tactile_mae":
            with torch.no_grad():
                cls = self.backbone.touch_model.embeddings.class_embedding
                self.queries.copy_(
                    cls[None, None] + torch.randn_like(self.queries) * 0.02
                )
        return report

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_encoder:
            self.backbone.eval()
        return self

    def forward(self, images):
        # [B,T,S,F,C,H,W] -> [B,T*S*N,D], with physical sensor identities [0,3].
        b, t, s, f, c, h, w = images.shape
        if (s, f, c, h, w) != (len(self.config.keys), self.config.frames, 3, 224, 224):
            raise ValueError(f"Invalid tactile image shape {images.shape}")
        dtype = self.projection.weight.dtype
        flat = images.reshape(b * t * s, f, c, h, w).to(dtype=dtype)
        if self.config.backend == "tactile_mae":
            features = self.backbone(flat, self.queries)
        else:
            raw = self.pool_norm(self.backbone(flat))
            q = self.queries.expand(len(raw), -1, -1)
            features = q + self.pool_attention(q, raw, raw, need_weights=False)[0]
        features = self.projection(self.norm(features)).reshape(
            b, t, s, self.config.num_tokens, -1
        )
        identities = self.sensor_embedding[list(self.config.sensor_indices)]
        return (features + identities[None, None]).flatten(1, 3)

    def apply_fsdp2(self, **kwargs):
        from torch.distributed.fsdp import fully_shard

        layers = (
            self.backbone.blocks
            if self.config.backend == "sparsh_vjepa"
            else self.backbone.touch_model.encoder.layers
        )
        for layer in layers:
            fully_shard(layer, **kwargs)
        fully_shard(self, **kwargs)
