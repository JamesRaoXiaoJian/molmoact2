#!/usr/bin/env python3
"""Compare real pretrained backbone features with the local LeRobot reference."""

import argparse
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from olmo.nn.tactile_config import DEFAULT_ENCODER_PATHS, TactileConfig
from olmo.nn.tactile_encoder import TactileEncoder, _CLIPBackbone, read_checkpoint
from olmo.preprocessing.tactile_preprocessor import preprocess_tactile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=tuple(DEFAULT_ENCODER_PATHS))
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--reference-root", type=Path, default=Path("/root/lerobot-v0.5.1")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = TactileConfig(
        backend=args.backend,
        encoder_path=str(args.weights or DEFAULT_ENCODER_PATHS[args.backend]),
    )
    ours = TactileEncoder(cfg, 16).eval()
    report = ours.initialize_pretrained()
    raw = np.random.default_rng(0).integers(
        0, 256, (1, 2, cfg.frames, 128, 192, 3), dtype=np.uint8
    )
    pixels = torch.from_numpy(preprocess_tactile(raw, cfg)).reshape(
        2, cfg.frames, 3, 224, 224
    )
    policies = args.reference_root / "src/lerobot/policies"
    if args.backend == "tactile_mae":
        package = types.ModuleType("reference_tactile_mae")
        package.__path__ = [str(policies / "starvla_groot/tactile_mae")]
        sys.modules[package.__name__] = package
        build = importlib.import_module("reference_tactile_mae.models.build")
        inference = importlib.import_module("reference_tactile_mae.inference")
        model = build.build_model(
            arch="vit_l", mask_ratio=0.0, use_sensor_token=True, use_same_patchemb=True
        )
        state = read_checkpoint(cfg.encoder_path)
        model.load_state_dict(
            {k: v for k, v in state.items() if k in model.state_dict()}, strict=False
        )
        reference = inference.TactileMAEFeatureExtractor(
            model, freeze=False, num_query_tokens=cfg.num_tokens, dtype=None
        ).eval()
        with torch.no_grad():
            reference.query_tokens.copy_(ours.queries[0])
            actual = ours.backbone(pixels, ours.queries)
            ref_raw = (
                torch.from_numpy(raw[:, :, 0].reshape(2, 128, 192, 3))
                .permute(0, 3, 1, 2)
                .float()
                / 255
            )
            expected = reference(ref_raw)
    else:
        import transformers.models.clip.modeling_clip as clip

        # The reference targets Transformers 4.57. In 5.x the equivalent CLIP
        # container was removed; use its same stable submodules for construction.
        if not hasattr(clip, "CLIPVisionTransformer"):
            clip.CLIPVisionTransformer = _CLIPBackbone
        spec = importlib.util.spec_from_file_location(
            "reference_tactile_backbones", policies / "pi05_tactile/backbones.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = (
            module.AnyTouch2Backbone()
            if args.backend == "anytouch2"
            else module.SparshBackbone(args.backend)
        )
        reference.load_pretrained(cfg.encoder_path)
        reference.eval()
        with torch.no_grad():
            actual = ours.backbone(pixels)
            expected = (
                reference(pixels, cfg.sensor_id)
                if args.backend == "anytouch2"
                else reference(pixels)
            )
    error = float((actual - expected).abs().max())
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    print(
        json.dumps(
            {
                "backend": args.backend,
                **report,
                "feature_shape": list(actual.shape),
                "float32_max_absolute_error": error,
                "reference_match": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
