#!/usr/bin/env python3
"""Decode an actual X5 observation and optionally preprocess it, without weights."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_ROOT))

import numpy as np
import torch
from launch_scripts.data_mixtures import build_molmoact2_x5
from olmo.data.lerobot_wrapper import build_lerobot_dataset
from olmo.data.tactile import ensure_tactile_image_capacity
from olmo.nn.tactile_config import BACKENDS, TactileConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--layout", choices=("two", "four"), default="two")
    parser.add_argument("--backend", choices=BACKENDS, default="as_image")
    parser.add_argument("--n-obs-steps", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--checkpoint", help="Optional local HF base for native preprocessing checks."
    )
    args = parser.parse_args()
    if args.n_obs_steps < 1:
        parser.error("--n-obs-steps must be positive")

    mixture, raw_metadata = build_molmoact2_x5(args.layout)
    raw_tag, repo_ids, _ = mixture[0]
    tag = raw_tag.removeprefix("lerobot:")
    repo_id = repo_ids[0].removeprefix("lerobot:")
    metadata = raw_metadata[raw_tag]
    metadata.update(tactile_backend=args.backend, tactile_history_stride=2)
    stats = json.loads((args.dataset_root / "meta" / "stats.json").read_text())
    root_alias = tempfile.TemporaryDirectory(prefix="molmoact2-x5-smoke-")
    repo_path = Path(root_alias.name) / repo_id
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    repo_path.symlink_to(args.dataset_root.resolve(), target_is_directory=True)
    # Only this smoke process gets these settings; no training/global env changes.
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "LEROBOT_DATA_ROOT": root_alias.name,
            "LEROBOT_STATS_BY_TAG": json.dumps(
                {tag: {k: stats[k] for k in ("action", "observation.state")}}
            ),
            "LEROBOT_REPO_TO_TAG": json.dumps({repo_id: tag}),
            "LEROBOT_TAG_METADATA": json.dumps({tag: metadata}),
            "LEROBOT_STYLE_SAMPLING_RATES": json.dumps({"robot_action": 1.0}),
            "LEROBOT_N_OBS_STEPS": str(args.n_obs_steps),
            "LEROBOT_MAX_ACTION_HORIZON": "70",
            "LEROBOT_MAX_ACTION_DIM": "14",
            "LEROBOT_NORM_MODE": "mean_std",
            "LEROBOT_ACTION_FORMAT": "continuous",
            "LEROBOT_STATE_FORMAT": "continuous",
            "LEROBOT_RANDOM_CAMERA_ORDER": "none",
            "LEROBOT_VIDEO_BACKEND": "pyav",
            "LEROBOT_ENABLE_DEPTH_REASONING": "0",
            "LEROBOT_ADD_DEPTH_TOKENS": "0",
            "LEROBOT_IMAGE_RESIZE": "None",
        }
    )
    for name in ("LEROBOT_STATS_BY_TAG", "LEROBOT_REPO_TO_TAG", "LEROBOT_TAG_METADATA"):
        os.environ.pop(f"{name}_PATH", None)
    dataset = build_lerobot_dataset(repo_ids[0], split="train")
    example = dataset.get(
        args.index, np.random.default_rng(0), allow_random_retry=False
    )
    images = example["image"]
    independent = args.backend != "as_image"
    expected_count = (
        3 if independent else 3 + len(metadata["tactile_keys"])
    ) * args.n_obs_steps
    if not isinstance(images, list) or len(images) != expected_count:
        raise AssertionError(f"Expected {expected_count} images, got {len(images)}")
    result = {
        "layout": args.layout,
        "backend": args.backend,
        "dataset_rows": len(dataset),
        "index": args.index,
        "n_obs_steps": args.n_obs_steps,
        "image_keys": example["metadata"]["image_keys_used"],
        "image_shapes": [list(image.shape) for image in images],
        "image_augmentation_mask": example["image_augmentation_mask"],
        "state_shape": list(example["state"].shape),
        "action_shape": list(example["action"].shape),
    }
    if independent:
        result["tactile_window_shape"] = list(example["tactile_windows"].shape)
    if args.checkpoint:
        from launch_scripts.lerobot_utils.hf import (
            _apply_lerobot_molmoact2_defaults,
            get_hf_model_config,
        )

        cfg = get_hf_model_config(args.checkpoint, frame_loading_backend="av")
        _apply_lerobot_molmoact2_defaults(cfg)
        cfg.n_obs_steps = args.n_obs_steps
        cfg.tactile = TactileConfig(backend=args.backend, layout=args.layout)
        cfg.vision_backbone.use_image_augmentation = "photometric"
        ensure_tactile_image_capacity(cfg, {tag: metadata})
        train_pp = cfg.build_preprocessor(for_inference=True, is_training=True)
        infer_pp = cfg.build_preprocessor(for_inference=True, is_training=False)
        train = train_pp(example, rng=np.random.RandomState(3))
        infer = infer_pp(example, rng=np.random.RandomState(3))
        if independent:
            np.testing.assert_array_equal(
                train["tactile_images"], infer["tactile_images"]
            )
            result["normalized_tactile_shape"] = list(train["tactile_images"].shape)
            result["tactile_prefix_tokens"] = int(train["tactile_token_mask"].sum())
        else:
            first_tactile = 3 * args.n_obs_steps
            np.testing.assert_array_equal(
                train["images"][first_tactile:], infer["images"][first_tactile:]
            )
        if train["images"].shape[0] != expected_count:
            raise AssertionError("Preprocessor dropped an observation.")
        collator = cfg.build_collator(train_pp.get_output_shapes(), pad_mode=None)
        batch = collator([train])
        if independent:
            if batch["tactile_token_mask"].dtype != torch.bool:
                raise AssertionError("Collator changed tactile mask dtype.")
            if int(batch["tactile_token_mask"].sum()) != train_pp.prefix_length:
                raise AssertionError(
                    "Collator dropped or padded tactile tokens incorrectly."
                )
            result["batch_tactile_shape"] = list(batch["tactile_images"].shape)
        result["batch_input_shape"] = list(batch["input_ids"].shape)
        result.update(
            preprocessed_shape=list(train["images"].shape),
            tactile_train_inference_equal=True,
            max_images=cfg.mm_preprocessor.image.max_images,
        )
    print(json.dumps(result, indent=2))
    root_alias.cleanup()


if __name__ == "__main__":
    main()
