#!/usr/bin/env python3
"""Explicit X5 v2 recipe: native units, training-only ranges and validation."""

import json
import os
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENTS), str(EXPERIMENTS / "lerobot/src")]

from launch_scripts.data_mixtures import (
    MOLMOACT2_LEROBOT_MIXTURES,
    build_single_lerobot_mixture,
)
from olmo.data.tactile import X5_RGB_IMAGE_KEYS, tactile_keys_for_layout

REPO = "local/hot_stamp_bag_merged_0918-20_cleaned"
NAMES = [
    f"{arm}_{name}.pos"
    for arm in ("left", "right")
    for name in (*[f"joint_{i}" for i in range(1, 7)], "gripper")
]


def register_recipe(mode, manifest):
    if mode not in ("rgb", "mae"):
        raise ValueError("Expected rgb or mae")
    name = f"x5_contract_v2_{mode}"
    tag = f"arx_x5_cleaned_v2_{mode}"
    splits = json.loads(Path(manifest).read_text())["splits"]

    def build():
        mixture, metadata = build_single_lerobot_mixture(
            name=name,
            tag=tag,
            repo_ids=[REPO + "@" + ",".join(map(str, splits["train"]))],
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=list(X5_RGB_IMAGE_KEYS),
            normalize_gripper=True,
            action_dim=14,
            action_horizon=30,
            n_action_steps=5,
            setup_type="bimanual ARX X5 robotic arms",
            control_mode="absolute joint pose",
        )
        entry = next(iter(metadata.values()))
        entry.update(
            state_mask=[True] * 14,
            action_mask=[True] * 14,
            normalization={
                "version": 2,
                "state": {"mode": "min_max", "clip": True},
                "action": {"mode": "min_max", "clip": False},
            },
            input_contract={
                "version": "x5-native-v2",
                "state_names": NAMES,
                "action_names": NAMES,
                "joint_units": "rad",
                "gripper_units": "calibrated native motor angle rad",
                "rgb_geometry": "native -> checkpoint SigLIP resize",
                "task_normalization": "robot_text_v1",
                "state_bins": 256,
                "action_representation": "absolute joint pose",
                "episode_padding": "masked",
            },
            training_episode_ids=splits["train"],
            validation_episode_ids=splits["val"],
            test_episode_ids=splits["test"],
        )
        if mode == "mae":
            entry.update(
                tactile_keys=tactile_keys_for_layout("two"),
                tactile_layout="two",
                tactile_backend="tactile_mae",
            )
        return mixture, metadata

    MOLMOACT2_LEROBOT_MIXTURES[name] = build
    return name, tag, splits


def main():
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: train_x5_contract_v2.py rgb|mae CHECKPOINT SPLITS_JSON [trainer flags]"
        )
    mode, checkpoint, manifest = sys.argv[1:4]
    name, tag, splits = register_recipe(mode, manifest)
    from launch_scripts import train_lerobot as launcher
    from olmo.data.data_loader import DataLoaderConfig
    from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig

    run_trainer = launcher.run_trainer

    def run_with_contract(cfg):
        meta = cfg.model.robot_processor.metadata_by_tag[tag]
        if meta.get("input_contract", {}).get("version") != "x5-native-v2":
            raise ValueError("Missing v2 contract in final model config")
        if (
            cfg.model.state_format != "discrete"
            or cfg.model.action_format != "continuous"
        ):
            raise ValueError(
                "X5 v2 recipe uses discrete state and continuous native absolute actions"
            )
        # A true held-out partition only for initialization that did not see our old 337 episodes.
        warmstart = os.environ.get("X5_INITIALIZATION") == "warmstart"
        label = "regression_seen_val" if warmstart else "validation"
        cfg.evaluators = [
            LossDatasetEvaluatorConfig(
                label=label,
                device_batch_size=2,
                max_examples=256,
                response_logits_only=True,
                reduce_loss_metrics_manually=True,
                data=DataLoaderConfig(
                    dataset="lerobot:" + REPO + "@" + ",".join(map(str, splits["val"])),
                    split="validation",
                    shuffle=True,
                    seed=20261014,
                    sequence_length=cfg.data.sequence_length,
                    num_workers=0,
                    drop_last=True,
                    pad=None,
                    pin_memory=True,
                ),
            )
        ]
        cfg.eval_interval = 1 if int(cfg.max_duration) == 2 else 1000
        run_trainer(cfg)

    launcher.run_trainer = run_with_contract
    sys.argv = [sys.argv[0], checkpoint, name, *sys.argv[4:]]
    launcher.main()


if __name__ == "__main__":
    main()
