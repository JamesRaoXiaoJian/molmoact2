#!/usr/bin/env python3
"""Register the matched RGB / tactile-MAE recipes, then run the native trainer."""

import os
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "lerobot" / "src"))

from launch_scripts.data_mixtures import (
    MOLMOACT2_LEROBOT_MIXTURES,
    build_single_lerobot_mixture,
)
from olmo.data.tactile import X5_RGB_IMAGE_KEYS, tactile_keys_for_layout


def register_recipe(mode):
    if mode not in ("rgb", "mae"):
        raise ValueError("Comparison mode must be rgb or mae.")
    name = f"x5_cleaned_{mode}"

    def build():
        mixture, metadata = build_single_lerobot_mixture(
            name=name,
            tag=f"arx_x5_cleaned_{mode}",
            repo_ids=["local/hot_stamp_bag_merged_0918-20_cleaned"],
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=list(X5_RGB_IMAGE_KEYS),
            normalize_gripper=False,
            action_dim=14,
            action_horizon=30,
            n_action_steps=5,
            setup_type="bimanual ARX X5 robotic arms",
            control_mode="absolute joint pose",
        )
        if mode == "mae":
            next(iter(metadata.values())).update(
                tactile_keys=tactile_keys_for_layout("two"),
                tactile_layout="two",
                tactile_backend="tactile_mae",
            )
        return mixture, metadata

    MOLMOACT2_LEROBOT_MIXTURES[name] = build
    return name


def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: train_x5_cleaned_comparison.py rgb|mae CHECKPOINT [trainer flags]"
        )
    mode, checkpoint = sys.argv[1:3]
    os.environ["MOLMOACT2_X5_REPO_ID"] = "local/hot_stamp_bag_merged_0918-20_cleaned"
    name = register_recipe(mode)
    sys.argv = [sys.argv[0], checkpoint, name, *sys.argv[3:]]
    from launch_scripts.train_lerobot import main as train

    train()


if __name__ == "__main__":
    main()
