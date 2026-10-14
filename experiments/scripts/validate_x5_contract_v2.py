#!/usr/bin/env python3
"""CPU real-data acceptance for X5 v2, including train/native token parity."""

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/lerobot/src")]

import numpy as np
import torch
from olmo.models.model_config import BaseModelConfig
from olmo.data.robot_processing import RobotProcessorConfig
from olmo.data.lerobot_wrapper import build_lerobot_dataset
from olmo.extra_tokens import discretize_normalized_state
from lerobot.policies.molmoact2.modeling_molmoact2 import _build_example
from train_x5_contract_v2 import register_recipe, REPO
from launch_scripts.data_mixtures import MOLMOACT2_LEROBOT_MIXTURES


def validate(cohort, output):
    stats = json.loads((cohort / "dataset/meta/stats.json").read_text())
    info = json.loads((cohort / "dataset/meta/info.json").read_text())
    report = {}
    with tempfile.TemporaryDirectory(prefix="x5-v2-preflight-") as temp:
        alias = Path(temp) / REPO
        alias.parent.mkdir(parents=True)
        alias.symlink_to((cohort / "dataset").resolve())
        for mode in ("rgb", "mae"):
            name, tag, splits = register_recipe(mode, cohort / "splits.json")
            _, registered = MOLMOACT2_LEROBOT_MIXTURES[name]()
            meta = copy.deepcopy(next(iter(registered.values())))
            feature_stats = {
                key: {
                    **stats[key],
                    "names": info["features"][key]["names"],
                    "mask": [True] * 14,
                }
                for key in ("observation.state", "action")
            }
            pcfg = RobotProcessorConfig.from_stats(
                {tag: feature_stats},
                tag_metadata={tag: meta},
                repo_to_tag={REPO: tag},
                norm_mode="min_max",
            )
            proc = pcfg.build_processor()
            cfg = BaseModelConfig.load(
                ROOT
                / f"outputs/runs/molmoact2-x5-cleaned-{mode}-fft20k-b64-s42-20261008/checkpoints/step20000-unsharded/config.yaml",
                key="model",
            )
            cfg.robot_processor = pcfg
            pp = cfg.build_preprocessor(
                for_inference=False, is_training=False, max_seq_len=16384
            )
            infer_pp = cfg.build_preprocessor(
                for_inference=True, is_training=False, max_seq_len=16384
            )
            env = {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "LEROBOT_DATA_ROOT": temp,
                "LEROBOT_PACKED_IMAGE_ROOT": str((cohort / "packed_cache").resolve()),
                "LEROBOT_REPO_TO_TAG": json.dumps({REPO: tag}),
                "LEROBOT_TAG_METADATA": json.dumps({tag: meta}),
                "LEROBOT_STATS_BY_TAG": json.dumps({tag: feature_stats}),
                "LEROBOT_NORM_MODE": "min_max",
                "LEROBOT_ACTION_FORMAT": "continuous",
                "LEROBOT_STATE_FORMAT": "discrete",
                "LEROBOT_STYLE_SAMPLING_RATES": json.dumps({"robot_action": 1.0}),
                "LEROBOT_N_OBS_STEPS": "1",
                "LEROBOT_MAX_ACTION_HORIZON": "30",
                "LEROBOT_MAX_ACTION_DIM": "32",
                "LEROBOT_IMAGE_RESIZE": "None",
                "LEROBOT_RANDOM_CAMERA_ORDER": "none",
                "LEROBOT_USE_ANNOTATED_TASK": "0",
                "LEROBOT_SAMPLE_ANNOTATED_TASK": "0",
                "LEROBOT_ENABLE_DEPTH_REASONING": "0",
                "LEROBOT_ADD_DEPTH_TOKENS": "0",
                "LEROBOT_ADD_SETUP_TOKENS": "1",
                "LEROBOT_ADD_CONTROL_TOKENS": "1",
                "LEROBOT_VIDEO_BACKEND": "pyav",
            }
            os.environ.update(env)
            for key in (
                "LEROBOT_REPO_TO_TAG",
                "LEROBOT_TAG_METADATA",
                "LEROBOT_STATS_BY_TAG",
            ):
                os.environ.pop(key + "_PATH", None)
            rows = []
            for split, ids in splits.items():
                ds = build_lerobot_dataset(
                    "lerobot:" + REPO + "@" + ",".join(map(str, ids)),
                    split="train" if split == "train" else "validation",
                )
                for i in (0, len(ds) - 1):
                    raw = ds.dataset[i]
                    item = ds.get(
                        i, np.random.default_rng(42), allow_random_retry=False
                    )
                    normalized = proc.normalize_state(raw["observation.state"], tag)
                    native = _build_example(
                        copy.deepcopy(item["image"]),
                        raw["task"],
                        normalized,
                        inference_action_mode="continuous",
                        state_format="discrete",
                        num_state_tokens=256,
                        style="robot_action",
                        setup_type=meta["setup_type"],
                        control_mode=meta["control_mode"],
                        add_setup_tokens=True,
                        add_control_tokens=True,
                    )
                    if "tactile_windows" in item:
                        native["tactile_windows"] = item["tactile_windows"].copy()
                    train_input = pp(copy.deepcopy(item), rng=np.random.RandomState(42))
                    native_input = infer_pp(native, rng=np.random.RandomState(42))
                    assert np.array_equal(
                        train_input["input_tokens"], native_input["input_tokens"]
                    )
                    assert np.array_equal(train_input["images"], native_input["images"])
                    assert np.array_equal(
                        item["action_horizon_is_pad"], np.asarray(raw["action_is_pad"])
                    )
                    action = np.asarray(raw["action"], np.float32)
                    np.testing.assert_allclose(
                        item["action"][:, :14],
                        proc.normalize_action(action, tag),
                        atol=1e-6,
                    )
                    np.testing.assert_allclose(
                        proc.unnormalize_action(item["action"][:, :14], tag),
                        action,
                        atol=1e-6,
                    )
                    rows.append(
                        {
                            "split": split,
                            "dataset_rows": len(ds),
                            "row": i,
                            "token_parity": True,
                            "image_parity": True,
                            "padding_count": int(item["action_horizon_is_pad"].sum()),
                            "action_roundtrip": True,
                        }
                    )
            probe = np.zeros((3, 14), np.float32)
            probe[:, 13] = [-2.2, -1.8, -1.2]
            bins = discretize_normalized_state(proc.normalize_state(probe, tag))[:, 13]
            assert len(set(bins.tolist())) == 3
            report[mode] = {"cases": rows, "right_gripper_probe_bins": bins.tolist()}
    assert not torch.cuda.is_initialized()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    validate(a.cohort, a.output)
