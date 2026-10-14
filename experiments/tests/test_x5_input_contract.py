"""Regressions for actual X5 representation failures, without hardware/model weights."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from olmo.data.robot_processing import RobotProcessorConfig
from olmo.data.robot_text import normalize_robot_task
from olmo.extra_tokens import discretize_normalized_state
from scripts.x5_openpi_control import build_molmo_observation


def config(v2=True):
    stats = {
        "min": [-3.5, -2.5],
        "max": [0.0, 0.0],
        "q01": [-3.4, -2.4],
        "q99": [-0.1, -0.1],
        "mask": [True, True],
    }
    meta = {"state_stats": copy.deepcopy(stats), "action_stats": copy.deepcopy(stats)}
    if v2:
        meta["normalization"] = {
            "version": 2,
            "state": {"mode": "min_max", "clip": True},
            "action": {"mode": "min_max", "clip": False},
        }
    else:
        meta["state_stats"]["mask"] = meta["action_stats"]["mask"] = [False, False]
    return RobotProcessorConfig(metadata_by_tag={"x5": meta}, norm_mode="q01_q99")


def test_gripper_states_are_distinguishable_and_action_units_roundtrip():
    raw = np.asarray([[-3.2, -2.2], [-2.0, -1.8], [-1.4, -1.2]], np.float32)
    old = config(False).build_processor()
    new = config().build_processor()
    np.testing.assert_array_equal(
        discretize_normalized_state(old.normalize_state(raw, "x5")), 0
    )
    bins = discretize_normalized_state(new.normalize_state(raw, "x5"))
    assert len(np.unique(bins[:, 0])) == len(np.unique(bins[:, 1])) == 3
    # Continuous targets can extrapolate beyond statistical bounds. Stats are not hardware limits.
    actions = np.asarray([[-3.8, 0.1], [0.0, -2.7]], np.float32)
    np.testing.assert_allclose(
        new.unnormalize_action(new.normalize_action(actions, "x5"), "x5"),
        actions,
        atol=1e-6,
    )
    np.testing.assert_allclose(old.unnormalize_action(actions, "x5"), actions, atol=0)


def test_legacy_quantile_checkpoint_retains_clipping_and_new_contract_serializes():
    cfg = config(False)
    for feature in ("state", "action"):
        cfg.metadata_by_tag["x5"][feature + "_stats"]["mask"] = [True, True]
    old = cfg.build_processor()
    np.testing.assert_allclose(
        old.unnormalize_action(np.array([[2.0, -2.0]]), "x5"), [[-0.1, -2.4]], atol=1e-6
    )
    cfg = config()
    loaded = RobotProcessorConfig(
        **json.loads(
            json.dumps(
                {"metadata_by_tag": cfg.metadata_by_tag, "norm_mode": cfg.norm_mode}
            )
        )
    )
    assert loaded.build_processor().action_normalizers["x5"].clip is False
    for bad in [
        {"version": 3, "state": {}, "action": {}},
        {"version": 2},
        {
            "version": 2,
            "state": {"mode": "min_max", "clip": "false"},
            "action": {"mode": "none", "clip": False},
        },
    ]:
        loaded.metadata_by_tag["x5"]["normalization"] = bad
        with pytest.raises(ValueError):
            loaded.build_processor()


def test_hf_and_native_normalization_contracts_match():
    from olmo.hf_model.modeling_molmoact2 import _RobotStats

    for v2 in (False, True):
        cfg = config(v2)
        native = cfg.build_processor()
        hf = _RobotStats(
            {"metadata_by_tag": cfg.metadata_by_tag, "norm_mode": cfg.norm_mode}
        )
        raw = np.array([[-3.8, -2.2], [-1.8, 0.2]], np.float32)
        np.testing.assert_array_equal(
            native.normalize_state(raw, "x5"), hf.normalize_state(raw, "x5")
        )
        np.testing.assert_array_equal(
            native.normalize_action(raw, "x5"),
            hf.action_normalizers["x5"].normalize(raw),
        )
        np.testing.assert_array_equal(
            native.unnormalize_action(raw, "x5"), hf.unnormalize_action(raw, "x5")
        )


@pytest.mark.parametrize("filename", ["model.yaml", "config.yaml"])
def test_native_finetuning_preserves_checkpoint_action_expert(tmp_path, filename):
    from launch_scripts.train_lerobot import get_model
    from olmo.models.molmoact2.molmoact2 import MolmoAct2Config
    from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
    from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
    from olmo.nn.action_expert import ActionExpertConfig

    cfg = MolmoAct2Config(
        mm_preprocessor=Molmo2PreprocessorConfig(image=MultiCropConfig()),
        action_expert=ActionExpertConfig(
            hidden_size=128, num_heads=4, num_layers=3, mlp_ratio=4.0
        ),
    )
    import textwrap

    cfg.save(tmp_path / filename)
    if filename == "config.yaml":
        path = tmp_path / filename
        path.write_text("model:\n" + textwrap.indent(path.read_text(), "  "))
    loaded = get_model(str(tmp_path), "video")
    assert loaded.action_expert.mlp_ratio == 4.0
    assert loaded.action_expert.hidden_size == 128
    assert loaded.action_expert.num_heads == 4
    assert loaded.action_expert.num_layers == 3


def test_native_weight_preflight_rejects_changed_layer_width(tmp_path):
    import torch
    from scripts.train_x5_contract_v2 import validate_native_weight_shapes

    torch.save(torch.nn.Linear(3, 4).state_dict(), tmp_path / "model.pt")
    validate_native_weight_shapes(
        SimpleNamespace(build_model=lambda: torch.nn.Linear(3, 4)), tmp_path
    )
    with pytest.raises(ValueError, match="shape mismatch before FSDP"):
        validate_native_weight_shapes(
            SimpleNamespace(build_model=lambda: torch.nn.Linear(3, 2)), tmp_path
        )


def test_client_snapshots_original_rgb_and_tactile_without_changing_geometry():
    rgb = np.full((480, 640, 3), 123, np.uint8)
    tactile = np.full((288, 384, 3), 42, np.uint8)
    camera = SimpleNamespace(read=lambda: rgb)
    state = np.zeros(14, np.float32)
    obs = build_molmo_observation(
        state=state,
        prompt="Task",
        left_camera=camera,
        right_camera=camera,
        head_camera=camera,
        tactile_cameras={"tactile": SimpleNamespace(read=lambda: tactile)},
    )
    assert obs["left_image"].shape == obs["head_image"].shape == rgb.shape
    assert obs["tactile"].shape == tactile.shape
    rgb.fill(0)
    tactile.fill(0)
    state.fill(1)
    assert np.all(obs["left_image"] == 123) and np.all(obs["tactile"] == 42)
    assert np.all(obs["state"] == 0)


@pytest.mark.parametrize(
    "task", [" Under the paper. ", "Task: Under the paper!", '"Goal: Under the paper."']
)
def test_native_prompt_uses_training_task_canonicalization(task):
    from lerobot.policies.molmoact2.prompt_utils import build_prompt_fields

    assert normalize_robot_task(task) == "under the paper"
    assert build_prompt_fields(task=task) == build_prompt_fields(task="under the paper")


@pytest.mark.parametrize("maximum", [30, 50])
def test_episode_end_padding_is_combined_with_horizon_padding(maximum):
    from olmo.data.lerobot_wrapper import LeRobotDatasetWrapper

    class Tiny:
        repo_id = "local/test"
        root = None
        revision = "main"
        meta = SimpleNamespace(camera_keys=["observation.images.cam"])

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return {
                "task": "close gripper",
                "action": np.zeros((30, 14), np.float32),
                "action_is_pad": np.arange(30) > 0,
                "observation.state": np.zeros(14, np.float32),
                "observation.images.cam": np.zeros((8, 12, 3), np.uint8),
                "index": 0,
                "frame_index": 0,
                "episode_index": 0,
            }

    wrapper = LeRobotDatasetWrapper(
        dataset=Tiny(),
        split="train",
        camera_keys=["observation.images.cam"],
        camera_keys_alternative=None,
        question_key="task",
        state_keys=["observation.state"],
        action_keys=["action"],
        observation_indices=[0],
        action_indices=list(range(30)),
        style="robot_action",
        action_format="continuous",
        state_format="discrete",
        tag_action_horizon=30,
        tag_n_action_steps=5,
        max_action_horizon=maximum,
        max_action_dim=32,
        metadata_repo_id="local/test",
    )
    item = wrapper.get(0, np.random.default_rng(42), allow_random_retry=False)
    np.testing.assert_array_equal(item["action_horizon_is_pad"], np.arange(maximum) > 0)
    np.testing.assert_array_equal(item["action_dim_is_pad"], np.arange(32) >= 14)
