from types import SimpleNamespace

import numpy as np
import pytest
import torch
from launch_scripts.data_mixtures import MOLMOACT2_LEROBOT_MIXTURES
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from olmo.data.lerobot_wrapper import LeRobotDatasetWrapper
from olmo.data.robot_processing import RobotProcessorConfig
from olmo.data.tactile import (
    TACTILE_IMAGE_KEYS,
    X5_RGB_IMAGE_KEYS,
    ensure_tactile_image_capacity,
    observation_image_keys,
    tactile_keys_for_layout,
)
from olmo.nn.tactile_config import TactileConfig
from olmo.preprocessing.image_preprocessor import ImagePreprocessor
from olmo.preprocessing.multicrop_preprocessor import (
    MultiCropImagePreprocessor,
    MultiImagePreprocessor,
)
from olmo.preprocessing.tactile_preprocessor import preprocess_tactile
from scripts.serve_policy import MolmoAct2Server, _build_observations


def _metadata(layout="two"):
    _, entries = MOLMOACT2_LEROBOT_MIXTURES[f"x5_tactile_{layout}"]()
    return next(iter(entries.values()))


def _robot_processor(metadata):
    return RobotProcessorConfig.from_stats(
        stats_by_tag={
            "x5": {
                "action": {"mean": [0.0] * 14, "std": [1.0] * 14},
                "observation.state": {"mean": [0.0] * 14, "std": [1.0] * 14},
            }
        },
        tag_metadata={"x5": metadata},
        repo_to_tag={"local/x5": "x5"},
        norm_mode="none",
    )


def _handles(metadata):
    return SimpleNamespace(
        robot_processor=_robot_processor(metadata).build_processor(),
        norm_tag="x5",
        state_format="continuous",
        inference_action_mode="continuous",
        num_state_tokens=256,
        default_style="robot_action",
        add_setup_tokens=False,
        add_control_tokens=False,
    )


def _policy(metadata):
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy._handles = _handles(metadata)
    return policy


def _row():
    row = {
        "task": "Stamp the bag.",
        "observation.state": np.zeros((2, 14), dtype=np.float32),
        "action": np.zeros((3, 14), dtype=np.float32),
        "index": 0,
        "episode_index": 0,
    }
    for i, key in enumerate((*X5_RGB_IMAGE_KEYS, *TACTILE_IMAGE_KEYS)):
        row[key] = np.stack(
            [np.full((3, 28, 28), 20 * i + t, np.uint8) for t in range(2)]
        )
    return row


def _wrapper(
    row, layout="two", random_camera_order="none", auto_rgb=False, backend="as_image"
):
    class Dataset:
        repo_id = "local/x5"
        root = None
        revision = "main"
        meta = SimpleNamespace(camera_keys=[*X5_RGB_IMAGE_KEYS, *TACTILE_IMAGE_KEYS])

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return dict(row)

    return LeRobotDatasetWrapper(
        dataset=Dataset(),
        split="train",
        camera_keys=None if auto_rgb else list(X5_RGB_IMAGE_KEYS),
        camera_keys_alternative=None,
        tactile_keys=tactile_keys_for_layout(layout),
        tactile_config=TactileConfig(backend=backend, layout=layout),
        question_key="task",
        state_keys=["observation.state"],
        action_keys=["action"],
        observation_indices=[-1, 0],
        action_indices=[-1, 0, 1],
        tag_action_horizon=3,
        tag_n_action_steps=1,
        max_action_horizon=3,
        max_action_dim=14,
        style="robot_action",
        state_format="continuous",
        random_camera_order=random_camera_order,
    )


@pytest.mark.parametrize("layout", ["two", "four"])
def test_train_and_policy_use_identical_sensor_and_history_order(layout):
    row = _row()
    expected = _wrapper(row, layout).get(
        0, np.random.default_rng(0), allow_random_retry=False
    )
    policy = _policy(_metadata(layout))
    observations = []
    for t in range(2):
        # Include unused sensors and deliberately reverse wire order.
        obs = {
            key: np.moveaxis(row[key][t], 0, -1)
            for key in reversed((*X5_RGB_IMAGE_KEYS, *TACTILE_IMAGE_KEYS))
        }
        obs.update(task=row["task"], state=row["observation.state"][t])
        observations.append(MolmoAct2Policy._obs_to_example(policy, obs))
    actual = MolmoAct2Policy._combine_history_examples(
        policy, observations, policy._handles, norm_tag="x5"
    )
    for train_image, infer_image in zip(
        expected["image"], actual["image"], strict=True
    ):
        np.testing.assert_array_equal(train_image, infer_image)
    assert expected["image_augmentation_mask"] == actual["image_augmentation_mask"]
    assert actual["image_augmentation_mask"] == [True] * 6 + [False] * (
        2 * len(tactile_keys_for_layout(layout))
    )


@pytest.mark.parametrize("order", ["all", "episode"])
def test_rgb_shuffling_and_autodetection_keep_sensor_slots_fixed(order):
    row = _row()
    wrapper = _wrapper(row, random_camera_order=order, auto_rgb=True)
    actual = wrapper.get(0, np.random.default_rng(4), allow_random_retry=False)
    assert set(actual["metadata"]["camera_keys_used"]) == set(X5_RGB_IMAGE_KEYS)
    assert actual["metadata"]["tactile_keys_used"] == tactile_keys_for_layout("two")
    assert [int(image[0, 0, 0]) for image in actual["image"][-4:]] == [60, 61, 120, 121]


def test_missing_tactile_data_is_rejected_in_training_and_policy():
    row = _row()
    row.pop(TACTILE_IMAGE_KEYS[3])
    with pytest.raises(ValueError, match="Missing required camera keys"):
        _wrapper(row).get(0, np.random.default_rng(0), allow_random_retry=False)
    policy = _policy(_metadata())
    example = MolmoAct2Policy._obs_to_example(
        policy,
        {X5_RGB_IMAGE_KEYS[0]: np.zeros((28, 28, 3), np.uint8), "state": np.zeros(14)},
    )
    with pytest.raises(ValueError, match="Missing required RGB/tactile"):
        MolmoAct2Policy._combine_history_examples(policy, [example], policy._handles)


@pytest.mark.parametrize("layout", ["two", "four"])
def test_checkpoint_metadata_restores_tactile_layout_and_image_capacity(
    tmp_path, layout
):
    cfg = _robot_processor(_metadata(layout))
    cfg.save(tmp_path / "robot.yaml")
    restored = RobotProcessorConfig.load(tmp_path / "robot.yaml")
    assert restored.build_processor().get_metadata("local/x5")[
        "tactile_keys"
    ] == tactile_keys_for_layout(layout)
    model = SimpleNamespace(
        mm_preprocessor=SimpleNamespace(image=SimpleNamespace(max_images=5)),
        n_obs_steps=2,
    )
    ensure_tactile_image_capacity(model, restored.tag_metadata)
    assert model.mm_preprocessor.image.max_images == 2 * (
        3 + len(tactile_keys_for_layout(layout))
    )


def _server(layout="two"):
    server = object.__new__(MolmoAct2Server)
    server.default_norm_tag = "x5"
    server.robot_processor = _handles(_metadata(layout)).robot_processor
    server.image_keys = [
        "unused_preset"
    ]  # Checkpoint metadata selects the actual inputs.
    server.image_resize = None
    server.n_obs_steps = 1
    server.default_n_action_steps = 5
    return server


@pytest.mark.parametrize("form", ["top_level", "mapping", "ordered_list"])
def test_server_accepts_reference_sensor_names_and_preserves_canonical_keys(form):
    keys = observation_image_keys(_metadata())
    imgs = [np.full((28, 28, 3), i, np.uint8) for i in range(len(keys))]
    mapping = {
        key.removeprefix("observation.images."): img for key, img in zip(keys, imgs)
    }
    payload = {"state": np.zeros(14), "instruction": "Stamp the bag."}
    if form == "top_level":
        payload.update(mapping)
    else:
        payload["images"] = mapping if form == "mapping" else imgs
    images, names, state, instruction, *_ = _server()._parse_request(payload)
    observations = _build_observations(images, names, state, instruction)
    assert [
        key for key in observations[0] if key.startswith("observation.images.")
    ] == keys
    assert [int(observations[0][key][0, 0, 0]) for key in keys] == list(
        range(len(keys))
    )


def test_server_rejects_missing_sensor_and_wrong_image_count():
    server = _server()
    with pytest.raises(ValueError, match="expects 5 images"):
        server._parse_request(
            {"images": [np.zeros((28, 28, 3), np.uint8)] * 3, "state": np.zeros(14)}
        )
    with pytest.raises(ValueError, match="Missing required RGB/tactile"):
        server._parse_request({"images": {}, "state": np.zeros(14)})


def _multi_image_preprocessor(max_images=7):
    tok = SimpleNamespace(
        image_patch_token_id=1,
        image_start_token_id=2,
        image_end_token_id=3,
        image_col_token_id=4,
        encode=lambda text: [5],
    )
    image = ImagePreprocessor(
        base_image_input_size=(28, 28),
        image_patch_size=14,
        normalize="siglip",
        resize="siglip",
        use_image_augmentation="photometric",
    )
    single = MultiCropImagePreprocessor(
        tok, image, use_col_tokens=False, image_pooling_h=1, image_pooling_w=1
    )
    return MultiImagePreprocessor(single, max_images)


def test_tactile_pixels_are_identical_in_train_and_inference_while_rgb_is_augmented():
    pp = _multi_image_preprocessor()
    image = np.random.default_rng(0).integers(0, 256, (28, 28, 3), dtype=np.uint8)
    train = pp(
        [image, image], True, np.random.default_rng(3), apply_augmentation=[True, False]
    )
    infer = pp(
        [image, image],
        False,
        np.random.default_rng(3),
        apply_augmentation=[True, False],
    )
    np.testing.assert_array_equal(train[1].images, infer[1].images)
    assert not np.array_equal(train[0].images, infer[0].images)


def test_image_cap_cannot_silently_drop_tactile_sensors():
    pp = _multi_image_preprocessor(max_images=5)
    images = [np.zeros((28, 28, 3), np.uint8)] * 7
    with pytest.raises(ValueError, match="truncate configured tactile"):
        pp(images, apply_augmentation=[True] * 3 + [False] * 4)


@pytest.mark.parametrize("backend", ["anytouch2", "sparsh_vjepa", "tactile_mae"])
def test_independent_windows_match_training_and_serving_without_future_frames(backend):
    cfg = TactileConfig(backend=backend)
    row = _row()
    offsets = sorted(
        {s - h * cfg.history_stride for s in (-1, 0) for h in range(cfg.frames)}
    )
    for i, key in enumerate(TACTILE_IMAGE_KEYS):
        row[key] = np.stack(
            [
                np.full((3, 28, 28), 40 + i * 20 + index, np.uint8)
                for index in range(len(offsets))
            ]
        )
    train = _wrapper(row, backend=backend).get(
        0, np.random.default_rng(0), allow_random_retry=False
    )
    metadata = {**_metadata(), "tactile_backend": backend, "tactile_history_stride": 2}
    policy = _policy(metadata)
    examples = []
    for t, outer in enumerate((-1, 0)):
        obs = {key: np.moveaxis(row[key][t], 0, -1) for key in X5_RGB_IMAGE_KEYS}
        for key in cfg.keys:
            obs[key] = np.stack(
                [
                    np.moveaxis(
                        row[key][offsets.index(outer - h * cfg.history_stride)], 0, -1
                    )
                    for h in reversed(range(cfg.frames))
                ]
            )
        obs.update(state=row["observation.state"][t], task=row["task"])
        examples.append(MolmoAct2Policy._obs_to_example(policy, obs))
    infer = MolmoAct2Policy._combine_history_examples(policy, examples, policy._handles)
    assert len(train["image"]) == len(infer["image"]) == 6
    np.testing.assert_array_equal(train["tactile_windows"], infer["tactile_windows"])
    np.testing.assert_array_equal(
        preprocess_tactile(train["tactile_windows"], cfg),
        preprocess_tactile(infer["tactile_windows"], cfg),
    )
    assert max(offsets) == 0
