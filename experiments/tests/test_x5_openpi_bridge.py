import json
from types import SimpleNamespace

import numpy as np
import pytest
from scripts.serve_x5_openpi import (
    ACTION_NAMES,
    TACTILE_KEYS,
    OpenPIBridge,
    pack,
    unpack,
)


class FakeMolmo:
    def __init__(self):
        self.payloads = []
        self.resets = []

    def predict_action(self, payload):
        self.payloads.append(payload)
        actions = np.tile(
            np.array(
                [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    -2.1,
                    1.1,
                    1.2,
                    1.3,
                    1.4,
                    1.5,
                    1.6,
                    -3.2,
                ],
                dtype=np.float32,
            ),
            (30, 1),
        )
        return SimpleNamespace(
            status_code=200,
            body=json.dumps(
                {"actions": actions.tolist(), "inference_calls": 1}
            ).encode(),
        )

    def reset_session(self, payload):
        self.resets.append(payload)


def observation(mode="rgb"):
    value = {
        "state": np.arange(14, dtype=np.float32),
        "prompt": "Recorded offline task",
    }
    for i, key in enumerate(("left_image", "right_image", "head_image")):
        value[key] = np.full((8, 9, 3), i, dtype=np.uint8)
    if mode == "mae":
        for i, key in enumerate(TACTILE_KEYS):
            value[key] = np.full((5, 7, 3), i + 10, dtype=np.uint8)
    return value


@pytest.mark.parametrize("mode", ["rgb", "mae"])
def test_mapping_keeps_pixels_state_action_units_and_order(mode):
    native = FakeMolmo()
    bridge = OpenPIBridge(native, mode)
    obs = observation(mode)
    result = bridge.infer(obs, "test")
    payload = native.payloads[0]
    assert payload["state"] is obs["state"]
    for source, dest in [
        ("left_image", "left_d405"),
        ("right_image", "right_d405"),
        ("head_image", "head_camera"),
    ]:
        assert payload[dest] is obs[source]
    for key in TACTILE_KEYS if mode == "mae" else ():
        assert payload[key] is obs[key]
    np.testing.assert_array_equal(
        result["actions"],
        np.tile(
            np.array(
                [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    -2.1,
                    1.1,
                    1.2,
                    1.3,
                    1.4,
                    1.5,
                    1.6,
                    -3.2,
                ],
                dtype=np.float32,
            ),
            (30, 1),
        ),
    )
    assert bridge.metadata["action_names"] == ACTION_NAMES
    assert bridge.metadata["action_horizon"] == 30
    assert bridge.metadata["n_action_steps"] == 5
    assert bridge.metadata["supports_rtc"] is False
    assert bridge.metadata["hardware_execution"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"state": np.zeros(13)},
        {"state": np.zeros(15)},
        {"state": np.zeros((1, 14))},
        {"state": np.full(14, np.nan, dtype=np.float32)},
        {"state": np.full(14, np.inf, dtype=np.float32)},
        {"prompt": ""},
        {"_rtc": {}},
        {"_reset": "true"},
        {"head_image": None},
        {"left_image": np.zeros((3, 224, 224), dtype=np.uint8)},
        {"left_image": np.zeros((224, 224, 3), dtype=np.float32)},
        {"base_image": np.zeros((8, 9, 3), dtype=np.uint8)},
        {"actions": np.zeros((30, 14))},
    ],
)
def test_invalid_inputs_fail_before_model(change):
    native = FakeMolmo()
    obs = observation()
    obs.update(change)
    with pytest.raises((ValueError, TypeError)):
        OpenPIBridge(native, "rgb").infer(obs, "test")
    assert not native.payloads


def test_missing_and_temporal_tactile_are_not_dropped():
    native = FakeMolmo()
    bridge = OpenPIBridge(native, "mae")
    for bad in [
        observation(),
        {**observation("mae"), TACTILE_KEYS[0]: np.zeros((4, 5, 7, 3), dtype=np.uint8)},
    ]:
        with pytest.raises((ValueError, TypeError)):
            bridge.infer(bad, "test")
    with pytest.raises((ValueError, TypeError)):
        OpenPIBridge(native, "rgb").infer(observation("mae"), "test")
    assert not native.payloads


def test_reset_is_explicit_and_loopback_required():
    native = FakeMolmo()
    bridge = OpenPIBridge(native, "rgb")
    bridge.infer({**observation(), "_reset": True}, "episode")
    assert native.resets == [{"session_id": "episode"}]
    assert bridge.reset_requests == 1
    with pytest.raises(ValueError, match="loopback"):
        bridge.create_server(8001, "0.0.0.0")


def test_codec_roundtrip_and_no_pickle_arrays():
    value = observation("mae")
    decoded = unpack(pack(value))
    for key in value:
        if isinstance(value[key], np.ndarray):
            np.testing.assert_array_equal(value[key], decoded[key])
            assert decoded[key].flags.writeable
        else:
            assert value[key] == decoded[key]
    with pytest.raises((ValueError, TypeError)):
        pack(np.array([object()], dtype=object))


def test_episode_policy_marks_reset_without_masquerading_model_type():
    from scripts.x5_openpi_control import EpisodePolicy

    class Client:
        requests = None

        def __init__(self):
            self.requests = []

        def get_server_metadata(self):
            return {
                "model_type": "molmoact2",
                "model_variant": "rgb",
                "supports_rtc": False,
                "action_horizon": 30,
                "n_action_steps": 5,
            }

        def infer(self, obs):
            self.requests.append(obs)
            return {"actions": np.zeros((30, 14), dtype=np.float32)}

    client = Client()
    policy = EpisodePolicy(client, "rgb")
    policy.infer({"state": np.zeros(14, dtype=np.float32)})
    policy.infer({"state": np.zeros(14, dtype=np.float32)})
    policy.reset()
    policy.infer({"state": np.zeros(14, dtype=np.float32)})
    assert [r["_reset"] for r in client.requests] == [True, False, True]
    assert policy.get_server_metadata()["model_type"] == "molmoact2"


@pytest.mark.parametrize(
    "override",
    [
        {"execute_steps": 37},
        {"rtc": True},
        {"enable_motion": True},
        {"auto_reset_on_startup": True},
    ],
)
def test_control_glue_rejects_unsafe_window_and_modes(override):
    from scripts.x5_openpi_control import validate_control_args

    args = SimpleNamespace(
        rtc=False,
        policy_transport="websocket",
        execute_steps=5,
        action_start_index=0,
        enable_motion=False,
        auto_reset_each_episode=False,
        auto_reset_on_startup=False,
        dagger=False,
    )
    for k, v in override.items():
        setattr(args, k, v)
    with pytest.raises(ValueError):
        validate_control_args(args)


def test_rtc_wire_forwards_raw_prefix_without_gripper_calibration():
    native = FakeMolmo()
    bridge = OpenPIBridge(native, "rgb", enable_rtc=True)
    obs = observation()
    prefix = np.ones((30, 14), dtype=np.float32)
    prefix[:, 6] = -3.5
    prefix[:, 13] = -2.4
    obs.update(actions=prefix, _rtc=dict(prefix_valid_steps=18, inference_delay=12,
               execution_horizon=20, max_guidance_weight=10.0))
    bridge.infer(obs, "rtc-test")
    payload = native.payloads[-1]
    np.testing.assert_array_equal(payload["rtc"]["previous_actions"], prefix)
    assert payload["rtc"]["previous_actions"] is not prefix
    assert payload["rtc"]["options"]["inference_delay"] == 12
    assert bridge.metadata["rtc_protocol"] == "molmo-rtc-v1"


@pytest.mark.parametrize("bad", ["missing", "shape", "dtype", "masked_nan", "options"])
def test_rtc_wire_rejects_invalid_prefix_before_model_call(bad):
    native = FakeMolmo()
    bridge = OpenPIBridge(native, "rgb", enable_rtc=True)
    obs = observation()
    obs.update(actions=np.zeros((30,14),np.float32),_rtc=dict(prefix_valid_steps=18,
        inference_delay=12,execution_horizon=20,max_guidance_weight=10.0))
    if bad == "missing": obs.pop("actions")
    if bad == "shape": obs["actions"] = np.zeros((30,15),np.float32)
    if bad == "dtype": obs["actions"] = obs["actions"].astype(np.float64)
    if bad == "masked_nan": obs["actions"][29,13] = np.nan
    if bad == "options": obs["_rtc"]["inference_delay"] = True
    with pytest.raises(ValueError): bridge.infer(obs,"rtc-test")
    assert not native.payloads
