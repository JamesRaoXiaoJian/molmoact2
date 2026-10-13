"""Loopback-only OpenPI wire adapter. Produces candidates; has no hardware path."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import msgpack
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve


# OpenPI's Apache-2.0 msgpack_numpy wire format; no pickle fallback.
def pack_array(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in "VOc":
        raise ValueError(f"Unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"Unsupported payload type: {type(value).__name__}")


def unpack_array(value):
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "VOc":
            raise ValueError(f"Unsupported dtype: {dtype}")
        return np.ndarray(
            buffer=value[b"data"], dtype=dtype, shape=value[b"shape"]
        ).copy()
    if b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "VOc":
            raise ValueError(f"Unsupported dtype: {dtype}")
        return dtype.type(value[b"data"])
    return value


def pack(value):
    return msgpack.packb(value, default=pack_array)


def unpack(value):
    return msgpack.unpackb(value, object_hook=unpack_array)


RGB_KEYS = {
    "left_image": "left_d405",
    "right_image": "right_d405",
    "head_image": "head_camera",
}
TACTILE_KEYS = ("left_gripper_left_tactile", "right_gripper_right_tactile")
ACTION_NAMES = [
    f"{arm}_{name}.pos"
    for arm in ("left", "right")
    for name in (*[f"joint_{i}" for i in range(1, 7)], "gripper")
]


class OpenPIBridge:
    """Delegate model semantics to the unchanged Molmo HTTP service in process."""

    def __init__(self, molmo, mode: str):
        if mode not in ("rgb", "mae"):
            raise ValueError("Select rgb or mae explicitly")
        self.molmo = molmo
        self.mode = mode
        self.inference_requests = 0
        self.closed_sessions = 0
        self.reset_requests = 0
        self.metadata = {
            "model_type": "molmoact2",
            "model_variant": mode,
            "protocol": "openpi-msgpack-numpy",
            "supports_rtc": False,
            "action_horizon": 30,
            "action_dim": 14,
            "n_action_steps": 5,
            "n_obs_steps": 1,
            "dataset_fps": 30,
            "action_index_0_frame_offset": 0,
            "camera_order": ["left_d405", "right_d405", "head_camera"],
            "image_layout": "HWC RGB uint8",
            "tactile_inputs": mode == "mae",
            "tactile_keys": list(TACTILE_KEYS) if mode == "mae" else [],
            "tactile_backend": "tactile_mae" if mode == "mae" else None,
            "tactile_frames": 1,
            "tactile_history_stride": 2,
            "state_names": ACTION_NAMES,
            "action_names": ACTION_NAMES,
            "arm_order": "left_then_right",
            "action_space": "absolute joint pose",
            "joint_units": "rad",
            "gripper_units": "calibrated native motor angle rad",
            "norm_tag": f"arx_x5_cleaned_{mode}",
            "reset_observation_flag": "_reset",
            "dry_run": True,
            "hardware_execution": False,
            "output_role": "candidate actions only; receiver controls any actuation",
        }

    def payload(self, observation: dict, session: str):
        if not isinstance(observation, dict):
            raise TypeError("Observation must be a mapping")
        if "_rtc" in observation:
            raise ValueError("MolmoAct2 adapter does not support RTC")
        allowed = {"state", "prompt", "base_image", "_reset", *RGB_KEYS, *TACTILE_KEYS}
        unknown = set(observation) - allowed
        if unknown:
            raise ValueError(f"Unsupported observation fields: {sorted(unknown)}")
        state = np.asarray(observation.get("state"))
        if (
            state.shape != (14,)
            or state.dtype != np.float32
            or not np.isfinite(state).all()
        ):
            raise ValueError(
                "state must be a finite float32 (14,) vector; no padding or truncation"
            )
        prompt = observation.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if not isinstance(observation.get("_reset", False), (bool, np.bool_)):
            raise TypeError("_reset must be boolean")
        result = {
            "state": state,
            "instruction": prompt,
            "session_id": session,
            "action_chunk": True,
            "n_action_steps": 30,
        }
        images = dict(RGB_KEYS)
        if self.mode == "mae":
            images.update({key: key for key in TACTILE_KEYS})
        elif any(key in observation for key in TACTILE_KEYS):
            raise ValueError("RGB model does not consume tactile inputs; select mae")
        if "base_image" in observation:
            raise ValueError(
                "This checkpoint has no base camera; refusing to ignore an extra stream"
            )
        for source, target in images.items():
            image = np.asarray(observation.get(source))
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[-1] != 3
                or min(image.shape[:2]) < 1
            ):
                raise ValueError(
                    f"{source} must be present HWC RGB uint8; temporal/CHW inputs unsupported"
                )
            result[target] = image
        return result

    def reset(self, session):
        self.molmo.reset_session({"session_id": session})

    def infer(self, observation, session):
        payload = self.payload(observation, session)
        if observation.get("_reset", False):
            self.reset(session)
            self.reset_requests += 1
        started = time.perf_counter()
        response = self.molmo.predict_action(payload)
        value = json.loads(response.body)
        if response.status_code != 200:
            raise ValueError(value.get("error", "Molmo inference failed"))
        actions = np.asarray(value["actions"], dtype=np.float32)
        if actions.shape != (30, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid native Molmo action output {actions.shape}")
        self.inference_requests += 1
        return {
            "actions": actions,
            "policy_timing": {"infer_ms": (time.perf_counter() - started) * 1000},
            "inference_calls": value["inference_calls"],
            "dry_run": True,
        }

    def handler(self, connection):
        session = f"openpi:{uuid.uuid4()}"
        try:
            connection.send(pack(self.metadata))
            while True:
                try:
                    request = connection.recv()
                    if not isinstance(request, bytes):
                        raise TypeError("Expected a binary msgpack observation")
                    started = time.perf_counter()
                    response = self.infer(unpack(request), session)
                    response["server_timing"] = {
                        "infer_ms": (time.perf_counter() - started) * 1000
                    }
                    connection.send(pack(response))
                except ConnectionClosed:
                    break
                except Exception as exc:  # noqa: BLE001 -- wire error boundary reports failure and closes
                    connection.send(f"{type(exc).__name__}: {exc}")
                    connection.close(1011, "Invalid observation or inference failure")
                    break
        finally:
            self.reset(session)
            self.closed_sessions += 1

    def create_server(self, port=0, host="127.0.0.1"):
        if host not in ("127.0.0.1", "::1"):
            raise ValueError("Only literal loopback interfaces are allowed")
        return serve(
            self.handler, host, port, compression=None, max_size=16 * 1024 * 1024
        )


def build_bridge(mode, checkpoint=None, norm_tag=None, tokenizer_dir=None):
    from scripts.serve_policy import MolmoAct2Server

    root = Path(__file__).resolve().parents[2]
    tag = norm_tag or f"arx_x5_cleaned_{mode}"
    optimization = os.environ.get("MOLMO_NATIVE_OPTIMIZATION", "graph")
    if optimization not in {"baseline", "context", "deferred", "modcache", "graph"}:
        raise ValueError("Unknown MOLMO_NATIVE_OPTIMIZATION")
    compile_setting = os.environ.get("MOLMO_COMPILE_ACTION_EXPERT", "0")
    if compile_setting not in {"0", "1"}:
        raise ValueError("MOLMO_COMPILE_ACTION_EXPERT must be 0 or 1")
    molmo = MolmoAct2Server(
        checkpoint=checkpoint
        or str(
            root
            / f"outputs/runs/molmoact2-x5-cleaned-{mode}-fft20k-b64-s42-20261008/checkpoints/step20000-unsharded"
        ),
        norm_tag=tag,
        image_keys="x5_tactile" if mode == "mae" else "x5",
        device="cuda:0",
        parameter_dtype="bfloat16",
        tokenizer_dir=tokenizer_dir or str(root / "data/hf-cache/hub"),
        enable_inference_cuda_graph=False,
    )
    metadata = molmo.robot_processor.get_metadata(tag)
    for key, expected in {
        "action_dim": 14,
        "action_horizon": 30,
        "n_action_steps": 5,
        "control_mode": "absolute joint pose",
        "normalize_gripper": False,
    }.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Checkpoint {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    handles = getattr(getattr(molmo, "policy", None), "_handles", None)
    if handles is not None:
        handles.model.native_inference_optimization = optimization
        if compile_setting == "1":
            from olmo.nn.native_action_compile import compile_action_blocks

            compile_action_blocks(handles.model)
    elif compile_setting == "1":
        raise ValueError("Action block compilation requires the native checkpoint backend")
    bridge = OpenPIBridge(molmo, mode)
    bridge.metadata["norm_tag"] = tag
    bridge.metadata["native_inference_optimization"] = (
        optimization if handles is not None else "hf_backend"
    )
    bridge.metadata["native_action_blocks_compiled"] = compile_setting == "1"
    return bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("rgb", "mae"))
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--checkpoint", help="Native directory containing model.pt and config.yaml"
    )
    parser.add_argument(
        "--norm-tag", help="Robot normalization tag from this checkpoint"
    )
    parser.add_argument(
        "--tokenizer-dir", help="Compatible local tokenizer assets/cache"
    )
    args = parser.parse_args()
    bridge = build_bridge(args.mode, args.checkpoint, args.norm_tag, args.tokenizer_dir)
    with bridge.create_server(args.port) as server:
        print(
            json.dumps(
                {"listening": server.socket.getsockname(), "metadata": bridge.metadata}
            ),
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
