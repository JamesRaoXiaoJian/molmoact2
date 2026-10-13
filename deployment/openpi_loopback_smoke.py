"""Existing OpenPI network client -> real Molmo -> pure control functions -> fake sink."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.abc
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from websockets.exceptions import ConnectionClosed

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/lerobot/src")]
from scripts.run_x5_inference_comparison import infer, test_http
from scripts.serve_x5_openpi import RGB_KEYS, TACTILE_KEYS, build_bridge


class RecordedCamera:
    def __init__(self, image):
        self.image = image

    def read(self):
        return self.image.copy()


class FakeSink:
    """Only records candidates; there is deliberately no native-send branch."""

    def __init__(self):
        self.records = []
        self.actual_action_sends = 0
        self.hardware_initializations = 0

    def record_candidate(self, action, duration):
        assert action.shape == (14,) and np.isfinite(action).all()
        self.records.append(
            {"action": action.copy(), "duration": duration, "at": time.monotonic()}
        )

    def set_joint_positions(self, *args, **kwargs):
        raise AssertionError("Hardware sends are forbidden")

    def set_gripper_pos(self, *args, **kwargs):
        raise AssertionError("Hardware sends are forbidden")


class HardwareImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "arx5_interface",
            "can",
            "rospy",
            "rclpy",
            "pyrealsense2",
            "pyorbbecsdk",
        }:
            raise AssertionError(f"Hardware module import forbidden: {fullname}")


def audit_guard(event, args):
    if event == "socket.connect":
        address = args[1]
        if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1"}:
            raise AssertionError(f"Non-loopback connection forbidden: {address}")
    if event == "socket.__new__" and args[1] == 29:
        raise AssertionError("CAN socket creation forbidden")
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = str(args[0])
        if path.startswith(("/dev/video", "/dev/input", "/dev/tty", "/dev/can")):
            raise AssertionError(f"Hardware device access forbidden: {path}")


def load_safe_control_functions(openpi):
    """Read exact reviewed pure functions, never import/run the hardware main module."""
    source = (openpi / "examples/x5_client/main.py").read_text()
    tree = ast.parse(source)
    names = {
        "build_policy_observation",
        "validate_action_chunk",
        "clip_action",
        "apply_stateless_gripper_close_offset",
        "command_duration",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    from openpi_client import image_tools

    namespace = {
        "np": np,
        "Any": Any,
        "image_tools": image_tools,
        "ACTION_DIM": 14,
        "JOINT_INDICES": np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]),
        "GRIPPER_INDICES": np.array([6, 13]),
    }
    safe_source = "from __future__ import annotations\n" + "\n\n".join(
        ast.get_source_segment(source, n) for n in nodes
    )
    exec(  # noqa: S102 -- only five reviewed pure AST functions, no hardware module import
        compile(safe_source, str(openpi / "examples/x5_client/main.py"), "exec"),
        namespace,
    )
    requested = {
        "--control-hz",
        "--duration-scale",
        "--max-step-delta",
        "--max-gripper-step-delta",
        "--gripper-min",
        "--gripper-max",
        "--reject-first-joint-delta",
        "--gripper-close-offset",
        "--gripper-close-threshold",
    }
    defaults = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ) and (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in requested
        ):
            for kw in node.keywords:
                if kw.arg == "default":
                    defaults[node.args[0].value[2:]] = ast.literal_eval(kw.value)
    assert set(defaults) == {n[2:] for n in requested}
    return namespace, defaults, hashlib.sha256(source.encode()).hexdigest()


def close(client):
    client._ws.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("rgb", "mae"))
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=Path.home() / "lwm_ws/openpi-x5-arx5-beta/openpi",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/inference/openpi_bridge"
    )
    args = parser.parse_args()
    sys.meta_path.insert(0, HardwareImportGuard())
    sys.addaudithook(audit_guard)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.openpi_root / "packages/openpi-client/src"))
    from openpi_client.action_chunk_broker import ActionChunkBroker
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    controls, defaults, source_hash = load_safe_control_functions(args.openpi_root)
    manifest = json.loads((ROOT / "data/offline_samples/manifest.json").read_text())
    started = time.monotonic()
    bridge = build_bridge(args.mode)
    load_s = time.monotonic() - started
    sink = FakeSink()
    sample_results = []
    raw_wire_chunks = []
    control_rejections = []
    torch.cuda.reset_peak_memory_stats()
    with bridge.create_server() as service:
        worker = threading.Thread(target=service.serve_forever, daemon=True)
        worker.start()
        port = service.socket.getsockname()[1]
        client = WebsocketClientPolicy("127.0.0.1", port)
        metadata = client.get_server_metadata()
        assert metadata["action_horizon"] == 30 and metadata["n_action_steps"] == 5
        assert not metadata["supports_rtc"] and not metadata["hardware_execution"]
        assert metadata["action_names"] == manifest["dimension_names"]
        for item in manifest["samples"]:
            data = np.load(
                ROOT / "data/offline_samples" / item["file"], allow_pickle=False
            )
            tactile = (
                {
                    k: RecordedCamera(data[f"observation.images.{k}"])
                    for k in TACTILE_KEYS
                }
                if args.mode == "mae"
                else {}
            )
            obs = controls["build_policy_observation"](
                state=data["observation.state"],
                prompt=item["task"],
                left_camera=RecordedCamera(data["observation.images.left_d405"]),
                right_camera=RecordedCamera(data["observation.images.right_d405"]),
                head_camera=RecordedCamera(data["observation.images.head_camera"]),
                tactile_cameras=tactile,
            )
            native = {
                f"observation.images.{dest}": obs[src] for src, dest in RGB_KEYS.items()
            }
            native.update({f"observation.images.{k}": obs[k] for k in tactile})
            native.update({"observation.state": obs["state"], "task": obs["prompt"]})
            expected, direct_s = infer(bridge.molmo.policy, native, item["seed"], 10)
            torch.manual_seed(item["seed"])
            started = time.monotonic()
            result = client.infer({**obs, "_reset": True})
            elapsed = time.monotonic() - started
            np.testing.assert_allclose(
                result["actions"], expected, rtol=1e-5, atol=1e-5
            )
            actions = controls["validate_action_chunk"](
                result["actions"],
                obs["state"],
                defaults["reject-first-joint-delta"],
                0,
                5,
                30,
            )
            nonfinite = actions.copy()
            nonfinite[0, 0] = np.nan
            distant = actions.copy()
            distant[0, 0] = obs["state"][0] + 1.0
            for label, candidate, stop in [
                ("window beyond horizon", actions, 31),
                ("wrong action dimensions", np.zeros((30, 15), dtype=np.float32), 5),
                ("nonfinite actions", nonfinite, 5),
                ("first joint delta beyond original limit", distant, 5),
            ]:
                try:
                    controls["validate_action_chunk"](
                        candidate,
                        obs["state"],
                        defaults["reject-first-joint-delta"],
                        0,
                        stop,
                        30,
                    )
                except RuntimeError as exc:
                    control_rejections.append({"case": label, "error": str(exc)})
                else:
                    raise AssertionError(f"Control boundary accepted {label}")
            raw_wire_chunks.append(actions.copy())
            # Real existing client broker consumes the checkpoint's first five rows.
            broker = ActionChunkBroker(client, action_horizon=5)
            before = bridge.inference_requests
            torch.manual_seed(item["seed"])
            current = obs["state"].copy()
            calls = []
            for step in range(6):
                tick = time.monotonic()
                prior = bridge.inference_requests
                single = broker.infer(obs)["actions"]
                calls.append(bridge.inference_requests - prior)
                if step < 5:
                    np.testing.assert_allclose(
                        single, actions[step], rtol=1e-5, atol=1e-5
                    )
                adjusted = controls["apply_stateless_gripper_close_offset"](
                    single,
                    defaults["gripper-close-offset"],
                    defaults["gripper-close-threshold"],
                    defaults["gripper-max"],
                )
                safe = controls["clip_action"](
                    adjusted,
                    current,
                    defaults["max-step-delta"],
                    defaults["max-gripper-step-delta"],
                    defaults["gripper-min"],
                    defaults["gripper-max"],
                )
                joint_indices = controls["JOINT_INDICES"]
                gripper_indices = controls["GRIPPER_INDICES"]
                assert (
                    np.abs(safe[joint_indices] - current[joint_indices]).max()
                    <= defaults["max-step-delta"] + 1e-6
                )
                assert (
                    np.abs(safe[gripper_indices] - current[gripper_indices]).max()
                    <= defaults["max-gripper-step-delta"] + 1e-6
                )
                assert np.all(safe[gripper_indices] >= defaults["gripper-min"])
                assert np.all(safe[gripper_indices] <= defaults["gripper-max"])
                sink.record_candidate(
                    safe,
                    controls["command_duration"](
                        defaults["control-hz"], defaults["duration-scale"]
                    ),
                )
                current = (
                    safe.copy()
                )  # simulated state only, never a hardware feedback source
                time.sleep(
                    max(0.0, 1 / defaults["control-hz"] - (time.monotonic() - tick))
                )
            assert (
                calls == [1, 0, 0, 0, 0, 1] and bridge.inference_requests - before == 2
            )
            broker.reset()
            prior = bridge.inference_requests
            broker.infer({**obs, "_reset": True})
            assert bridge.inference_requests == prior + 1
            sample_results.append(
                {
                    "episode": item["episode"],
                    "frame": item["frame"],
                    "wire_latency_s": elapsed,
                    "direct_latency_s": direct_s,
                    "shape": list(actions.shape),
                    "finite": True,
                    "direct_native_max_abs_delta": float(
                        np.abs(actions - expected).max()
                    ),
                    "queue_calls": calls,
                    "broker_and_wire_reset": "passed",
                    "original_rgb_shapes": {
                        k: list(data[f"observation.images.{v}"].shape)
                        for k, v in RGB_KEYS.items()
                    },
                    "client_rgb_shapes": {k: list(obs[k].shape) for k in RGB_KEYS},
                    "tactile_shapes": {k: list(obs[k].shape) for k in tactile},
                }
            )
        # The unchanged native HTTP single-action queue must also remain correct.
        http_sample = {**manifest["samples"][-1], "obs": native}
        native_http = test_http(bridge.molmo, http_sample, expected, item["seed"])
        bridge.molmo.reset_session({"session_id": "inference-test"})
        other = WebsocketClientPolicy("127.0.0.1", port)
        other.infer(obs)
        assert len(bridge.molmo._session_states) == 2
        client.infer({**obs, "_reset": True})
        assert len(bridge.molmo._session_states) == 2
        close(other)
        time.sleep(0.05)
        assert len(bridge.molmo._session_states) == 1
        close(client)
        try:
            client.infer(obs)
        except ConnectionClosed:
            pass
        else:
            raise AssertionError("Closed client unexpectedly reconnected")
        time.sleep(0.05)
        reconnect = WebsocketClientPolicy("127.0.0.1", port)
        assert reconnect.get_server_metadata() == metadata
        reconnect.infer({**obs, "_reset": True})
        close(reconnect)
        invalid = [
            {**obs, "state": np.zeros(15, dtype=np.float32)},
            {**obs, "state": np.full(14, np.nan, dtype=np.float32)},
            {**obs, "head_image": None},
            {**obs, "_rtc": {}},
            {**obs, "left_image": np.zeros((3, 224, 224), dtype=np.uint8)},
        ]
        if args.mode == "mae":
            invalid.append({k: v for k, v in obs.items() if k != TACTILE_KEYS[0]})
        failures = []
        for bad in invalid:
            bad_client = WebsocketClientPolicy("127.0.0.1", port)
            prior = bridge.inference_requests
            try:
                bad_client.infer(bad)
            except RuntimeError as exc:
                failures.append(str(exc))
            else:
                raise AssertionError("Invalid observation accepted")
            assert bridge.inference_requests == prior
            close(bad_client)
        service.shutdown()
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert not bridge.molmo._session_states, bridge.molmo._session_states
    assert sink.actual_action_sends == 0 and sink.hardware_initializations == 0
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / f"{args.mode}_candidates.npz",
        safe_candidates=np.stack([v["action"] for v in sink.records]),
    )
    np.savez_compressed(
        args.output / f"{args.mode}_actions.npz", actions=np.stack(raw_wire_chunks)
    )
    report = {
        "status": "passed",
        "mode": args.mode,
        "load_s": load_s,
        "metadata": metadata,
        "transport": "existing OpenPI WebsocketClientPolicy over real loopback TCP WebSocket",
        "openpi_control_source_sha256": source_hash,
        "control_defaults": defaults,
        "samples": sample_results,
        "median_wire_latency_s": statistics.median(
            r["wire_latency_s"] for r in sample_results
        ),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "native_http_queue_regression": native_http,
        "malformed_input_errors": failures,
        "original_control_boundary_rejections": control_rejections,
        "fake_candidate_times_s": [
            v["at"] - sink.records[0]["at"] for v in sink.records
        ],
        "fake_command_durations_s": [v["duration"] for v in sink.records],
        "disconnect_reconnect": "passed; closed client raises ConnectionClosed; explicit new connection required",
        "two_connection_session_isolation": "passed; reset and close do not remove other session",
        "closed_sessions": bridge.closed_sessions,
        "wire_reset_requests": bridge.reset_requests,
        "client_reset_method": "existing WebsocketClientPolicy.reset is a no-op; broker reset plus _reset tested",
        "fake_sink_candidate_count": len(sink.records),
        "raw_actions_npz": f"{args.mode}_actions.npz",
        "raw_first_action": raw_wire_chunks[0][0].tolist(),
        "actual_action_sends": sink.actual_action_sends,
        "hardware_initializations": sink.hardware_initializations,
        "hardware_import_and_nonloopback_connect_guards": "enabled; violations raise immediately",
        "hardware_main_module_imported": False,
        "held_out_or_robot_evaluation": False,
        "cadence_note": "30Hz fake playback only; synchronous inference adds chunk gaps; no RTC or real-time hardware guarantee",
    }
    (args.output / f"{args.mode}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "mode",
                    "median_wire_latency_s",
                    "peak_allocated_gib",
                    "actual_action_sends",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
