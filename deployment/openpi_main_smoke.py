"""Run the original OpenPI main with import traps and simulated hardware only."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "experiments"),
    str(ROOT / "experiments/lerobot/src"),
    str(ROOT / "deployment"),
]
from openpi_loopback_smoke import HardwareImportGuard, audit_guard
from scripts.run_x5_inference_comparison import image_payload, infer
from scripts.serve_x5_openpi import RGB_KEYS, TACTILE_KEYS, build_bridge
from scripts.x5_openpi_control import EpisodePolicy, load_mock_main


class Trap(ModuleType):
    __rxj_mock__ = True

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"Real hardware access forbidden: {self.__name__}.{name}")


def marked(value):
    value.__rxj_mock__ = True
    return value


def install_import_traps(openpi):
    for name in (
        "cv2",
        "pyrealsense2",
        "arx5_interface",
        "can",
        "rospy",
        "rclpy",
        "dmrobotics",
        "pyorbbecsdk",
    ):
        sys.modules[name] = Trap(name)
    for name, path in [
        ("examples", openpi / "examples"),
        ("examples.x5_client", openpi / "examples/x5_client"),
        ("openpi", openpi / "src/openpi"),
        ("openpi.shared", openpi / "src/openpi/shared"),
    ]:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    rtc = Trap("openpi.rtc_client")
    sys.modules[rtc.__name__] = rtc
    tactile = ModuleType("openpi.shared.tactile")
    tactile.TACTILE_KEYS = tuple(
        f"{arm}_gripper_{finger}_tactile"
        for arm in ("left", "right")
        for finger in ("left", "right")
    )
    sys.modules[tactile.__name__] = tactile
    for suffix, attribute in [
        ("dagger", "DaggerControl"),
        ("episodic_recorder", "EpisodicRecorder"),
        ("rerun_visualizer", "RerunVisualizer"),
        ("orbbec_camera", "OrbbecRgbCamera"),
        ("tactile_camera", "DmTactileCamera"),
    ]:
        module = ModuleType(f"examples.x5_client.{suffix}")
        setattr(
            module,
            attribute,
            marked(
                lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("Unexpected optional hardware path")
                )
            ),
        )
        sys.modules[module.__name__] = module


class Scenario:
    def __init__(self, kind, samples, bridge, mode):
        self.kind, self.samples, self.bridge, self.mode = kind, samples, bridge, mode
        self.episode = 0
        self.paused = False
        self.pending_pause = False
        self.failed = False
        self.candidates = []
        self.requests = []
        self.events = []
        self.cameras = []
        self.robot_closed = False
        self.controls_stopped = False
        self.actual_sends = 0
        self.real_hardware_initializations = 0

    @property
    def sample(self):
        return self.samples[min(max(self.episode - 1, 0), 1)]

    def replacements(self):
        context = self

        @marked
        class Robot:
            def __init__(self, *args, **kwargs):
                context.events.append("mock robot created")

            def get_joint_positions(self):
                state = context.sample["observation.state"]
                return {"left": state[:7].copy(), "right": state[7:].copy()}

            def check_faults(self):
                return None

            def gravity_compensation(self):
                context.events.append("mock gravity")
                return True

            def set_joint_positions(self, *a, **k):
                raise AssertionError("Motor command path forbidden")

            def set_gripper_pos(self, *a, **k):
                raise AssertionError("Gripper command path forbidden")

            def close(self):
                context.robot_closed = True
                context.events.append("mock robot closed")

        @marked
        class Camera:
            def __init__(self, serial, *args, **kwargs):
                self.serial = serial
                self.started = False
                self.stopped = False
                context.cameras.append(self)

            def start(self):
                self.started = True

            def read(self):
                key = {
                    "mock-left": "left_d405",
                    "mock-right": "right_d405",
                    "mock-head": "head_camera",
                    "mock-touch-left": TACTILE_KEYS[0],
                    "mock-touch-right": TACTILE_KEYS[1],
                }[self.serial]
                return context.sample[f"observation.images.{key}"].copy()

            def reset_history(self):
                context.events.append("mock tactile history reset")

            def stop(self):
                self.stopped = True

        @marked
        class Controls:
            def __init__(self, *args, **kwargs):
                self.calls = 0

            def start(self):
                context.events.append("mock pedal started")

            def stop(self):
                context.controls_stopped = True

            def flush(self):
                pass

            def poll(self, timeout=0.0):
                self.calls += 1
                if self.calls > 100:
                    raise AssertionError("Mock loop exceeded bounded event script")
                if timeout:
                    if context.failed:
                        return "quit"
                    if context.episode == 0 or (
                        context.episode == 1 and context.paused
                    ):
                        context.episode += 1
                        context.events.append(
                            "start" if context.episode == 1 else "resume"
                        )
                        return "toggle_episode"
                    return "quit"
                if (
                    context.kind == "late_pause"
                    and context.pending_pause
                    and not context.paused
                ):
                    context.paused = True
                    context.events.append("late pause")
                    return "toggle_episode"
                if (
                    context.kind == "normal"
                    and len(context.candidates) == 2
                    and not context.paused
                ):
                    context.paused = True
                    context.events.append("pause mid-window")
                    return "toggle_episode"
                if (
                    context.episode == 2
                    and sum(x["episode"] == 2 for x in context.candidates) >= 5
                ):
                    context.events.append("normal exit")
                    return "quit"
                return None

        @marked
        class Filter:
            def __init__(self, *a, **k):
                pass

            def start(self):
                pass

            def stop(self):
                pass

        return {
            "load_bimanual_arm": marked(lambda: Robot),
            "require_can_interface": marked(lambda name: None),
            "RealSenseRgbCamera": Camera,
            "OpenCvRgbCamera": Camera,
            "OrbbecRgbCamera": Camera,
            "DmTactileCamera": Camera,
            "_TerminalKeys": Controls,
            "_PedalControls": Controls,
            "_SdkStderrFilter": Filter,
        }


def fingerprint(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, Image.Image):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        return [
            str(value.dtype),
            list(value.shape),
            hashlib.sha256(value.tobytes()).hexdigest(),
        ]
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [fingerprint(v) for v in value]
    return value


def test_history_and_queue_isolation(bridge, samples):
    policy = bridge.molmo
    references = []
    payloads = []
    for index, sample in enumerate(samples):
        obs = {k: v for k, v in sample.items() if k.startswith("observation.")}
        if bridge.mode == "rgb":
            obs = {k: v for k, v in obs.items() if "tactile" not in k}
        obs["task"] = sample["task"]
        expected, _ = infer(policy.policy, obs, 42 + index * 12, 10)
        references.append(expected)
        image_keys = [
            *RGB_KEYS.values(),
            *(TACTILE_KEYS if bridge.mode == "mae" else ()),
        ]
        p = {k: image_payload(obs[f"observation.images.{k}"]) for k in image_keys}
        p.update(
            state=obs["observation.state"].tolist(),
            instruction=obs["task"],
            session_id=f"isolated-{index}",
            single_action=True,
        )
        payloads.append(p)
    with TestClient(policy.app) as client:
        for index, payload in enumerate(payloads):
            torch.manual_seed(42 + index * 12)
            r = client.post("/act", json=payload)
            r.raise_for_status()
            np.testing.assert_allclose(
                r.json()["action"], references[index][0], rtol=1e-5, atol=1e-5
            )
            if index == 0:
                first = fingerprint(policy._session_states["isolated-0"].__dict__)
        assert fingerprint(policy._session_states["isolated-0"].__dict__) == first
        second = fingerprint(policy._session_states["isolated-1"].__dict__)
        queues = policy._session_states["isolated-0"].action_queues
        assert sum(len(v) for v in queues.values()) == 4
        queue_rows = [
            v.detach().cpu().numpy() for queue in queues.values() for v in queue
        ]
        assert all(row.shape == (1, 14) for row in queue_rows)
        queued = np.concatenate(queue_rows, axis=0)
        np.testing.assert_allclose(queued, references[0][1:5], rtol=1e-5, atol=1e-5)
        assert (
            sum(
                len(v)
                for v in policy._session_states["isolated-0"].obs_history.values()
            )
            == 1
        )
        r = client.post("/act", json=payloads[0])
        r.raise_for_status()
        assert r.json()["inference_calls"] == 0
        np.testing.assert_allclose(
            r.json()["action"], references[0][1], rtol=1e-5, atol=1e-5
        )
        assert fingerprint(policy._session_states["isolated-1"].__dict__) == second
        policy.reset_session({"session_id": "isolated-0"})
        assert fingerprint(policy._session_states["isolated-1"].__dict__) == second
        for row in (1, 2):
            r = client.post("/act", json=payloads[1])
            r.raise_for_status()
            assert r.json()["inference_calls"] == 0
            np.testing.assert_allclose(
                r.json()["action"], references[1][row], rtol=1e-5, atol=1e-5
            )
        policy.reset_session({"session_id": "isolated-1"})
    return {
        "status": "passed",
        "history_and_queue_content_unchanged_by_other_session": True,
        "initial_remaining_queue_rows": 4,
        "queue_rows_match_native_reference": True,
        "captured_history_frames": 1,
        "other_session_reset_preserves_queue_and_outputs": True,
    }


def run_scenario(kind, samples, bridge, openpi, port):
    from openpi_client import websocket_client_policy

    original_client = websocket_client_policy.WebsocketClientPolicy
    context = Scenario(kind, samples, bridge, bridge.mode)
    module = load_mock_main(openpi, context.replacements())
    original_clip = module.clip_action

    def clip(*args, **kwargs):
        result = original_clip(*args, **kwargs)
        context.candidates.append({"episode": context.episode, "action": result.copy()})
        return result

    module.clip_action = clip

    class RecordingPolicy(EpisodePolicy):
        def infer(self, obs):
            if kind == "error" and len(context.requests) == 1:
                context.failed = True
                obs = {**obs, "state": np.zeros(15, dtype=np.float32)}
            if kind == "late_pause" and context.episode == 1:
                threading.Timer(
                    0.05, lambda: setattr(context, "pending_pause", True)
                ).start()
            torch.manual_seed(42 + 12 * (context.episode - 1))
            started = time.monotonic()
            reset = self.pending_reset
            result = super().infer(obs)
            context.requests.append(
                {
                    "observation": obs,
                    "reset": reset,
                    "actions": result["actions"].copy(),
                    "latency_s": time.monotonic() - started,
                }
            )
            return result

    wrappers = []

    def factory(*args, **kwargs):
        wrapper = RecordingPolicy(original_client(*args, **kwargs), bridge.mode)
        wrappers.append(wrapper)
        return wrapper

    websocket_client_policy.WebsocketClientPolicy = factory
    argv = sys.argv
    handlers = {
        s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP)
    }
    sys.argv = [
        "molmo-mocked-control",
        "--host=127.0.0.1",
        f"--port={port}",
        "--policy-transport=websocket",
        "--execute-steps=5",
        "--action-start-index=0",
        "--control-hz=30",
        "--max-chunks=0",
        "--gravity-compensation-setup",
        "--episode-control=pedal",
        "--left-camera=mock-left",
        "--right-camera=mock-right",
        "--head-camera=mock-head",
        "--startup-settle-s=0.01",
        "--startup-gripper-valid-frames=1",
        "--startup-gripper-timeout-s=0.2",
        "--prompt=" + samples[0]["task"],
    ]
    if bridge.mode == "mae":
        sys.argv.extend(["--tactile-serials", "mock-touch-left", "mock-touch-right"])
    error = None
    try:
        module.main()
    except RuntimeError as exc:
        if kind != "error":
            raise
        error = str(exc)
    finally:
        sys.argv = argv
        websocket_client_policy.WebsocketClientPolicy = original_client
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for wrapper in wrappers:
            wrapper.client._ws.close()
    assert (
        context.robot_closed
        and context.controls_stopped
        and all(c.started and c.stopped for c in context.cameras)
    )
    assert context.actual_sends == 0 and context.real_hardware_initializations == 0
    if kind == "error":
        assert error and len(context.candidates) == 5
    else:
        assert [r["reset"] for r in context.requests] == [True, True]
        assert sum(c["episode"] == 1 for c in context.candidates) == (
            2 if kind == "normal" else 0
        )
        assert sum(c["episode"] == 2 for c in context.candidates) == 5
    deltas = []
    for index, request in enumerate(context.requests):
        obs = request["observation"]
        native = {
            f"observation.images.{dest}": obs[src] for src, dest in RGB_KEYS.items()
        }
        native.update(
            {f"observation.images.{k}": obs[k] for k in TACTILE_KEYS if k in obs}
        )
        native.update({"observation.state": obs["state"], "task": obs["prompt"]})
        expected, _ = infer(bridge.molmo.policy, native, 42 + index * 12, 10)
        np.testing.assert_allclose(request["actions"], expected, rtol=1e-5, atol=1e-5)
        deltas.append(float(np.abs(request["actions"] - expected).max()))
    return {
        "case": kind,
        "status": "passed",
        "episode_requests": len(context.requests),
        "wire_resets": [r["reset"] for r in context.requests],
        "candidate_count": len(context.candidates),
        "candidate_episodes": [c["episode"] for c in context.candidates],
        "latencies_s": [r["latency_s"] for r in context.requests],
        "native_output_max_abs_deltas": deltas,
        "events": context.events,
        "expected_error": error,
        "robot_closed": context.robot_closed,
        "controls_stopped": context.controls_stopped,
        "all_mock_cameras_stopped": True,
        "actual_action_sends": 0,
        "real_hardware_initializations": 0,
        "original_main_reset_hook_count": module.__rxj_episode_reset_hook_count__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rgb", "mae"), required=True)
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=Path.home() / "lwm_ws/openpi-x5-arx5-beta/openpi",
    )
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, HardwareImportGuard())
    sys.addaudithook(audit_guard)
    sys.path.insert(0, str(args.openpi_root / "packages/openpi-client/src"))
    manifest = json.loads((ROOT / "data/offline_samples/manifest.json").read_text())
    samples = []
    for item in manifest["samples"]:
        with np.load(ROOT / "data/offline_samples" / item["file"]) as data:
            sample = {
                k: data[k].copy() for k in data.files if k.startswith("observation.")
            }
        sample["task"] = item["task"]
        samples.append(sample)
    bridge = build_bridge(args.mode)
    install_import_traps(args.openpi_root)
    with bridge.create_server() as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        cases = [
            run_scenario(
                kind, samples, bridge, args.openpi_root, server.socket.getsockname()[1]
            )
            for kind in ("normal", "late_pause", "error")
        ]
        server.shutdown()
        worker.join(5)
        assert not worker.is_alive()
    isolation = test_history_and_queue_isolation(bridge, samples)
    assert not bridge.molmo._session_states
    report = {
        "status": "passed",
        "mode": args.mode,
        "original_main_executed": True,
        "openpi_files_modified": False,
        "cases": cases,
        "history_queue_isolation": isolation,
        "actual_action_sends": 0,
        "real_hardware_initializations": 0,
        "control_window": [0, 5],
        "rtc": False,
        "model_type": "molmoact2",
        "compiled_blocks": bridge.metadata.get("native_action_blocks_compiled", False),
        "original_main_sha256": hashlib.sha256(
            (args.openpi_root / "examples/x5_client/main.py").read_bytes()
        ).hexdigest(),
        "scope": "Whole original main control loop with one in-memory episode reset hook; SDK/cameras/pedal/robot/optional modules mocked before import/init. No real hardware lifecycle or robot performance evaluation.",
    }
    output = ROOT / "outputs/inference/openpi_bridge"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.mode}_main.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": "passed",
                "full_original_main": True,
                "cases": [c["case"] for c in cases],
                "actual_action_sends": 0,
                "compiled_blocks": report["compiled_blocks"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
