"""Thin Molmo episode/window glue for the unchanged OpenPI main control loop."""

from __future__ import annotations

import ast
import importlib.util
import socket
import sys
import threading
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def build_molmo_observation(*, state, prompt, left_camera, right_camera,
                            base_camera=None, head_camera=None, tactile_cameras=None,
                            rgb_geometry="native"):
    """Snapshot native RGB geometry; the checkpoint owns the only image resize."""
    observation = {"state": np.asarray(state).copy(), "prompt": prompt}
    cameras = {"left_image": left_camera, "right_image": right_camera}
    if base_camera is not None:
        cameras["base_image"] = base_camera
    if head_camera is not None:
        cameras["head_image"] = head_camera
    cameras.update(tactile_cameras or {})
    for key, camera in cameras.items():
        observation[key] = np.asarray(camera.read()).copy()
        if key in {"left_image", "right_image", "head_image", "base_image"} and rgb_geometry == "legacy_224_pad":
            from openpi_client import image_tools
            observation[key] = image_tools.resize_with_pad(observation[key], 224, 224)
    if rgb_geometry not in {"native", "legacy_224_pad"}:
        raise ValueError(f"Unsupported Molmo RGB geometry {rgb_geometry!r}")
    return observation


class EpisodePolicy:
    def __init__(self, client, mode: str):
        self.client = client
        self.metadata = client.get_server_metadata()
        if (
            self.metadata.get("model_type") != "molmoact2"
            or self.metadata.get("model_variant") != mode
            or not isinstance(self.metadata.get("supports_rtc"), bool)
            or self.metadata.get("action_horizon") != 30
            or self.metadata.get("n_action_steps") != 5
        ):
            raise ValueError(
                "Expected selected Molmo 30-step/5-step metadata"
            )
        if self.metadata["supports_rtc"] and self.metadata.get("rtc_protocol") != "molmo-rtc-v1":
            raise ValueError("Unsupported Molmo RTC protocol")
        self._reset_lock = threading.Lock()
        self._reset_generation = 0
        self._ack_generation = -1

    def get_server_metadata(self):
        return self.metadata

    @property
    def pending_reset(self):
        with self._reset_lock:
            return self._reset_generation != self._ack_generation

    def reset(self):
        with self._reset_lock:
            self._reset_generation += 1

    def infer(self, observation):
        with self._reset_lock:
            generation = self._reset_generation
            reset = generation != self._ack_generation or observation.get("_reset", False)
        request = {**observation, "_reset": reset}
        result = self.client.infer(request)
        with self._reset_lock:
            if generation == self._reset_generation:
                self._ack_generation = generation
        return result

    def close(self):
        ws = getattr(self.client, "_ws", None)
        if ws is not None:
            transport = getattr(ws, "socket", None)
            if transport is not None:
                try:
                    transport.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            ws.close()


def _validate_rtc_args(args):
    if not getattr(args, "rtc", False):
        return
    threshold = getattr(args, "rtc_queue_threshold", None)
    horizon = getattr(args, "rtc_execution_horizon", None)
    if (getattr(args, "control_hz", None) != 30 or args.action_start_index != 0
            or not isinstance(threshold, int) or not 1 <= threshold < 30
            or not isinstance(horizon, int) or not threshold <= horizon <= 30):
        raise ValueError("X5 RTC requires 30 Hz, index zero, and 1 <= queue threshold <= execution horizon <= 30")
    if not 0 < args.rtc_max_guidance_weight <= 10:
        raise ValueError("X5 RTC guidance must be in (0, 10]")


def _configure_rtc_defaults(args):
    flags = {arg.split("=", 1)[0] for arg in sys.argv[1:]}
    if args.rtc:
        if "--rtc-queue-threshold" not in flags:
            args.rtc_queue_threshold = 18
        if "--rtc-execution-horizon" not in flags:
            args.rtc_execution_horizon = 20
    return args


def validate_control_args(args):
    if args.policy_transport != "websocket":
        raise ValueError("Molmo control entry requires WebSocket mode")
    _validate_rtc_args(args)
    if args.execute_steps != 5 or args.action_start_index != 0:
        raise ValueError(
            "Molmo control entry requires action window [0:5], never 37 steps"
        )
    if (
        args.enable_motion
        or args.auto_reset_each_episode
        or args.auto_reset_on_startup
        or args.dagger
    ):
        raise ValueError(
            "This entry only supports hardware-mocked dry-run; motion/reset/DAgger forbidden"
        )


def load_mock_main(openpi_root: Path, replacements: dict):
    """Fail closed unless all import and initialization boundaries are pre-mocked."""
    roots = (
        "cv2",
        "pyrealsense2",
        "arx5_interface",
        "can",
        "rospy",
        "rclpy",
        "dmrobotics",
        "pyorbbecsdk",
    )
    if any(not getattr(sys.modules.get(name), "__rxj_mock__", False) for name in roots):
        raise RuntimeError(
            "Install all hardware import traps before loading original main"
        )
    required = {
        "load_bimanual_arm",
        "require_can_interface",
        "RealSenseRgbCamera",
        "OpenCvRgbCamera",
        "OrbbecRgbCamera",
        "DmTactileCamera",
        "_TerminalKeys",
        "_PedalControls",
        "_SdkStderrFilter",
    }
    if not required.issubset(replacements) or any(
        not getattr(replacements[name], "__rxj_mock__", False) for name in required
    ):
        raise RuntimeError(
            "All hardware initialization boundaries must use marked mocks"
        )
    module = _load_episode_main(openpi_root)
    for name, value in replacements.items():
        setattr(module, name, value)
    original_parse = module.parse_args

    def safe_parse(**kwargs):
        args = _configure_rtc_defaults(original_parse(**kwargs))
        validate_control_args(args)
        return args

    module.parse_args = safe_parse
    return module


def _load_episode_main(openpi_root: Path):
    """Load original control body and inject only the per-episode reset hook."""
    path = openpi_root / "examples/x5_client/main.py"
    spec = importlib.util.spec_from_file_location("molmo_mock_openpi_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if __package__:
        from .x5_rtc_client import RTCInferenceEngine
    else:
        from x5_rtc_client import RTCInferenceEngine

    # Only replace this loaded control module's engine reference. No edits or
    # global monkeypatches to the user's OpenPI package/hardware implementation.
    module.rtc_client = SimpleNamespace(RTCInferenceEngine=RTCInferenceEngine)
    module.build_policy_observation = partial(build_molmo_observation, rgb_geometry="legacy_224_pad")
    original = ast.parse(path.read_text())
    main = next(
        n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )

    class EpisodeReset(ast.NodeTransformer):
        count = 0

        def visit_For(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and isinstance(node.iter.func.value, ast.Name)
                and node.iter.func.value.id == "tactile_cameras"
                and any(
                    isinstance(n, ast.Attribute) and n.attr == "reset_history"
                    for n in ast.walk(node)
                )
            ):
                self.count += 1
                reset = ast.Expr(
                    ast.Call(
                        ast.Attribute(
                            ast.Name("client", ast.Load()), "reset", ast.Load()
                        ),
                        [],
                        [],
                    )
                )
                return [ast.copy_location(reset, node), node]
            return node

    injector = EpisodeReset()
    main = injector.visit(main)
    if injector.count != 1:
        raise RuntimeError(
            "Original episode hook changed; refuse unreviewed control source"
        )
    tree = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), main],
        type_ignores=[],
    )
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102 -- one reviewed reset hook in original function
    module.__rxj_episode_reset_hook_count__ = injector.count
    return module


def validate_hardware_control_args(args):
    if args.policy_transport != "websocket":
        raise ValueError("Molmo X5 requires WebSocket transport")
    _validate_rtc_args(args)
    if (
        not 1 <= args.execute_steps <= 30
        or not 0 <= args.action_start_index < 30
        or args.action_start_index + args.execute_steps > 30
    ):
        raise ValueError(
            "Requested execution window must fit the final X5 checkpoint horizon 30"
        )
    if not args.gravity_compensation_setup:
        raise ValueError(
            "Molmo client requires operator-gated gravity-compensation setup"
        )
    if args.dagger:
        raise ValueError("Molmo DAgger takeover is not supported by this entry")


def load_control_main(openpi_root: Path, mode: str):
    """Real hardware entry; original validation and all motion gates stay active."""
    if mode not in ("rgb", "mae"):
        raise ValueError("Select rgb or mae")
    module = _load_episode_main(openpi_root)
    original_parse = module.parse_args

    def parse(**kwargs):
        args = _configure_rtc_defaults(original_parse(**kwargs))
        flags = {arg.split("=", 1)[0] for arg in sys.argv[1:]}
        if "--max-step-delta" not in flags:
            args.max_step_delta = 0.1
        validate_hardware_control_args(args)
        return args

    module.parse_args = parse
    original_main = module.main

    def main():
        from openpi_client import websocket_client_policy

        original_type = websocket_client_policy.WebsocketClientPolicy
        clients = []

        def factory(*args, **kwargs):
            client = original_type(*args, **kwargs)
            clients.append(client)
            policy = EpisodePolicy(client, mode)
            geometry = policy.metadata.get("rgb_geometry", "legacy_224_pad")
            if geometry not in {"native", "legacy_224_pad"}:
                raise ValueError(f"Unsupported server RGB geometry {geometry!r}")
            module.build_policy_observation = partial(build_molmo_observation, rgb_geometry=geometry)
            return policy

        websocket_client_policy.WebsocketClientPolicy = factory
        try:
            original_main()
        finally:
            websocket_client_policy.WebsocketClientPolicy = original_type
            for client in clients:
                ws = getattr(client, "_ws", None)
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        module.LOGGER.exception(
                            "Failed to close Molmo policy connection"
                        )

    module.main = main
    return module
