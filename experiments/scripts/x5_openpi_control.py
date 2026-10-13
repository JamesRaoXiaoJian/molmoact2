"""Thin Molmo episode/window glue for the unchanged OpenPI main control loop."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


class EpisodePolicy:
    def __init__(self, client, mode: str):
        self.client = client
        self.metadata = client.get_server_metadata()
        if (
            self.metadata.get("model_type") != "molmoact2"
            or self.metadata.get("model_variant") != mode
            or self.metadata.get("supports_rtc") is not False
            or self.metadata.get("action_horizon") != 30
            or self.metadata.get("n_action_steps") != 5
        ):
            raise ValueError(
                "Expected selected Molmo synchronous 30-step/5-step metadata"
            )
        self.pending_reset = True

    def get_server_metadata(self):
        return self.metadata

    def reset(self):
        self.pending_reset = True

    def infer(self, observation):
        request = {**observation, "_reset": self.pending_reset}
        result = self.client.infer(request)
        self.pending_reset = False
        return result


def validate_control_args(args):
    if args.rtc or args.policy_transport != "websocket":
        raise ValueError("Molmo control entry requires synchronous WebSocket mode")
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
        args = original_parse(**kwargs)
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
    if args.rtc or args.policy_transport != "websocket":
        raise ValueError(
            "This X5 bridge currently supports synchronous WebSocket; upstream Molmo async/RTC requires a different adapter"
        )
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
        args = original_parse(**kwargs)
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
            return EpisodePolicy(client, mode)

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
