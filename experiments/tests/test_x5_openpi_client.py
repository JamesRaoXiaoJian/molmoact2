"""Targeted production-entry tests with all hardware boundaries pre-mocked."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments/scripts"), str(ROOT / "deployment")]
import run_x5_openpi_client as entry
from openpi_main_smoke import Scenario, install_import_traps
from serve_x5_openpi import OpenPIBridge
from x5_openpi_control import load_control_main, load_mock_main

OPENPI = Path(os.environ.get("OPENPI_ROOT", Path.home() / "lwm_ws/openpi-x5-arx5-beta/openpi"))
DEVICE_ARGS = [
    "--left-camera=mock-left",
    "--right-camera=mock-right",
    "--head-camera=mock-head",
    "--left-gripper-open-readout=-3.5",
    "--right-gripper-open-readout=-2.49695",
]
CONTROL_ARGS = [
    "--policy-transport=websocket",
    "--execute-steps=5",
    "--action-start-index=0",
    "--control-hz=30",
    "--max-chunks=0",
    "--gravity-compensation-setup",
    "--episode-control=pedal",
    "--no-auto-reset-on-startup",
    "--no-auto-reset-each-episode",
    "--startup-settle-s=0.01",
    "--startup-gripper-valid-frames=1",
    "--startup-gripper-timeout-s=0.2",
]


def prepare():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(OPENPI / "packages/openpi-client/src"))
    install_import_traps(OPENPI)


def samples():
    manifest = json.loads((ROOT / "data/offline_samples/manifest.json").read_text())
    out = []
    for item in manifest["samples"]:
        with np.load(ROOT / "data/offline_samples" / item["file"]) as data:
            sample = {
                k: data[k].copy() for k in data.files if k.startswith("observation.")
            }
        sample["task"] = item["task"]
        out.append(sample)
    return out


def test_help_and_device_parse(monkeypatch):
    with pytest.raises(SystemExit) as error:
        entry.parse_entry(["--help"])
    assert error.value.code == 0
    with pytest.raises(SystemExit):
        entry.parse_entry(["--mode=rgb"])
    with pytest.raises(SystemExit):
        entry.parse_entry(["--mode=mae", *DEVICE_ARGS])
    args, remaining = entry.parse_entry(["--mode=rgb", *DEVICE_ARGS, "--prompt=task"])
    assert args.mode == "rgb" and "--prompt=task" in remaining


@pytest.mark.parametrize(
    "extra,allowed",
    [
        ([], True),
        (["--enable-motion"], False),
        (["--enable-motion", "--confirm-motion=WRONG"], False),
        (["--enable-motion", "--confirm-motion=MOVE_X5"], True),
        (["--rtc"], False),
        (["--execute-steps=37"], False),
        (["--action-start-index=1"], True),
        (["--auto-reset-on-startup"], False),
    ],
)
def test_original_motion_gates(monkeypatch, extra, allowed):
    prepare()
    module = load_control_main(OPENPI, "rgb")
    monkeypatch.setattr(sys, "argv", ["client", *DEVICE_ARGS, *CONTROL_ARGS, *extra])
    if not allowed:
        with pytest.raises(ValueError):
            args = module.parse_args()
            module.validate_args(args)
    else:
        args = module.parse_args()
        module.validate_args(args)
        assert not args.auto_reset_on_startup and not args.auto_reset_each_episode
        assert (
            args.execute_steps == 5
            and args.action_start_index in (0, 1)
            and not args.rtc
        )
        assert args.reject_first_joint_delta == 0.25 and args.max_step_delta == 0.02
        assert args.max_gripper_step_delta == 0.1


@pytest.mark.parametrize("mode", ["rgb", "mae"])
@pytest.mark.parametrize("kind", ["normal", "late_pause", "error"])
def test_formal_cli_original_main_lifecycle(monkeypatch, mode, kind):
    prepare()
    from openpi_client import websocket_client_policy

    context = Scenario(kind, samples(), SimpleNamespace(mode=mode), mode)
    sockets = []

    class Socket:
        closed = False

        def close(self):
            self.closed = True

    class FakePolicy:
        def __init__(self, **kwargs):
            self._ws = Socket()
            sockets.append(self._ws)

        def get_server_metadata(self):
            return OpenPIBridge(None, mode).metadata

        def infer(self, observation):
            if kind == "error" and len(context.requests) == 1:
                context.failed = True
                raise RuntimeError("mock inference failure")
            if kind == "late_pause" and context.episode == 1:
                context.pending_pause = True
            context.requests.append(
                {"reset": observation["_reset"], "prompt": observation["prompt"]}
            )
            actions = np.repeat(observation["state"][None], 30, axis=0).astype(
                np.float32
            )
            return {"actions": actions}

    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", FakePolicy)
    import x5_openpi_control

    original_loader = x5_openpi_control.load_control_main

    def mocked_loader(root, selected):
        module = original_loader(root, selected)
        for name, value in context.replacements().items():
            setattr(module, name, value)
        original_clip = module.clip_action

        def clip(*args, **kwargs):
            action = original_clip(*args, **kwargs)
            context.candidates.append(
                {"episode": context.episode, "action": action.copy()}
            )
            return action

        module.clip_action = clip
        return module

    monkeypatch.setattr(x5_openpi_control, "load_control_main", mocked_loader)
    argv = [
        f"--mode={mode}",
        f"--openpi-root={OPENPI}",
        *DEVICE_ARGS,
        *CONTROL_ARGS,
        "--prompt=authorized offline task",
    ]
    if mode == "mae":
        argv += ["--tactile-serials", "mock-touch-left", "mock-touch-right"]
    handlers = {
        s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP)
    }
    try:
        if kind == "error":
            with pytest.raises(RuntimeError, match="mock inference failure"):
                entry.main(argv)
        else:
            entry.main(argv)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    assert context.robot_closed and context.controls_stopped
    assert all(c.started and c.stopped for c in context.cameras)
    assert sockets and all(s.closed for s in sockets)
    assert all(r["reset"] for r in context.requests)
    assert all(r["prompt"] == "authorized offline task" for r in context.requests)
    assert len(context.candidates) == (7 if kind == "normal" else 5)
    if kind != "error":
        assert [c["episode"] for c in context.candidates][-5:] == [2] * 5
        assert len(context.requests) == 2
    assert context.actual_sends == context.real_hardware_initializations == 0
    assert all(np.isfinite(c["action"]).all() for c in context.candidates)
    import torch

    assert not torch.cuda.is_initialized()


def test_original_mock_loader_still_rejects_motion(monkeypatch):
    prepare()
    context = Scenario("normal", samples(), SimpleNamespace(mode="rgb"), "rgb")
    module = load_mock_main(OPENPI, context.replacements())
    monkeypatch.setattr(
        sys,
        "argv",
        ["mock", *CONTROL_ARGS, "--enable-motion", "--confirm-motion=MOVE_X5"],
    )
    with pytest.raises(ValueError, match="hardware-mocked"):
        module.parse_args()
