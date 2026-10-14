"""Logging preserves control calls and does not require robot modules."""

import json
import logging
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from x5_client_logging import ClientRunLog, _NonblockingHandler


def traces(path):
    return [
        json.loads(line.split("control_trace ", 1)[1])
        for line in path.read_text().splitlines()
        if "control_trace {" in line
    ]


def test_control_calls_unchanged_and_logs_flushed_on_exception(tmp_path):
    target, current, safe = (np.full(14, x, np.float32) for x in (0.3, 0.1, 0.12))
    sentinel, robot, calls = object(), object(), []

    def original_clip(t, c, *args, **kwargs):
        assert t is target and c is current
        calls.append(("clip", args, kwargs))
        return safe

    def original_send(r, a, duration):
        assert r is robot and a is safe
        calls.append(("send", duration))
        return sentinel

    module = SimpleNamespace(
        clip_action=original_clip,
        send_native_action=original_send,
        calculate_control_metrics=lambda *a: {"joint_clip_count": 12},
    )
    root = logging.getLogger()
    before = (root.level, list(root.handlers))
    with (
        pytest.raises(RuntimeError, match="diagnostic failure"),
        ClientRunLog("rgb", ["--rtc", "--max-step-delta=0.02"], tmp_path) as run,
    ):
        run.attach_control(module)
        assert module.clip_action(target, current, 0.02, 0.1, -3.4, 0) is safe
        assert module.send_native_action(robot, safe, duration=0.8 / 30) is sentinel
        # An unsent candidate must never be reported as an SDK send.
        module.clip_action(target, current, 0.02, 0.1, -3.4, 0)
        raise RuntimeError("diagnostic failure")
    assert (
        module.clip_action is original_clip
        and module.send_native_action is original_send
    )
    assert (root.level, list(root.handlers)) == before
    records = traces(run.path)
    assert [r["event"] for r in records] == ["candidate", "sent", "candidate"]
    assert [r["seq"] for r in records] == [1, 1, 2]
    assert records[0]["adjusted_target"] == target.tolist()
    assert records[0]["current"] == current.tolist()
    assert records[0]["safe_target"] == safe.tolist()
    assert records[0]["joint_clip_count"] == 12
    assert records[1]["command_duration_s"] == 0.8 / 30
    assert records[1]["sdk_call_ms"] >= 0 and records[2]["interval_ms"] >= 0
    text = run.path.read_text()
    assert '"--rtc"' in text and "Traceback" in text and "diagnostic failure" in text
    assert text.splitlines()[-1].endswith("dropped_records=0")
    assert len(calls) == 3 and not run.listener._thread


def test_failed_send_not_recorded_as_sent(tmp_path):
    def fail(*a, **kw):
        raise RuntimeError("SDK rejected command")

    module = SimpleNamespace(clip_action=lambda *a: None, send_native_action=fail)
    with (
        pytest.raises(RuntimeError, match="SDK rejected"),
        ClientRunLog("mae", [], tmp_path) as run,
    ):
        run.attach_control(module)
        module.send_native_action(None, np.zeros(14), duration=0.02)
    assert not traces(run.path)
    assert "SDK rejected command" in run.path.read_text()


def test_full_queue_drops_instead_of_blocking():
    records = queue.Queue(maxsize=1)
    handler = _NonblockingHandler(records)
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "one", (), None)
    handler.handle(record)
    handler.handle(record)
    assert records.qsize() == 1 and handler.dropped == 1
    handler.close()


def test_per_run_paths_and_log_directory_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLMO_CLIENT_LOG_DIR", str(tmp_path))
    with ClientRunLog("rgb", []) as first:
        logging.getLogger("test.client").info("first run")
    with ClientRunLog("rgb", []) as second:
        logging.getLogger("test.client").info("second run")
    assert first.path != second.path
    assert first.path.parent == second.path.parent == tmp_path
    assert (
        "first run" in first.path.read_text()
        and "second run" not in first.path.read_text()
    )
    assert "second run" in second.path.read_text()


def test_unwritable_log_location_fails_before_control_import(tmp_path, monkeypatch):
    import run_x5_openpi_client as entry

    bad = tmp_path / "file"
    bad.write_text("not a directory")
    monkeypatch.setenv("MOLMO_CLIENT_LOG_DIR", str(bad))
    monkeypatch.setattr(
        entry, "_run_control", lambda *a: pytest.fail("Control must not load")
    )
    with pytest.raises(FileExistsError):
        entry.main(
            [
                "--mode=rgb",
                "--left-camera=mock",
                "--right-camera=mock",
                "--head-camera=mock",
                "--left-gripper-open-readout=-3.5",
                "--right-gripper-open-readout=-2.5",
            ]
        )
