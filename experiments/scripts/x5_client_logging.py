"""Per-run Python logs and control traces; never initializes hardware."""

from __future__ import annotations

import json
import logging
import os
import queue
import time
from datetime import UTC, datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path

LOGGER = logging.getLogger(__name__)
TRACE = logging.getLogger("molmo.control")


class _NonblockingHandler(QueueHandler):
    def __init__(self, records):
        super().__init__(records)
        self.dropped = 0

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # A slow disk must not stall the control loop or grow memory forever.
            self.dropped += 1


class _DrainListener(QueueListener):
    def enqueue_sentinel(self):
        # Shutdown happens after original control cleanup, outside the action loop.
        self.queue.put(self._sentinel)


class ClientRunLog:
    def __init__(self, mode, argv, directory=None):
        directory = Path(
            directory
            or os.environ.get("MOLMO_CLIENT_LOG_DIR")
            or Path(__file__).resolve().parents[2] / "logs"
        ).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S-%f")
        self.path = (
            directory.resolve() / f"molmo-{mode}-client-{stamp}-{os.getpid()}.log"
        )
        # Fail before loading hardware if the requested log cannot be opened.
        self.file = logging.FileHandler(self.path, mode="x", encoding="utf-8")
        self.file.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        self.handler = _NonblockingHandler(queue.Queue(maxsize=8192))
        self.listener = _DrainListener(self.handler.queue, self.file)
        self.mode, self.argv = mode, list(argv)
        self._restores = []

    def __enter__(self):
        root = logging.getLogger()
        self._root_level = root.level
        self._trace_state = TRACE.level, TRACE.propagate, list(TRACE.handlers)
        root.setLevel(min(root.level, logging.INFO))
        root.addHandler(self.handler)
        TRACE.setLevel(logging.INFO)
        TRACE.propagate = False
        TRACE.handlers = [self.handler]
        self.listener.start()
        LOGGER.info("Client log: %s", self.path)
        LOGGER.info(
            "client_start mode=%s cwd=%s argv=%s",
            self.mode,
            Path.cwd(),
            json.dumps(self.argv, ensure_ascii=False),
        )
        return self

    def attach_control(self, module):
        """Observe original inputs/results without changing commands or limits."""
        original_clip, original_send = module.clip_action, module.send_native_action
        sequence, pending, last_clip, last_send = 0, None, None, None

        def clip(target, current, *args, **kwargs):
            nonlocal sequence, pending, last_clip
            safe = original_clip(target, current, *args, **kwargs)
            now = time.monotonic_ns()
            sequence += 1
            pending = sequence
            payload = {
                "event": "candidate",
                "seq": sequence,
                "monotonic_ns": now,
                "interval_ms": None if last_clip is None else (now - last_clip) / 1e6,
                "current": current.tolist(),
                "adjusted_target": target.tolist(),
                "safe_target": safe.tolist(),
                **module.calculate_control_metrics(target, current, safe),
            }
            last_clip = now
            TRACE.info("control_trace %s", json.dumps(payload, separators=(",", ":")))
            return safe

        def send(robot, action, duration):
            nonlocal pending, last_send
            started = time.monotonic_ns()
            result = original_send(robot, action, duration=duration)
            finished = time.monotonic_ns()
            payload = {
                "event": "sent",
                "seq": pending,
                "monotonic_ns": finished,
                "interval_ms": None
                if last_send is None
                else (finished - last_send) / 1e6,
                "sdk_call_ms": (finished - started) / 1e6,
                "command_duration_s": duration,
                "safe_target": action.tolist(),
            }
            pending, last_send = None, finished
            TRACE.info("control_trace %s", json.dumps(payload, separators=(",", ":")))
            return result

        module.clip_action, module.send_native_action = clip, send
        self._restores.append((module, original_clip, original_send))

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            LOGGER.error(
                "Client exited with %s",
                kind.__name__,
                exc_info=(kind, value, traceback),
            )
        else:
            LOGGER.info("Client exited normally")
        for module, clip, send in reversed(self._restores):
            module.clip_action, module.send_native_action = clip, send
        root = logging.getLogger()
        root.removeHandler(self.handler)
        root.setLevel(self._root_level)
        TRACE.level, TRACE.propagate, TRACE.handlers = self._trace_state
        self.listener.stop()
        # This last record is emitted after draining, even if the queue was full.
        record = logging.LogRecord(
            __name__,
            logging.INFO,
            __file__,
            0,
            "client_log_closed dropped_records=%d",
            (self.handler.dropped,),
            None,
        )
        self.file.handle(record)
        if self.handler.dropped:
            LOGGER.warning(
                "Client log dropped %d records: %s", self.handler.dropped, self.path
            )
        self.handler.close()
        self.file.close()
        return False
