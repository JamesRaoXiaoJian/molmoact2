"""NumPy-only RTC worker for the existing OpenPI X5 control-loop interface.

Adapted from OpenPI X5 rtc_client and LeRobot ActionQueue (Apache-2.0).
The worker owns the sole in-flight RPC. The control thread alone consumes
actions. Raw model targets are retained separately from calibrated targets.
"""
from __future__ import annotations

from collections import deque
import copy
import logging
import math
import threading
import time

import numpy as np

LOGGER = logging.getLogger(__name__)


class RTCActionQueue:
    def __init__(self, *, action_horizon, action_dim, control_hz, clock=time.monotonic):
        self.action_horizon, self.action_dim = action_horizon, action_dim
        self.hz, self.clock = control_hz, clock
        self._lock = threading.Lock()
        self._raw = self._processed = None
        self._index = self._revision = 0
        self._origin = 0.0
        self.expired_steps = 0

    def clear(self):
        with self._lock:
            self._raw = self._processed = None
            self._index = 0
            self._revision += 1

    def _expire(self):
        if self._raw is not None:
            due = max(0, int(math.floor((self.clock() - self._origin) * self.hz + 1e-7)))
            index = min(len(self._raw), max(self._index, due))
            self.expired_steps += index - self._index
            self._index = index

    def qsize(self):
        with self._lock:
            self._expire()
            return 0 if self._raw is None else max(0, len(self._raw) - self._index)

    @property
    def revision(self):
        with self._lock:
            return self._revision

    def get_with_revision(self):
        with self._lock:
            self._expire()
            if self._raw is None or self._index >= len(self._raw):
                return None, self._revision
            result = self._processed[self._index].copy()
            self._index += 1
            return result, self._revision

    def snapshot_prefix(self, execution_horizon):
        with self._lock:
            self._expire()
            prefix = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
            if self._raw is None:
                return prefix, 0
            valid = min(execution_horizon, len(self._raw) - self._index)
            prefix[:valid] = self._raw[self._index:self._index + valid]
            return prefix, valid

    def replace(self, raw, processed, *, origin, delay):
        expected = (self.action_horizon, self.action_dim)
        for array in (raw, processed):
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"RTC requires finite actions shaped {expected}")
        if not 0 <= delay < self.action_horizon:
            raise RuntimeError("RTC result is older than its entire action horizon")
        with self._lock:
            self._raw, self._processed = raw.copy(), processed.copy()
            self._origin, self._index = origin, delay
            self._revision += 1


class RTCInferenceEngine:
    """Drop-in engine for OpenPI main; it never initializes or commands hardware."""
    def __init__(self, client, *, control_hz, action_horizon, action_dim,
                 execution_horizon=20, queue_threshold=18, max_guidance_weight=10.0,
                 initial_action_skip=0, action_transform=None, clock=None,
                 initial_latency_s=0.45):
        if control_hz != 30 or action_horizon != 30 or action_dim != 14:
            raise ValueError("X5 RTC requires the checkpoint's 30 Hz, 30-frame, 14D contract")
        if not 0 < queue_threshold < action_horizon or not 1 <= execution_horizon <= action_horizon:
            raise ValueError("Invalid RTC queue threshold or execution horizon")
        if initial_action_skip != 0:
            raise ValueError("RTC starts at action index zero; stale frames are removed by measured delay")
        if not math.isfinite(max_guidance_weight) or not 0 < max_guidance_weight <= 10:
            raise ValueError("RTC guidance must be in (0, 10]")
        self.client = client
        self._clock = clock or time.monotonic
        self._hz, self._horizon = control_hz, action_horizon
        self._execution_horizon, self._threshold = execution_horizon, queue_threshold
        self._guidance, self._transform = max_guidance_weight, action_transform
        self._initial_latency = initial_latency_s
        self.queue = RTCActionQueue(action_horizon=action_horizon, action_dim=action_dim,
                                    control_hz=control_hz, clock=self._clock)
        self._condition = threading.Condition()
        self._active = self._busy = self._closed = self._has_chunk = False
        self._generation = self._count = 0
        self._busy_since = 0.0
        self._warmup_attempts = 0
        self._request = self._error = None
        self._latencies = deque(maxlen=64)
        self._metrics = {"discarded_responses": 0, "underflows": 0, "failures": 0, "budget_misses": 0,
                         "warmup_discards": 0,
                         "dropped_prefix_frames": 0, "initial_request_ms": []}
        self._thread = threading.Thread(target=self._loop, name="molmo-rtc-inference", daemon=True)
        self._thread.start()

    @property
    def busy(self):
        with self._condition:
            return self._busy

    @property
    def inference_count(self):
        with self._condition:
            return self._count

    @property
    def metrics(self):
        with self._condition:
            return {**copy.deepcopy(self._metrics), "inferences": self._count, "generation": self._generation,
                    "queue_frames": self.queue.qsize(), "expired_steps": self.queue.expired_steps,
                    "latency_p95_ms": None if not self._latencies else float(np.percentile(self._latencies, 95) * 1000)}

    def resume(self):
        with self._condition:
            if self._closed:
                raise RuntimeError("RTC worker is closed")
            if not self._active:
                self._active = True
                self._error = None
                self._has_chunk = False
                self._warmup_attempts = 0
            self._condition.notify_all()

    def pause(self):
        with self._condition:
            self._active = False
            self._generation += 1
            if self._request is not None:
                self._busy = False
            self._request = None
            self._has_chunk = False
            self.queue.clear()
            self._condition.notify_all()

    def close(self):
        self.pause()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        # Closing the socket also wakes a worker blocked in recv; no RPC joins on
        # the action thread during normal execution or episode pause.
        try:
            closer = getattr(self.client, "close", None)
            if closer is not None:
                closer()
            else:
                ws = getattr(self.client, "_ws", None)
                if ws is not None:
                    ws.close()
        except Exception:
            LOGGER.exception("RTC transport close failed")
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            LOGGER.error("RTC RPC worker did not terminate; all results remain invalidated")

    def raise_if_failed(self):
        with self._condition:
            if self._active and self._busy and not self._has_chunk and self._clock() - self._busy_since > 30:
                self._error = RuntimeError("RTC startup request exceeded 30 seconds")
                self._active = False
                self._generation += 1
                self.queue.clear()
            if self._active and self._has_chunk and self._busy and self.queue.qsize() == 0:
                self._metrics["underflows"] += 1
                self._error = RuntimeError("RTC action queue depleted before the next prediction")
                self._active = False
                self._generation += 1
                self.queue.clear()
            error = self._error
        if error is not None:
            raise RuntimeError("RTC inference stopped: " + str(error)) from error

    def maybe_request(self, observation):
        with self._condition:
            if not self._active or self._closed or self._busy or self.queue.qsize() > self._threshold:
                return False
            raw, valid = self.queue.snapshot_prefix(self._execution_horizon)
            if self._has_chunk and valid == 0:
                self._error = RuntimeError("RTC cannot restart from an exhausted live queue")
                self._metrics["underflows"] += 1
                self._active = False
                self.queue.clear()
                return False
            budget = float(np.percentile(self._latencies, 95)) if self._latencies else self._initial_latency
            delay = math.ceil(budget * self._hz) + 1 if valid else 0
            if valid and delay >= valid:
                self._error = RuntimeError("RTC latency budget does not fit the remaining prefix")
                self._metrics["budget_misses"] += 1
                self._active = False
                self.queue.clear()
                return False
            request = copy.deepcopy(observation)
            request["actions"] = raw
            request["_reset"] = not self._has_chunk
            request["_rtc"] = {"prefix_valid_steps": valid, "inference_delay": delay,
                               "execution_horizon": self._execution_horizon,
                               "max_guidance_weight": self._guidance}
            self._busy_since = self._clock()
            self._request = (request, self._generation, not self._has_chunk, self._busy_since)
            self._busy = True
            self._condition.notify_all()
            return True

    def _loop(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._request is not None)
                if self._closed:
                    return
                request, generation, initial, submitted = self._request
                self._request = None
            try:
                response = self.client.infer(request)
                raw = np.asarray(response.get("actions"), dtype=np.float32)
                if raw.shape != (30, 14) or not np.isfinite(raw).all():
                    raise ValueError("Invalid RTC action chunk")
                processed = raw.copy() if self._transform is None else np.asarray(self._transform(raw.copy()), dtype=np.float32)
                now = self._clock()
                latency = now - submitted
                delay = 0 if initial else math.ceil(latency * self._hz)
                with self._condition:
                    if self._closed or not self._active or generation != self._generation:
                        self._metrics["discarded_responses"] += 1
                        continue
                    if initial:
                        self._metrics["initial_request_ms"] = (self._metrics["initial_request_ms"] + [latency * 1000])[-64:]
                        # Discard cold graph/compile results. Take a new observation
                        # next tick, and start motion only once inference is warm.
                        if latency >= self._horizon / (2 * self._hz):
                            self._warmup_attempts += 1
                            self._metrics["warmup_discards"] += 1
                            if self._warmup_attempts >= 3:
                                raise RuntimeError("RTC inference cannot meet the 30-frame/30-Hz startup budget")
                            continue
                    # Atomic ownership check and publish: pause cannot interleave.
                    self.queue.replace(raw, processed, origin=now if initial else submitted, delay=delay)
                    self._has_chunk = True
                    self._count += 1
                    self._metrics["dropped_prefix_frames"] += delay
                    self._latencies.append(latency)
                LOGGER.info("RTC request_ms=%.1f skipped=%d queue=%d generation=%d",
                            latency * 1000, delay, self.queue.qsize(), generation)
            except Exception as exc:
                with self._condition:
                    if generation == self._generation and self._active and not self._closed:
                        self._error = exc
                        self._active = False
                        self.queue.clear()
                        self._metrics["failures"] += 1
                    else:
                        self._metrics["discarded_responses"] += 1
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()
