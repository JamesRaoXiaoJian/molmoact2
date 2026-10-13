"""Optional graph for the already-prepared native action flow, never the VLM."""

from __future__ import annotations

import time

import torch


def _signature(value):
    if torch.is_tensor(value):
        return tuple(value.shape), tuple(value.stride()), value.dtype, value.device
    if isinstance(value, dict):
        return tuple((k, _signature(v)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_signature(v) for v in value)
    return value


def _clone(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    if isinstance(value, list):
        return [_clone(v) for v in value]
    return value


def _copy(target, source):
    if torch.is_tensor(target):
        target.copy_(source)
    elif isinstance(target, dict):
        for key in target:
            _copy(target[key], source[key])
    elif isinstance(target, (tuple, list)):
        for a, b in zip(target, source):
            _copy(a, b)


def _flow(model, expert, trajectory, context, modulations, padding, steps, rtc_condition=None):
    from olmo.nn.action_expert import _DeferredFiniteChecks
    from olmo.nn.rtc_guidance import rtc_euler_step

    checks = _DeferredFiniteChecks()
    if rtc_condition is not None:
        checks.record(rtc_condition["prefix"], "rtc_prefix")
        checks.record(rtc_condition["weights"], "rtc_weights")
    for index in range(steps):
        time_value = torch.full(
            (trajectory.shape[0],), index / steps, device=trajectory.device
        )
        trajectory = model._mask_action_dim_tensor(
            trajectory,
            action_dim_is_pad=padding,
            enabled=model.config.mask_action_dim_padding,
        )
        velocity = expert(
            trajectory,
            time_value,
            prepared_context=context,
            step_modulations=modulations[index],
            finite_checks=checks,
        )
        velocity = model._mask_action_dim_tensor(
            velocity,
            action_dim_is_pad=padding,
            enabled=model.config.mask_action_dim_padding,
        )
        trajectory = rtc_euler_step(trajectory, velocity, index, steps, rtc_condition)
        if rtc_condition is not None:
            checks.record(trajectory, f"rtc_trajectory_{index}")
        trajectory = model._mask_action_dim_tensor(
            trajectory,
            action_dim_is_pad=padding,
            enabled=model.config.mask_action_dim_padding,
        )
    checks.record(trajectory, "integrated_trajectory")
    return trajectory, torch.stack(checks.flags).all()


class NativeActionFlowGraph:
    """One graph key; changing shapes/precision/weights recaptures explicitly."""

    def __init__(self):
        self.key = None
        self.graph = None
        self.capture_count = 0
        self.capture_seconds = []

    def run(self, model, expert, trajectory, context, modulations, padding, steps, rtc_condition=None):
        if expert.training or torch.is_grad_enabled() or not trajectory.is_cuda:
            raise RuntimeError("Action graph requires CUDA eval/no-grad inference")
        versions = tuple(
            (id(p), p._version, p.dtype, p.device) for p in expert.parameters()
        )
        key = (
            steps,
            _signature(trajectory),
            _signature(context),
            _signature(padding),
            _signature(rtc_condition),
            getattr(expert, "_inference_modulation_key", None),
            versions,
        )
        if key != self.key or self.graph is None:
            started = time.perf_counter()
            # Build a complete candidate without changing the last healthy cache.
            static_trajectory = trajectory.clone()
            static_context = _clone(context)
            static_padding = _clone(padding)
            static_rtc = _clone(rtc_condition)
            rtc_kwargs = {} if static_rtc is None else {"rtc_condition": static_rtc}
            current = torch.cuda.current_stream()
            warmup = torch.cuda.Stream()
            warmup.wait_stream(current)
            try:
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        _, valid = _flow(
                            model,
                            expert,
                            static_trajectory,
                            static_context,
                            modulations,
                            static_padding,
                            steps,
                            **rtc_kwargs,
                        )
            finally:
                # Also join on a Python/CUDA error before candidate buffers expire.
                current.wait_stream(warmup)
                torch.cuda.synchronize()
            if not bool(valid):
                raise RuntimeError("Non-finite graph warmup; no actions returned")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, valid = _flow(
                    model,
                    expert,
                    static_trajectory,
                    static_context,
                    modulations,
                    static_padding,
                    steps,
                    **rtc_kwargs,
                )
            graph.replay()
            if not bool(valid):
                raise RuntimeError("Non-finite action flow graph; no actions returned")
            # Publish only after capture and the first replay both succeed.
            self.static_trajectory = static_trajectory
            self.static_context = static_context
            self.static_padding = static_padding
            self.static_rtc = static_rtc
            self.modulations = modulations
            self.output = output
            self.valid = valid
            self.graph = graph
            self.key = key
            self.capture_count += 1
            self.capture_seconds.append(time.perf_counter() - started)
        else:
            self.static_trajectory.copy_(trajectory)
            _copy(self.static_context, context)
            _copy(self.static_padding, padding)
            _copy(self.static_rtc, rtc_condition)
            self.graph.replay()
            if not bool(self.valid):
                raise RuntimeError("Non-finite action flow graph; no actions returned")
        return self.output.clone()
