"""CPU fake-CUDA regression for atomic graph replacement and stream cleanup."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from olmo.nn import native_action_graph as module


class Tensor:
    is_cuda = True
    shape = (1, 1, 1)
    dtype = "bf16"
    device = "cuda:0"

    def __init__(self, value):
        self.value = value

    def stride(self):
        return (1, 1, 1)

    def clone(self):
        return Tensor(self.value)

    def copy_(self, other):
        self.value = other.value

    def __bool__(self):
        return bool(self.value)


class Stream:
    def __init__(self):
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other)


@pytest.mark.parametrize(
    "failure",
    ["warmup_false", "warmup_raise", "capture_raise", "replay_false", "replay_raise"],
)
def test_failed_recapture_preserves_previous_graph_and_joins_stream(
    monkeypatch, failure
):
    state = SimpleNamespace(phase=None, failure=None, graph=None, syncs=0)
    current = Stream()

    @contextmanager
    def stream(_stream):
        state.phase = "warmup"
        try:
            yield
        finally:
            state.phase = None

    @contextmanager
    def graph_context(graph):
        state.phase = "capture"
        state.graph = graph
        try:
            yield
        finally:
            state.phase = None
            state.graph = None

    class Graph:
        def replay(self):
            if state.failure == "replay_raise":
                raise RuntimeError("injected first replay error")
            self.output.value = self.trajectory.value + self.context["kv"].value
            self.valid.value = state.failure != "replay_false"

    def synchronize():
        state.syncs += 1

    def flow(_model, _expert, trajectory, context, _mods, _padding, _steps):
        if state.failure == f"{state.phase}_raise":
            raise RuntimeError("injected flow error")
        output = Tensor(trajectory.value + context["kv"].value)
        valid = Tensor(state.failure != "warmup_false")
        if state.phase == "capture":
            state.graph.trajectory = trajectory
            state.graph.context = context
            state.graph.output = output
            state.graph.valid = valid
        return output, valid

    fake_torch = SimpleNamespace(
        is_tensor=lambda value: isinstance(value, Tensor),
        is_grad_enabled=lambda: False,
        cuda=SimpleNamespace(
            current_stream=lambda: current,
            Stream=Stream,
            stream=stream,
            synchronize=synchronize,
            CUDAGraph=Graph,
            graph=graph_context,
        ),
    )
    monkeypatch.setattr(module, "torch", fake_torch)
    monkeypatch.setattr(module, "_flow", flow)
    expert = SimpleNamespace(training=False, parameters=lambda: ())
    manager = module.NativeActionFlowGraph()
    healthy = manager.run(None, expert, Tensor(2), {"kv": Tensor(3)}, (), None, 2)
    old_key, old_graph, old_context = manager.key, manager.graph, manager.static_context
    assert healthy.value == 5
    previous_waits, previous_syncs = len(current.waits), state.syncs
    state.failure = failure
    with pytest.raises(RuntimeError):
        manager.run(None, expert, Tensor(10), {"kv": Tensor(20)}, (), None, 3)
    assert manager.key == old_key
    assert manager.graph is old_graph
    assert manager.static_context is old_context
    assert manager.capture_count == 1
    assert len(current.waits) == previous_waits + 1
    assert state.syncs == previous_syncs + 1

    state.failure = None
    recovered = manager.run(None, expert, Tensor(4), {"kv": Tensor(6)}, (), None, 2)
    assert recovered.value == 10
    assert healthy.value == 5
    assert manager.capture_count == 1
    replacement = manager.run(None, expert, Tensor(7), {"kv": Tensor(8)}, (), None, 3)
    assert replacement.value == 15
    assert manager.capture_count == 2
