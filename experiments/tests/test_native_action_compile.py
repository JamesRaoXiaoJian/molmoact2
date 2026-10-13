from types import SimpleNamespace

import pytest

from olmo.nn import native_action_compile as module


def test_compile_is_explicit_graph_free_idempotent_and_invalidates_old_capture(monkeypatch):
    calls = []

    def compile_(forward, **kwargs):
        calls.append(kwargs)
        return lambda *a, **k: forward(*a, **k)

    monkeypatch.setattr(module.torch, "compile", compile_)
    blocks = [SimpleNamespace(forward=lambda x: x + 1) for _ in range(2)]
    model = SimpleNamespace(action_expert=SimpleNamespace(training=False, blocks=blocks),
                            _native_action_flow_graph=object())
    module.compile_action_blocks(model)
    assert model._native_action_flow_graph is None
    assert all(c["options"]["triton.cudagraphs"] is False for c in calls)
    assert all(c["options"]["emulate_precision_casts"] for c in calls)
    assert blocks[0].forward(2) == 3
    module.compile_action_blocks(model)
    assert len(calls) == 2


def test_compile_rejects_training_model():
    with pytest.raises(RuntimeError, match="eval model"):
        module.compile_action_blocks(SimpleNamespace(action_expert=SimpleNamespace(training=True)))
