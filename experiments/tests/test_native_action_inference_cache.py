"""Nontrivial original-vs-cached flow parity and numerical safety regressions."""

from types import SimpleNamespace

import pytest
import torch
from olmo.models.molmoact2.molmoact2 import MolmoAct2
from olmo.nn.action_expert import ActionExpert, ActionExpertConfig
from olmo.nn.native_action_graph import _flow


def setup_expert(state_present=True):
    torch.manual_seed(12)
    config = ActionExpertConfig(
        max_horizon=4,
        max_action_dim=3,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        timestep_embed_dim=8,
        ffn_multiple_of=8,
        compile=None,
    )
    expert = ActionExpert(
        config, llm_dim=16, llm_kv_dim=16, llm_num_kv_heads=4, llm_num_layers=2
    ).eval()
    with torch.no_grad():
        for parameter in expert.parameters():
            parameter.uniform_(-0.3, 0.3)
    kv = [(torch.randn(1, 5, 16), torch.randn(1, 5, 16)) for _ in range(2)]
    state = torch.randn(1, 3) if state_present else None
    mask = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.bool)

    def masked(value, *, action_dim_is_pad=None, enabled=False):
        if not enabled or action_dim_is_pad is None:
            return value
        return value * (~action_dim_is_pad).to(value.dtype)[:, None, :]

    model = SimpleNamespace(
        config=SimpleNamespace(
            state_format="continuous" if state_present else "discrete",
            flow_matching_num_steps=10,
            action_horizon=4,
            max_action_dim=3,
            mask_action_dim_padding=True,
        ),
        _require_action_expert=lambda: expert,
        _select_layer_kv_states=lambda x: x,
        _depth_gate_from_condition=lambda **kwargs: (None, None),
        _apply_depth_gate_to_layer_states=lambda x, *args: x,
        _apply_depth_gate_to_layer_kv_states=lambda x, *args: x,
        _maybe_detach_action_expert_condition=lambda x, y: (x, y),
        _mask_action_dim_tensor=masked,
    )
    return model, expert, kv, state, mask


def generate(model, kv, state, mask, mode):
    model.native_inference_optimization = mode
    return MolmoAct2.generate_actions(
        model,
        input_ids=None,
        states=state,
        encoder_kv_states=kv,
        encoder_attention_mask=mask,
        action_dim_is_pad=torch.tensor([[False, False, True]]),
        generator=torch.Generator().manual_seed(321),
    )


@pytest.mark.parametrize("state_present", [True, False])
@pytest.mark.parametrize("mode", ["context", "deferred", "modcache"])
def test_nontrivial_masked_flow_bitwise_parity(mode, state_present):
    model, _expert, kv, state, mask = setup_expert(state_present)
    with torch.no_grad():
        original = generate(model, kv, state, mask, "baseline")
        optimized = generate(model, kv, state, mask, mode)
    assert original.std() > 0.01
    assert original[:, :, 2].count_nonzero() == 0
    torch.testing.assert_close(optimized, original, rtol=0, atol=0)


def test_modulation_cache_reuse_and_weight_invalidation():
    _model, expert, _kv, _state, _mask = setup_expert()
    with torch.no_grad():
        a = expert.inference_modulations(10, 1, torch.device("cpu"), torch.float32)
        assert (
            expert.inference_modulations(10, 1, torch.device("cpu"), torch.float32) is a
        )
        expert.blocks[0].modulation.linear.bias.add_(0.01)
        assert (
            expert.inference_modulations(10, 1, torch.device("cpu"), torch.float32)
            is not a
        )


@pytest.mark.parametrize("mode", ["context", "deferred", "modcache"])
def test_masked_invalid_context_still_rejected(mode):
    model, _expert, kv, state, mask = setup_expert()
    kv[0][0][0, 2, 0] = float("nan")
    with torch.no_grad(), pytest.raises(RuntimeError, match="Non-finite"):
        generate(model, kv, state, mask, mode)


def test_no_inference_cache_in_training_and_normal_backward():
    _model, expert, kv, state, mask = setup_expert()
    expert.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval/no-grad"):
        expert.prepare_inference_context(
            kv, mask, state, 1, 4, torch.device("cpu"), torch.float32
        )
    output = expert(
        torch.randn(1, 4, 3),
        torch.tensor([0.3]),
        kv,
        encoder_attention_mask=mask,
        state_embeddings=state,
    )
    output.square().mean().backward()
    assert expert.action_embed.weight.grad is not None
    assert torch.isfinite(expert.action_embed.weight.grad).all()


def test_graph_flow_callable_matches_same_eager_flow_on_cpu():
    model, expert, kv, state, mask = setup_expert()
    padding = torch.tensor([[False, False, True]])
    with torch.no_grad():
        original = generate(model, kv, state, mask, "modcache")
        context = expert.prepare_inference_context(
            kv, mask, state, 1, 4, torch.device("cpu"), torch.float32
        )
        modulations = expert.inference_modulations(
            10, 1, torch.device("cpu"), torch.float32
        )
        noise = torch.randn(1, 4, 3, generator=torch.Generator().manual_seed(321))
        noise = model._mask_action_dim_tensor(
            noise, action_dim_is_pad=padding, enabled=True
        )
        actual, valid = _flow(model, expert, noise, context, modulations, padding, 10)
    assert bool(valid)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
