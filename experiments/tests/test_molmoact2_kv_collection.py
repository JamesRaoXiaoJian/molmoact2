import pytest
import torch

from olmo.models.model import OLMoOutput
from olmo.models.molmoact2.molmoact2 import MolmoAct2
from olmo.models.molmo2.molmo2 import Molmo2
from olmo.nn.llm import (
    AttentionLayerNormType,
    BlockType,
    LlmActivationCheckpointMode,
    LlmConfig,
    OLMoBlock,
)
from olmo.torch_util import BufferCache


def _build_block(block_type: BlockType, *, qwen3_qk_norm: bool = False) -> tuple[OLMoBlock, LlmConfig]:
    cfg = LlmConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        block_type=block_type,
        rope=True,
        max_sequence_length=16,
        attention_dropout=0.0,
        residual_dropout=0.0,
        response_residual_dropout=0.0,
        attention_layer_norm=True,
        attention_layer_norm_type=(
            AttentionLayerNormType.qwen3 if qwen3_qk_norm else AttentionLayerNormType.olmo
        ),
        qkv_bias=qwen3_qk_norm,
    )
    block = OLMoBlock.build(0, cfg, BufferCache(), device="cpu")
    return block, cfg


@pytest.mark.parametrize(
    ("block_type", "qwen3_qk_norm"),
    [
        pytest.param(BlockType.sequential, False, id="sequential-olmo-qknorm"),
        pytest.param(BlockType.llama, True, id="llama-qwen3-qknorm"),
    ],
)
def test_collect_kv_states_matches_use_cache(block_type: BlockType, qwen3_qk_norm: bool) -> None:
    torch.manual_seed(0)
    block, cfg = _build_block(block_type, qwen3_qk_norm=qwen3_qk_norm)
    x = torch.randn(2, 6, cfg.d_model)
    position_ids = torch.arange(6, dtype=torch.long).unsqueeze(0).expand(2, -1)

    block.eval()
    with torch.no_grad():
        out_with_cache, cache = block(x.clone(), position_ids=position_ids, use_cache=True)
        out_with_collect, collected = block(
            x.clone(),
            position_ids=position_ids,
            collect_kv_states=True,
        )

    assert cache is not None
    assert collected is not None
    assert torch.allclose(out_with_cache, out_with_collect)
    assert torch.allclose(cache[0], collected[0])
    assert torch.allclose(cache[1], collected[1])


@pytest.mark.parametrize(
    ("block_type", "qwen3_qk_norm"),
    [
        pytest.param(BlockType.sequential, False, id="sequential-olmo-qknorm"),
        pytest.param(BlockType.llama, True, id="llama-qwen3-qknorm"),
    ],
)
def test_collect_kv_states_survives_fine_grained_checkpointing(
    block_type: BlockType,
    qwen3_qk_norm: bool,
) -> None:
    torch.manual_seed(0)
    block, cfg = _build_block(block_type, qwen3_qk_norm=qwen3_qk_norm)
    cfg.activation_checkpoint = LlmActivationCheckpointMode.fine_grained
    block.apply_activation_checkpointing(cfg.activation_checkpoint)
    block.train()

    x = torch.randn(2, 6, cfg.d_model, requires_grad=True)
    position_ids = torch.arange(6, dtype=torch.long).unsqueeze(0).expand(2, -1)

    out, collected = block(
        x,
        position_ids=position_ids,
        collect_kv_states=True,
    )

    assert collected is not None
    assert torch.isfinite(collected[0]).all()
    assert torch.isfinite(collected[1]).all()

    loss = out.square().mean() + collected[0].square().mean() + collected[1].square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_run_backbone_collects_kv_without_forcing_use_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs = {}
    expected_cache = [(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))]
    expected_hidden_states = (torch.randn(1, 3, 4),)

    def fake_forward(self, **kwargs):
        captured_kwargs.update(kwargs)
        return OLMoOutput(
            logits=torch.zeros(1, 1, 1),
            attn_key_values=expected_cache,
            hidden_states=None,
            internal={"layer_hidden_states": expected_hidden_states},
        )

    fake_model = object.__new__(MolmoAct2)
    monkeypatch.setattr(Molmo2, "forward", fake_forward)

    base_output, layer_states, layer_kv_states = MolmoAct2._run_backbone(
        fake_model,
        output_hidden_states=False,
        collect_layer_hidden_states=True,
        collect_layer_kv_states=True,
        input_ids=torch.zeros(1, 1, dtype=torch.long),
        use_cache=False,
    )

    assert captured_kwargs["use_cache"] is False
    assert captured_kwargs["collect_layer_kv_states"] is True
    assert layer_states == expected_hidden_states
    assert layer_kv_states == expected_cache
    assert base_output.attn_key_values is None


@pytest.mark.parametrize("qwen_norm", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_prepared_kv_reuse_preserves_outputs_masks_and_cache(monkeypatch, qwen_norm, bidirectional):
    torch.manual_seed(47)
    block, cfg = _build_block(BlockType.sequential, qwen3_qk_norm=qwen_norm)
    block.eval()
    x = torch.randn(2, 6, cfg.d_model)
    positions = torch.arange(6).expand(2, -1)
    allowed = torch.ones(6, 6, dtype=torch.bool).tril()
    if bidirectional:
        allowed[:3, :3] = True
    bias = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)[None, None]
    calls = []
    prepare = block._prepare_attention_qkv

    def counted(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(block, "_prepare_attention_qkv", counted)
    with torch.no_grad():
        original, original_kv = block(x, attention_bias=bias, position_ids=positions, collect_kv_states=True)
        assert len(calls) == 2
        calls.clear()
        actual, actual_kv = block(x, attention_bias=bias, position_ids=positions,
                                  collect_kv_states=True, reuse_prepared_kv=True)
        assert len(calls) == 1
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    for actual_tensor, original_tensor in zip(actual_kv, original_kv):
        torch.testing.assert_close(actual_tensor, original_tensor, rtol=0, atol=0)
