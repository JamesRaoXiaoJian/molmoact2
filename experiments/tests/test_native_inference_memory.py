from types import SimpleNamespace

import torch

from olmo import tokenizer
from olmo.models.molmoact2.molmoact2 import MolmoAct2, MolmoAct2Config
from olmo.nn.action_expert import ActionExpertConfig
from olmo.nn.llm import BlockType, LlmConfig


def test_last_logits_keeps_full_action_conditioning(monkeypatch):
    """Last-token logits must preserve every KV position and the resulting actions."""
    ids = {text: i + 10 for i, text in enumerate(tokenizer.EXTRA_TOKENS)}
    fake = SimpleNamespace(encode=lambda text: [ids[t] for t in text.split(' ')],
                           bos_token_id=1, eos_token_id=2)
    monkeypatch.setattr(MolmoAct2Config, 'build_tokenizer', lambda _: fake)
    config = MolmoAct2Config(
        llm=LlmConfig(d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, mlp_ratio=2,
                      block_type=BlockType.llama, rope=True, vocab_size=128,
                      embedding_size=128, additional_vocab_size=16,
                      weight_tying=False, max_sequence_length=128,
                      residual_dropout=0, embedding_dropout=0, attention_dropout=0),
        vision_backbone=None, max_action_dim=14, action_horizon=3,
        state_format='continuous',
        action_expert=ActionExpertConfig(hidden_size=16, num_heads=4, num_layers=1,
                                         max_action_dim=14, max_horizon=3),
    )
    torch.manual_seed(42)
    model = MolmoAct2(config, device='cpu')
    model.reset_parameters()
    model.eval()
    inputs = dict(input_ids=torch.tensor([[1, 3, 4, 5, 6]]), states=torch.zeros(1, 14))
    with torch.no_grad():
        full, _, full_kv = model._run_backbone(output_hidden_states=False,
                                             collect_layer_hidden_states=False,
                                             collect_layer_kv_states=True,
                                             input_ids=inputs['input_ids'], last_logits_only=False)
        last, _, last_kv = model._run_backbone(output_hidden_states=False,
                                             collect_layer_hidden_states=False,
                                             collect_layer_kv_states=True,
                                             input_ids=inputs['input_ids'], last_logits_only=True)
        assert full.logits.shape[1] == 5 and last.logits.shape[1] == 1
        for left, right in zip(full_kv, last_kv):
            torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
            torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)
        actions_full = model.generate_actions(**inputs, num_steps=2, last_logits_only=False,
                                              generator=torch.Generator().manual_seed(3))
        actions_last = model.generate_actions(**inputs, num_steps=2, last_logits_only=True,
                                              generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(actions_full, actions_last, rtol=0, atol=0)
