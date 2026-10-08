import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from olmo import tokenizer
from olmo.models.molmoact2.molmoact2 import MolmoAct2, MolmoAct2Config
from olmo.nn import tactile_encoder
from olmo.nn.action_expert import ActionExpertConfig
from olmo.nn.image_vit import VisionBackboneType, VitConfig
from olmo.nn.llm import BlockType, LlmConfig
from olmo.nn.tactile_config import TactileConfig
from olmo.nn.vision_backbone import (
    ImagePooling2DType,
    ImageProjectType,
    MolmoVisionBackboneConfig,
)
from olmo.preprocessing.tactile_preprocessor import (
    TactileExamplePreprocessor,
    preprocess_tactile,
)
from olmo.train.checkpoint_loading import initialize_tactile_state_for_checkpoint
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig

BACKENDS = ("anytouch2", "sparsh_vjepa", "tactile_mae")


def _tiny_vision_config():
    return MolmoVisionBackboneConfig(
        vit=VitConfig(
            image_model_type=VisionBackboneType.siglip,
            image_default_input_size=(28, 28),
            image_emb_dim=16,
            image_num_heads=2,
            image_num_key_value_heads=2,
            image_head_dim=8,
            image_mlp_dim=32,
            image_num_layers=1,
            image_num_pos=4,
            normalize="siglip",
        ),
        image_pooling_2d=ImagePooling2DType.none,
        image_projector=ImageProjectType.linear,
        compile_vit=None,
        compile_connector=None,
    )


def _tiny_config(monkeypatch, backend):
    monkeypatch.setattr(tactile_encoder, "backbone_spec", lambda _: (16, 1, 2, 16))
    ids = {text: i + 10 for i, text in enumerate(tokenizer.EXTRA_TOKENS)}
    fake_tokenizer = SimpleNamespace(
        encode=lambda text: [ids[t] for t in text.split(" ")],
        bos_token_id=1,
        eos_token_id=2,
    )
    monkeypatch.setattr(MolmoAct2Config, "build_tokenizer", lambda _: fake_tokenizer)
    return MolmoAct2Config(
        llm=LlmConfig(
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            mlp_ratio=2,
            block_type=BlockType.llama,
            rope=True,
            vocab_size=128,
            embedding_size=128,
            additional_vocab_size=16,
            weight_tying=False,
            max_sequence_length=128,
            residual_dropout=0,
            embedding_dropout=0,
            attention_dropout=0,
        ),
        vision_backbone=_tiny_vision_config(),
        max_action_dim=14,
        action_horizon=3,
        state_format="continuous",
        action_expert=ActionExpertConfig(
            hidden_size=16, num_heads=4, num_layers=1, max_action_dim=14, max_horizon=3
        ),
        tactile=TactileConfig(backend=backend, num_tokens=2),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_independent_tactile_drives_actions_has_gradients_and_restores_without_external_weights(
    monkeypatch, tmp_path, backend
):
    torch.manual_seed(42)
    cfg = _tiny_config(monkeypatch, backend)
    model = MolmoAct2(cfg, device="cpu")
    model.reset_parameters()
    # Make the action expert's initially zero output/gates nonzero so this checks
    # the entire conditioning path on the first backward pass.
    for p in model.action_expert.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.1)
    pixels = torch.randn(1, 1, 2, cfg.tactile.frames, 3, 224, 224)
    mask = torch.tensor([[False, True, True, True, True, False, False, False]])
    inputs = {
        "input_ids": torch.tensor([[1, 0, 0, 0, 0, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 8, dtype=torch.bool),
        "states": torch.zeros(1, 1, 14),
        "tactile_images": pixels,
        "tactile_token_mask": mask,
    }
    before = model.tactile_encoder.projection.weight.detach().clone()
    optimizer = OptimizerConfig(
        name=OptimizerType.adamw,
        llm_learning_rate=0.001,
        action_expert_learning_rate=0.001,
    ).build_optimizer(None, None, model)
    out = model(**inputs, actions=torch.zeros(1, 3, 14))
    loss = out.internal["action_flow_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grad = model.tactile_encoder.projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    backbone_grads = [
        p.grad for p in model.tactile_encoder.backbone.parameters() if p.requires_grad
    ]
    assert any(g is not None and g.abs().sum() > 0 for g in backbone_grads)
    optimizer.step()
    assert not torch.equal(before, model.tactile_encoder.projection.weight)
    model.eval()
    first = model.generate_actions(
        **inputs, num_steps=2, generator=torch.Generator().manual_seed(3)
    )
    changed = model.generate_actions(
        **{**inputs, "tactile_images": pixels + 0.5},
        num_steps=2,
        generator=torch.Generator().manual_seed(3),
    )
    assert first.shape == (1, 3, 14) and torch.isfinite(first).all()
    assert not torch.equal(first, changed)
    cfg.tactile.encoder_path = "/missing/initialization/weights"
    cfg.save(tmp_path / "config.yaml")
    torch.save(model.state_dict(), tmp_path / "model.pt")
    restored_cfg = MolmoAct2Config.load(tmp_path / "config.yaml")
    restored = MolmoAct2(restored_cfg, device="cpu").eval()
    state = torch.load(tmp_path / "model.pt", weights_only=True)
    initialize_tactile_state_for_checkpoint(state, restored)
    restored.load_state_dict(state, strict=True)
    actual = restored.generate_actions(
        **inputs, num_steps=2, generator=torch.Generator().manual_seed(3)
    )
    torch.testing.assert_close(actual, first, rtol=0, atol=0)
    state.pop("tactile_encoder.projection.weight")
    initialize_tactile_state_for_checkpoint(state, restored)
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(state, strict=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_shared_cpu_preprocessing_matches_expected_reference_geometry(backend):
    cfg = TactileConfig(backend=backend)
    raw = np.full((1, 2, cfg.frames, 16, 28, 3), 128, dtype=np.uint8)
    actual = preprocess_tactile(raw, cfg)
    assert actual.shape == (1, 2, cfg.frames, 3, 224, 224)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, preprocess_tactile(raw, cfg))


def test_prefix_tokens_have_no_loss_and_rgb_tokens_keep_their_order():
    cfg = TactileConfig(backend="sparsh_vjepa", num_tokens=2)

    class Base:
        preprocessor = SimpleNamespace(
            text_preprocessor=SimpleNamespace(max_sequence_length=100)
        )

        def __call__(self, example, **kwargs):
            return {
                "input_tokens": np.array([1, 9, 8]),
                "target_tokens": np.array([9, 8, 2]),
                "loss_masks": np.array([0.0, 0.0, 1.0]),
                "position_ids": np.arange(3),
            }

    pp = TactileExamplePreprocessor(Base(), cfg, 1)
    out = pp({"tactile_windows": np.zeros((1, 2, 4, 16, 28, 3), dtype=np.uint8)})
    assert out["input_tokens"].tolist() == [1, 0, 0, 0, 0, 9, 8]
    assert out["loss_masks"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert out["tactile_token_mask"].sum() == 4
    assert out["position_ids"].tolist() == list(range(7))


def test_rgb_and_tactile_both_reach_the_native_action_loss(monkeypatch):
    cfg = _tiny_config(monkeypatch, "tactile_mae")
    cfg.vision_backbone = MolmoVisionBackboneConfig(
        vit=VitConfig(
            image_model_type=VisionBackboneType.siglip,
            image_default_input_size=(28, 28),
            image_emb_dim=16,
            image_num_heads=2,
            image_num_key_value_heads=2,
            image_head_dim=8,
            image_mlp_dim=32,
            image_num_layers=1,
            image_num_pos=4,
            normalize="siglip",
        ),
        image_pooling_2d=ImagePooling2DType.none,
        image_projector=ImageProjectType.linear,
        compile_vit=None,
        compile_connector=None,
    )
    model = MolmoAct2(cfg, device="cpu")
    model.reset_parameters()
    for p in model.action_expert.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.1)
    rgb = torch.randn(1, 3, 4, 588, requires_grad=True)
    tactile = torch.randn(1, 1, 2, 1, 3, 224, 224, requires_grad=True)
    ids = torch.tensor([[1] + [0] * 4 + [model._image_patch_id] * 12 + [3, 4]])
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 1:5] = True
    output = model(
        input_ids=ids,
        images=rgb,
        token_pooling=torch.arange(12).view(1, 12, 1),
        tactile_images=tactile,
        tactile_token_mask=mask,
        states=torch.zeros(1, 1, 14),
        actions=torch.zeros(1, 3, 14),
    )
    output.internal["action_flow_loss"].backward()
    assert rgb.grad is not None and rgb.grad.abs().sum() > 0
    assert tactile.grad is not None and tactile.grad.abs().sum() > 0


def test_initial_base_checkpoint_materializes_tactile_from_cpu_weights_for_meta_model(
    monkeypatch, tmp_path
):
    cfg = _tiny_config(monkeypatch, "anytouch2")
    cpu_encoder = tactile_encoder.TactileEncoder(cfg.tactile, 16)
    path = tmp_path / "weights.pth"
    torch.save({"model": cpu_encoder.backbone.state_dict()}, path)
    cfg.tactile.encoder_path = str(path)
    with torch.device("meta"):
        target = torch.nn.Module()
        target.tactile_encoder = tactile_encoder.TactileEncoder(cfg.tactile, 16)
    state = {}
    initialize_tactile_state_for_checkpoint(state, target)
    assert state and all(value.device.type == "cpu" for value in state.values())
    target.load_state_dict(state, strict=True, assign=True)
    for key, expected in cpu_encoder.backbone.state_dict().items():
        torch.testing.assert_close(
            target.tactile_encoder.backbone.state_dict()[key], expected
        )


def test_tactile_backbone_and_adapter_get_distinct_native_learning_rates(monkeypatch):
    model = MolmoAct2(_tiny_config(monkeypatch, "tactile_mae"), device="cpu")
    model.reset_parameters()
    config = OptimizerConfig(
        tactile_backbone_learning_rate=5e-6, tactile_adapter_learning_rate=5e-5
    )
    groups = config.get_param_groups(None, None, model)
    backbone = [
        group for group in groups if group["group_name"].startswith("vit_tactile")
    ]
    adapter = [
        group
        for group in groups
        if group["group_name"].startswith("action_expert_tactile")
    ]
    assert backbone and adapter
    assert all(group["lr"] == 5e-6 for group in backbone)
    assert all(group["lr"] == 5e-5 for group in adapter)
    assert any(
        model.tactile_encoder.projection.weight is p
        for group in adapter
        for p in group["params"]
    )
    assert any(
        model.tactile_encoder.backbone.video_patch_embedding.weight is p
        for group in backbone
        for p in group["params"]
    )
    scheduler = SchedulerConfig().build()
    assert scheduler.get_lr(5e-6, 1000, 20000, "vit_tactile") > 0
    assert scheduler.get_lr(5e-5, 1000, 20000, "action_expert_tactile") > 0


def test_tactile_preprocessor_can_be_restored_by_spawn_workers():
    original = TactileExamplePreprocessor(
        SimpleNamespace(tokenizer="tokenizer-probe"),
        TactileConfig(backend="tactile_mae"),
        1,
    )
    restored = pickle.loads(pickle.dumps(original))
    assert restored.tokenizer == "tokenizer-probe"
    assert restored.prefix_length == 64
