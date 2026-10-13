"""Opt-in action-block compilation for an already loaded inference model."""
import torch


def compile_action_blocks(model):
    """Keep graph capture explicit; compilation can change floating-point results.

    Call before serving requests. The ordinary training/loader path never calls
    this function, and the uncompiled deployment remains the default.
    """
    expert = model.action_expert
    if expert.training:
        raise RuntimeError("Action block compilation requires an eval model")
    if getattr(expert, "_native_blocks_compiled", False):
        return
    forwards = [torch.compile(block.forward, fullgraph=True, dynamic=False, options={
        "triton.cudagraphs": False,
        "emulate_precision_casts": True,
        "force_same_precision": True,
        "use_fast_math": False,
    }) for block in expert.blocks]
    for block, forward in zip(expert.blocks, forwards):
        block.forward = forward
    # A previously captured graph would still contain uncompiled kernels.
    model._native_action_flow_graph = None
    expert._native_blocks_compiled = True
