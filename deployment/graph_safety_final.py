"""Small real-CUDA graph safety/recovery test; offline, no hardware modules."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

# Register the same hardware import/device/network audit as the full benchmark.
import benchmark_x5_native  # noqa: F401
import torch
from olmo.nn import native_action_graph as graph_module
from tests.test_native_action_inference_cache import setup_expert


def main():
    model, expert, kv, state, mask = setup_expert()
    expert.to(device="cuda", dtype=torch.bfloat16)
    kv = [(k.to("cuda", torch.bfloat16), v.to("cuda", torch.bfloat16)) for k, v in kv]
    state = state.to("cuda", torch.bfloat16)
    mask = mask.to("cuda")
    padding = torch.tensor([[False, False, True]], device="cuda")
    noise = torch.randn(1, 4, 3, device="cuda", dtype=torch.bfloat16)
    manager = graph_module.NativeActionFlowGraph()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        context = expert.prepare_inference_context(
            kv, mask, state, 1, 4, torch.device("cuda"), torch.bfloat16
        )
        mods = expert.inference_modulations(10, 1, torch.device("cuda"), torch.bfloat16)
        eager, valid = graph_module._flow(model, expert, noise, context, mods, padding, 10)
        assert bool(valid)
        original = manager.run(model, expert, noise, context, mods, padding, 10)
        torch.testing.assert_close(original, eager, rtol=0, atol=0)
        old_key, old_graph = manager.key, manager.graph
        original_flow = graph_module._flow
        failure_results = []
        for name in ("nonfinite_warmup", "warmup_exception"):
            def injected(*args, **kwargs):
                if name == "warmup_exception":
                    raise RuntimeError("intentional offline warmup exception")
                result, _valid = original_flow(*args, **kwargs)
                return result, torch.tensor(False, device="cuda")

            try:
                with patch.object(graph_module, "_flow", injected):
                    manager.run(model, expert, noise, context, mods, padding, 9)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Injected failure returned actions")
            assert manager.key == old_key and manager.graph is old_graph
            recovered = manager.run(model, expert, noise, context, mods, padding, 10)
            torch.testing.assert_close(recovered, original, rtol=0, atol=0)
            failure_results.append({"failure": name, "recovered_bitwise": True})

        changed_context = {
            key: value for key, value in context.items()
        }
        # Construct a new prepared context from genuinely changed KV observations.
        changed_kv = [(k + 0.125, v - 0.125) for k, v in kv]
        changed_context = expert.prepare_inference_context(
            changed_kv, mask, state, 1, 4, torch.device("cuda"), torch.bfloat16
        )
        expected, valid = original_flow(model, expert, noise, changed_context, mods, padding, 10)
        assert bool(valid)
        changed = manager.run(model, expert, noise, changed_context, mods, padding, 10)
        torch.testing.assert_close(changed, expected, rtol=0, atol=0)
        torch.testing.assert_close(original, eager, rtol=0, atol=0)
        assert manager.capture_count == 1

        invalid_kv = [(k.clone(), v.clone()) for k, v in kv]
        invalid_kv[0][0][0, 2, 0] = float("nan")
        try:
            expert.prepare_inference_context(
                invalid_kv, mask, state, 1, 4, torch.device("cuda"), torch.bfloat16
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("Masked NaN context was accepted")

        # Bypass version invalidation deliberately to prove checks inside replay remain.
        with torch.no_grad():
            expert.final_layer.linear.bias.data.fill_(float("nan"))
        try:
            manager.run(model, expert, noise, context, mods, padding, 10)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Internal NaN graph returned actions")
    report = {
        "status": "passed", "device": str(torch.cuda.get_device_name()),
        "dtype": "bfloat16", "graph_eager_bitwise": True,
        "failure_recovery": failure_results, "new_context_bitwise": True,
        "old_output_no_alias": True, "masked_nan_context_rejected": True,
        "internal_replay_nan_rejected": True, "capture_count": manager.capture_count,
        "actual_action_sends": 0, "hardware_initializations": 0,
    }
    path = Path(__file__).parent / "perf_20261012" / "graph_safety_final.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
