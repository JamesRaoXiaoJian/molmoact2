"""Paired 5080 optimization measurements. Recorded inputs only; hardware guarded."""
import argparse
import json
import platform
from pathlib import Path

import benchmark_x5_native as base  # installs hardware/network guards
import numpy as np
import torch
from capture_x5_device_state import snapshot
from olmo.nn.native_action_compile import compile_action_blocks


def configure(model, variant):
    model.native_inference_optimization = "graph" if variant.endswith("graph") else "modcache"
    model.action_expert.optimize_static_inference = variant not in {"previous_eager", "previous_graph"}
    model.optimize_backbone_inference = variant not in {"previous_eager", "previous_graph"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rgb", "mae"), required=True)
    parser.add_argument("--phase", choices=("eager", "graph"), default="eager")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-blocks", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    native, _wire = base.load_inputs(args.mode)
    bridge = base.build_bridge(args.mode)
    policy = bridge.molmo.policy
    model = policy._handles.model
    variants = (["previous_eager", "static_eager"] if args.phase == "eager"
                else ["previous_graph", "static_graph"])
    if args.compile_blocks:
        variants.append("compiled_" + args.phase)
    gold = []
    configure(model, "previous_eager")
    for i, observation in enumerate(native):
        action, _, _ = base.run_native(policy, observation, 42 + 12 * i)
        gold.append(action)
    rows = []
    for variant in variants:
        configure(model, variant)
        if variant.startswith("compiled_"):
            compile_action_blocks(model)
        # Separate capture/cold initialization from steady-state timing.
        cold_ms = None
        for i in range(4):
            _, warm_ms, _ = base.run_native(policy, native[i % 2], 42 + 12 * (i % 2))
            if cold_ms is None:
                cold_ms = warm_ms
        torch.cuda.reset_peak_memory_stats()
        device_before = snapshot()
        timings, max_delta, bitwise = [], 0.0, True
        for i in range(args.iterations):
            index = i % 2
            result, elapsed, _ = base.run_native(policy, native[index], 42 + 12 * index)
            delta = float(np.abs(result - gold[index]).max())
            max_delta = max(max_delta, delta)
            bitwise = bitwise and bool(np.array_equal(result, gold[index]))
            if not bitwise and not variant.startswith("compiled_"):
                raise AssertionError(f"{variant}: output changed (max abs {delta})")
            timings.append(elapsed)
        manager = getattr(model, "_native_action_flow_graph", None)
        if args.phase == "eager":
            assert manager is None, "CUDA Graph was instantiated during the no-Graph test"
        row = {"variant": variant, **base.stats(timings), "bitwise_equal": bitwise,
               "max_abs_delta": max_delta, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "graph_capture_count": 0 if manager is None else manager.capture_count,
               "first_variant_call_ms": cold_ms, "latencies_ms": timings,
               "device_before": device_before, "device_after": snapshot()}
        rows.append(row)
        print(json.dumps({"mode": args.mode, **{k:v for k,v in row.items()
                                              if k not in {"latencies_ms", "device_before", "device_after"}}}), flush=True)
        (args.output / f"{args.mode}_{args.phase}.json").write_text(json.dumps({
            "mode": args.mode, "device": torch.cuda.get_device_name(0), "phase": args.phase,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "solver_steps": 10, "horizon": 30, "samples": 2, "rows": rows,
            "hardware_initializations": 0, "actual_action_sends": 0}, indent=2) + "\n")


if __name__ == "__main__":
    main()
