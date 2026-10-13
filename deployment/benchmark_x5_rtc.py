"""Recorded-input RTC eager/graph parity and timings; hardware guarded."""
import argparse
import json
from pathlib import Path
import time

import benchmark_x5_native as base
import numpy as np
import torch
from olmo.nn.native_action_compile import compile_action_blocks


def infer(policy, observation, seed, rtc=None):
    policy.reset()
    gen = torch.Generator(device=policy._handles.device).manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = policy.generate_inference_result_from_observations(
        observation, norm_tag=policy._handles.norm_tag, num_steps=10,
        n_action_steps=30, generator=gen, rtc=rtc)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000
    actions = result.actions.cpu().numpy()[0].copy()
    assert actions.shape == (30, 14) and np.isfinite(actions).all()
    return actions, ms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("rgb","mae"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--compile-blocks", action="store_true")
    args = p.parse_args()
    native, _ = base.load_inputs(args.mode)
    bridge = base.build_bridge(args.mode)
    policy = bridge.molmo.policy
    model = policy._handles.model
    if args.compile_blocks: compile_action_blocks(model)
    model.native_inference_optimization = "modcache"
    baseline = [infer(policy, obs, 42 + 12*j)[0] for j, obs in enumerate(native)]
    requests, references, shifts = [], [], []
    for j in range(2):
        variants = []
        for case in range(3):
            prefix = baseline[j].copy()
            if case == 2: prefix[:20, 0] += .04
            variants.append({"previous_actions":prefix, "options":{
                "prefix_valid_steps":0 if case == 0 else 20, "inference_delay":10,
                "execution_horizon":20, "max_guidance_weight":10.0}})
        requests.append(variants)
        normalized = policy._handles.robot_processor.normalize_action(variants[1]["previous_actions"],
                                                                       repo_id=policy._handles.norm_tag)
        np.testing.assert_array_equal(normalized[:, [6, 13]], variants[1]["previous_actions"][:, [6, 13]])
        values = [infer(policy, native[j], 42+12*j, req)[0] for req in variants]
        assert np.array_equal(values[0].view(np.uint32), baseline[j].view(np.uint32)), "Empty RTC prefix changed baseline"
        assert not np.array_equal(values[1], values[2]), "RTC ignores updated prefix values"
        references.append(values)
        shifts.append(float(np.abs(values[1]-values[2]).max()))
    rows = []
    for mode in ("modcache", "graph"):
        model.native_inference_optimization = mode
        # Graph first capture is separated from steady-state measurements.
        cold_start = time.perf_counter()
        for j in range(2): infer(policy, native[j], 42+12*j, requests[j][0])
        cold_ms = (time.perf_counter()-cold_start)*1000
        torch.cuda.reset_peak_memory_stats()
        timings=[]
        for i in range(args.iterations):
            j, case = i % 2, i % 3
            actual, ms = infer(policy, native[j], 42+12*j, requests[j][case])
            np.testing.assert_array_equal(actual.view(np.uint32), references[j][case].view(np.uint32))
            timings.append(ms)
        manager = getattr(model, "_native_action_flow_graph", None)
        captures = 0 if manager is None else manager.capture_count
        if mode == "graph": assert captures == 1, "RTC values caused graph recapture"
        rows.append({"mode":mode, **base.stats(timings), "eager_bitwise_equal":True,
                     "graph_captures":captures, "two_warmup_calls_ms":cold_ms,
                     "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30})
    report={"model":args.mode,"compiled":args.compile_blocks,"rows":rows,
            "empty_prefix_bitwise_equal":True,"raw_gripper_prefix_preserved":True,"prefix_sensitivity_max_rad":shifts,
            "scope":"Two recorded observations; inference correctness and latency only, not task success",
            "real_robot_initializations":0,"actual_action_sends":0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == "__main__": main()
