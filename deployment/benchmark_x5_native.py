"""Recorded-input GPU performance test; no hardware modules or motion paths."""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import json
import sys
import threading
import time
from pathlib import Path


class HardwareLoader(importlib.abc.Loader):
    def create_module(self, spec):
        raise AssertionError("Real hardware module import forbidden: " + spec.name)

    def exec_module(self, module):
        raise AssertionError("Hardware execution forbidden")


class HardwareGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "arx5_interface",
            "can",
            "rospy",
            "rclpy",
            "pyrealsense2",
            "pyorbbecsdk",
            "dmrobotics",
        }:
            return importlib.machinery.ModuleSpec(fullname, HardwareLoader())


def audit(event, args):
    if event == "socket.connect":
        address = args[1]
        if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1"}:
            raise AssertionError("Only loopback test connections allowed")
    if event == "socket.__new__" and args[1] == 29:
        raise AssertionError("CAN socket forbidden")
    if (
        event == "open"
        and isinstance(args[0], (str, bytes))
        and str(args[0]).startswith(
            ("/dev/video", "/dev/input", "/dev/tty", "/dev/can")
        )
    ):
        raise AssertionError("Robot/camera device access forbidden")


sys.meta_path.insert(0, HardwareGuard())
sys.addaudithook(audit)
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "experiments"),
    str(ROOT / "experiments/lerobot/src"),
    str(Path.home() / "lwm_ws/openpi-x5-arx5-beta/openpi/packages/openpi-client/src"),
]

import numpy as np
import torch
from openpi_client import image_tools, websocket_client_policy
from scripts.serve_x5_openpi import RGB_KEYS, TACTILE_KEYS, build_bridge


def stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def load_inputs(mode):
    manifest = json.loads((ROOT / "data/offline_samples/manifest.json").read_text())
    tasks = json.loads((ROOT / "configs/x5_molmo_tasks.json").read_text())
    prompt = next(t["task"] for t in tasks["task_rows"] if t["task_index"] == 1)
    native, wire = [], []
    for entry in manifest["samples"]:
        with np.load(ROOT / "data/offline_samples" / entry["file"]) as data:
            observation = {
                k: data[k].copy() for k in data.files if k.startswith("observation.")
            }
        observation = {
            k: v for k, v in observation.items() if mode == "mae" or "tactile" not in k
        }
        observation["task"] = prompt
        request = {"state": observation["observation.state"], "prompt": prompt}
        for source, dest in RGB_KEYS.items():
            image = image_tools.resize_with_pad(
                observation["observation.images." + dest], 224, 224
            )
            observation["observation.images." + dest] = image
            request[source] = image
        if mode == "mae":
            for key in TACTILE_KEYS:
                request[key] = observation["observation.images." + key]
        native.append(observation)
        wire.append(request)
    return native, wire


def run_native(policy, observation, seed):
    policy.reset()
    generator = torch.Generator(device=policy._handles.device).manual_seed(seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = policy.generate_inference_result_from_observations(
        observation,
        norm_tag=policy._handles.norm_tag,
        num_steps=10,
        n_action_steps=30,
        generator=generator,
    )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    actions = result.actions.detach().cpu().numpy().copy()
    post = (time.perf_counter() - started) * 1000
    assert actions.shape == (1, 30, 14) and np.isfinite(actions).all()
    return actions[0], elapsed, post


class Segments:
    def __init__(self, policy):
        self.policy = policy
        self.saved = []
        self.events = []
        self.host = {}
        self.shapes = {}

    def patch(self, owner, attr, label, gpu=False):
        old = getattr(owner, attr)
        self.saved.append((owner, attr, old))

        def call(*args, **kwargs):
            if gpu:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
            t = time.perf_counter()
            value = old(*args, **kwargs)
            self.host[label] = (
                self.host.get(label, 0.0) + (time.perf_counter() - t) * 1000
            )
            if gpu:
                end.record()
                self.events.append((label, start, end))
            if label == "h2d" and args and isinstance(args[0], dict):
                self.shapes.update(
                    {k: list(v.shape) for k, v in args[0].items() if torch.is_tensor(v)}
                )
            return value

        setattr(owner, attr, call)

    def __enter__(self):
        p, model = self.policy, self.policy._handles.model
        self.patch(p, "_obs_to_example", "observation_cpu")
        self.patch(p, "_combine_history_examples", "history_cpu")
        self.patch(p, "_collate_example", "preprocess_and_h2d")
        self.patch(p, "_move_to_device", "h2d", True)
        self.patch(model, "generate_actions", "generate_total", True)
        self.patch(model, "_run_backbone", "vlm_prefill", True)
        self.patch(model.vision_backbone, "forward", "vision", True)
        self.patch(model.action_expert, "forward", "expert_forward", True)
        return self

    def __exit__(self, *args):
        for owner, attr, old in reversed(self.saved):
            setattr(owner, attr, old)

    def clear(self):
        self.events, self.host = [], {}

    def result(self):
        output = {"host_" + k: v for k, v in self.host.items()}
        for label, start, end in self.events:
            output["gpu_" + label] = output.get(
                "gpu_" + label, 0.0
            ) + start.elapsed_time(end)
        output["gpu_flow_and_context"] = (
            output["gpu_generate_total"] - output["gpu_vlm_prefill"]
        )
        output["gpu_llm_without_vision"] = output["gpu_vlm_prefill"] - output.get(
            "gpu_vision", 0.0
        )
        output["host_preprocess_without_h2d_enqueue"] = (
            output["host_preprocess_and_h2d"] - output["host_h2d"]
        )
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rgb", "mae"), required=True)
    parser.add_argument("--scheme", default="baseline")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "deployment/perf_20261012")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    native, wire = load_inputs(args.mode)
    start = time.perf_counter()
    bridge = build_bridge(args.mode)
    bridge.molmo.policy._handles.model.native_inference_optimization = args.scheme
    load_s = time.perf_counter() - start
    policy = bridge.molmo.policy
    _actions, first_ms, _first_post = run_native(policy, native[0], 42)
    for i in range(args.warmup):
        run_native(policy, native[i % 2], 42 + 12 * (i % 2))
    torch.cuda.reset_peak_memory_stats()
    samples, outputs = [], []
    for i in range(args.iterations):
        a, elapsed, post = run_native(policy, native[i % 2], 42 + 12 * (i % 2))
        samples.append(
            {
                "sample": i % 2,
                "seed": 42 + 12 * (i % 2),
                "native_ms": elapsed,
                "d2h_ms": post,
            }
        )
        outputs.append(a)
    comparison = None
    baseline_file = output / f"{args.mode}_baseline_outputs.npz"
    if args.scheme != "baseline" and baseline_file.exists():
        with np.load(baseline_file) as data:
            reference = data["actions"][: args.iterations]
        current = np.stack(outputs)
        np.testing.assert_allclose(current, reference, rtol=1e-5, atol=1e-5)
        comparison = {
            "max_abs_delta": float(np.abs(current - reference).max()),
            "bitwise_equal": bool(np.array_equal(current, reference)),
        }
    print(
        json.dumps(
            {
                "phase": "native_complete",
                "mode": args.mode,
                "scheme": args.scheme,
                "native": stats([r["native_ms"] for r in samples]),
                "baseline_comparison": comparison,
            }
        ),
        flush=True,
    )
    sections = []
    with Segments(policy) as hooks:
        for i in range(4):
            hooks.clear()
            run_native(policy, native[i % 2], 42 + 12 * (i % 2))
            sections.append(hooks.result())
        shapes = hooks.shapes
    profile_counts = None
    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            run_native(policy, native[0], 42)
        profile_counts = {
            e.key: {
                "count": e.count,
                "self_cpu_us": e.self_cpu_time_total,
                "cpu_total_us": e.cpu_time_total,
                "self_device_us": e.self_device_time_total,
            }
            for e in prof.key_averages()
            if e.key
            in {
                "aten::_local_scalar_dense",
                "aten::isfinite",
                "aten::item",
                "aten::to",
                "aten::_to_copy",
                "aten::scaled_dot_product_attention",
            }
        }
        prof.export_chrome_trace(str(output / f"{args.mode}_{args.scheme}_trace.json"))
    wire_times, schedules = [], []
    with bridge.create_server() as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        client = websocket_client_policy.WebsocketClientPolicy(
            host="127.0.0.1", port=server.socket.getsockname()[1]
        )
        for i in range(args.warmup + args.iterations):
            j = i % 2
            torch.manual_seed(42 + 12 * j)
            start = time.perf_counter()
            result = client.infer({**wire[j], "_reset": i == 0})
            elapsed = (time.perf_counter() - start) * 1000
            np.testing.assert_allclose(
                result["actions"], outputs[j], rtol=1e-5, atol=1e-5
            )
            if i >= args.warmup:
                wire_times.append(elapsed)
        for steps in (5, 15, 30):
            start = time.perf_counter()
            result = client.infer(wire[0])
            request_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            for action in result["actions"][:steps]:
                assert np.isfinite(action).all()
                time.sleep(1 / 30)
            execution_ms = (time.perf_counter() - start) * 1000
            schedules.append(
                {
                    "steps": steps,
                    "request_ms": request_ms,
                    "mock_playback_ms": execution_ms,
                    "blocking_cycle_ms": request_ms + execution_ms,
                }
            )
        client._ws.close()
        server.shutdown()
        worker.join(5)
        assert not worker.is_alive()
    torch.cuda.synchronize()
    summary = {
        "status": "passed",
        "baseline_output_comparison": comparison,
        "graph": None
        if not hasattr(policy._handles.model, "_native_action_flow_graph")
        else {
            "capture_count": policy._handles.model._native_action_flow_graph.capture_count,
            "capture_seconds": policy._handles.model._native_action_flow_graph.capture_seconds,
        },
        "mode": args.mode,
        "scheme": args.scheme,
        "parameter_dtype": policy.config.parameter_dtype,
        "solver_steps": 10,
        "horizon": 30,
        "recorded_samples": 2,
        "task_index": 1,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_s": load_s,
        "first_call_ms": first_ms,
        "native": stats([r["native_ms"] for r in samples]),
        "d2h": stats([r["d2h_ms"] for r in samples]),
        "websocket": stats(wire_times),
        "segments": {key: stats([r[key] for r in sections]) for key in sections[0]},
        "input_shapes": shapes,
        "profiler_counts": profile_counts,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "scheduling": schedules,
        "actual_action_sends": 0,
        "hardware_initializations": 0,
        "scope": "Recorded observations, input RGB resized to same client 224x224; same seeds and fixed task; segmented events may add instrumentation overhead. Native includes CPU preprocessing and synchronization; D2H listed separately. WS includes packing/service/postprocess. Mock playback has no SDK/camera/CAN.",
    }
    np.savez(
        output / f"{args.mode}_{args.scheme}_outputs.npz", actions=np.stack(outputs)
    )
    (output / f"{args.mode}_{args.scheme}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "mode",
                    "scheme",
                    "native",
                    "websocket",
                    "peak_allocated_gib",
                    "profiler_counts",
                    "actual_action_sends",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
