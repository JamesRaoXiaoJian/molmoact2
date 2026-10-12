#!/usr/bin/env python3
"""Synchronized native-vs-official HF latency benchmark; never controls hardware."""
import argparse
import functools
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/lerobot/src")]
os.environ.setdefault("WANDB_MODE", "disabled")
import numpy as np
import torch
import yaml

from scripts.run_x5_inference_comparison import prepare_samples
from scripts.serve_policy import MolmoAct2Server
from olmo.data.tactile import X5_RGB_IMAGE_KEYS


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def official_copy(source, output, native_checkpoint):
    """Keep official weights intact; add evaluation-only ARX normalization metadata."""
    target = output / "official_hf_eval"
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        dst = target / path.name
        if dst.exists():
            continue
        if path.suffix in {".py", ".json", ".jinja"}:
            shutil.copy2(path, dst)
        else:
            dst.symlink_to(path.resolve(), target_is_directory=path.is_dir())
    model_file = target / "modeling_molmoact2.py"
    text = model_file.read_text()
    # This is the same NumPy compatibility fix used by the official robot server.
    text = text.replace("value.detach().cpu().numpy().astype(np.float32, copy=False)",
                        "value.detach().cpu().float().numpy().astype(np.float32, copy=False)")
    model_file.write_text(text)
    cfg = yaml.safe_load((native_checkpoint / "config.yaml").read_text())["model"]["robot_processor"]
    norm = json.loads((target / "norm_stats.json").read_text())
    metadata = cfg["metadata_by_tag"]["arx_x5_cleaned_rgb"]
    norm["metadata_by_tag"]["benchmark_arx"] = metadata
    dump(target / "norm_stats.json", norm)
    return target


class Timings:
    """CUDA events for GPU stages; wall times for CPU wrappers. Profile separately."""
    def __init__(self):
        self.events, self.wall, self.tokens = {}, {}, None

    def wrap(self, obj, method, name, gpu=True):
        original = getattr(obj, method)
        @functools.wraps(original)
        def call(*args, **kwargs):
            if name == "backbone":
                ids = kwargs.get("input_ids")
                if ids is not None:
                    self.tokens = int(ids.shape[-1])
            if gpu:
                first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                first.record()
            start = time.perf_counter()
            value = original(*args, **kwargs)
            self.wall.setdefault(name, []).append((time.perf_counter() - start) * 1000)
            if gpu:
                last.record()
                self.events.setdefault(name, []).append((first, last))
            return value
        if method == "__call__":
            cls = type(obj)
            class_original = cls.__call__
            def dispatch(instance, *args, **kwargs):
                if instance is obj:
                    return call(*args, **kwargs)
                return class_original(instance, *args, **kwargs)
            cls.__call__ = dispatch
        else:
            setattr(obj, method, call)

    def clear(self):
        self.events, self.wall = {}, {}

    def result(self):
        torch.cuda.synchronize()
        return {"gpu_stage_ms": {k: sum(a.elapsed_time(b) for a, b in v) for k, v in self.events.items()},
                "host_stage_ms": {k: sum(v) for k, v in self.wall.items()},
                "stage_calls": {k: len(v) for k, v in self.wall.items()}, "input_tokens": self.tokens,
                "note": "Nested stages overlap; profiling events are excluded from latency medians."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("rgb", "mae", "official"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[4, 10, 30])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cohort", type=Path, default=ROOT / "outputs/dlc/x5_cleaned_ablation_20261008")
    parser.add_argument("--official-model", type=Path, default=ROOT / "artifacts/models/MolmoAct2")
    parser.add_argument("--samples-dir", type=Path, help="Portable offline_samples manifest/NPZ directory, including the 5080 deployment samples")
    parser.add_argument("--reuse-mixed-log", type=Path, help="Reuse completed timing rows while adding a separate fixed-shape and profiling pass")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.samples_dir is not None:
        manifest = json.loads((args.samples_dir / "manifest.json").read_text())
        samples = []
        for item in manifest["samples"]:
            with np.load(args.samples_dir / item["file"], allow_pickle=False) as data:
                obs = {k: data[k].copy() for k in data.files if k.startswith("observation.")}
            obs["task"] = item["task"]
            samples.append({"episode": item["episode"], "frame": item["frame"], "obs": obs})
    else:
        samples, _ = prepare_samples(args.cohort / "dataset", args.cohort / "packed_cache", 2)
    if args.route != "mae":
        for s in samples:
            s["obs"] = {k: v for k, v in s["obs"].items() if not k.startswith("observation.images.") or k in X5_RGB_IMAGE_KEYS}
    checkpoint = ROOT / f"outputs/runs/molmoact2-x5-cleaned-{'mae' if args.route == 'mae' else 'rgb'}-fft20k-b64-s42-20261008/checkpoints/step20000-unsharded"
    print(json.dumps({"stage": "loading", "route": args.route}), flush=True)
    load_start = time.perf_counter()
    if args.route == "official":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        source = official_copy(args.official_model, args.output, checkpoint)
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, extra_special_tokens={})
        model = AutoModelForImageTextToText.from_pretrained(source, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
        def move_cast(inputs, device):
            return {k: v.to(device=device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                    if torch.is_tensor(v) else v for k, v in inputs.items()}
        model._move_inputs_to_device = move_cast
        def invoke(sample, steps, graph, count=30):
            images = [sample["obs"][k] for k in X5_RGB_IMAGE_KEYS]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                return model.predict_action(processor=processor, images=images,
                    task=sample["obs"]["task"], state=sample["obs"]["observation.state"],
                    norm_tag="benchmark_arx", inference_action_mode="continuous", enable_depth_reasoning=False,
                    normalize_language=False, num_steps=steps, n_action_steps=count,
                    generator=torch.Generator(device="cuda").manual_seed(42), enable_cuda_graph=graph).actions
        graph_options = [False, True]
    else:
        server = MolmoAct2Server(checkpoint=str(checkpoint), norm_tag=f"arx_x5_cleaned_{args.route}",
            image_keys="x5_tactile" if args.route == "mae" else "x5", device="cuda:0",
            parameter_dtype="bfloat16", enable_inference_cuda_graph=False)
        model = server.policy._handles.model
        def invoke(sample, steps, graph, count=30):
            return server.policy.generate_inference_result_from_observations(sample["obs"],
                norm_tag=server.default_norm_tag, num_steps=steps, n_action_steps=count,
                generator=torch.Generator(device="cuda").manual_seed(42)).actions
        graph_options = [False]
    load_s = time.perf_counter() - load_start
    print(json.dumps({"stage": "loaded", "route": args.route, "load_s": load_s}), flush=True)
    results = []
    if args.reuse_mixed_log is not None:
        for line in args.reuse_mixed_log.read_text(errors="replace").splitlines():
            if line.startswith("{"):
                row = json.loads(line)
                if row.get("stage") == "timing" and row.get("route") == args.route:
                    results.append({k: v for k, v in row.items() if k not in {"stage", "route"}})
    for graph in graph_options:
        for steps in args.steps:
            if args.reuse_mixed_log is not None:
                continue
            cold = []
            for s in samples:
                torch.cuda.synchronize(); start = time.perf_counter()
                invoke(s, steps, graph); torch.cuda.synchronize()
                cold.append((time.perf_counter() - start) * 1000)
            times = []
            for _ in range(args.repeats):
                for s in samples:
                    torch.cuda.synchronize(); start = time.perf_counter()
                    action = invoke(s, steps, graph); torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
                    assert action.shape == (1, 30, 14) and torch.isfinite(action).all()
            captured = bool(model.model.action_cuda_graph_manager.action_flow_graph) if args.route == "official" else False
            row = {"flow_steps": steps, "generated_horizon": 30, "returned_steps": 30,
                "cuda_graph_requested": graph, "cuda_graph_present": captured,
                "calls": len(times), "median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, .95)),
                "min_ms": min(times), "max_ms": max(times), "warmup_ms": cold}
            results.append(row); print(json.dumps({"stage": "timing", "route": args.route, **row}), flush=True)
    fixed_shape = []
    if args.route == "official":
        for graph in graph_options:
            for steps in args.steps:
                invoke(samples[0], steps, graph); torch.cuda.synchronize()
                times, recaptures = [], 0
                for _ in range(8):
                    manager = model.model.action_cuda_graph_manager
                    before = manager.action_flow_graph
                    torch.cuda.synchronize(); start = time.perf_counter()
                    invoke(samples[0], steps, graph); torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
                    recaptures += int(graph and manager.action_flow_graph is not before)
                row = {"flow_steps": steps, "cuda_graph_requested": graph,
                    "median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, .95)),
                    "calls": len(times), "graph_recaptures_during_measured_calls": recaptures}
                fixed_shape.append(row)
                print(json.dumps({"stage": "fixed_shape", "route": args.route, **row}), flush=True)
    # Return-length overrides slice the generated chunk; they do not skip denoising.
    execute_timings = []
    for count in (5, 15, 30):
        times = []
        for _ in range(3):
            torch.cuda.synchronize(); start = time.perf_counter()
            action = invoke(samples[0], 10, args.route == "official", count); torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            assert action.shape == (1, count, 14)
        execute_timings.append({"returned_steps": count, "generated_horizon": 30,
            "median_ms": float(np.median(times)), "execution_duration_ms_at_30hz": count / 30 * 1000})
    timer = Timings()
    if args.route != "official":
        timer.wrap(server.policy, "_collate_example", "preprocess", gpu=False)
        timer.wrap(model, "_run_backbone", "backbone")
        timer.wrap(model.vision_backbone, "forward", "rgb_encoder")
        timer.wrap(model.action_expert, "forward", "action_expert")
        timer.wrap(model.action_expert, "_prepare_kv_context", "kv_preparation")
        if args.route == "mae":
            timer.wrap(model.tactile_encoder, "forward", "tactile_encoder")
    else:
        timer.wrap(processor, "__call__", "preprocess", gpu=False)
        timer.wrap(model.model, "forward", "backbone")
        timer.wrap(model.model, "generate_actions_from_inputs", "model_generation")
        timer.wrap(model.model, "_run_action_flow_loop", "action_expert")
        timer.wrap(model.model._require_action_expert(), "prepare_context", "kv_preparation")
    timer.clear(); invoke(samples[0], 10, False); torch.cuda.synchronize()
    profiling = timer.result()
    report = {"route": args.route, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "dtype": str(next(model.parameters()).dtype), "samples": [{"episode": s["episode"], "frame": s["frame"]} for s in samples],
        "load_s": load_s, "results": results, "fixed_shape_results": fixed_shape,
        "mixed_shape_log_source": str(args.reuse_mixed_log) if args.reuse_mixed_log else None,
        "returned_length_timings": execute_timings, "profile": profiling,
        "official_weights_unchanged": args.route == "official", "official_input_note":
        "Official base weights and HF predict_action code; evaluation-only ARX metadata added to a copied norm_stats.json for identical 3-RGB/14-state inputs. Output quality is not evaluated.",
        "gpu_synchronized": True, "robot_commands_sent": False, "network_listener_started": False,
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(args.output / f"{args.route}.json", report)
    print(json.dumps({"stage": "complete", "route": args.route, "profile": profiling}), flush=True)


if __name__ == "__main__":
    main()
