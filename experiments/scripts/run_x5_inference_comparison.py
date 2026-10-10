#!/usr/bin/env python3
"""Replay actual cleaned observations through native final checkpoints.

This is a training-data replay and API smoke test, not a held-out evaluation or
a robot rollout. It never sends commands to a robot or opens a network listener.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/lerobot/src")]
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from fastapi.testclient import TestClient

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from olmo.data.packed_images import PackedImageReader
from olmo.data.tactile import X5_RGB_IMAGE_KEYS, tactile_keys_for_layout
from scripts.serve_policy import MolmoAct2Server


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_samples(dataset, cache, count):
    meta = LeRobotDatasetMetadata("local/hot_stamp_bag_merged_0918-20_cleaned", root=dataset)
    episodes = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    tasks = {r["task_index"]: r["task"] for r in pq.read_table(dataset / "meta/tasks.parquet").to_pylist()}
    reader = PackedImageReader(cache)
    selected = np.linspace(0, len(episodes) - 1, count, dtype=int).tolist()
    tables, samples = {}, []
    keys = [*X5_RGB_IMAGE_KEYS, *tactile_keys_for_layout("two")]
    for ep_idx in selected:
        ep = episodes[ep_idx]
        data_file = dataset / meta.get_data_file_path(ep_idx)
        if data_file not in tables:
            tables[data_file] = pq.read_table(data_file, columns=["episode_index", "frame_index", "observation.state", "action", "task_index"]).to_pandas()
        rows = tables[data_file]
        rows = rows[rows.episode_index == ep_idx].sort_values("frame_index").reset_index(drop=True)
        assert len(rows) == ep["length"]
        for fraction in (0.1, 0.5, 0.9):
            frame = int((len(rows) - 30) * fraction)
            images = reader.query(meta, {k: [frame / meta.fps] for k in keys}, ep_idx)
            obs = {k: v.permute(1, 2, 0).numpy() for k, v in images.items()}
            obs["observation.state"] = np.asarray(rows.iloc[frame]["observation.state"], dtype=np.float32)
            obs["task"] = tasks[int(rows.iloc[frame].task_index)]
            target = np.stack(rows.iloc[frame:frame + 30].action.to_list()).astype(np.float32)
            assert target.shape == (30, 14)
            samples.append({"episode": ep_idx, "frame": frame, "obs": obs, "target": target})
    reader.close()
    return samples, meta.info["features"]["action"]["names"]


def infer(policy, observation, seed, num_steps):
    generator = torch.Generator(device=policy._handles.device).manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = policy.generate_inference_result_from_observations(
        observation, norm_tag=policy._handles.norm_tag,
        num_steps=num_steps, n_action_steps=30, generator=generator
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    action = result.actions.detach().cpu().numpy()
    assert action.shape == (1, 30, 14), action.shape
    assert np.isfinite(action).all(), "Non-finite action output"
    return action[0], elapsed


def metrics(prediction, target, processor, tag):
    pred = processor.normalize_action(prediction, tag)
    gt = processor.normalize_action(target, tag)
    error = pred - gt
    raw_error = prediction - target
    joints = [*range(6), *range(7, 13)]
    return {
        "policy_space_mse": float(np.mean(error ** 2)),
        "policy_space_mae": float(np.mean(np.abs(error))),
        "policy_space_first5_mse": float(np.mean(error[:5] ** 2)),
        "policy_space_first_mse": float(np.mean(error[0] ** 2)),
        "joint_mae_raw": float(np.mean(np.abs(raw_error[:, joints]))),
        "joint_normalized_mae": float(np.mean(np.abs(error[:, joints]))),
        "gripper_mae_raw": float(np.mean(np.abs(raw_error[:, [6, 13]]))),
        "per_dimension_policy_space_mae": np.abs(error).mean(axis=0).tolist(),
    }


def image_payload(image):
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def test_http(server, sample, expected, seed):
    obs = sample["obs"]
    tag = server.default_norm_tag
    metadata = server.robot_processor.get_metadata(tag)
    keys = [*metadata["camera_keys"], *metadata.get("tactile_keys", [])]
    payload = {k.removeprefix("observation.images."): image_payload(obs[k]) for k in keys}
    payload.update(state=obs["observation.state"].tolist(), instruction=obs["task"], session_id="inference-test")
    with TestClient(server.app) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok"
        client.post("/reset", json={"session_id": "inference-test"}).raise_for_status()
        torch.manual_seed(seed)
        full = client.post("/act", json={**payload, "action_chunk": True, "n_action_steps": 30})
        full.raise_for_status()
        full_body = full.json()
        actual = np.asarray(full_body["actions"])
        assert actual.shape == (30, 14) and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        client.post("/reset", json={"session_id": "inference-test"}).raise_for_status()
        torch.manual_seed(seed)
        calls, queue_actions = [], []
        for _ in range(6):
            response = client.post("/act", json={**payload, "single_action": True})
            response.raise_for_status()
            body = response.json()
            vector = np.asarray(body["action"])
            assert vector.shape == (14,) and np.isfinite(vector).all()
            queue_actions.append(vector)
            calls.append(body["inference_calls"])
        assert calls == [1, 0, 0, 0, 0, 1], calls
        np.testing.assert_allclose(queue_actions[:5], expected[:5], rtol=1e-5, atol=1e-5)
        missing_sensor = None
        if metadata.get("tactile_keys"):
            missing = {k: v for k, v in payload.items() if k != metadata["tactile_keys"][0].removeprefix("observation.images.")}
            response = client.post("/act", json=missing)
            assert response.status_code >= 400 and "Missing required" in response.json()["error"]
            missing_sensor = {"status": response.status_code, "error": response.json()["error"]}
    return {"transport": "in-process ASGI HTTP; no network listener", "health": health.json(),
            "full_chunk_shape": full_body["action_shape"], "full_chunk_latency_ms": full_body["latency_ms"],
            "single_action_inference_calls": calls, "reset": "passed", "direct_api_parity": "passed",
            "missing_tactile_rejected": missing_sensor}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rgb", "mae"), required=True)
    parser.add_argument("--cohort", type=Path, default=ROOT / "outputs/dlc/x5_cleaned_ablation_20261008")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = args.output / args.mode
    out.mkdir(parents=True, exist_ok=True)
    samples, names = prepare_samples(args.cohort / "dataset", args.cohort / "packed_cache", args.episodes)
    if args.mode == "rgb":
        for sample in samples:
            sample["obs"] = {k: v for k, v in sample["obs"].items()
                             if not k.startswith("observation.images.") or k in X5_RGB_IMAGE_KEYS}
    save_json(out / "samples.json", [{"episode": s["episode"], "frame": s["frame"], "task": s["obs"]["task"]} for s in samples])
    checkpoint = ROOT / f"outputs/runs/molmoact2-x5-cleaned-{args.mode}-fft20k-b64-s42-20261008/checkpoints/step20000-unsharded"
    tag = f"arx_x5_cleaned_{args.mode}"
    start = time.perf_counter()
    print(json.dumps({"stage": "loading", "mode": args.mode, "checkpoint": str(checkpoint)}), flush=True)
    server = MolmoAct2Server(checkpoint=str(checkpoint), norm_tag=tag,
                            image_keys="x5_tactile" if args.mode == "mae" else "x5", device="cuda:0")
    policy = server.policy
    load_s = time.perf_counter() - start
    example = policy._combine_history_examples(
        [policy._obs_to_example(samples[0]["obs"])], policy._handles, norm_tag=tag
    )
    assert len(example["image"]) == 3
    if args.mode == "mae":
        assert example["tactile_windows"].shape[:3] == (1, 2, 1)
    print(json.dumps({"stage": "loaded", "mode": args.mode, "seconds": load_s}), flush=True)
    warmup, cold_s = infer(policy, samples[0]["obs"], args.seed, args.num_steps)
    torch.cuda.reset_peak_memory_stats()
    results, predictions = [], []
    for i, sample in enumerate(samples):
        prediction, elapsed = infer(policy, sample["obs"], args.seed + i, args.num_steps)
        record = {"episode": sample["episode"], "frame": sample["frame"], "latency_s": elapsed,
                  **metrics(prediction, sample["target"], server.robot_processor, tag)}
        results.append(record)
        predictions.append(prediction)
        print(json.dumps({"stage": "sample", "mode": args.mode, "index": i, **record}), flush=True)
        save_json(out / "sample_results.json", results)
    repeated, _ = infer(policy, samples[0]["obs"], args.seed, args.num_steps)
    np.testing.assert_allclose(repeated, predictions[0], rtol=1e-6, atol=1e-6)
    controls = []
    if args.mode == "mae":
        for i in (1, len(samples) // 2):
            sample = samples[i]
            for condition in ("blank", "swap"):
                obs = dict(sample["obs"])
                left, right = tactile_keys_for_layout("two")
                if condition == "blank":
                    obs[left], obs[right] = np.zeros_like(obs[left]), np.zeros_like(obs[right])
                else:
                    obs[left], obs[right] = obs[right], obs[left]
                action, _ = infer(policy, obs, args.seed + i, args.num_steps)
                delta = server.robot_processor.normalize_action(action, tag) - server.robot_processor.normalize_action(predictions[i], tag)
                controls.append({"episode": sample["episode"], "frame": sample["frame"], "condition": condition,
                                 "policy_space_output_delta_mae": float(np.abs(delta).mean()),
                                 **metrics(action, sample["target"], server.robot_processor, tag)})
    save_json(out / "tactile_controls.json", controls)
    http = test_http(server, samples[0], predictions[0], args.seed)
    # Profile BF16 using the same native call and thread-local autocast inside
    # the call itself, so direct and ASGI worker-thread inference both use it.
    original_call = policy._call_generate_inference_result
    def bf16_call(*call_args, **call_kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return original_call(*call_args, **call_kwargs)
    policy._call_generate_inference_result = bf16_call
    infer(policy, samples[0]["obs"], args.seed, args.num_steps)
    bf16_results = []
    bf16_first = None
    for i in np.linspace(0, len(samples) - 1, 4, dtype=int):
        action, elapsed = infer(policy, samples[i]["obs"], args.seed + int(i), args.num_steps)
        if i == 0:
            bf16_first = action
        bf16_results.append({"index": int(i), "latency_s": elapsed,
                             **metrics(action, samples[i]["target"], server.robot_processor, tag)})
    bf16_http = test_http(server, samples[0], bf16_first, args.seed)
    policy._call_generate_inference_result = original_call
    save_json(out / "bf16_profile.json", {"samples": bf16_results, "http": bf16_http,
              "median_latency_s": float(np.median([r["latency_s"] for r in bf16_results]))})
    np.savez_compressed(out / "actions.npz", predictions=np.stack(predictions), targets=np.stack([s["target"] for s in samples]))
    latencies = [r["latency_s"] for r in results]
    summary = {"mode": args.mode, "checkpoint": str(checkpoint), "dataset": str(args.cohort / "dataset"),
               "split": "training-data replay (all 337 episodes were used in training)",
               "episodes": args.episodes, "samples": len(samples), "seed": args.seed,
               "flow_steps": args.num_steps, "action_shape": [30, 14], "dimension_names": names,
               "rgb_cameras": list(X5_RGB_IMAGE_KEYS), "tactile_sensors": list(tactile_keys_for_layout("two")) if args.mode == "mae" else [],
               "device": torch.cuda.get_device_name(0), "parameter_dtype": str(next(policy._handles.model.parameters()).dtype),
               "load_s": load_s, "cold_inference_s": cold_s,
               "median_latency_s": float(np.median(latencies)), "p95_latency_s": float(np.quantile(latencies, .95)),
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "reproducibility": "passed", "finite_actions": True,
               "mean_metrics": {k: float(np.mean([r[k] for r in results])) for k in
                                ["policy_space_mse", "policy_space_mae", "policy_space_first5_mse", "policy_space_first_mse", "joint_mae_raw", "joint_normalized_mae", "gripper_mae_raw"]},
               "metric_scale": "12 joint dimensions q01/q99 normalized; 2 unnormalized grippers, matching training metadata",
               "bf16_profile_median_s": float(np.median([r["latency_s"] for r in bf16_results])),
               "http": http}
    save_json(out / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
