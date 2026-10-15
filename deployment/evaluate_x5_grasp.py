"""Recorded-observation grasp-transition inference, never a robot success test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from evaluate_x5_actions import ROOT, action_metrics, infer, save

THRESHOLD = -0.75  # Existing right-gripper close-offset trigger, not contact detection.


def crossing(values, threshold=THRESHOLD, dwell=3):
    """First sustained threshold crossing; zero means already closed at window start."""
    values = np.asarray(values)
    hits = np.flatnonzero(
        np.convolve((values >= threshold).astype(int), np.ones(dwell, int), "valid")
        == dwell
    )
    return int(hits[0]) if len(hits) else None


def reversals(values, band=0.1):
    """Count direction reversals with at least band radians of opposite excursion."""
    values = np.asarray(values)
    if not len(values):
        return 0
    origin = extreme = float(values[0])
    direction = count = 0
    for v in values[1:]:
        v = float(v)
        if direction == 0:
            if abs(v - origin) >= band:
                direction = 1 if v > origin else -1
                extreme = v
        elif direction > 0:
            extreme = max(extreme, v)
            if extreme - v >= band:
                count += 1
                direction = -1
                extreme = v
        else:
            extreme = min(extreme, v)
            if v - extreme >= band:
                count += 1
                direction = 1
                extreme = v
    return count


def closure_metrics(predictions, targets):
    p, t = np.asarray(predictions), np.asarray(targets)
    if (
        p.shape != t.shape
        or p.ndim != 2
        or not np.isfinite(p).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError(
            "Expected matching finite [windows, time] gripper trajectories"
        )
    pc, tc = [crossing(v) for v in p], [crossing(v) for v in t]
    eligible = [i for i, c in enumerate(tc) if c is not None]
    detected = [i for i in eligible if pc[i] is not None]
    delays = np.array([pc[i] - tc[i] for i in detected], dtype=float)
    timely = [i for i in detected if abs(pc[i] - tc[i]) <= 5]
    rev, refrev = (
        np.array([reversals(v) for v in p]),
        np.array([reversals(v) for v in t]),
    )
    reopen = [i for i in detected if np.any(p[i, pc[i] + 3 :] < THRESHOLD - 0.1)]
    return {
        "windows": len(p),
        "reference_closing_windows": len(eligible),
        "predicted_crossings": len(detected),
        "threshold_arrival_rate": len(detected) / len(eligible) if eligible else None,
        "arrival_within_5_frames_rate": len(timely) / len(eligible)
        if eligible
        else None,
        "missed_closures": len(eligible) - len(detected),
        "matched_delay_median_frames": float(np.median(delays))
        if len(delays)
        else None,
        "matched_absolute_timing_error_mean_frames": float(abs(delays).mean())
        if len(delays)
        else None,
        "right_gripper_mae_rad": float(abs(p - t).mean()),
        "endpoint_mae_rad": float(abs(p[:, -1] - t[:, -1]).mean()),
        "endpoint_within_0p1_rad": float((abs(p[:, -1] - t[:, -1]) <= 0.1).mean()),
        "mean_reversals_over_0p1_rad": float(rev.mean()),
        "reference_mean_reversals_over_0p1_rad": float(refrev.mean()),
        "mean_excess_reversals": float(np.maximum(rev - refrev, 0).mean()),
        "reopen_after_predicted_close_count": len(reopen),
    }


def select_events(states, episodes, rows, partitions):
    events = []
    for episode, indices in rows.items():
        grip = states[indices, 13]
        # Search the first 20 seconds for an open->closed transition. No phase model or labels.
        for f in range(15, min(600, len(indices) - 16)):
            if (
                grip[f - 1] < THRESHOLD <= grip[f]
                and np.all(grip[f : f + 3] >= THRESHOLD)
                and np.min(grip[max(0, f - 90) : f]) < -1.0
            ):
                events.append(
                    {
                        "episode": episode,
                        "frame": f - 15,
                        "crossing_frame": f,
                        "split": partitions[episode],
                        "seed": 20261015 + len(events),
                    }
                )
                break
    return events


def run(args):
    import benchmark_x5_native as guarded
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from olmo.data.packed_images import PackedImageReader
    from olmo.data.tactile import X5_RGB_IMAGE_KEYS
    from openpi_client import image_tools

    out = args.output / args.variant
    out.mkdir(parents=True, exist_ok=True)
    if (out / "grasp_summary.json").exists():
        raise FileExistsError("Completed inference results must not be overwritten")
    cohort = ROOT / "outputs/dlc/x5_cleaned_ablation_20261008"
    dataset = cohort / "dataset"
    meta = LeRobotDatasetMetadata(
        "local/hot_stamp_bag_merged_0918-20_cleaned", root=dataset
    )
    table = pq.read_table(dataset / "data/chunk-000/file-000.parquet")
    states = np.array(table["observation.state"].to_pylist(), np.float32)
    actions = np.array(table["action"].to_pylist(), np.float32)
    ep = np.asarray(table["episode_index"])
    task = np.asarray(table["task_index"])
    tasks = {
        r["task_index"]: r["task"]
        for r in pq.read_table(dataset / "meta/tasks.parquet").to_pylist()
    }
    rows = {int(e): np.flatnonzero(ep == e) for e in np.unique(ep)}
    splits = json.loads(
        (ROOT / "outputs/dlc/x5_contract_v2_20261014/splits.json").read_text()
    )["splits"]
    partitions = {e: s for s, ids in splits.items() for e in ids}
    events = select_events(states, ep, rows, partitions)
    grouped = []
    for split in ["train", "val", "test"]:
        candidates = [e for e in events if e["split"] == split]
        grouped.extend(
            candidates[i]
            for i in np.linspace(
                0, len(candidates) - 1, min(4, len(candidates)), dtype=int
            )
        )
    plan = {
        "events": events,
        "sequence_events": grouped,
        "threshold_rad": THRESHOLD,
        "threshold_dwell_frames": 3,
        "reversal_hysteresis_rad": 0.1,
        "timely_tolerance_frames": 5,
        "fps": 30,
        "horizon": 30,
        "sequence_requests": 10,
        "request_stride": 12,
        "assumed_latency_frames": 12,
        "scope": "Demonstration observations remain fixed; no simulated state feedback, physical contact, or robot success measurement. All old/warmstart weights saw these episodes before adaptation.",
        "hardware_initializations": 0,
        "robot_action_sends": 0,
    }
    save(out / "grasp_protocol.json", plan)
    reader = PackedImageReader(cohort / "packed_cache")
    legacy = args.variant == "legacy"
    os.environ["MOLMO_INPUT_CONTRACT"] = "legacy" if legacy else "aligned"
    os.environ["MOLMO_NATIVE_OPTIMIZATION"] = "graph"
    os.environ["MOLMO_COMPILE_ACTION_EXPERT"] = "0"
    checkpoint = args.checkpoint
    tag = (
        "arx_x5_cleaned_v2_rgb"
        if args.variant.startswith("step")
        else "arx_x5_cleaned_rgb"
    )
    print(
        json.dumps(
            {"stage": "loading", "variant": args.variant, "events": len(events)}
        ),
        flush=True,
    )
    bridge = guarded.build_bridge("rgb", checkpoint=checkpoint, norm_tag=tag)
    policy = bridge.molmo.policy
    plan.update(
        checkpoint=str(policy.config.checkpoint_dir),
        checkpoint_config_sha256=hashlib.sha256(
            (policy.config.checkpoint_dir / "config.yaml").read_bytes()
        ).hexdigest(),
        norm_tag=tag,
        rgb_geometry=bridge.metadata["rgb_geometry"],
        task_normalization=bridge.metadata["task_normalization"],
        flow_steps=10,
        precision="bfloat16",
        optimization="native CUDA Graph",
    )
    save(out / "grasp_protocol.json", plan)

    def observe(episode, frame):
        indices = rows[episode]
        if frame < 0 or frame + 30 > len(indices):
            raise ValueError("Invalid episode window")
        images = reader.query(
            meta, {k: [frame / 30] for k in X5_RGB_IMAGE_KEYS}, episode
        )
        obs = {k: v.permute(1, 2, 0).numpy() for k, v in images.items()}
        if legacy:
            for k in X5_RGB_IMAGE_KEYS:
                obs[k] = image_tools.resize_with_pad(obs[k], 224, 224)
        obs["observation.state"] = states[indices[frame]].copy()
        obs["task"] = tasks[int(task[indices[frame]])]
        return obs, actions[indices[frame : frame + 30]].copy()

    predictions = []
    targets = []
    initial = []
    for i, event in enumerate(events):
        obs, target = observe(event["episode"], event["frame"])
        pred, _ = infer(policy, obs, event["seed"])
        predictions.append(pred)
        targets.append(target)
        initial.append(obs["observation.state"])
        if i % 50 == 0:
            print(
                json.dumps({"stage": "closure", "done": i + 1, "total": len(events)}),
                flush=True,
            )
    p, t, s = map(np.stack, (predictions, targets, initial))
    np.savez_compressed(out / "closure_actions.npz", predictions=p, targets=t, states=s)
    report = {
        "variant": args.variant,
        "checkpoint": plan["checkpoint"],
        "scope": plan["scope"],
        "closure": closure_metrics(p[:, :, 13], t[:, :, 13]),
        "all_actions_at_closure": action_metrics(p, t, s),
        "by_split": {},
    }
    for split in splits:
        mask = np.array([e["split"] == split for e in events])
        report["by_split"][split] = {
            "closure": closure_metrics(p[mask, :, 13], t[mask, :, 13]),
            "actions": action_metrics(p[mask], t[mask], s[mask]),
        }
    save(out / "closure_summary.json", report)

    seq_obs = []
    seq_targets = []
    seq_states = []
    seq_plain = []
    seq_rtc = []
    seq_samples = []
    for group, event in enumerate(grouped):
        start = max(0, event["crossing_frame"] - 60)
        for k in range(10):
            f = start + 12 * k
            obs, target = observe(event["episode"], f)
            seed = 20262015 + group * 10 + k
            raw, _ = infer(policy, obs, seed)
            seq_obs.append(obs)
            seq_targets.append(target)
            seq_states.append(obs["observation.state"])
            seq_plain.append(raw)
            seq_samples.append(
                {
                    "episode": event["episode"],
                    "frame": f,
                    "request": k,
                    "seed": seed,
                    "split": event["split"],
                }
            )
    for group, event in enumerate(grouped):
        previous = None
        for k in range(10):
            i = group * 10 + k
            prefix = np.zeros((30, 14), np.float32)
            if previous is not None:
                prefix[:18] = previous[12:]
            options = {
                "prefix_valid_steps": 0 if previous is None else 18,
                "inference_delay": 0 if previous is None else 12,
                "execution_horizon": 20,
                "max_guidance_weight": 10.0,
            }
            pred, _ = infer(
                policy,
                seq_obs[i],
                seq_samples[i]["seed"],
                {"previous_actions": prefix, "options": options},
            )
            if previous is None:
                assert np.array_equal(pred, seq_plain[i])
            seq_rtc.append(pred)
            previous = pred
        print(
            json.dumps({"stage": "sequence", "done": group + 1, "total": len(grouped)}),
            flush=True,
        )
    plain, rtc, truth, obsstates = map(
        np.stack, (seq_plain, seq_rtc, seq_targets, seq_states)
    )
    np.savez_compressed(
        out / "grasp_sequence_actions.npz",
        plain=plain,
        rtc=rtc,
        targets=truth,
        states=obsstates,
    )
    save(out / "grasp_sequence_samples.json", seq_samples)
    report["sequence"] = {}
    gt_stitched = truth[:, 12:24, 13].reshape(len(grouped), -1)
    for name, pred in [("plain", plain), ("rtc", rtc)]:
        stitched = pred[:, 12:24, 13].reshape(len(grouped), -1)
        boundary = abs(np.diff(stitched, axis=1)[:, 11::12])
        report["sequence"][name] = {
            "closure": closure_metrics(stitched, gt_stitched),
            "future_actions": action_metrics(
                pred[:, 12:24], truth[:, 12:24], obsstates
            ),
            "right_chunk_boundary_abs_p95_rad": float(np.quantile(boundary, 0.95)),
            "right_chunk_boundary_abs_max_rad": float(boundary.max()),
            "right_chunk_boundary_over_0p1_fraction": float((boundary > 0.1).mean()),
        }
    # Encoding sensitivity: it diagnoses observability, not task correctness.
    probe = []
    for angle in [-2.2, -1.8, -1.2]:
        obs, _ = observe(events[0]["episode"], events[0]["frame"])
        obs["observation.state"][13] = angle
        pred, _ = infer(policy, obs, 42)
        probe.append(pred)
    report["right_state_counterfactual"] = {
        "angles_rad": [-2.2, -1.8, -1.2],
        "first_right_gripper_target_rad": [float(v[0, 13]) for v in probe],
        "last_right_gripper_target_rad": [float(v[-1, 13]) for v in probe],
        "max_action_difference_rad": float(
            max(abs(probe[0] - v).max() for v in probe[1:])
        ),
        "outputs_all_equal": all(np.array_equal(probe[0], v) for v in probe[1:]),
    }
    save(out / "grasp_summary.json", report)
    print(
        json.dumps(
            {
                "stage": "complete",
                "variant": args.variant,
                "closure": report["closure"],
                "sequence": report["sequence"],
            }
        ),
        flush=True,
    )
    reader.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        choices=["legacy", "aligned", "step2000", "step5000"],
        required=True,
    )
    p.add_argument("--checkpoint")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
