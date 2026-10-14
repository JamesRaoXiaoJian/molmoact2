"""Matched training-data action regression and sequential RTC replay, without hardware."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPPERS = [6, 13]


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def action_metrics(predictions, targets, states):
    """States are the observation poses, broadcast along each prediction horizon."""
    p, t, s = (np.asarray(x, dtype=np.float64) for x in (predictions, targets, states))
    if p.shape != t.shape or p.shape[-1] != 14 or s.shape != p.shape[:-2] + (14,):
        raise ValueError(
            "Expected matching (..., horizon, 14) predictions/targets and (..., 14) states"
        )
    if not all(np.isfinite(x).all() for x in (p, t, s)):
        raise ValueError("All action values must be finite")
    error = p - t
    joints = np.abs(error[..., JOINTS])
    motion = (t - s[..., None, :])[..., JOINTS]
    predicted_motion = (p - s[..., None, :])[..., JOINTS]
    moving = np.abs(motion) > 0.03
    moving_frame = moving.any(-1)
    hold_mse = float(np.mean(motion**2))
    return {
        "action_vectors": int(np.prod(p.shape[:-1])),
        "joint_mae_rad": float(joints.mean()),
        "joint_mae_deg": float(np.rad2deg(joints.mean())),
        "joint_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(joints**2)))),
        "joint_abs_error_p95_deg": float(np.rad2deg(np.quantile(joints, 0.95))),
        "worst_joint_per_action_p95_deg": float(
            np.rad2deg(np.quantile(joints.max(-1), 0.95))
        ),
        "all_12_joints_within_degrees": {
            str(deg): float((joints <= np.deg2rad(deg)).all(-1).mean())
            for deg in (1, 2, 5)
        },
        "individual_joint_within_degrees": {
            str(deg): float((joints <= np.deg2rad(deg)).mean()) for deg in (1, 2, 5)
        },
        "gripper_mae_rad": float(np.abs(error[..., GRIPPERS]).mean()),
        "per_gripper_mae_rad": np.abs(error[..., GRIPPERS])
        .reshape(-1, 2)
        .mean(0)
        .tolist(),
        "both_grippers_within_0p1_rad": float(
            (np.abs(error[..., GRIPPERS]) <= 0.1).all(-1).mean()
        ),
        "hold_baseline_joint_mae_deg": float(np.rad2deg(np.abs(motion).mean())),
        "joint_mse_reduction_vs_hold": None
        if hold_mse == 0
        else 1 - float(np.mean(joints**2)) / hold_mse,
        "moving_action_vectors": int(moving_frame.sum()),
        "moving_all_12_within_2deg": None
        if not moving_frame.any()
        else float((joints[moving_frame] <= np.deg2rad(2)).all(-1).mean()),
        "moving_component_count": int(moving.sum()),
        "moving_component_mae_deg": None
        if not moving.any()
        else float(np.rad2deg(joints[moving].mean())),
        "moving_direction_agreement": None
        if not moving.any()
        else float((motion[moving] * predicted_motion[moving] > 0).mean()),
        "moving_amplitude_median_ratio": None
        if not moving.any()
        else float(np.median(np.abs(predicted_motion[moving] / motion[moving]))),
        "moving_amplitude_projection_ratio": None
        if not moving.any()
        else float(
            np.sum(motion[moving] * predicted_motion[moving])
            / np.sum(motion[moving] ** 2)
        ),
        "per_joint_mae_deg": np.rad2deg(joints.reshape(-1, 12).mean(0)).tolist(),
    }


def infer(policy, observation, seed, rtc=None):
    import torch

    policy.reset()
    generator = torch.Generator(device=policy._handles.device).manual_seed(seed)
    started = time.perf_counter()
    result = policy.generate_inference_result_from_observations(
        observation,
        norm_tag=policy._handles.norm_tag,
        num_steps=10,
        n_action_steps=30,
        generator=generator,
        **({"rtc": rtc} if rtc is not None else {}),
    )
    action = result.actions.detach().cpu().numpy().copy()
    if action.shape != (1, 30, 14) or not np.isfinite(action).all():
        raise ValueError("Invalid predicted action chunk")
    return action[0], time.perf_counter() - started


def run(args):
    # Installs the existing device/import/network guards before model imports.
    import benchmark_x5_native as guarded
    import pyarrow.parquet as pq
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from olmo.data.packed_images import PackedImageReader
    from olmo.data.tactile import X5_RGB_IMAGE_KEYS, tactile_keys_for_layout
    from openpi_client import image_tools

    out = args.output / args.mode
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(
            "Use a new output directory instead of overwriting a completed evaluation"
        )
    dataset = args.cohort / "dataset"
    meta = LeRobotDatasetMetadata(
        "local/hot_stamp_bag_merged_0918-20_cleaned", root=dataset
    )
    tables = [
        pq.read_table(
            p,
            columns=[
                "episode_index",
                "frame_index",
                "observation.state",
                "action",
                "task_index",
            ],
        )
        for p in sorted((dataset / "data").rglob("*.parquet"))
    ]
    import pyarrow as pa

    table = pa.concat_tables(tables)
    ep = np.asarray(table["episode_index"])
    frame = np.asarray(table["frame_index"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    task = np.asarray(table["task_index"])
    tasks = {
        r["task_index"]: r["task"]
        for r in pq.read_table(dataset / "meta/tasks.parquet").to_pylist()
    }
    rows = {int(e): np.flatnonzero(ep == e) for e in np.unique(ep)}
    for indices in rows.values():
        assert np.array_equal(frame[indices], np.arange(len(indices)))
    episodes = np.linspace(0, len(rows) - 1, min(args.episodes, len(rows)), dtype=int)
    samples = []
    for episode in episodes:
        for bucket, fraction in enumerate((0.1, 0.5, 0.9)):
            f = int((len(rows[int(episode)]) - 30) * fraction)
            index = rows[int(episode)][f]
            samples.append(
                {
                    "episode": int(episode),
                    "frame": f,
                    "bucket": bucket,
                    "task_index": int(task[index]),
                    "seed": args.seed + len(samples),
                }
            )
    protocol = {
        "scope": "Training-data replay: all 337 episodes were used for training; not held-out or closed-loop",
        "sample_count": len(samples),
        "episodes": len(episodes),
        "flow_steps": 10,
        "precision": "bfloat16",
        "optimization": "graph",
        "compile_action_blocks": False,
        "action_horizon": 30,
        "rgb_resize": [224, 224],
        "prompt": "recorded dataset task",
        "moving_component_threshold_rad": 0.03,
        "seed": args.seed,
        "rtc": {
            "episodes": args.rtc_episodes,
            "requests_per_episode": args.rtc_requests,
            "observation_advance_frames": 12,
            "assumed_inference_delay_frames": 12,
            "assumed_inference_delay_ms": 400,
            "prefix_valid_steps": 18,
            "execution_horizon": 20,
            "max_guidance_weight": 10,
            "state_source": "recorded demonstration, not simulated policy feedback",
        },
        "hardware_initializations": 0,
        "robot_action_sends": 0,
        "device": torch.cuda.get_device_name(0),
        "samples": samples,
    }
    save(out / "protocol.json", protocol)
    reader = PackedImageReader(args.cohort / "packed_cache")
    keys = list(X5_RGB_IMAGE_KEYS) + (
        list(tactile_keys_for_layout("two")) if args.mode == "mae" else []
    )

    def observation(episode, f):
        indices = rows[int(episode)]
        assert 0 <= f and f + 30 <= len(indices)
        images = reader.query(meta, {key: [f / meta.fps] for key in keys}, int(episode))
        obs = {key: im.permute(1, 2, 0).numpy() for key, im in images.items()}
        for key in X5_RGB_IMAGE_KEYS:
            obs[key] = image_tools.resize_with_pad(obs[key], 224, 224)
        obs["observation.state"] = state[indices[f]].copy()
        obs["task"] = tasks[int(task[indices[f]])]
        return obs, action[indices[f : f + 30]].copy()

    os.environ["MOLMO_NATIVE_OPTIMIZATION"] = "graph"
    os.environ["MOLMO_COMPILE_ACTION_EXPERT"] = "0"
    print(
        json.dumps({"stage": "loading", "mode": args.mode, "samples": len(samples)}),
        flush=True,
    )
    bridge = guarded.build_bridge(args.mode)
    policy = bridge.molmo.policy
    print(json.dumps({"stage": "loaded", "mode": args.mode}), flush=True)
    first, _ = observation(samples[0]["episode"], samples[0]["frame"])
    infer(policy, first, args.seed)
    predictions, targets, states, elapsed = [], [], [], []
    with (out / "sample_metrics.jsonl").open("w") as records:
        for i, sample in enumerate(samples):
            obs, target = observation(sample["episode"], sample["frame"])
            prediction, seconds = infer(policy, obs, sample["seed"])
            predictions.append(prediction)
            targets.append(target)
            states.append(obs["observation.state"])
            elapsed.append(seconds)
            row = {
                **sample,
                "latency_s": seconds,
                **action_metrics(prediction[None], target[None], states[-1][None]),
            }
            records.write(json.dumps(row) + "\n")
            records.flush()
            if i % 25 == 0 or i + 1 == len(samples):
                progress = {
                    "stage": "independent",
                    "mode": args.mode,
                    "completed": i + 1,
                    "total": len(samples),
                    "mean_inference_s": float(np.mean(elapsed)),
                }
                save(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    predictions, targets, states = map(np.stack, (predictions, targets, states))
    repeated, _ = infer(policy, first, args.seed)
    assert np.array_equal(repeated, predictions[0]), "Seeded replay changed outputs"
    np.savez_compressed(
        out / "actions.npz", predictions=predictions, targets=targets, states=states
    )
    independent = action_metrics(predictions, targets, states)
    independent["hold_baseline"] = action_metrics(
        np.broadcast_to(states[:, None], targets.shape), targets, states
    )
    independent["by_horizon_index"] = {
        str(k): action_metrics(predictions[:, k : k + 1], targets[:, k : k + 1], states)
        for k in (0, 4, 9, 14, 19, 29)
    }
    independent["by_task"] = {
        str(t): action_metrics(predictions[mask], targets[mask], states[mask])
        for t in tasks
        if (mask := np.array([s["task_index"] == t for s in samples])).any()
    }
    save(out / "independent_summary.json", independent)

    rtc_predictions, unconstrained, rtc_targets, rtc_states, rtc_samples = (
        [],
        [],
        [],
        [],
        [],
    )
    rtc_observations = []
    rtc_episodes = np.linspace(0, len(rows) - 1, args.rtc_episodes, dtype=int)
    for group, episode in enumerate(rtc_episodes):
        needed = 30 + 12 * (args.rtc_requests - 1)
        start = int((len(rows[int(episode)]) - needed) * 0.5)
        for k in range(args.rtc_requests):
            f = start + 12 * k
            obs, target = observation(int(episode), f)
            seed = args.seed + 100000 + group * args.rtc_requests + k
            raw, _ = infer(policy, obs, seed)
            rtc_observations.append(obs)
            unconstrained.append(raw)
            rtc_targets.append(target)
            rtc_states.append(obs["observation.state"])
            rtc_samples.append(
                {"episode": int(episode), "frame": f, "request": k, "seed": seed}
            )
    # Both conditions reset observation history and use explicit per-sample RNG.
    # Group calls to avoid alternating/re-capturing conditional/unconditional graphs.
    for group, _episode in enumerate(rtc_episodes):
        previous = None
        for k in range(args.rtc_requests):
            index = group * args.rtc_requests + k
            obs, raw, seed = (
                rtc_observations[index],
                unconstrained[index],
                rtc_samples[index]["seed"],
            )
            prefix = np.zeros((30, 14), dtype=np.float32)
            if previous is not None:
                prefix[:18] = previous[12:]
            options = {
                "prefix_valid_steps": 0 if previous is None else 18,
                "inference_delay": 0 if previous is None else 12,
                "execution_horizon": 20,
                "max_guidance_weight": 10.0,
            }
            guided, _ = infer(
                policy, obs, seed, {"previous_actions": prefix, "options": options}
            )
            if previous is None:
                assert np.array_equal(raw, guided), "Empty RTC prefix changed baseline"
            previous = guided
            rtc_predictions.append(guided)
        print(
            json.dumps(
                {
                    "stage": "rtc",
                    "mode": args.mode,
                    "episodes_completed": group + 1,
                    "episodes_total": args.rtc_episodes,
                }
            ),
            flush=True,
        )
    rp, up, rt, rs = map(
        np.stack, (rtc_predictions, unconstrained, rtc_targets, rtc_states)
    )
    np.savez_compressed(
        out / "rtc_actions.npz", guided=rp, unguided=up, targets=rt, states=rs
    )
    save(out / "rtc_samples.json", rtc_samples)
    continuing = np.array([s["request"] > 0 for s in rtc_samples])
    rtc_summary = {
        "assumptions": protocol["rtc"],
        "requests": len(rp),
        "guided_full_chunk": action_metrics(rp, rt, rs),
        "unguided_full_chunk": action_metrics(up, rt, rs),
        "guided_future_after_delay": action_metrics(
            rp[continuing, 12:], rt[continuing, 12:], rs[continuing]
        ),
        "unguided_future_after_delay": action_metrics(
            up[continuing, 12:], rt[continuing, 12:], rs[continuing]
        ),
    }
    save(
        out / "summary.json",
        {
            "scope": protocol["scope"],
            "mode": args.mode,
            "independent": independent,
            "rtc": rtc_summary,
            "seed_reproducibility": "exact array equality passed",
            "empty_rtc_prefix_parity": "exact array equality passed",
        },
    )
    reader.close()
    print(
        json.dumps({"stage": "complete", "mode": args.mode, **independent}), flush=True
    )


def summarize(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {
        mode: json.loads((args.output / mode / "summary.json").read_text())
        for mode in ("rgb", "mae")
    }
    a = {mode: np.load(args.output / mode / "actions.npz") for mode in rows}
    assert np.array_equal(a["rgb"]["targets"], a["mae"]["targets"])
    assert np.array_equal(a["rgb"]["states"], a["mae"]["states"])
    manifest = json.loads((args.output / "rgb/protocol.json").read_text())
    other = json.loads((args.output / "mae/protocol.json").read_text())
    assert manifest["samples"] == other["samples"]
    paired = {}
    for mode, row in rows.items():
        err = np.abs(a[mode]["predictions"] - a[mode]["targets"])[:, :, JOINTS]
        paired[mode] = err.mean((1, 2))
        p, t, s = (a[mode][key] for key in ("predictions", "targets", "states"))
        limiter = {}
        for limit in (0.02, 0.03, 0.04, 0.05):
            current = s.copy()
            replay = p.copy()
            clipped_frames = 0
            for k in range(30):
                delta = p[:, k, JOINTS] - current[:, JOINTS]
                clipped_frames += int((np.abs(delta) > limit).any(-1).sum())
                current[:, JOINTS] += np.clip(delta, -limit, limit)
                replay[:, k, JOINTS] = current[:, JOINTS]
            result = action_metrics(replay, t, s)
            limiter[str(limit)] = {
                "frame_clip_fraction": clipped_frames / (len(p) * 30),
                "joint_mae_deg": result["joint_mae_deg"],
                "moving_all_12_within_2deg": result["moving_all_12_within_2deg"],
                "end_of_chunk_joint_mae_deg": float(
                    np.rad2deg(np.abs(replay[:, -1, JOINTS] - t[:, -1, JOINTS]).mean())
                ),
            }
        row["ideal_joint_limiter_replay"] = {
            "scope": "Each chunk starts from recorded state; instantaneous perfect actuator; joints only; no RTC, SDK or wall-clock delay",
            "limits_rad": limiter,
        }
        row["independent"]["first_target_exceeds_0p25_rad_fraction"] = float(
            (np.abs(p[:, 0, JOINTS] - s[:, JOINTS]) > 0.25).any(-1).mean()
        )
        import yaml

        config_path = (
            ROOT
            / f"outputs/runs/molmoact2-x5-cleaned-{mode}-fft20k-b64-s42-20261008/checkpoints/step20000-unsharded/config.yaml"
        )
        processor = yaml.safe_load(config_path.read_text())["model"]["robot_processor"]
        assert processor["norm_mode"] == "q01_q99"
        stats = processor["metadata_by_tag"][f"arx_x5_cleaned_{mode}"]["action_stats"]
        lower, upper = np.array(stats["q01"]), np.array(stats["q99"])
        bounded = np.clip(t, lower, upper)
        bounds_error = np.abs(bounded - t)[:, :, JOINTS]
        assert (p[:, :, JOINTS] >= lower[JOINTS] - 1e-6).all()
        assert (p[:, :, JOINTS] <= upper[JOINTS] + 1e-6).all()
        examples = []
        for index in np.argsort(bounds_error.mean((1, 2)))[-3:][::-1]:
            joint = JOINTS[int(bounds_error[index].mean(0).argmax())]
            examples.append(
                {
                    **manifest["samples"][index],
                    "joint": ("left_" if joint < 7 else "right_")
                    + f"joint_{joint % 7 + 1}",
                    "allowed_angle_deg": np.rad2deg(
                        [lower[joint], upper[joint]]
                    ).tolist(),
                    "target_angle_deg_range": np.rad2deg(
                        [t[index, :, joint].min(), t[index, :, joint].max()]
                    ).tolist(),
                    "predicted_angle_deg_range": np.rad2deg(
                        [p[index, :, joint].min(), p[index, :, joint].max()]
                    ).tolist(),
                }
            )
        row["normalization_bounds"] = {
            "source": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "mode": "q01_q99",
            "inference_clips_normalized_values": True,
            "training_normalization_also_clips": True,
            "target_action_fraction_any_joint_outside_bounds": float(
                (bounds_error > 1e-6).any(-1).mean()
            ),
            "minimum_possible_joint_mae_deg_under_bounds": float(
                np.rad2deg(bounds_error.mean())
            ),
            "maximum_possible_all12_within2deg_under_bounds": float(
                (bounds_error <= np.deg2rad(2)).all(-1).mean()
            ),
            "examples": examples,
        }
    worst = np.argsort(np.maximum(paired["rgb"], paired["mae"]))[-10:][::-1]
    report = {
        "protocol": manifest,
        "results": rows,
        "paired_mae_better_chunk_fraction": float(
            (paired["mae"] < paired["rgb"]).mean()
        ),
        "worst_chunks": [
            {
                **manifest["samples"][i],
                "rgb_joint_mae_deg": float(np.rad2deg(paired["rgb"][i])),
                "mae_joint_mae_deg": float(np.rad2deg(paired["mae"][i])),
            }
            for i in worst
        ],
        "evaluation_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    save(args.output / "comparison.json", report)
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "joint_mae_deg",
                "all_12_joints_within_2deg_percent",
                "moving_all_12_within_2deg_percent",
                "gripper_mae_rad",
                "moving_direction_agreement_percent",
                "moving_amplitude_median_ratio",
                "rtc_future_joint_mae_deg",
                "matched_unguided_future_joint_mae_deg",
            ]
        )
        for mode, row in rows.items():
            i, r = row["independent"], row["rtc"]
            writer.writerow(
                [
                    mode,
                    i["joint_mae_deg"],
                    100 * i["all_12_joints_within_degrees"]["2"],
                    100 * i["moving_all_12_within_2deg"],
                    i["gripper_mae_rad"],
                    100 * i["moving_direction_agreement"],
                    i["moving_amplitude_median_ratio"],
                    r["guided_future_after_delay"]["joint_mae_deg"],
                    r["unguided_future_after_delay"]["joint_mae_deg"],
                ]
            )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for mode, label, color in (
        ("rgb", "RGB", "#2563eb"),
        ("mae", "Tactile MAE", "#d97706"),
    ):
        d = a[mode]
        e = np.rad2deg(np.abs(d["predictions"] - d["targets"])[:, :, JOINTS])
        axes[0, 0].plot(np.arange(30), e.mean((0, 2)), label=label, color=color)
        axes[0, 1].plot(
            np.arange(30), 100 * (e <= 2).all(-1).mean(0), label=label, color=color
        )
        i = rows[mode]["independent"]
        axes[1, 0].plot(
            np.arange(12), i["per_joint_mae_deg"], "o-", label=label, color=color
        )
    baseline = np.rad2deg(
        np.abs(a["rgb"]["targets"] - a["rgb"]["states"][:, None])[:, :, JOINTS]
    )
    axes[0, 0].plot(
        np.arange(30),
        baseline.mean((0, 2)),
        "--",
        color="gray",
        label="Hold current pose",
    )
    axes[0, 1].plot(
        np.arange(30),
        100 * (baseline <= 2).all(-1).mean(0),
        "--",
        color="gray",
        label="Hold current pose",
    )
    for mode, x, color in (("rgb", 0, "#2563eb"), ("mae", 1, "#d97706")):
        r = rows[mode]["rtc"]
        axes[1, 1].bar(
            [x - 0.17, x + 0.17],
            [
                r["unguided_future_after_delay"]["joint_mae_deg"],
                r["guided_future_after_delay"]["joint_mae_deg"],
            ],
            width=0.32,
            color=[color, color],
            alpha=0.8,
        )
    axes[0, 0].set(
        title="Joint error by horizon",
        xlabel="Prediction index (30 Hz)",
        ylabel="Mean absolute error (degrees)",
    )
    axes[0, 1].set(
        title="All 12 joints within 2 degrees",
        xlabel="Prediction index (30 Hz)",
        ylabel="Action vectors (%)",
    )
    axes[1, 0].set(
        title="Per-joint error",
        xlabel="Left joints 1-6, right joints 1-6",
        ylabel="Mean absolute error (degrees)",
    )
    axes[1, 0].set_xticks(
        np.arange(12), [f"{s}{j}" for s in ("L", "R") for j in range(1, 7)]
    )
    axes[1, 1].set(
        title="Sequential replay: unconstrained / RTC",
        ylabel="Future joint MAE after 400 ms (degrees)",
    )
    axes[1, 1].set_xticks(
        [-0.17, 0.17, 0.83, 1.17], ["RGB plain", "RGB RTC", "MAE plain", "MAE RTC"]
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    for axis in axes.flat[:3]:
        axis.legend()
    fig.suptitle(
        "X5 training-data replay | not held-out accuracy or robot task success",
        fontsize=13,
    )
    fig.savefig(args.output / "action_accuracy.png", dpi=170)
    fig.savefig(args.output / "action_accuracy.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(
        min(4, len(worst)), 2, figsize=(12, 10), squeeze=False, constrained_layout=True
    )
    for pair, index in zip(axes, worst[:4], strict=False):
        errors = np.maximum(
            np.abs(a["rgb"]["predictions"][index] - a["rgb"]["targets"][index]),
            np.abs(a["mae"]["predictions"][index] - a["mae"]["targets"][index]),
        )
        joint_indices = np.asarray(JOINTS)[np.argsort(errors[:, JOINTS].mean(0))[-2:]]
        sample = manifest["samples"][index]
        for axis, joint in zip(pair, joint_indices, strict=True):
            axis.plot(
                np.arange(30) / 30,
                np.rad2deg(a["rgb"]["targets"][index, :, joint]),
                "k--",
                label="Demonstration",
            )
            for mode, color in (("rgb", "#2563eb"), ("mae", "#d97706")):
                axis.plot(
                    np.arange(30) / 30,
                    np.rad2deg(a[mode]["predictions"][index, :, joint]),
                    color=color,
                    label=mode.upper(),
                )
            joint_name = ("L" if joint < 7 else "R") + str(joint % 7 + 1)
            axis.set(
                title=f"Episode {sample['episode']}, frame {sample['frame']}, {joint_name}",
                xlabel="Horizon (seconds)",
                ylabel="Joint angle (degrees)",
            )
            axis.grid(alpha=0.2)
            axis.legend()
    fig.suptitle(
        "Largest-error sampled chunks | raw model output, before client limiting"
    )
    fig.savefig(args.output / "worst_action_chunks.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                m: {"independent": rows[m]["independent"], "rtc": rows[m]["rtc"]}
                for m in rows
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rgb", "mae", "summarize"), required=True)
    parser.add_argument(
        "--cohort", type=Path, default=ROOT / "outputs/dlc/x5_cleaned_ablation_20261008"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=337)
    parser.add_argument("--rtc-episodes", type=int, default=8)
    parser.add_argument("--rtc-requests", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261014)
    arguments = parser.parse_args()
    (summarize if arguments.mode == "summarize" else run)(arguments)
