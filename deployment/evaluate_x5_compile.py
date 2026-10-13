"""Compare compilation with recorded actions; never a held-out success-rate test."""
import argparse
import json
from pathlib import Path

import benchmark_x5_native as base  # installs hardware and network guards
import numpy as np
import torch
from benchmark_x5_incremental import configure
from olmo.nn.native_action_compile import compile_action_blocks
from openpi_client import image_tools
from scripts.run_x5_inference_comparison import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rgb", "mae"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.samples / "manifest.json").read_text())
    observations, targets = [], []
    for entry in manifest["samples"]:
        with np.load(args.samples / entry["file"]) as data:
            obs = {k: data[k].copy() for k in data.files if k.startswith("observation.")
                   and (args.mode == "mae" or "tactile" not in k)}
            targets.append(data["target"].copy())
        obs["task"] = entry["task"]
        for name in base.RGB_KEYS.values():
            key = "observation.images." + name
            obs[key] = image_tools.resize_with_pad(obs[key], 224, 224)
        observations.append(obs)
    bridge = base.build_bridge(args.mode)
    policy = bridge.molmo.policy
    model = policy._handles.model
    rows, arrays = [], {}
    for variant in ("previous_eager", "static_eager", "compiled_eager"):
        configure(model, variant)
        if variant == "compiled_eager":
            compile_action_blocks(model)
        predictions = []
        errors = []
        for i, (obs, target, entry) in enumerate(zip(observations, targets, manifest["samples"])):
            actions, _, _ = base.run_native(policy, obs, entry["seed"])
            predictions.append(actions)
            errors.append(metrics(actions, target, bridge.molmo.robot_processor, policy._handles.norm_tag))
        current = np.stack(predictions)
        arrays[variant] = current
        delta = np.abs(current - arrays["previous_eager"])
        joints = delta[:, :, [*range(6), *range(7, 13)]]
        grippers = delta[:, :, [6, 13]]
        row = {
            "variant": variant, "samples": len(current),
            "bitwise_equal": bool(np.array_equal(current, arrays["previous_eager"])),
            "max_abs_delta_rad": float(delta.max()), "mean_abs_delta_rad": float(delta.mean()),
            "joint_delta_max_rad": float(joints.max()), "joint_delta_p99_rad": float(np.percentile(joints, 99)),
            "gripper_delta_max_rad": float(grippers.max()),
            "policy_space_mse": float(np.mean([r["policy_space_mse"] for r in errors])),
            "first5_policy_space_mse": float(np.mean([r["policy_space_first5_mse"] for r in errors])),
            "joint_mae_to_recorded_rad": float(np.mean([r["joint_mae_raw"] for r in errors])),
            "per_sample_metrics": errors,
        }
        if variant == "static_eager" and not row["bitwise_equal"]:
            raise AssertionError("Exact optimization changed recorded actions")
        assert getattr(model, "_native_action_flow_graph", None) is None
        rows.append(row)
        print(json.dumps({"mode": args.mode, **{k:v for k,v in row.items() if k != "per_sample_metrics"}}), flush=True)
        (args.output / f"{args.mode}_compile_quality.json").write_text(json.dumps({
            "mode": args.mode, "scope": manifest["scope"], "rows": rows,
            "solver_steps": 10, "horizon": 30, "rgb_client_resize": [224, 224],
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
            "hardware_initializations": 0, "actual_action_sends": 0}, indent=2) + "\n")
    np.savez_compressed(args.output / f"{args.mode}_compile_actions.npz", **arrays, targets=np.stack(targets))


if __name__ == "__main__":
    main()
