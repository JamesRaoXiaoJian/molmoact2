#!/usr/bin/env python3
"""Summarize the matched native X5 replay, including standalone plots."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reports, metrics, samples, arrays = {}, {}, {}, {}
    for mode in ("rgb", "mae"):
        root = args.output / mode
        reports[mode] = json.loads((root / "summary.json").read_text())
        metrics[mode] = json.loads((root / "sample_results.json").read_text())
        samples[mode] = json.loads((root / "samples.json").read_text())
        arrays[mode] = np.load(root / "actions.npz")
    assert samples["rgb"] == samples["mae"]
    np.testing.assert_array_equal(arrays["rgb"]["targets"], arrays["mae"]["targets"])
    assert reports["rgb"]["seed"] == reports["mae"]["seed"]
    assert reports["rgb"]["flow_steps"] == reports["mae"]["flow_steps"]
    paired = []
    for index, sample in enumerate(samples["rgb"]):
        paired.append({**sample, "rgb_policy_space_mse": metrics["rgb"][index]["policy_space_mse"],
                       "mae_policy_space_mse": metrics["mae"][index]["policy_space_mse"],
                       "mae_minus_rgb_mse": metrics["mae"][index]["policy_space_mse"] - metrics["rgb"][index]["policy_space_mse"]})
    with (args.output / "paired_errors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    controls = json.loads((args.output / "mae/tactile_controls.json").read_text())
    result = {
        "scope": "Matched replay on training data; no held-out or physical robot success measurement.",
        "samples": len(paired), "reports": reports, "tactile_controls": controls,
        "mae_lower_replay_mse_samples": sum(p["mae_minus_rgb_mse"] < 0 for p in paired),
        "paired_mean_mse_difference": float(np.mean([p["mae_minus_rgb_mse"] for p in paired])),
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    names = reports["rgb"]["dimension_names"]
    labels = [f"L{i+1}" for i in range(6)] + ["L grip"] + [f"R{i+1}" for i in range(6)] + ["R grip"]
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(14)
    for mode, offset, color in [("rgb", -.18, "#0072B2"), ("mae", .18, "#E69F00")]:
        errors = np.asarray([r["per_dimension_policy_space_mae"] for r in metrics[mode]])
        ax.bar(x + offset, errors.mean(0), width=.36, label="RGB" if mode == "rgb" else "Tactile-MAE", color=color)
    ax.set(xticks=x, xticklabels=labels, ylabel="Mean absolute error (joints normalized, grippers raw)",
           title=f"Training-data replay: {len(paired)} observations, identical flow noise seeds")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "action_error.png", dpi=180)
    fig.savefig(args.output / "action_error.pdf")
    plt.close(fig)

    index = len(paired) // 2 + 1
    fig, axes = plt.subplots(7, 2, figsize=(12, 14), sharex=True)
    for dim, ax in enumerate(axes.flat):
        ax.plot(arrays["rgb"]["targets"][index, :, dim], color="black", label="Recorded target")
        ax.plot(arrays["rgb"]["predictions"][index, :, dim], color="#0072B2", label="RGB")
        ax.plot(arrays["mae"]["predictions"][index, :, dim], color="#E69F00", label="Tactile-MAE")
        ax.set_title(names[dim], fontsize=9)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Action horizon step (30 Hz)")
    fig.suptitle(f"Training-data replay: episode {paired[index]['episode']}, frame {paired[index]['frame']}\nRaw joint/gripper values; 30-step predicted chunk", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.output / "action_chunk.png", dpi=160)
    fig.savefig(args.output / "action_chunk.pdf")
    plt.close(fig)
    print(json.dumps({"samples": len(paired), "mean_metrics": {m: r["mean_metrics"] for m, r in reports.items()},
                      "median_latency_s": {m: r["median_latency_s"] for m, r in reports.items()},
                      "mae_lower_replay_mse_samples": result["mae_lower_replay_mse_samples"]}, indent=2))


if __name__ == "__main__":
    main()
