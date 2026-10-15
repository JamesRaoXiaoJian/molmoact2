"""Compare actions in physical units and render a standalone inference report."""

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from evaluate_x5_actions import ROOT, JOINTS, action_metrics, save

LABELS = {
    "legacy": "Old / deployed inputs",
    "aligned": "Old / aligned inputs",
    "step2000": "Adapted 2k",
    "step5000": "Adapted 5k",
}
COLORS = {
    "legacy": "#777777",
    "aligned": "#d99000",
    "step2000": "#229966",
    "step5000": "#2675cf",
}


def paired_ci(before, after):
    # Three sampled windows are correlated within each episode; resample episodes.
    diff = np.rad2deg(
        abs(after["predictions"][..., JOINTS] - after["targets"][..., JOINTS]).mean(
            (1, 2)
        )
        - abs(before["predictions"][..., JOINTS] - before["targets"][..., JOINTS]).mean(
            (1, 2)
        )
    )
    per_episode = diff.reshape(337, 3).mean(1)
    rng = np.random.default_rng(42)
    boot = per_episode[rng.integers(0, 337, size=(2000, 337))].mean(1)
    return {
        "mean_difference_deg": float(per_episode.mean()),
        "episode_bootstrap_ci95_deg": np.quantile(boot, [0.025, 0.975]).tolist(),
        "lower_error_episode_fraction": float((per_episode < 0).mean()),
    }


def main(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = {
        "legacy": ROOT / "outputs/inference/action_accuracy_20261014/rgb",
        "aligned": ROOT / "outputs/inference/input_contract_v2_20261014/rgb",
        "step2000": output / "step2000/rgb",
        "step5000": output / "step5000/rgb",
    }
    arrays = {k: np.load(v / "actions.npz") for k, v in sources.items()}
    protocols = {
        k: json.loads((v / "protocol.json").read_text()) for k, v in sources.items()
    }
    reference = protocols["legacy"]["samples"]
    assert all(p["samples"] == reference for p in protocols.values())
    assert all(
        np.array_equal(v["targets"], arrays["legacy"]["targets"])
        and np.array_equal(v["states"], arrays["legacy"]["states"])
        for v in arrays.values()
    )
    splits = json.loads(
        (ROOT / "outputs/dlc/x5_contract_v2_20261014/splits.json").read_text()
    )["splits"]
    report = {
        "scope": "Matched recorded-observation inference. All models inherit weights that saw all 337 episodes; val/test mean excluded from adaptation, not unseen to original weights. No closed-loop success inference.",
        "sampled_episodes": 337,
        "general_observations_per_model": 1011,
        "horizon": 30,
        "flow_steps": 10,
        "precision": "BF16 native CUDA Graph",
        "device": protocols["step5000"]["device"],
        "hardware_initializations": 0,
        "robot_action_sends": 0,
        "models": {},
        "paired_joint_mae": {},
    }
    for k, v in arrays.items():
        g = json.loads((output / k / "grasp_summary.json").read_text())
        report["models"][k] = {
            "general": action_metrics(v["predictions"], v["targets"], v["states"]),
            "grasp": g,
            "by_split": {},
            "source": str(sources[k]),
            "actions_sha256": hashlib.sha256(
                (sources[k] / "actions.npz").read_bytes()
            ).hexdigest(),
        }
        for split, eps in splits.items():
            mask = np.array([s["episode"] in eps for s in reference])
            report["models"][k]["by_split"][split] = action_metrics(
                v["predictions"][mask], v["targets"][mask], v["states"][mask]
            )
        report["models"][k]["by_right_state"] = {}
        for group, mask in {
            "below_minus_one": v["states"][:, 13] < -1.0,
            "at_or_above_minus_one": v["states"][:, 13] >= -1.0,
        }.items():
            assert mask.any()
            report["models"][k]["by_right_state"][group] = {
                "observations": int(mask.sum()),
                "right_gripper_mae_rad": float(
                    abs(
                        v["predictions"][mask, :, 13] - v["targets"][mask, :, 13]
                    ).mean()
                ),
                "first_right_gripper_mae_rad": float(
                    abs(
                        v["predictions"][mask, 0, 13] - v["targets"][mask, 0, 13]
                    ).mean()
                ),
                "first_right_target_differs_from_state_over_0p1_fraction": float(
                    (
                        abs(v["predictions"][mask, 0, 13] - v["states"][mask, 13]) > 0.1
                    ).mean()
                ),
            }
        report["models"][k]["first_joint_target_over_0p25_rad"] = int(
            (
                abs(v["predictions"][:, 0, JOINTS] - v["states"][:, JOINTS]).max(-1)
                > 0.25
            ).sum()
        )
    for k in ["step2000", "step5000"]:
        report["paired_joint_mae"][k] = {
            baseline: paired_ci(arrays[baseline], arrays[k])
            for baseline in ["legacy", "aligned"]
        }
    report["paired_joint_mae"]["5k_minus_2k"] = paired_ci(
        arrays["step2000"], arrays["step5000"]
    )
    example_index = next(
        i for i, s in enumerate(reference) if s["episode"] == 247 and s["frame"] == 1047
    )
    report["former_quantile_limit_example"] = {
        "sample": reference[example_index],
        "right_joint": 5,
        "models": {},
    }
    for key, values in arrays.items():
        pred = np.rad2deg(values["predictions"][example_index, :, 11])
        truth = np.rad2deg(values["targets"][example_index, :, 11])
        report["former_quantile_limit_example"]["models"][key] = {
            "prediction_min_max_deg": [float(pred.min()), float(pred.max())],
            "reference_min_max_deg": [float(truth.min()), float(truth.max())],
            "mae_deg": float(abs(pred - truth).mean()),
        }
    save(output / "comparison.json", report)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    keys = list(LABELS)
    for ax, key, title, source in zip(
        axs,
        ["joint_mae_deg", "right_gripper_mae_rad", "arrival_within_5_frames_rate"],
        [
            "Joint MAE (degrees)",
            "Right gripper MAE at closure (rad)",
            "Closing on time (within 5 frames)",
        ],
        ["general", "closure", "closure"],
    ):
        values = [
            report["models"][k]["general"][key]
            if source == "general"
            else report["models"][k]["grasp"]["closure"][key]
            for k in keys
        ]
        bars = ax.bar(range(4), values, color=[COLORS[k] for k in keys])
        ax.set_xticks(
            range(4),
            ["Old\n(deployed)", "Old\n(aligned)", "Adapted\n2k", "Adapted\n5k"],
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, max(values) * 1.15)
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                f"{v:.1%}" if key == "arrival_within_5_frames_rate" else f"{v:.3f}",
                ha="center",
                va="bottom",
            )
    fig.suptitle("Recorded-observation inference | not physical grasp success")
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)

    plans = {
        k: json.loads((output / k / "grasp_protocol.json").read_text()) for k in keys
    }
    assert all(p["events"] == plans["legacy"]["events"] for p in plans.values())
    assert all(
        p["sequence_events"] == plans["legacy"]["sequence_events"]
        for p in plans.values()
    )
    events = plans["legacy"]["events"]
    selected = [plans["legacy"]["sequence_events"][i] for i in [0, 3, 4, 7, 8, 11]]
    closures = {k: np.load(output / k / "closure_actions.npz") for k in keys}
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for ax, event in zip(axs.flat, selected):
        i = next(i for i, e in enumerate(events) if e["episode"] == event["episode"])
        x = np.arange(30) / 30
        truth = closures["legacy"]["targets"][i, :, 13]
        ax.plot(x, truth, color="black", lw=2.5, label="Demonstration")
        for k in keys:
            ax.plot(
                x,
                closures[k]["predictions"][i, :, 13],
                color=COLORS[k],
                label=LABELS[k],
            )
        ax.axhline(-0.75, color="red", ls=":", lw=0.8)
        ax.axvline(0.5, color="gray", ls=":", lw=0.8)
        ax.set(
            title=f"Episode {event['episode']} / {event['split']}",
            xlabel="Seconds after observation",
            ylabel="Right gripper angle (rad)",
        )
        ax.grid(alpha=0.2)
    axs.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Preselected first-closure examples | increasing angle closes the gripper"
    )
    fig.savefig(output / "closure_examples.png", dpi=160)
    fig.savefig(output / "closure_examples.pdf")
    plt.close(fig)

    seq = {k: np.load(output / k / "grasp_sequence_actions.npz") for k in keys}
    fig, axs = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    for ax, g in zip(axs, [0, 4, 8]):
        event = plans["legacy"]["sequence_events"][g]
        x = np.arange(120) / 30
        ax.plot(
            x,
            seq["legacy"]["targets"][g * 10 : (g + 1) * 10, 12:24, 13].reshape(-1),
            color="black",
            lw=2.5,
            label="Demonstration",
        )
        for k in keys:
            ax.plot(
                x,
                seq[k]["rtc"][g * 10 : (g + 1) * 10, 12:24, 13].reshape(-1),
                color=COLORS[k],
                label=LABELS[k],
            )
        for boundary in np.arange(0.4, 4, 0.4):
            ax.axvline(boundary, color="gray", alpha=0.2)
        ax.axhline(-0.75, color="red", ls=":", lw=0.8)
        ax.set(
            title=f"RTC sequence / episode {event['episode']} / {event['split']}",
            xlabel="Seconds of stitched predictions",
            ylabel="Right gripper (rad)",
        )
        ax.grid(alpha=0.2)
    axs[0].legend(fontsize=8)
    fig.suptitle("RTC replay | recorded states, assumed 400 ms delay; not closed-loop")
    fig.savefig(output / "rtc_sequences.png", dpi=160)
    fig.savefig(output / "rtc_sequences.pdf")
    plt.close(fig)

    def table(headers, rows):
        return (
            "<table><tr>"
            + "".join("<th>" + html.escape(x) + "</th>" for x in headers)
            + "</tr>"
            + "".join(
                "<tr>"
                + "".join("<td>" + html.escape(str(x)) + "</td>" for x in row)
                + "</tr>"
                for row in rows
            )
            + "</table>"
        )

    body = "<h1>X5 适配模型动作推理评估</h1><p>2026-10-15 · A100 · BF16 · CUDA Graph · 10 次 flow 积分 · 30 帧动作块</p>"
    body += "<p>全量 337 条轨迹各取 3 个观测，共 1011 次预测；另评估 337 次首次右夹爪闭合与 12 段连续回放。所有模型继承的旧权重都见过这些轨迹，验证/测试划分仅未参与本次适配。这里没有真实机器人动作或抓取成功率。</p>"
    body += table(
        ["模型", "关节 MAE / °", "右夹爪 MAE / rad", "全部12关节 ≤2°"],
        [
            (
                LABELS[k],
                f"{report['models'][k]['general']['joint_mae_deg']:.4f}",
                f"{report['models'][k]['general']['per_gripper_mae_rad'][1]:.4f}",
                f"{report['models'][k]['general']['all_12_joints_within_degrees']['2']:.1%}",
            )
            for k in keys
        ],
    )
    body += "<p>闭合阈值为既有控制配置的 −0.75 rad；连续 3 帧越过算到达，在示范时刻 ±5 帧内算时机匹配。反向运动需累计 ≥0.1 rad 才计为反复；这些是动作轨迹指标，不表示接触或抓取成功。</p>"
    body += table(
        ["模型", "闭合窗口右夹爪 MAE", "阈值到达率", "时机匹配率", "末帧误差 ≤0.1 rad"],
        [
            (
                LABELS[k],
                f"{report['models'][k]['grasp']['closure']['right_gripper_mae_rad']:.4f}",
                f"{report['models'][k]['grasp']['closure']['threshold_arrival_rate']:.1%}",
                f"{report['models'][k]['grasp']['closure']['arrival_within_5_frames_rate']:.1%}",
                f"{report['models'][k]['grasp']['closure']['endpoint_within_0p1_rad']:.1%}",
            )
            for k in keys
        ],
    )
    body += "<p>下面单独检查右夹爪低于 −1 rad、原来会发生状态编码合并的 337 个观测。首帧真实动作标签等于当前状态，因此首帧目标偏差可诊断 chunk 开头是否偏离当前开度。</p>"
    body += table(
        ["模型", "首帧右夹爪误差 / rad", "首帧偏离当前开度 >0.1 rad"],
        [
            (
                LABELS[k],
                f"{report['models'][k]['by_right_state']['below_minus_one']['first_right_gripper_mae_rad']:.4f}",
                f"{report['models'][k]['by_right_state']['below_minus_one']['first_right_target_differs_from_state_over_0p1_fraction']:.1%}",
            )
            for k in keys
        ],
    )
    body += "<p>连续回放每种模型使用同样的 12 段观测。目标跳变量包含正常闭合运动，不能将每个超过 0.1 rad 的相邻目标变化都称为错误。</p>"
    body += table(
        [
            "模型",
            "预测方式",
            "平均反向运动次数 / 段",
            "chunk 边界跳变 P95 / rad",
            "闭合时机匹配",
        ],
        [
            (
                LABELS[k],
                method,
                f"{report['models'][k]['grasp']['sequence'][method]['closure']['mean_reversals_over_0p1_rad']:.2f}",
                f"{report['models'][k]['grasp']['sequence'][method]['right_chunk_boundary_abs_p95_rad']:.4f}",
                f"{report['models'][k]['grasp']['sequence'][method]['closure']['arrival_within_5_frames_rate']:.1%}",
            )
            for k in keys
            for method in ["plain", "rtc"]
        ],
    )
    body += "<p>原审计已定位的 episode 247 / frame 1047：右关节 5 的示范目标为 19.22–25.74°，旧模型输出固定在 8.16°；适配模型已能预测这一范围内的动作。详细数值保存在 comparison.json 的 former_quantile_limit_example 中。</p>"
    for name in ["comparison", "closure_examples", "rtc_sequences"]:
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode(
            "ascii"
        )
        body += f'<img src="data:image/png;base64,{encoded}" alt="{name}">'
    body += '<p>RTC 使用记录观测，固定假设推理延迟 12 帧（400ms）、请求间隔 12 帧、重叠容量 20、引导权重 10。未将预测反馈为后续观测，不能推断闭环稳定性或 5080 实际延迟。</p><p><a href="comparison.json">完整数据、分区指标与按 episode bootstrap 的配对置信区间</a></p>'
    style = "body{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#183044}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border:1px solid #ccd5dd;text-align:left}th{background:#eef3f6}img{width:100%;margin:20px 0}a{color:#086591}"
    (output / "report.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>X5 推理评估</title><style>'
        + style
        + "</style><body>"
        + body
        + "</body></html>"
    )
    print(
        json.dumps(
            {
                k: {
                    "general": report["models"][k]["general"],
                    "closure": report["models"][k]["grasp"]["closure"],
                    "sequence": report["models"][k]["grasp"]["sequence"],
                }
                for k in keys
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
