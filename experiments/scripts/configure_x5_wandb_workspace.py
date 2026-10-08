#!/usr/bin/env python3
"""Curate live X5 ablation dashboards without attaching to training processes.

Run with a separate environment containing wandb-workspaces. This only reads run
history and saves workspace layouts; it never initializes or finishes a run.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces._graphql import execute_graphql
from wandb_workspaces.workspaces import internal

LOSS = "train/action_flow_loss"
VIEW_NAME = "X5 cleaned RGB vs Tactile-MAE"
VIEWS_QUERY = """
query Views($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    allViews(viewType: "project-view") {
      edges { node { id name displayName spec } }
    }
  }
}
"""


def load_view(node, entity, project):
    # Personal workspaces use a '-w' suffix. Workspace.from_url currently only
    # handles saved '-v' views, so preserve the exact backend identity here.
    return ws.Workspace._from_model(
        internal.View(
            entity=entity,
            project=project,
            id=node["id"],
            name=node["name"],
            display_name=node["displayName"],
            spec=json.loads(node["spec"]),
        )
    )


def make_sections(keys, runs, gpu_keys):
    titles = {
        f"{r.id}:{key}": f"{'Tactile-MAE' if '-mae-' in r.name else 'RGB'} / {key}"
        for r in runs
        for key in keys | gpu_keys
    }

    def line(title, y, **kwargs):
        return wr.LinePlot(
            title=title,
            x="Step",
            y=y,
            title_x="Optimizer step",
            smoothing_type="none",
            ignore_outliers=False,
            aggregate=False,
            max_runs_to_show=2,
            line_titles=titles,
            **kwargs,
        )

    def section(name, panels, pinned=False):
        return ws.Section(
            name=name,
            panels=panels,
            is_open=True,
            pinned=pinned,
            layout_settings=ws.SectionLayoutSettings(columns=3, rows=2),
        )

    raw = line("动作主损失 / Action flow loss（原始）", [LOSS], title_y="Flow matching loss")
    smooth = line("动作主损失 / Loss 趋势（保留原始曲线）", [LOSS], title_y="Flow matching loss")
    smooth.smoothing_type = "exponential"
    smooth.smoothing_factor = 0.6
    smooth.smoothing_show_original = True
    sections = [section("01 任务效果 / Loss", [
        raw, smooth, wr.ScalarChart(title="最新动作损失 / Latest loss", metric=LOSS)
    ], pinned=True)]

    progress = line("训练进度：更新步 vs 已用时间（目标 20,000）", ["Step"], title_y="Optimizer step")
    progress.x = "RelativeTime(Process)"
    progress.title_x = "Elapsed seconds"
    sections.append(section("02 训练进度 / Progress", [
        wr.ScalarChart(title="当前更新步 / 20,000", metric="Step"),
        progress,
        wr.ScalarChart(title="已运行时长 / Elapsed seconds", metric="RelativeTime(Process)"),
    ]))

    optimizer = []
    groups = [
        ("动作专家学习率 / Action expert LR", ["optim/action_expert_lr"]),
        ("视觉与语言学习率 / VLM LR", ["optim/llm_lr", "optim/vit_lr", "optim/connector_lr"]),
        ("MAE 主干与适配器学习率 / Tactile LR", ["optim/vit_tactile_lr", "optim/action_expert_tactile_lr"]),
        ("梯度范数 / Gradient norms", ["optim/action_expert_grad_norm", "optim/llm_grad_norm", "optim/vit_grad_norm"]),
        ("触觉梯度范数 / Tactile gradient norms", ["optim/vit_tactile_grad_norm", "optim/action_expert_tactile_grad_norm"]),
    ]
    for title, candidates in groups:
        available = [k for k in candidates if k in keys]
        if available:
            optimizer.append(line(title, available))
    sections.append(section("03 学习率与稳定性 / Optimizer", optimizer))

    efficiency = []
    for title, key in [
        ("训练速度 / Optimizer steps per second", "throughput/device/batches_per_second"),
        ("PyTorch 峰值显存 / Peak allocated memory (MB)", "System/Peak GPU Memory (MB)"),
    ]:
        if key in keys:
            efficiency.append(line(title, [key]))
    # System metrics live in a separate stream; use elapsed time, not step.
    if "system.gpu.0.gpu" in gpu_keys:
        gpu = line("GPU 0 利用率 / Utilization (%)", ["system.gpu.0.gpu"], title_y="Percent")
        gpu.x = "RelativeTime(Process)"
        gpu.title_x = "Elapsed seconds"
        efficiency.append(gpu)
    sections.append(section("04 速度与资源 / Efficiency", efficiency))
    sections.append(ws.Section(name="05 实验说明 / Recipe", is_open=False, panels=[wr.MarkdownPanel(
        markdown=(
            "337 条轨迹，382,422 帧。两组均为 8×A800、global batch 64、"
            "20,000 optimizer updates、seed 42，约 3.35 次数据遍历。"
            "同一官方初始化；RGB 3 路相机，MAE 另外使用 2 路触觉。\n\n"
            "Loss 是真实的动作 flow matching 目标，数值越低表示训练目标拟合越好；"
            "实际操作成功率仍需部署评测。本轮未配置验证集，不显示验证损失或成功率。"
            "Loss 每 10 步上传；时间以各自训练进程启动为起点。"
            "GPU 0 利用率来自 W&B 系统采集；全部 8 张卡的系统指标仍可在 run 的 System 页面查看。"
        )
    )]))
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--update-personal", action="store_true", help="Also curate the API account's personal workspace")
    args = parser.parse_args()
    args.record_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=45)
    runs = [r for r in api.runs(f"{args.entity}/{args.project}")
            if "-fft20k-b64-s42-20261008" in r.name and not r.name.endswith("-smoke")]
    if not runs:
        raise SystemExit("No formal comparison runs found")
    keys, gpu_keys, evidence = set(), set(), []
    for run in runs:
        rows = list(run.scan_history(keys=["_step", LOSS], page_size=1000))
        if not rows:  # A just-started run will join the view when its first log arrives.
            continue
        if any(not math.isfinite(row[LOSS]) for row in rows):
            raise ValueError(f"Non-finite optimized loss in {run.id}")
        steps = [row["_step"] for row in rows]
        if any(a >= b for a, b in zip(steps, steps[1:])):
            raise ValueError(f"Non-increasing logged steps in {run.id}")
        keys.update(dict(run.summary))
        gpu_keys.update(run.system_metrics)
        evidence.append({"id": run.id, "url": run.url, "state": run.state,
                         "loss_points": len(rows), "first_step": steps[0],
                         "last_step": steps[-1], "latest_loss": rows[-1][LOSS]})
    if not evidence:
        raise SystemExit("No uploaded primary loss yet; wait for first training log")

    variables = {"entity": args.entity, "project": args.project}
    response = execute_graphql(api, VIEWS_QUERY, variables)
    nodes = [e["node"] for e in response["project"]["allViews"]["edges"]]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (args.record_dir / f"workspaces_before_{stamp}.json").write_text(json.dumps(response, indent=2))
    node = next((n for n in nodes if n["displayName"] == VIEW_NAME), None)
    workspace = load_view(node, args.entity, args.project) if node else ws.Workspace(
        entity=args.entity, project=args.project, name=VIEW_NAME, auto_generate_panels=False
    )
    workspace.sections = make_sections(keys, runs, gpu_keys)
    workspace.settings = ws.WorkspaceSettings(x_axis="Step", sort_panels_alphabetically=False, max_runs=2)
    workspace.runset_settings = ws.RunsetSettings(
        query="-fft20k-b64-s42-20261008",
        filters="State != 'crashed' and State != 'failed'",
        run_settings={r.id: ws.RunSettings(color="#E69F00" if "-mae-" in r.name else "#0072B2") for r in runs},
        pinned_columns=["summary:train/action_flow_loss", "summary:_step", "run:state"],
    )
    workspace.save()
    personal_name = None
    if args.update_personal:
        for node in nodes:
            if node["name"].endswith("-w") and api.viewer.username in node["name"]:
                personal = load_view(node, args.entity, args.project)
                # Preserve existing diagnostic groups, collapsed after curated sections.
                managed_names = {s.name for s in workspace.sections}
                diagnostics = [s for s in personal.sections if s.name not in managed_names]
                for section in diagnostics:
                    section.is_open = False
                    section.pinned = False
                personal.sections = [*workspace.sections, *diagnostics]
                personal.settings = workspace.settings
                personal.runset_settings = workspace.runset_settings
                personal.save()
                personal_name = node["name"]

    loaded = ws.Workspace.from_url(workspace.url)
    assert [s.name for s in loaded.sections] == [s.name for s in workspace.sections]
    assert loaded.sections[0].is_open and loaded.sections[0].pinned
    assert loaded.sections[0].panels[0].y[0].name == LOSS
    assert not loaded.settings.sort_panels_alphabetically
    if personal_name is not None:
        response = execute_graphql(api, VIEWS_QUERY, variables)
        node = next(e["node"] for e in response["project"]["allViews"]["edges"] if e["node"]["name"] == personal_name)
        personal = load_view(node, args.entity, args.project)
        assert personal.sections[0].panels[0].y[0].name == LOSS
        names = [s.name for s in personal.sections]
        assert names[:len(loaded.sections)] == [s.name for s in loaded.sections]
        assert len(names) == len(set(names)), "Repeated dashboard sections"
    result = {"workspace_url": workspace.url, "verified_at": stamp, "runs": evidence,
              "sections": [{"name": s.name, "titles": [getattr(p, "title", None) for p in s.panels]} for s in loaded.sections]}
    (args.record_dir / "workspace_verified.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
