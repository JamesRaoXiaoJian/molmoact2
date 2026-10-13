"""Resolve the independent Molmo registry without loading models or hardware."""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_profile(
    model_select, role, registry=None, tasks=None, environ=None, rng=None
):
    registry = registry or json.loads(
        (ROOT / "configs/x5_molmo_policies.json").read_text()
    )
    env = os.environ if environ is None else environ
    selected = next(
        (p for p in registry["policies"] if p["id"] == int(model_select)), None
    )
    if selected is None:
        raise ValueError(
            f"MODEL_SELECT={model_select} is not registered in configs/x5_molmo_policies.json"
        )
    if selected.get("backend") != "molmoact2" or selected.get("mode") not in (
        "rgb",
        "mae",
    ):
        raise ValueError("This entry accepts registered Molmo RGB/MAE X5 policies only")
    tasks = tasks or json.loads((ROOT / selected["tasks_file"]).read_text())
    result = {**selected, "control": dict(selected["control"])}
    for variable, key, cast in (
        ("CONTROL_HZ", "control_hz", float),
        ("EXECUTE_STEPS", "execute_steps", int),
        ("ACTION_START_INDEX", "action_start_index", int),
    ):
        if variable in env:
            result["control"][key] = cast(env[variable])
    c = result["control"]
    if (
        c["control_hz"] <= 0
        or not 1 <= c["execute_steps"] <= 30
        or not 0 <= c["action_start_index"] < 30
        or c["execute_steps"] + c["action_start_index"] > 30
    ):
        raise ValueError(
            "Execution window must fit this checkpoint's 30-action horizon"
        )
    if role == "client":
        if env.get("PROMPT", "").strip():
            result["prompt"] = env["PROMPT"]
            result["task_index"] = None
        elif selected.get("prompt_selection") == "process_random":
            task = (rng or random.SystemRandom()).choice(tasks["task_rows"])
            result.update(prompt=task["task"], task_index=task["task_index"])
        elif selected.get("prompt_selection") == "fixed_data_task":
            task = next(
                t
                for t in tasks["task_rows"]
                if t["task_index"] == selected["task_index"]
            )
            result.update(prompt=task["task"], task_index=task["task_index"])
        else:
            raise ValueError(
                "Prompt selection awaits clarification: current OpenPI code is fixed, not random; no task 1 default was invented"
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("server", "client"), required=True)
    parser.add_argument("--model-select", type=int, required=True)
    args = parser.parse_args()
    profile = resolve_profile(args.model_select, args.role)
    print("MODE=" + shlex.quote(profile["mode"]))
    print("MODEL_DESCRIPTION=" + shlex.quote(profile["description"]))
    if args.role == "server":
        argv = [
            "--checkpoint=" + profile["checkpoint"],
            "--norm-tag=" + profile["norm_tag"],
            "--tokenizer-dir=" + profile["tokenizer_dir"],
        ]
        print("SERVER_ARGS=(" + " ".join(shlex.quote(arg) for arg in argv) + ")")
    if args.role == "client":
        c = profile["control"]
        argv = ["--prompt=" + profile["prompt"]]
        for key in (
            "control_hz",
            "action_start_index",
            "execute_steps",
            "duration_scale",
            "max_step_delta",
            "reject_first_joint_delta",
            "max_gripper_step_delta",
            "gripper_min",
            "gripper_max",
        ):
            argv.append("--" + key.replace("_", "-") + "=" + str(c[key]))
        print("MODEL_TASK_INDEX=" + shlex.quote(str(profile["task_index"])))
        print("MODEL_ARGS=(" + " ".join(shlex.quote(arg) for arg in argv) + ")")


if __name__ == "__main__":
    main()
