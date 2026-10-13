# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Adapted under Apache-2.0 from LeRobot RTC and cijerezg/lerobot (6617d375).
# Sources and numerical convention: experiments/docs/x5_rtc_zh.md.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Inference-only RTC prefix guidance for Molmo's noise-to-action flow.

Use the community identity-Jacobian approximation, avoiding a backward pass.
Inputs stay fixed-size for native CUDA Graph replay. The ordinary Euler update
is evaluated first so an empty prefix is exactly the existing BF16 baseline.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch


def validate_rtc_options(options: dict, horizon: int) -> dict:
    fields = {"prefix_valid_steps", "inference_delay", "execution_horizon", "max_guidance_weight"}
    if not isinstance(options, dict) or set(options) != fields:
        raise ValueError(f"RTC requires exactly these fields: {sorted(fields)}")
    result = dict(options)
    for name in fields - {"max_guidance_weight"}:
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"RTC {name} must be an integer")
        result[name] = int(value)
    if not 0 <= result["prefix_valid_steps"] <= horizon:
        raise ValueError("RTC prefix length is outside the action horizon")
    if not 0 <= result["inference_delay"] < horizon:
        raise ValueError("RTC inference delay is outside the action horizon")
    if not 1 <= result["execution_horizon"] <= horizon:
        raise ValueError("RTC execution horizon is outside the action horizon")
    weight = options["max_guidance_weight"]
    if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or not 0 < weight <= 10:
        raise ValueError("RTC max_guidance_weight must be finite and in (0, 10]")
    result["max_guidance_weight"] = float(weight)
    return result


def prepare_rtc_condition(prefix: torch.Tensor, options: dict) -> dict:
    if prefix.ndim != 3 or not torch.is_floating_point(prefix) or not bool(torch.isfinite(prefix).all()):
        raise ValueError("RTC prefix must be a finite floating [batch, horizon, action_dim] tensor")
    horizon = prefix.shape[1]
    options = validate_rtc_options(options, horizon)
    end = min(options["prefix_valid_steps"], options["execution_horizon"])
    start = min(options["inference_delay"], end)
    # LeRobot LINEAR schedule: committed prefix=1, soft tail, unknown future=0.
    weights = torch.zeros(horizon, device=prefix.device, dtype=torch.float32)
    weights[:start] = 1
    if end > start:
        weights[start:end] = torch.linspace(1, 0, end - start + 2, device=prefix.device, dtype=torch.float32)[1:-1]
    return {"prefix": prefix.to(torch.float32), "weights": weights[None, :, None],
            "max_guidance_weight": options["max_guidance_weight"]}


def rtc_euler_step(trajectory, velocity, index: int, steps: int, condition=None):
    dt = 1.0 / steps
    result = trajectory + dt * velocity
    if condition is None:
        return result
    # Molmo integrates t=0(noise) -> 1(action); no velocity/time sign ambiguity.
    tau = index / steps
    remaining = 1.0 - tau
    max_weight = condition["max_guidance_weight"]
    gain = max_weight if tau == 0 else min(
        remaining / tau * (remaining * remaining + tau * tau) / (remaining * remaining), max_weight)
    weights = condition["weights"]
    endpoint = trajectory.float() + remaining * velocity.float()
    # Keep unconditioned/padded elements inert even if an unused estimate overflows.
    error = torch.where(weights != 0, condition["prefix"] - endpoint, 0.0)
    corrected = result + dt * gain * error * weights
    return torch.where(weights != 0, corrected, result)
