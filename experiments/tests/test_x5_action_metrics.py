"""Analytic checks for reported regression accuracy, direction and amplitude."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "action_accuracy",
    Path(__file__).resolve().parents[2] / "deployment/evaluate_x5_actions.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "scale,direction,reduction", [(1, 1, 1), (0, 0, 0), (0.5, 1, 0.75), (-1, 0, -3)]
)
def test_known_amplitude_and_hold_baseline(scale, direction, reduction):
    targets = np.full((2, 30, 14), 0.1)
    states = np.zeros((2, 14))
    metrics = module.action_metrics(targets * scale, targets, states)
    assert metrics["moving_amplitude_median_ratio"] == abs(scale)
    assert metrics["moving_direction_agreement"] == direction
    assert metrics["joint_mse_reduction_vs_hold"] == pytest.approx(reduction)
    assert metrics["joint_mae_deg"] == pytest.approx(np.rad2deg(0.1 * abs(scale - 1)))


def test_joint_vector_accuracy_differs_from_individual_accuracy():
    target = np.zeros((1, 1, 14))
    prediction = target.copy()
    prediction[0, 0, 0] = np.deg2rad(3)
    metrics = module.action_metrics(prediction, target, np.zeros((1, 14)))
    assert metrics["all_12_joints_within_degrees"]["2"] == 0
    assert metrics["individual_joint_within_degrees"]["2"] == pytest.approx(11 / 12)
    assert metrics["all_12_joints_within_degrees"]["5"] == 1
    assert metrics["gripper_mae_rad"] == 0
    assert metrics["moving_component_count"] == 0
    assert metrics["moving_amplitude_median_ratio"] is None
    assert metrics["joint_mse_reduction_vs_hold"] is None


def test_nonfinite_and_shape_errors_are_rejected():
    target = np.zeros((1, 30, 14))
    with pytest.raises(ValueError):
        module.action_metrics(target[..., :13], target, np.zeros((1, 14)))
    bad = target.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        module.action_metrics(bad, target, np.zeros((1, 14)))
