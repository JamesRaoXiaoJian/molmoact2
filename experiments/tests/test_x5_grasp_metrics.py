import sys
from pathlib import Path

import numpy as np
import pytest

DEPLOYMENT = Path(__file__).resolve().parents[2] / "deployment"
sys.path.insert(0, str(DEPLOYMENT))
from evaluate_x5_grasp import crossing, reversals, closure_metrics, select_events


def test_crossing_requires_dwell_and_identifies_missing():
    assert crossing([-1, -0.5, -1, -1]) is None
    assert crossing([-1, -1, -0.5, -0.5, -0.5]) == 2
    assert crossing([0, 0, 0]) == 0


def test_reversals_ignore_noise_and_count_real_reopening():
    assert reversals([-2, -1, 0]) == 0
    assert reversals([-2, -1, -1.01, -1, 0]) == 0
    assert reversals([-2, -1, 0, -1, 0]) == 2


def test_closure_metrics_penalize_missing_and_early_closure():
    target = np.tile(np.r_[np.full(15, -2.0), np.zeros(15)], (3, 1))
    predictions = target.copy()
    predictions[1] = 0
    predictions[2] = -2
    metrics = closure_metrics(predictions, target)
    assert metrics["threshold_arrival_rate"] == pytest.approx(2 / 3)
    assert metrics["arrival_within_5_frames_rate"] == pytest.approx(1 / 3)
    assert metrics["missed_closures"] == 1
    assert metrics["matched_delay_median_frames"] == -7.5


def test_event_selector_uses_first_open_close_transition():
    states = np.zeros((100, 14))
    states[:40, 13] = -2
    events = select_events(states, np.zeros(100), {0: np.arange(100)}, {0: "test"})
    assert (
        len(events) == 1
        and events[0]["frame"] == 25
        and events[0]["crossing_frame"] == 40
    )
