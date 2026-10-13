import threading
import time

import numpy as np
import pytest
import torch

from olmo.nn.rtc_guidance import prepare_rtc_condition, rtc_euler_step, validate_rtc_options
from scripts.x5_rtc_client import RTCActionQueue, RTCInferenceEngine
from scripts.x5_openpi_control import EpisodePolicy
from tests.test_native_action_inference_cache import setup_expert
from olmo.models.molmoact2.molmoact2 import MolmoAct2


def options(valid=20, delay=10, horizon=20, weight=10):
    return dict(prefix_valid_steps=valid, inference_delay=delay,
                execution_horizon=horizon, max_guidance_weight=weight)


def test_linear_prefix_schedule_matches_lerobot():
    c = prepare_rtc_condition(torch.ones(1, 30, 14), options(valid=5, delay=2, horizon=20))
    torch.testing.assert_close(c["weights"].flatten()[:7], torch.tensor([1, 1, .75, .5, .25, 0, 0]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_empty_prefix_preserves_existing_euler_bits(dtype):
    torch.manual_seed(7)
    x, v = torch.randn(1, 30, 14), torch.randn(1, 30, 14).to(dtype)
    c = prepare_rtc_condition(torch.randn_like(x), options(valid=0))
    for step in range(10):
        actual, expected = rtc_euler_step(x, v, step, 10, c), x + .1 * v
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_guidance_has_correct_molmo_time_sign_and_only_uses_prefix():
    x = torch.zeros(1, 30, 14)
    v = torch.zeros_like(x)
    c = prepare_rtc_condition(torch.ones_like(x), options(valid=8, delay=3, horizon=8))
    actual = rtc_euler_step(x, v, 0, 10, c)
    torch.testing.assert_close(actual[:, :3], torch.ones(1, 3, 14))
    assert actual[:, 8:].count_nonzero() == 0
    assert (actual[:, 3:8] > 0).all() and (actual[:, 3:8] < 1).all()


@pytest.mark.parametrize("bad", [dict(prefix_valid_steps=True), dict(inference_delay=30),
    dict(execution_horizon=31), dict(max_guidance_weight=float("nan")), dict(max_guidance_weight=11)])
def test_bad_rtc_parameters_rejected(bad):
    with pytest.raises(ValueError):
        validate_rtc_options({**options(), **bad}, 30)


def test_guided_native_flow_and_zero_prefix_baseline_parity():
    model, expert, kv, state, mask = setup_expert()
    model.native_inference_optimization = "modcache"
    def sample(condition=None):
        return MolmoAct2.generate_actions(model, input_ids=None, encoder_kv_states=kv,
            encoder_attention_mask=mask, states=state, rtc_condition=condition,
            action_dim_is_pad=torch.tensor([[False, False, True]]),
            generator=torch.Generator().manual_seed(321))
    with torch.no_grad():
        base = sample()
        empty = prepare_rtc_condition(torch.zeros_like(base), options(valid=0, delay=0, horizon=4))
        assert torch.equal(sample(empty), base)
        target = base.clone(); target[:, :2, :2] += .1
        c = prepare_rtc_condition(target, options(valid=2, delay=1, horizon=2))
        guided = sample(c)
        assert ((guided[:, :1, :2]-target[:, :1, :2]).abs().mean()
                < (base[:, :1, :2]-target[:, :1, :2]).abs().mean())
        assert guided[:, :, 2].count_nonzero() == 0
        assert torch.isfinite(guided).all()


class Clock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value


def test_queue_keeps_raw_prefix_and_drops_time_expired_frames():
    clock = Clock()
    q = RTCActionQueue(action_horizon=30, action_dim=14, control_hz=30, clock=clock)
    raw = np.repeat(np.arange(30, dtype=np.float32)[:, None], 14, axis=1)
    processed = raw + 100
    q.replace(raw, processed, origin=0, delay=0)
    action, rev = q.get_with_revision()
    np.testing.assert_array_equal(action, processed[0])
    prefix, valid = q.snapshot_prefix(20)
    assert valid == 20 and prefix[0, 0] == 1
    clock.value = .21
    action, rev2 = q.get_with_revision()
    assert rev == rev2 and action[0] == 106
    assert q.expired_steps == 5
    raw.fill(-1)
    assert q.snapshot_prefix(20)[0][0, 0] == 7
    clock.value = 1.01
    assert q.qsize() == 0
    assert q.get_with_revision()[0] is None


class BlockingClient:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.requests = []
        self.response = np.ones((30, 14), np.float32)
        self.error = None
    def infer(self, obs):
        self.requests.append(obs)
        self.entered.set()
        assert self.release.wait(2)
        if self.error: raise self.error
        return {"actions": self.response}
    def close(self): self.release.set()


def settled(engine):
    deadline = time.monotonic() + 2
    while engine.busy and time.monotonic() < deadline:
        time.sleep(.002)
    assert not engine.busy


def engine(client, clock=None):
    return RTCInferenceEngine(client, control_hz=30, action_horizon=30, action_dim=14, clock=clock)


def test_pause_discards_inflight_result_and_resume_sends_fresh_reset():
    client = BlockingClient()
    e = engine(client)
    try:
        e.resume(); assert e.maybe_request({"state":np.zeros(14, np.float32)})
        assert client.entered.wait(1)
        e.pause(); e.resume(); client.release.set(); settled(e)
        assert e.queue.qsize() == 0 and e.metrics["discarded_responses"] == 1
        assert e.maybe_request({"state":np.ones(14, np.float32)})
        settled(e)
        assert client.requests[-1]["_reset"] is True
        assert client.requests[-1]["_rtc"]["prefix_valid_steps"] == 0
        assert e.inference_count == 1 and e.queue.qsize() > 0
    finally: e.close()
    assert not e._thread.is_alive()


@pytest.mark.parametrize("bad", ["error", "nan", "shape"])
def test_worker_failure_clears_queue_and_surfaces_to_controller(bad):
    client = BlockingClient(); client.release.set()
    if bad == "error": client.error = RuntimeError("injected failure")
    if bad == "nan": client.response[0, 0] = float("nan")
    if bad == "shape": client.response = client.response[:5]
    e = engine(client)
    try:
        e.resume(); e.maybe_request({}); settled(e)
        with pytest.raises(RuntimeError, match="RTC inference stopped"): e.raise_if_failed()
        assert e.queue.qsize() == 0
    finally: e.close()


def test_network_wait_does_not_block_consumption_and_underflow_stops():
    clock = Clock(); client = BlockingClient(); client.release.set()
    e = engine(client, clock)
    try:
        e.resume(); e.maybe_request({}); settled(e)
        clock.value = .4  # eighteen frames remain
        client.release.clear(); client.entered.clear()
        assert e.maybe_request({}) and client.entered.wait(1)
        assert e.queue.get_with_revision()[0] is not None
        assert e.busy
        clock.value = 1.01
        with pytest.raises(RuntimeError, match="depleted"): e.raise_if_failed()
        client.release.set(); settled(e)
        assert e.queue.qsize() == 0 and e.metrics["discarded_responses"] == 1
    finally: e.close()


def test_refill_aligns_by_elapsed_time_and_never_reuses_gripper_calibration():
    clock = Clock(); client = BlockingClient(); client.release.set()
    client.response = np.repeat(np.arange(30, dtype=np.float32)[:, None], 14, axis=1)
    e = RTCInferenceEngine(client, control_hz=30, action_horizon=30, action_dim=14,
                          clock=clock, action_transform=lambda raw: raw + 100)
    try:
        e.resume(); e.maybe_request({}); settled(e)
        clock.value = .4
        client.release.clear(); client.entered.clear()
        observation = {"state":np.zeros(14,np.float32)}
        assert e.maybe_request(observation) and client.entered.wait(1)
        observation["state"].fill(123)
        assert np.all(client.requests[-1]["state"] == 0)
        assert client.requests[-1]["actions"][0, 6] == 12  # raw, never +100
        clock.value = .65
        client.response = client.response + 1000
        client.release.set(); settled(e)
        assert e.metrics["dropped_prefix_frames"] == 8
        action, _ = e.queue.get_with_revision()
        assert action[6] == 1108  # time-aligned + exactly one calibration
    finally: e.close()


def test_episode_policy_does_not_ack_reset_that_arrived_during_rpc():
    client = BlockingClient()
    client.get_server_metadata = lambda: {"model_type":"molmoact2", "model_variant":"rgb",
        "supports_rtc":False, "action_horizon":30, "n_action_steps":5}
    policy = EpisodePolicy(client, "rgb")
    worker = threading.Thread(target=lambda: policy.infer({}))
    worker.start(); assert client.entered.wait(1)
    policy.reset(); client.release.set(); worker.join(1)
    policy.infer({})
    assert [r["_reset"] for r in client.requests] == [True, True]


def test_cold_start_result_is_discarded_before_any_action_and_observation_refreshed():
    clock = Clock(); client = BlockingClient(); e = engine(client, clock)
    try:
        e.resume(); e.maybe_request({"fresh":False}); assert client.entered.wait(1)
        clock.value = 1.5; client.release.set(); settled(e)
        assert e.inference_count == 0 and e.queue.qsize() == 0
        assert e.metrics["warmup_discards"] == 1
        e.maybe_request({"fresh":True}); settled(e)
        assert e.inference_count == 1 and client.requests[-1]["fresh"] is True
    finally:e.close()
