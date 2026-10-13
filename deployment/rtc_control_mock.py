"""Exercise the real OpenPI RTC control branch with preinstalled hardware traps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import benchmark_x5_native as guarded
import numpy as np
from openpi_main_smoke import Scenario, install_import_traps, marked
from scripts.x5_openpi_control import EpisodePolicy, load_mock_main
from scripts.x5_rtc_client import RTCInferenceEngine
from scripts.serve_x5_openpi import OpenPIBridge


def run_case(kind, samples, mode, openpi, port=None):
    from openpi_client import websocket_client_policy
    context = Scenario(kind, samples, SimpleNamespace(mode=mode), mode)
    context.resume_event = threading.Event()
    replacements = context.replacements()

    @marked
    class Controls:
        def __init__(self, *a, **kw): self.calls = 0
        def start(self): pass
        def stop(self): context.controls_stopped = True
        def flush(self): pass
        def poll(self, timeout=0):
            self.calls += 1
            if self.calls > 1000:
                raise RuntimeError("RTC mock exceeded its bounded event script")
            if timeout:
                if context.episode == 0:
                    context.episode = 1
                    return "toggle_episode"
                if context.episode == 1 and context.paused and not context.failed:
                    context.episode = 2
                    context.resume_event.set()
                    return "toggle_episode"
                return "quit"
            count = sum(c["episode"] == context.episode for c in context.candidates)
            if context.episode == 1 and (context.pending_pause or count >= 45):
                context.paused = True
                return "toggle_episode"
            if context.episode == 2 and count >= 25:
                return "toggle_episode"
            return None

    replacements["_TerminalKeys"] = replacements["_PedalControls"] = Controls
    module = load_mock_main(openpi, replacements)
    engines = []

    def engine_factory(*a, **kw):
        engine = RTCInferenceEngine(*a, **kw)
        engines.append(engine)
        return engine

    module.rtc_client = SimpleNamespace(RTCInferenceEngine=engine_factory)
    original_clip = module.clip_action

    def clip(target, *a, **kw):
        safe = original_clip(target, *a, **kw)
        context.candidates.append({"episode":context.episode,"time":time.monotonic(),
                                   "action":safe.copy(),"target":target.copy()})
        return safe

    module.clip_action = clip
    original_type = websocket_client_policy.WebsocketClientPolicy
    clients = []

    class FakeWire:
        def get_server_metadata(self): return OpenPIBridge(None, mode, enable_rtc=True).metadata
        def infer(self, observation):
            time.sleep(.08)
            return {"actions":np.repeat(observation["state"][None],30,axis=0)}

    class RecordingPolicy(EpisodePolicy):
        def infer(self, observation):
            episode = context.episode
            valid = observation["_rtc"]["prefix_valid_steps"]
            request = {"episode":episode,"valid_prefix":valid,"reset":self.pending_reset or observation.get("_reset",False),
                       "actions_already_consumed":sum(c["episode"]==episode for c in context.candidates)}
            context.requests.append(request)
            if episode == 1 and valid:
                if kind == "error":
                    context.failed = True
                    raise RuntimeError("injected RTC RPC failure")
                if kind == "underflow":
                    time.sleep(.85)
                if kind == "late_pause":
                    context.pending_pause = True
                    assert context.resume_event.wait(2)
            result = super().infer(observation)
            request["completed"] = True
            return result

    def client_factory(*a, **kw):
        client = original_type(*a, **kw) if port is not None else FakeWire()
        wrapped = RecordingPolicy(client, mode)
        clients.append(wrapped)
        return wrapped

    websocket_client_policy.WebsocketClientPolicy = client_factory
    old_argv = sys.argv
    handlers = {s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGTSTP)}
    sys.argv = ["rtc-hardware-mock", "--host=127.0.0.1",f"--port={port or 1}",
        "--rtc","--rtc-queue-threshold=18","--rtc-execution-horizon=20",
        "--control-hz=30","--execute-steps=5","--action-start-index=0","--max-chunks=0",
        "--gravity-compensation-setup","--episode-control=pedal",
        "--left-camera=mock-left","--right-camera=mock-right","--head-camera=mock-head",
        "--startup-settle-s=0.01","--startup-gripper-valid-frames=1","--startup-gripper-timeout-s=0.2",
        "--prompt="+samples[0]["task"]]
    if mode == "mae": sys.argv += ["--tactile-serials","mock-touch-left","mock-touch-right"]
    error = None
    try:
        module.main()
    except RuntimeError as exc:
        if kind not in {"error","underflow"}: raise
        error = str(exc)
    finally:
        sys.argv = old_argv
        websocket_client_policy.WebsocketClientPolicy = original_type
        for s,h in handlers.items(): signal.signal(s,h)
        for client in clients: client.close()
    assert context.robot_closed and context.controls_stopped
    assert all(c.started and c.stopped for c in context.cameras)
    assert all(not e._thread.is_alive() for e in engines)
    assert context.actual_sends == context.real_hardware_initializations == 0
    assert any(r["valid_prefix"] > 0 and r["actions_already_consumed"] > 0 for r in context.requests)
    if kind in {"error","underflow"}: assert error
    else:
        assert sum(c["episode"]==2 for c in context.candidates)==25
        assert all(r["reset"] for r in context.requests if r["valid_prefix"]==0)
    if kind == "late_pause": assert engines[0].metrics["discarded_responses"] >= 1
    if kind == "underflow": assert engines[0].metrics["underflows"] >= 1
    gaps = [(b["time"]-a["time"])*1000 for a,b in zip(context.candidates,context.candidates[1:]) if a["episode"]==b["episode"]]
    return {"case":kind,"mode":mode,"status":"passed","candidate_actions":len(context.candidates),
            "requests":context.requests,"metrics":engines[0].metrics,
            "control_interval_ms":guarded.stats(gaps),"expected_error":error,
            "real_robot_initializations":0,"actual_action_sends":0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rgb","mae"),required=True)
    parser.add_argument("--openpi-root",type=Path,default=Path.home()/"lwm_ws/openpi-x5-arx5-beta/openpi")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--real-model",action="store_true")
    args=parser.parse_args()
    native,_=guarded.load_inputs(args.mode)
    bridge=guarded.build_bridge(args.mode) if args.real_model else None
    install_import_traps(args.openpi_root)
    rows=[]
    if bridge is None:
        for kind in ("normal","late_pause","error","underflow"):
            rows.append(run_case(kind,native,args.mode,args.openpi_root))
    else:
        with bridge.create_server() as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                for kind in ("normal","late_pause","error","underflow"):
                    rows.append(run_case(kind,native,args.mode,args.openpi_root,server.socket.getsockname()[1]))
            finally: server.shutdown();thread.join(5)
            deadline = time.monotonic() + 5
            while bridge.closed_sessions < len(rows) and time.monotonic() < deadline:
                time.sleep(.01)
            assert bridge.closed_sessions == len(rows) and not bridge.molmo._session_states
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"rows":rows,"real_model":args.real_model},indent=2)+"\n")
    print(json.dumps({"mode":args.mode,"cases":[r["case"] for r in rows],"status":"passed","actual_action_sends":0}))


if __name__=="__main__": main()
