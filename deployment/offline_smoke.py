#!/usr/bin/env python3
"""Two-observation replay through final native checkpoints. Never controls hardware."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ[key] = '1'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['MPLBACKEND'] = 'Agg'
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/lerobot/src')]
import numpy as np
import torch
from olmo.data.tactile import X5_RGB_IMAGE_KEYS, tactile_keys_for_layout
from scripts.serve_policy import MolmoAct2Server
from scripts.run_x5_inference_comparison import infer, metrics, test_http


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('rgb', 'mae'), required=True)
    p.add_argument('--samples', type=Path, default=ROOT/'data/offline_samples')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/inference/5080_deployment')
    p.add_argument('--parameter-dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.samples/'manifest.json').read_text())
    samples = []
    for item in manifest['samples']:
        data = np.load(args.samples/item['file'], allow_pickle=False)
        obs = {k: data[k] for k in data.files if k.startswith('observation.')}
        if args.mode == 'rgb':
            obs = {k:v for k,v in obs.items() if not k.startswith('observation.images.') or k in X5_RGB_IMAGE_KEYS}
        obs['task'] = item['task']
        samples.append(dict(item, obs=obs, target=data['target'], baseline=data[f'baseline_{args.mode}']))
    checkpoint = ROOT/f'outputs/runs/molmoact2-x5-cleaned-{args.mode}-fft20k-b64-s42-20261008/checkpoints/step20000-unsharded'
    tag = f'arx_x5_cleaned_{args.mode}'
    start = time.perf_counter()
    print(json.dumps({'stage':'loading', 'mode':args.mode, 'parameter_dtype':args.parameter_dtype}), flush=True)
    server = MolmoAct2Server(checkpoint=str(checkpoint), norm_tag=tag,
                            image_keys='x5_tactile' if args.mode=='mae' else 'x5',
                            device='cuda:0', parameter_dtype=args.parameter_dtype,
                            tokenizer_dir=str(ROOT/'data/hf-cache/hub'),
                            enable_inference_cuda_graph=False)
    policy = server.policy
    load_s = time.perf_counter()-start
    dtypes = sorted({str(v.dtype) for v in policy._handles.model.parameters()})
    assert dtypes == [f'torch.{args.parameter_dtype}'], dtypes
    warmup, cold_s = infer(policy, samples[0]['obs'], samples[0]['seed'], 10)
    torch.cuda.reset_peak_memory_stats()
    records, actions = [], []
    for sample in samples:
        action, elapsed = infer(policy, sample['obs'], sample['seed'], 10)
        normalized = server.robot_processor.normalize_action(action, tag)
        reference = server.robot_processor.normalize_action(sample['baseline'], tag)
        record = {k:sample[k] for k in ('episode','frame','seed','baseline_index')}
        record.update(latency_s=elapsed, shape=list(action.shape), finite=bool(np.isfinite(action).all()),
                      baseline_fp32_policy_space_delta_mae=float(np.abs(normalized-reference).mean()),
                      baseline_fp32_policy_space_delta_max=float(np.abs(normalized-reference).max()),
                      **metrics(action, sample['target'], server.robot_processor, tag))
        records.append(record); actions.append(action)
        print(json.dumps({'stage':'inferred', 'mode':args.mode, **record}), flush=True)
    repeat, _ = infer(policy, samples[0]['obs'], samples[0]['seed'], 10)
    np.testing.assert_allclose(repeat, actions[0], rtol=1e-6, atol=1e-6)
    http = test_http(server, samples[0], actions[0], samples[0]['seed'])
    controls = []
    if args.mode=='mae':
        left, right = tactile_keys_for_layout('two')
        for condition in ('blank','swap'):
            obs = dict(samples[0]['obs'])
            if condition=='blank': obs[left],obs[right] = np.zeros_like(obs[left]),np.zeros_like(obs[right])
            else: obs[left],obs[right] = obs[right],obs[left]
            action, elapsed = infer(policy, obs, samples[0]['seed'], 10)
            delta = server.robot_processor.normalize_action(action,tag)-server.robot_processor.normalize_action(actions[0],tag)
            controls.append({'condition':condition, 'latency_s':elapsed,
                             'policy_space_output_delta_mae':float(np.abs(delta).mean())})
    report = {'scope':manifest['scope'], 'mode':args.mode, 'samples':len(samples),
              'checkpoint':str(checkpoint), 'flow_steps':10, 'device':torch.cuda.get_device_name(0),
              'cuda_capability':list(torch.cuda.get_device_capability(0)), 'torch':torch.__version__,
              'cuda_runtime':torch.version.cuda, 'parameter_dtypes':dtypes,
              'cuda_graph_enabled':False, 'load_s':load_s, 'cold_inference_s':cold_s,
              'median_latency_s':statistics.median(r['latency_s'] for r in records),
              'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
              'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
              'reproducibility':'passed', 'finite_actions':True, 'sample_results':records,
              'http':http, 'tactile_controls':controls, 'network_listener_started':False,
              'robot_commands_sent':False, 'held_out_evaluation':False, 'physical_robot_evaluation':False}
    np.savez_compressed(args.output/f'{args.mode}_actions.npz', actions=np.stack(actions))
    (args.output/f'{args.mode}.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'stage':'passed','mode':args.mode,'median_latency_s':report['median_latency_s'],
                      'peak_allocated_gib':report['peak_allocated_gib']}), flush=True)


if __name__=='__main__':
    main()
