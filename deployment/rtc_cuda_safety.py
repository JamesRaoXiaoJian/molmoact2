"""Real CUDA checks for prefix replay freshness, finite guards and output ownership."""
import argparse
import json
from pathlib import Path

import benchmark_x5_native  # hardware/network guards
import torch
from olmo.nn.native_action_graph import NativeActionFlowGraph, _flow
from olmo.nn.native_action_compile import compile_action_blocks
from olmo.nn.rtc_guidance import prepare_rtc_condition
from tests.test_native_action_inference_cache import setup_expert


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--compile-blocks",action="store_true")
    args=p.parse_args()
    model,expert,kv,state,mask=setup_expert()
    expert.to('cuda',torch.bfloat16); model.action_expert=expert
    if args.compile_blocks: compile_action_blocks(model)
    kv=[(k.to('cuda',torch.bfloat16),v.to('cuda',torch.bfloat16)) for k,v in kv]
    state=state.to('cuda',torch.bfloat16);mask=mask.cuda()
    noise=torch.randn(1,4,3,device='cuda')
    padding=torch.tensor([[False,False,True]],device='cuda')
    manager=NativeActionFlowGraph()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        context=expert.prepare_inference_context(kv,mask,state,1,4,torch.device('cuda'),torch.bfloat16)
        mods=expert.inference_modulations(10,1,torch.device('cuda'),torch.bfloat16)
        saved=[]
        for valid,value in [(0,.1),(3,.2),(2,-.3),(3,.1)]:
            prefix=torch.full_like(noise,value);prefix[:,:,2]=0
            cond=prepare_rtc_condition(prefix,dict(prefix_valid_steps=valid,inference_delay=1,
                execution_horizon=3,max_guidance_weight=3.0))
            expected,ok=_flow(model,expert,noise,context,mods,padding,10,cond)
            assert ok
            actual=manager.run(model,expert,noise,context,mods,padding,10,rtc_condition=cond)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            for result,backup in saved: assert torch.equal(result,backup)
            saved.append((actual,actual.clone()))
        assert manager.capture_count==1
        bad=prepare_rtc_condition(torch.zeros_like(noise),dict(prefix_valid_steps=3,inference_delay=1,
            execution_horizon=3,max_guidance_weight=3.0))
        bad['prefix'].fill_(float('nan'))
        try: manager.run(model,expert,noise,context,mods,padding,10,rtc_condition=bad)
        except RuntimeError: pass
        else: raise AssertionError('Invalid RTC graph returned actions')
        recovered=manager.run(model,expert,noise,context,mods,padding,10,rtc_condition=cond)
        torch.testing.assert_close(recovered,saved[-1][0],rtol=0,atol=0)
    report={'status':'passed','compiled':args.compile_blocks,'rtc_graph_eager_bitwise':True,
            'prefix_change_without_recapture':True,'empty_prefix_supported':True,'old_output_no_alias':True,
            'invalid_prefix_rejected_and_recovered':True,'actual_action_sends':0,'real_robot_initializations':0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
