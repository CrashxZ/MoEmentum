#!/usr/bin/env python3
"""Focused existing-hook capture; run under shared worker.lock."""
import os,json,subprocess as sp,time,re,sys
from pathlib import Path
root=Path('/home/inference/moe-bench');build=root/'strata-build-loaderfix';variant=root/'results/strata-qsum-green-v1';out=root/'results'/sys.argv[1];out.mkdir(exist_ok=False)
ids=json.loads((root/'prompt-short-ids.json').read_text());model=root/'models/Qwen3.8-35B-A3B-Q4_K_M.gguf'
base=(root/'results/strata-qsum-model-v1/generation-Q4_K_M-0.log').read_text();forced=re.search(r'^gen: (.*)$',base,re.M)[1].split()[:16]
env=dict(os.environ,LD_LIBRARY_PATH=f'{root}/cuda/lib:{build}',STRATA_MMQ_KQ='1',STRATA_TEST_Q36='1')
receipt=[]
def run(cmd,name,value):
 with (out/(name+'.log')).open('w') as f:p=sp.run(cmd,cwd=build,env=dict(env,STRATA_MMQ_Q4_K_QSUM=value,STRATA_DUMP_MOE_INPUT=str(out/f'moe20-{value}.bin'),STRATA_DUMP_MOE_LAYER='20'),stdout=f,stderr=sp.STDOUT,timeout=240)
 receipt.append(dict(name=name,argv=cmd,exit=p.returncode));print(name,p.returncode,flush=True)
initial=sp.check_output(['systemctl','is-active','llama-inference.service'],text=True).strip()
try:
 sp.run(['sudo','systemctl','stop','llama-inference.service'],check=True)
 for value in ['0','1']:
  for quant in ['Q4_K_M','IQ2_M']:run([str(root/'results/strata-qsum-model-v1/parity'),str(root/'models'/f'Qwen3.8-35B-A3B-{quant}.gguf'),'0','20','39'],f'weights-{quant}-{value}',value)
  cmd=[str(variant/'strata-q36'),'--model',str(model),'--tokens',','.join(map(str,ids)),'--force',','.join(forced),'--dump-full-logits',str(out/f'logits-{value}.bin'),'--max-context','4096','--chunk','128','--expert-vram','3','--cpu-threads','5','--mtp','0','--hybrid-rows','1']
  run(cmd,'logits-'+value,value)
finally:
 if initial=='active':sp.run(['sudo','systemctl','start','llama-inference.service'],check=True)
 (out/'receipt.json').write_text(json.dumps(dict(initial=initial,final=sp.check_output(['systemctl','is-active','llama-inference.service'],text=True).strip(),runs=receipt),indent=2)+'\n')
