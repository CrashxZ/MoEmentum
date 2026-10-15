#!/usr/bin/env python3
"""Bounded guest model validation. Invoke ONLY under shared worker.lock."""
import os,sys,json,subprocess as sp,shlex,hashlib,time,re
from pathlib import Path
root=Path('/home/inference/moe-bench');build=root/'strata-build-loaderfix';variant=root/'results/strata-qsum-green-v1';out=root/'results'/sys.argv[1];out.mkdir(exist_ok=False)
env=dict(os.environ,LD_LIBRARY_PATH=f'{root}/cuda/lib:{build}',STRATA_MMQ_KQ='1',STRATA_TEST_Q36='1')
receipts=[]
def run(argv,name,value='0',timeout=240):
    start=time.monotonic()
    with (out/(name+'.log')).open('w') as f:
        f.write(shlex.join(argv)+'\n');f.flush()
        try:p=sp.run(argv,cwd=build,env=dict(env,STRATA_MMQ_Q4_K_QSUM=value),stdout=f,stderr=sp.STDOUT,timeout=timeout);rc=p.returncode
        except sp.TimeoutExpired:rc=124
    r=dict(name=name,argv=argv,optin=value,exit=rc,wall_s=time.monotonic()-start);receipts.append(r);(out/'runs.json').write_text(json.dumps(receipts,indent=2)+'\n');print(name,rc,flush=True);return rc
initial=sp.check_output(['systemctl','is-active','llama-inference.service'],text=True).strip()
try:
    sp.run(['sudo','systemctl','stop','llama-inference.service'],check=True)
    (out/'libstrata_kq.so').symlink_to(build/'libstrata_kq.so')
    text=(build/'build.ninja').read_text();obj=text.split('build CMakeFiles/native_expert_parity.dir/src/kernels/native_expert_parity.cpp.o:',1)[1].split('\n\n',1)[0];link=text.split('build native_expert_parity:',1)[1].split('\n\n',1)[0]
    def var(b,n):return re.search(r'^  '+n+r' = (.*)$',b,re.M).group(1).replace('$','')
    cmd=['c++']+shlex.split(var(obj,'DEFINES')+' '+var(obj,'FLAGS')+' '+var(obj,'INCLUDES'))+[str(root/'strata/src/kernels/native_expert_parity.cpp'),'-o',str(out/'parity')]+shlex.split(var(link,'LINK_PATH')+' '+var(link,'LINK_LIBRARIES'));cmd=[str(variant/'libstrata_mmq.a') if x=='libstrata_mmq.a' else x for x in cmd]
    assert run(cmd,'compile-parity')==0
    pairs=['iq2_s/iq2_s','iq3_s/iq3_s','iq2_s/q4_K','iq3_s/q4_K','q4_K/q4_K','q4_K/q6_K','q6_K/q6_K']
    for value in ['0','1']:
        rc=run([str(out/'parity'),'--synthetic']+pairs,'synthetic-'+value,value)
        assert rc==(1 if value=='0' else 0), 'Unexpected retained synthetic gate status'
        for quant in ['Q4_K_M','IQ2_M']:
            assert run([str(out/'parity'),str(root/'models'/f'Qwen3.8-35B-A3B-{quant}.gguf'),'0','20','39'],f'weights-{quant}-{value}',value)==0
    ids=json.loads((root/'prompt-short-ids.json').read_text());assert len(ids)==105 and ids[-1]==248068
    for quant in ['Q4_K_M','IQ2_M']:
        for value in ['0','1']:
            cmd=[str(variant/'strata-q36'),'--model',str(root/'models'/f'Qwen3.8-35B-A3B-{quant}.gguf'),'--tokens',','.join(map(str,ids)),'--n','256','--max-context','4096','--chunk','128','--expert-vram','3','--cpu-threads','5','--mtp','0','--hybrid-rows','1']
            run(cmd,f'generation-{quant}-{value}',value,300)
finally:
    if initial=='active':sp.run(['sudo','systemctl','start','llama-inference.service'],check=True)
    final=sp.check_output(['systemctl','is-active','llama-inference.service'],text=True).strip()
    (out/'service.json').write_text(json.dumps(dict(initial=initial,final=final),indent=2)+'\n')
