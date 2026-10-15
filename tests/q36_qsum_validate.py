#!/usr/bin/env python3
"""Run under worker.lock. Isolated object/archive/link reuse, no Ninja writes."""
import os,sys,json,subprocess as sp,shlex,hashlib,time,re,shutil
from pathlib import Path
root=Path('/home/inference/moe-bench');build=root/'strata-build-loaderfix'
out=root/'results'/sys.argv[2];out.mkdir(exist_ok=False)
env=dict(os.environ,LD_LIBRARY_PATH=f'{root}/cuda/lib:{build}',STRATA_MMQ_KQ='1',STRATA_TEST_Q36='1')
def run(argv,name,check=True,timeout=240,e=None):
    with (out/(name+'.log')).open('w') as f:
        f.write(shlex.join(argv)+'\n');f.flush();p=sp.run(argv,cwd=build,env=e or env,stdout=f,stderr=sp.STDOUT,timeout=timeout)
    print(name,p.returncode,flush=True)
    if check and p.returncode: raise RuntimeError(name)
    return p.returncode
def compile_test(src,dst,archive=None):
    text=(build/'build.ninja').read_text();obj=text.split('build CMakeFiles/native_expert_parity.dir/src/kernels/native_expert_parity.cpp.o:',1)[1].split('\n\n',1)[0];link=text.split('build native_expert_parity:',1)[1].split('\n\n',1)[0]
    def var(b,n):return re.search(r'^  '+n+r' = (.*)$',b,re.M).group(1).replace('$','')
    cmd=['c++']+shlex.split(var(obj,'DEFINES')+' '+var(obj,'FLAGS')+' '+var(obj,'INCLUDES'))+[str(src),'-o',str(out/dst)]+shlex.split(var(link,'LINK_PATH')+' '+var(link,'LINK_LIBRARIES'))
    if archive:cmd=[str(archive) if x=='libstrata_mmq.a' else x for x in cmd]
    run(cmd,'build-'+dst)
protected=[build/'strata-q36',build/'libstrata_kq.so',build/'libstrata_mmq.a']
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
before=hashes();initial=sp.check_output(['systemctl','is-active','llama-inference.service'],text=True).strip();receipt={'initial_service':initial,'before':before}
try:
    sp.run(['sudo','systemctl','stop','llama-inference.service'],check=True)
    (out/'libstrata_kq.so').symlink_to(build/'libstrata_kq.so')
    if sys.argv[1]=='red':
        compile_test(root/'strata/tests/cuda/q36_qsum_contract.cpp','contract')
        receipt['default_exit']=run([str(out/'contract'),'0'],'default')
        receipt['optin_exit']=run([str(out/'contract'),'1'],'optin',False,e=dict(env,STRATA_MMQ_Q4_K_QSUM='1'))
        assert receipt['optin_exit']==1,'Expected missing feature failure'
    else:
        commands=sp.check_output(['ninja','-t','commands','strata-q36'],cwd=build,text=True).splitlines()
        cmd=shlex.split(next(x for x in commands if ' -c ' in x and 'moe_mmq.cu.o' in x))
        for flag in ['-o','-MF','-MT']:
            idx=cmd.index(flag);cmd[idx+1]=str(out/('moe_mmq.cu.o.d' if flag=='-MF' else 'moe_mmq.cu.o'))
        run(cmd,'compile-mmq',timeout=360)
        shutil.copy2(build/'libstrata_mmq.a',out/'libstrata_mmq.a');run(['ar','r',str(out/'libstrata_mmq.a'),str(out/'moe_mmq.cu.o')],'archive')
        link=shlex.split(next(x for x in commands if ' -o strata-q36 ' in x).replace(': && ','').replace(' && :',''))
        link[link.index('-o')+1]=str(out/'strata-q36');link=[str(out/'libstrata_mmq.a') if x=='libstrata_mmq.a' else x for x in link];run(link,'link')
        compile_test(root/'strata/tests/cuda/q36_qsum_contract.cpp','contract',out/'libstrata_mmq.a')
        for value in ['0','1']:
            run([str(out/'contract'),value],'contract-'+value,e=dict(env,STRATA_MMQ_Q4_K_QSUM=value))
        compile_test(root/'strata/tests/cuda/q36_mmq_numerics.cpp','numerics',out/'libstrata_mmq.a')
        for value in ['0','1']:
            for seed in [6,2,3,4,5,7,8]:
                rc=run([str(out/'numerics'),str(seed)],f'numerics-{value}-{seed}',False,e=dict(env,STRATA_MMQ_Q4_K_QSUM=value))
                assert rc==0, 'Semantic gate failed'
                if value=='1':
                    assert 'retained FP gate 3e-2: PASS' in (out/f'numerics-{value}-{seed}.log').read_text(), 'Unchanged 3% FP gate failed'
    receipt['after']=hashes();receipt['protected_unchanged']=before==receipt['after']
finally:
    if initial=='active':sp.run(['sudo','systemctl','start','llama-inference.service'],check=True)
    receipt['final_service']=sp.check_output(['systemctl','is-active','llama-inference.service'],text=True).strip()
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
