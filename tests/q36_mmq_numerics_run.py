#!/usr/bin/env python3
"""Run ONLY under the shared worker flock; never rebuild existing artifacts."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
build = root / 'strata-build-loaderfix'
out = root / 'results' / (sys.argv[2] if len(sys.argv) > 2 else 'strata-mmq-numerics')
out.mkdir(parents=True, exist_ok=False)
# The isolated loader resolves its library NEXT TO /proc/self/exe, not via LD_LIBRARY_PATH.
(out / 'libstrata_kq.so').symlink_to(build / 'libstrata_kq.so')
source = root / 'strata/tests/cuda/q36_mmq_numerics.cpp'
text = (build / 'build.ninja').read_text()
obj = text.split('build CMakeFiles/native_expert_parity.dir/src/kernels/native_expert_parity.cpp.o:', 1)[1].split('\n\n',1)[0]
link = text.split('build native_expert_parity:',1)[1].split('\n\n',1)[0]
def var(block, name):
    return re.search(r'^  '+name+r' = (.*)$',block,re.M).group(1).replace('$','')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
protected = [build/'strata-q36', build/'libstrata_kq.so', root/'strata/src/prefill/moe_mmq.cu',root/'strata/src/kernels/native_expert_parity.cpp']
before = {str(p):sha(p) for p in protected}
cmd = ['c++'] + shlex.split(var(obj,'DEFINES')+' '+var(obj,'FLAGS')+' '+var(obj,'INCLUDES'))
cmd += [str(source),'-o',str(out/'q36_mmq_numerics')]
cmd += shlex.split(var(link,'LINK_PATH')+' '+var(link,'LINK_LIBRARIES'))
env = dict(os.environ,STRATA_TEST_Q36='1',STRATA_MMQ_KQ='1',LD_LIBRARY_PATH=f'{root}/cuda/lib:{build}')
receipts=[]
with (out/'build.log').open('w') as f:
    f.write(shlex.join(cmd)+'\n'); f.flush()
    p = subprocess.run(cmd,cwd=build,env=env,stdout=f,stderr=subprocess.STDOUT)
if p.returncode: sys.exit(p.returncode)
for seed in [6,2,3,4,5,7,8]:
    cmd = [str(out/'q36_mmq_numerics'), str(seed)]
    with (out/f'seed-{seed}.log').open('w') as f:
        p = subprocess.run(cmd,cwd=build,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
    receipts.append({'seed':seed,'argv':cmd,'exit':p.returncode})
after = {str(p):sha(p) for p in protected}
receipt={'protected_before':before,'protected_after':after,'protected_unchanged':before==after,'runs':receipts,'diagnostic_sha256':sha(out/'q36_mmq_numerics'),'source_sha256':sha(source)}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
sys.exit(0 if before==after and all(r['exit']==0 for r in receipts) else 1)
