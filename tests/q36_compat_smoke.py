#!/usr/bin/env python3
"""Bounded serial compatibility smokes, not throughput benchmarks.
Usage: q36_compat_smoke.py REMOTE_BENCH_ROOT
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
out = root / 'results/strata-loader-fix/final'
out.mkdir(parents=True, exist_ok=True)
ids = json.loads((root / 'prompt-short-ids.json').read_text())
work = json.loads((root / 'workloads.json').read_text())
assert ids == work['prompt_ids'] and len(ids) == 105 and ids[-1] == 248068
binary = root / 'strata-build-loaderfix/strata-q36'
env = dict(os.environ, LD_LIBRARY_PATH=str(root/'cuda/lib')+':'+str(binary.parent))
summary = []
for quant in ['IQ2_M', 'IQ3_M', 'Q4_K_M']:
    dest = out / ('smoke-' + quant)
    dest.mkdir(exist_ok=False)
    argv = [str(binary), '--model', str(root/'models'/('Qwen3.8-35B-A3B-'+quant+'.gguf')),
            '--tokens', ','.join(map(str, ids)), '--n', '16', '--max-context', '4096',
            '--chunk', '128', '--expert-vram', '3', '--cpu-threads', '6', '--mtp', '0']
    receipt = dict(argv=argv, env_overrides={'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']},
                   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   prompt_ids=ids, prompt_count=len(ids), output_limit=16, timeout_s=180,
                   lock='/home/inference/ralphi-voice/worker.lock')
    (dest/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    with open(receipt['lock'], 'a') as lock:
        deadline = time.monotonic()+600
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()>deadline:
                    raise TimeoutError('exclusive GPU lock unavailable')
                time.sleep(1)
        start = time.monotonic()
        with open(dest/'stdout.log','wb') as stdout, open(dest/'stderr.log','wb') as stderr:
            try:
                p = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr, timeout=180)
                receipt.update(exit_code=p.returncode, status='exited')
            except subprocess.TimeoutExpired:
                receipt.update(exit_code=None, status='timeout')
        receipt['wall_s'] = time.monotonic()-start
    text = (dest/'stdout.log').read_text(errors='replace')+'\n'+(dest/'stderr.log').read_text(errors='replace')
    gen = re.search(r'^gen:\s*([^\n]*)', text, re.M)
    receipt['generated_ids'] = list(map(int, gen[1].split())) if gen else []
    receipt['generated_count'] = len(receipt['generated_ids'])
    (dest/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    summary.append(receipt)
    print(quant, receipt['status'], receipt['exit_code'], 'generated', receipt['generated_count'], flush=True)
    print(text[-3500:], flush=True)
(out/'smoke-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
