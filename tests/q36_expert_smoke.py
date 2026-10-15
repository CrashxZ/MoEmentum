#!/usr/bin/env python3
"""Run ONLY under worker.lock, enclosing build and numerical tests too."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
root = Path(sys.argv[1]).resolve()
streamed = len(sys.argv) > 2 and sys.argv[2] == '--streamed'
out = root / ('results/strata-expert-support/smokes-streamed' if streamed else 'results/strata-expert-support/smokes')
out.mkdir(parents=True, exist_ok=False)
ids = json.loads((root/'prompt-short-ids.json').read_text())
assert len(ids) == 105 and ids[-1] == 248068
binary = root/'strata-build-loaderfix/strata-q36'
env = dict(os.environ, LD_LIBRARY_PATH=f'{root}/cuda/lib:{binary.parent}')
summary = []
for quant in ['IQ2_M', 'IQ3_M', 'Q4_K_M']:
    dest = out/quant
    dest.mkdir()
    argv = [str(binary), '--model', str(root/'models'/f'Qwen3.8-35B-A3B-{quant}.gguf'),
            '--tokens', ','.join(map(str, ids)), '--n', '16', '--max-context', '4096',
            '--chunk', '128', '--expert-vram', '3', '--cpu-threads', '6', '--mtp', '0']
    if streamed:
        argv += ['--hybrid-rows', '1']
    receipt: dict = dict(quant=quant, argv=argv, prompt_ids=ids, output_limit=16,
                   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   env_overrides={'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']})
    start = time.monotonic()
    with (dest/'stdout.log').open('wb') as stdout, (dest/'stderr.log').open('wb') as stderr:
        try:
            p = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr, timeout=240)
            receipt.update(exit_code=p.returncode, status='exited')
        except subprocess.TimeoutExpired:
            receipt.update(exit_code=None, status='timeout')
    receipt['wall_s'] = time.monotonic()-start
    text = (dest/'stdout.log').read_text(errors='replace')+'\n'+(dest/'stderr.log').read_text(errors='replace')
    gen = re.search(r'^gen:\s*([^\n]*)', text, re.M)
    receipt['generated_ids'] = list(map(int, gen[1].split())) if gen else []
    receipt['passed_generation_gate'] = receipt['exit_code'] == 0 and len(receipt['generated_ids']) == 16
    (dest/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    summary.append(receipt)
    print(quant, receipt['status'], receipt['exit_code'], 'generated', len(receipt['generated_ids']), flush=True)
    print(text[-4000:], flush=True)
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
sys.exit(0 if all(r['passed_generation_gate'] for r in summary) else 1)
