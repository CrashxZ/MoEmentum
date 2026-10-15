#!/usr/bin/env python3
"""Retokenize saved llama output; compare 16 generated IDs, not logit parity.
Run under the same worker.lock as all guest work. No generation or downloads.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1]).resolve()
receipts = json.loads((root/'results/strata-expert-support/smokes/summary.json').read_text())
results = []
for r in receipts:
    quant = r['quant']
    baseline = root/f'results/formal-llama-105t/llama-{quant}/1-measured/stdout.log'
    argv = ['/opt/llama.cpp/b11429/llama-tokenize', '-m', str(root/f'models/Qwen3.8-35B-A3B-{quant}.gguf'),
            '--stdin', '--ids', '--no-bos', '--offline']
    p = subprocess.run(argv, input=baseline.read_bytes(), capture_output=True,
                       env=dict(os.environ, LD_LIBRARY_PATH='/opt/llama.cpp/b11429'), timeout=60)
    dest = root/f'results/strata-expert-support/tokenize-{quant}'
    dest.with_suffix('.stdout.log').write_bytes(p.stdout)
    dest.with_suffix('.stderr.log').write_bytes(p.stderr)
    p.check_returncode()
    ids = json.loads(p.stdout)
    got = r['generated_ids']
    prefix = 0
    for a,b in zip(ids, got):
        if a != b: break
        prefix += 1
    results.append(dict(quant=quant, argv=argv, baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                        llama_retokenized_prefix=ids[:len(got)], strata_ids=got,
                        matching_prefix_count=prefix, exact_prefix_match=ids[:len(got)] == got,
                        caveat='Saved rendered output retokenized, not raw llama generation IDs or logits.'))
(root/'results/strata-expert-support/llama-prefix-parity.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
