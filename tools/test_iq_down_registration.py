#!/usr/bin/env python3
"""Host-only registration check, NOT a CUDA/HIP build or device parity test.

Compile the actual format-list block and d_qk switch in both Q6_K configurations.
Fmt stand-ins expose only geometry. Device execution is covered by the separately
built native_grouped_parity executable, not by this check.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = (ROOT / 'src/kernels/cuda/iq_kernels.cu').read_text()
    lists = source[source.index('#ifdef STRATA_Q6K_EXPERTS'):source.index('#define STRATA_MMVQ_FMTS')]
    query = source[source.index('int d_qk(int t) {'):source.index('bool gu_split(int t) {')]
    fixture = '''#include <cstdio>
#include <initializer_list>
template<int T> struct Fmt { static constexpr int qk = 256; };
''' + lists + query + '''
int main() {
    int checked=0, failures=0;
    for (int t : {20,21,22,23,42,7,6,2,3,8,13,40,144,146}) {
        ++checked; if(d_qk(t)!=256) { ++failures; std::printf("missing down type %d\\n",t); }
    }
#ifdef STRATA_Q6K_EXPERTS
    ++checked; failures += d_qk(14)!=256;
#else
    ++checked; failures += d_qk(14)!=0;
#endif
    ++checked; failures += d_qk(999)!=0;
    std::printf("host registration checked=%d failures=%d (not device compilation)\\n",checked,failures);
    return failures ? 1 : 0;
}
'''
    (out / 'registration.cpp').write_text(fixture)
    receipts = []
    for q6 in (False, True):
        exe = out / ('q6-on' if q6 else 'q6-off')
        commands = [[os.environ.get('CXX', 'g++'), '-std=c++20',
                     *(['-DSTRATA_Q6K_EXPERTS'] if q6 else []),
                     str(out / 'registration.cpp'), '-o', str(exe)], [str(exe)]]
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            receipts.append(dict(argv=command, exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
            (out / 'receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
            print(result.stdout, end='')
            print(result.stderr, end='')
            if result.returncode:
                return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
