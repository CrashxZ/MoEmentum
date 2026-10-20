#!/usr/bin/env python3
"""Compile the actual parity translation unit on the host, without GPU execution.

Requires existing CUDA-compatible runtime and ggml headers. This checks C++/API
and optional symbol dependencies, not device compilation or full backend linking.
"""
import argparse
from pathlib import Path
import subprocess


def run(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-include', required=True)
    parser.add_argument('--ggml-include', required=True)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    common = [args.cxx, '-std=c++17', '-O0', '-I' + str(root / 'include'),
              '-I' + args.runtime_include, '-I' + args.ggml_include]
    source = str(root / 'src/kernels/native_expert_parity.cpp')
    for enabled in [False, True]:
        output = args.work_dir / ('mmq-on.o' if enabled else 'mmq-off.o')
        flags = ['-DSTRATA_PARITY_MMQ=1'] if enabled else []
        run(common + flags + ['-c', source, '-o', str(output)])
        symbols = run(['nm', '-u', '-C', str(output)])
        (output.with_suffix('.symbols')).write_text(symbols)
        references = [line for line in symbols.splitlines() if 'strata::prefill::mmq::' in line]
        assert bool(references) == enabled, f'MMQ={enabled}: unexpected dependencies: {references}'
        preprocessed = run(common + flags + ['-E', source])
        unavailable = 'Q36 MMQ: unavailable (fixture not built)' in preprocessed
        assert unavailable == (not enabled), f'MMQ={enabled}: unavailable diagnostic={unavailable}'
        print(f'MMQ={enabled}: real translation unit compiled; MMQ unresolved references={len(references)}; unavailable diagnostic={unavailable}')


if __name__ == '__main__':
    main()
