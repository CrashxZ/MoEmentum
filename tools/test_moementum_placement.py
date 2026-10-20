#!/usr/bin/env python3
"""CPU-only tests of the actual bounded parser and extracted request/placement seams.

CUDA calls use checked host-memory stand-ins: this is NOT GPU/graph/model proof.
No dependency downloads, model access, services or remote operations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['parser', 'request'], default='request')
    p.add_argument('--map', type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    header = ROOT / 'src/qwen35/q36_calibration_map_v4.hpp'
    assert header.exists(), 'missing calibrated-placement parser'
    valid = 'STRATA_CALIBRATION_MAP_V1\nmodel_sha256 ' + 'a' * 64 + '\nlayers 41\nexperts 256\n'
    valid += ''.join(f'layer {i} slots 23 experts ' + ' '.join(map(str, range(100,123))) + '\n' for i in range(41))
    cases = [('valid', valid, True),
             ('huge-layers', valid.replace('layers 41', 'layers 9223372036854775807'), False),
             ('huge-experts', valid.replace('experts 256', 'experts 9223372036854775807'), False),
             ('negative', valid.replace('layers 41', 'layers -1'), False),
             ('missing-mtp', valid.replace('layers 41', 'layers 40'), False),
             ('slots', valid.replace('slots 23', 'slots 24', 1), False),
             ('duplicate', valid.replace('experts 100 101 102', 'experts 100 100 102', 1), False),
             ('range', valid.replace('experts 100 101 102', 'experts 256 101 102', 1), False),
             ('bad-sha', valid.replace('a' * 64, 'A' * 64), False),
             ('truncated', '\n'.join(valid.splitlines()[:-1]), False),
             ('trailing', valid + 'unexpected\n', False),
             ('extra-token', valid.replace('layers 41', 'layers 41 extra'), False),
             ('magic', valid.replace('MAP_V1', 'MAP_V2'), False)]
    for name, data, _ in cases:
        (out / (name + '.map')).write_text(data)
    cpp = r'''
#include "q36_calibration_map_v4.hpp"
#include <cassert>
#include <cstring>
#include <cstdio>
using namespace q36_calibration;
int main(int argc, char** argv) {
    CalibrationMap m; std::string err;
    const bool ok = load_calibration_map_file(argv[1], m, err);
    assert(ok == (std::string(argv[2]) == "yes"));
    if (ok) {
        assert(m.layer_targets.size()==41 && m.layer_targets[40].size()==23);
        assert(validate_topology(m,41,256,std::vector<size_t>(41,23),err));
        assert(!validate_topology(m,40,256,std::vector<size_t>(40,23),err));
        assert(!validate_topology(m,41,255,std::vector<size_t>(41,23),err));
        assert(!validate_topology(m,41,256,std::vector<size_t>(41,22),err));
        assert(verify_model_sha256(m,m.model_sha256,err));
        assert(!verify_model_sha256(m,std::string(64,'b'),err));
        assert(!verify_model_sha256(m,"",err));
        assert(!verify_model_sha256(m,std::string(64,'A'),err));
    }
    std::printf("parser %s: expected result verified\n", argv[1]);
}
'''
    receipts = []

    def run(argv):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('Q36_', 'STRATA_'))}
        r = subprocess.run(argv, text=True, capture_output=True, timeout=60, env=env)
        receipts.append(dict(argv=list(map(str, argv)), exit=r.returncode, stdout=r.stdout, stderr=r.stderr))
        (out / 'receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
        print(r.stdout + r.stderr, end='')
        assert r.returncode == 0, f'failed: {argv}'

    (out / 'parser.cpp').write_text(cpp)
    run([os.environ.get('CXX', 'g++'), '-std=c++17', '-Wall', '-Wextra', '-Wno-unused-parameter', '-I', str(header.parent), str(out / 'parser.cpp'), '-o', str(out / 'parser')])
    for name, _, ok in cases:
        run([str(out / 'parser'), str(out / (name + '.map')), 'yes' if ok else 'no'])
    if a.map:
        assert hashlib.sha256(a.map.read_bytes()).hexdigest() == '920b9538d7e86a82ca9eed444482cd5a396442acc94792d34f352418ded67e0d'
        run([str(out / 'parser'), str(a.map.resolve()), 'yes'])
    if a.stage == 'request':
        source = (ROOT / 'src/qwen35/q36_engine.cu').read_text()
        assert 'void request_local_placement()' in source, 'missing request-local placement helper'
        # The actual production methods/block, not reimplemented reset logic.
        helper = source[source.index('    bool v4_map_loaded'):source.index('    /// The GPU half of a hybrid MoE:')]
        guard = source[source.index('    bool placement_request_allowed('):source.index('    void request(const std::string& line)')]
        boundary = source[source.index('        // [Q36-PLACEMENT-BOUNDARY]'):source.index('        if (adm) {   // another session holding')]
        guard_call = source.index('if (!placement_request_allowed(adm, geni))')
        assert source.index('    void request(const std::string& line)') < guard_call < source.index('        char* p = const_cast<char*>(line.c_str())')
        fixture = (ROOT / 'tools/moementum_placement_test.cpp').read_text()
        reset = source[source.index('    void reset() {'):source.index('    /// y [T][n_out] = W x')]
        fixture = fixture.replace('// EXTRACT_HELPER', helper).replace('// EXTRACT_GUARD', guard).replace('// EXTRACT_BOUNDARY', boundary).replace('// EXTRACT_RESET', reset)
        (out / 'request.cpp').write_text(fixture)
        run([os.environ.get('CXX', 'g++'), '-std=c++17', '-Wall', '-Wextra', '-I', str(header.parent), str(out / 'request.cpp'), '-o', str(out / 'request')])
        run([str(out / 'request'), str(out / 'valid.map')] + ([str(a.map.resolve())] if a.map else []))
    print('PASS: CPU parser' + (' and actual extracted placement/request seams' if a.stage == 'request' else '') + '; no GPU proof')


if __name__ == '__main__':
    main()
