#!/usr/bin/env python3
"""Execute the actual host quantize_scatter routing with instrumented backends.

Control-flow regression only, not GPU arithmetic or DS4 layout validation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = (ROOT / 'src/prefill/moe_mmq.cu').read_text()
    begin = source.index('void quantize_scatter(')
    end = source.index('\nContext::Context()', begin)
    prelude = r'''
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
using cudaStream_t = void*;
enum ggml_type { GGML_TYPE_Q4_K=12, GGML_TYPE_Q8_0=8 };
static int gathers=0, scatters=0;
int64_t pad512(int64_t n) { return (n+511)/512*512; }
void ck(int,const char*) {}
int cudaGetLastError() { return 0; }
void quantize(const float*,const int32_t*,void*,int,int64_t,int64_t,int64_t,void*) { ++gathers; }
template<class... A> void quantize_scatter_mmq_q8_1_cuda(A...) { ++scatters; }
'''
    driver = r'''
int main(int argc,char** argv) {
    if(argc!=4) return 2;
    setenv("STRATA_MMQ_Q4_K_QSUM",argv[1],1);
    setenv("STRATA_QUANT_GATHER",argv[2],1);
    const bool qsum=std::strcmp(argv[1],"1")==0, gather=std::strcmp(argv[2],"1")==0;
    int fail=0, checks=0;
    for(int t:{12,8}) {
        gathers=scatters=0;
        quantize_scatter(nullptr,nullptr,nullptr,nullptr,t,512,512,3,2,nullptr);
        const bool expected=gather || (qsum && t==12);
        fail += gathers != int(expected) || scatters != int(!expected); ++checks;
        quantize_scatter(nullptr,nullptr,nullptr,nullptr,t,512,512,0,2,nullptr);
        fail += gathers != int(expected) || scatters != int(!expected); ++checks;
    }
    std::printf("qsum=%s gather=%s checks=%d failures=%d (routing only)\n",argv[1],argv[2],checks,fail);
    return fail?1:0;
}
'''
    cpp = out / 'routing.cpp'
    cpp.write_text('#include <initializer_list>\n' + prelude + source[begin:end] + driver)
    commands = [[os.environ.get('CXX', 'g++'), '-std=c++17', str(cpp), '-o', str(out / 'routing')]]
    commands += [[str(out / 'routing'), q, g, 'test'] for q, g in [('0','0'),('1','0'),('true','0'),('1','1'),('0','1')]]
    receipts = []
    failed = False
    for command in commands:
        p = subprocess.run(command, text=True, capture_output=True, timeout=30)
        receipts.append(dict(argv=command, exit=p.returncode, stdout=p.stdout, stderr=p.stderr))
        (out / 'receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
        print(p.stdout + p.stderr, end='')
        failed |= p.returncode != 0
        if p.returncode and command == commands[0]:
            break
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
