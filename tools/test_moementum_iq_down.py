#!/usr/bin/env python3
"""Compile/run a CPU-only real-IQ-TU regression. No downloads, GPU or model load."""
from pathlib import Path
import argparse,json,os,subprocess,time

ROOT=Path(__file__).resolve().parents[1]
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--llama-dir',type=Path,required=True,help='llama.cpp checkout; tested commit 3cf03257f219afbe7334045ff7c6a06ac68c627d')
    a.add_argument('--output',type=Path,required=True,help='fresh, nonexistent output directory')
    o=a.parse_args();g=o.llama_dir.resolve()/'ggml';out=o.output.resolve()
    for p in [g/'src/ggml-cpu/arch/x86/quants.c',g/'src/ggml-quants.c',g/'include/ggml.h']:
        if not p.is_file():a.error('missing GGML dependency: '+str(p))
    out.mkdir(parents=True,exist_ok=False)
    flags=['-O3','-fno-fast-math','-ffp-contract=off','-mavx2','-mfma','-mf16c','-ffunction-sections','-fdata-sections']
    inc=sum((['-I',str(p)] for p in [ROOT/'src/kernels/cpu',ROOT/'include',g/'include',g/'src',g/'src/ggml-cpu']),[])
    cc=os.environ.get('CC','gcc');cxx=os.environ.get('CXX','g++')
    commands=[[cc,*flags,*inc,'-c',str(g/'src/ggml-cpu/arch/x86/quants.c'),'-o',str(out/'ref.o')],
              [cc,*flags,*inc,'-c',str(g/'src/ggml-quants.c'),'-o',str(out/'quant.o')],
              [cxx,'-std=c++20',*flags,*inc,str(ROOT/'tools/moementum_iq_down_test.cpp'),str(out/'ref.o'),str(out/'quant.o'),'-Wl,--gc-sections','-o',str(out/'test')],
              [str(out/'test')]]
    receipts=[]
    for i,cmd in enumerate(commands):
        started=time.monotonic()
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=180 if i<3 else 30)
            row=dict(argv=cmd,seconds=time.monotonic()-started,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
        except subprocess.TimeoutExpired:
            row=dict(argv=cmd,seconds=time.monotonic()-started,exit=None,error='timeout')
        receipts.append(row);(out/'receipt.json').write_text(json.dumps(receipts,indent=2))
        print(row.get('stdout',''),end='');print(row.get('stderr',''),end='')
        if row['exit']!=0:return row['exit'] if row['exit'] is not None else 1
    return 0
if __name__=='__main__':raise SystemExit(main())
