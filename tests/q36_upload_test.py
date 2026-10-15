#!/usr/bin/env python3
"""Compile the actual host upload_f32 from q36_engine.cu; use real CUDA readback.
Usage: q36_upload_test.py BUILD_DIR OUTPUT_DIR MODEL.gguf [MODEL.gguf ...]
Requires the existing native-expert GGML dependency and CUDA toolkit on PATH.
No copied conversion implementation and no synthetic model files.
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[1]
build, out = map(pathlib.Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
source = (root / 'src/qwen35/q36_engine.cu').read_text()
body = source[source.index('float bf16_to_f32('):source.index('\nbool load(', source.index('float bf16_to_f32('))]
prelude = r'''
#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/wait.h>
#include <unistd.h>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); exit(2); } } while(0)
'''
main = r'''
void check(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> ref(t.elements()), got(t.elements());
    const auto* traits = ggml_get_type_traits((ggml_type)t.type);
    if(t.type == GGML_TYPE_F32) memcpy(ref.data(),src,ref.size()*4);
    else traits->to_float(src,ref.data(),ref.size());
    float* d=upload_f32(t,src);
    CK(cudaMemcpy(got.data(),d,got.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaFree(d));
    if(memcmp(ref.data(),got.data(),ref.size()*4)) {
        fprintf(stderr,"FAIL %s type=%u CPU-reference/readback mismatch\n",t.name.c_str(),t.type); exit(3);
    }
    printf("PASS %s type=%u elements=%zu bit-exact GGML CPU -> GPU -> CPU\n",t.name.c_str(),t.type,got.size());
    fflush(stdout);
}
int main(int argc,char** argv) {
    // Invalid dimensions must fail before reading src or attempting CUDA allocation.
    if(argc==2 && std::string(argv[1])=="--invalid") {
        strata::TensorInfo t; t.name="invalid-row"; t.type=GGML_TYPE_Q4_K; t.shape={255,2};
        upload_f32(t,nullptr); return 0;
    }
    if(argc==2 && std::string(argv[1])=="--overflow") {
        strata::TensorInfo t; t.name="overflow"; t.type=GGML_TYPE_F32; t.shape={uint64_t(1)<<63,2};
        upload_f32(t,nullptr); return 0;
    }
    if(argc==2 && std::string(argv[1])=="--removed-type") {
        strata::TensorInfo t; t.name="removed-type"; t.type=4; t.shape={256,2};
        upload_f32(t,nullptr); return 0;
    }
    for(int type : {0,1,30,8}) {
        strata::TensorInfo t; t.name="regression"; t.type=type; t.shape={256,2};
        std::vector<float> f(512); for(size_t i=0;i<f.size();++i) f[i]=std::sin(float(i))*.25f;
        std::vector<uint8_t> q(ggml_row_size((ggml_type)type,f.size()));
        if(type==0) memcpy(q.data(),f.data(),q.size());
        else ggml_get_type_traits((ggml_type)type)->from_float_ref(f.data(),q.data(),f.size());
        check(t,q.data());
    }
    for(int i=1;i<argc;++i) {
        auto gm=strata::GgufModel::open(argv[i]);
        for(const char* name : {"blk.0.ssm_alpha.weight","blk.0.ssm_beta.weight"}) {
            size_t shard=0; auto* t=gm.find(name,&shard);
            if(!t) { fprintf(stderr,"missing %s\n",name); return 4; }
            check(*t,gm.shard(shard).tensor_data(*t));
        }
    }
}
'''
cpp = out / 'q36_upload_test.cu'
cpp.write_text(prelude + body + main)
gg = build / '_deps/strata_llamacpp-src/ggml'
exe = out / 'q36_upload_test'
subprocess.run(['nvcc', '-std=c++17', '-DSTRATA_Q36_CPU_POOL=1', '-I'+str(root/'include'), '-I'+str(gg/'include'), str(cpp), str(build/'ggml/src/libggml-base.a'), '-lpthread', '-ldl', '-o', str(exe)], check=True)
rc = subprocess.run([str(exe), *sys.argv[3:]]).returncode
if rc:
    sys.exit(rc)
for flag in ['--invalid', '--overflow', '--removed-type']:
    result = subprocess.run([str(exe), flag])
    if result.returncode != 1:
        sys.exit('FAIL '+flag+': expected controlled rejection exit 1, got '+str(result.returncode))
    print('PASS controlled size rejection '+flag)
