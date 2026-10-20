// CPU-only synthetic IQ3_S/IQ2_S down-row regression. No model or GPU.
// Tests actual public iq256_rows_v against GGML's AVX2 dot, not a duplicate kernel.
#include <iq_avx2.cpp>
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
extern "C" { float ggml_table_f32_f16[1 << 16]; }
namespace cpu = strata::kernels::cpu;
static uint64_t next(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static uint32_t bits(float x) { uint32_t n; std::memcpy(&n, &x, sizeof(n)); return n; }
int main() {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") || !__builtin_cpu_supports("f16c")) {
        std::fprintf(stderr, "SKIP: AVX2/FMA/F16C required\n"); return 77;
    }
    for (int i=0;i<65536;++i) ggml_table_f32_f16[i]=GGML_COMPUTE_FP16_TO_FP32((ggml_fp16_t)i);
    constexpr int N=512, R=9, T=8;
    uint64_t checked=0, mismatches=0, preservation=0;
    for (int type : {21,22}) for (uint64_t seed : {1ULL,42ULL,871ULL,1097ULL}) for (int pad : {0,16}) {
        uint64_t state=seed;
        const size_t block=type==21?110:82, stride=2*block+pad;
        std::vector<uint8_t> w(R*stride+4096,0xa5);
        for (int r=0;r<R;++r) for (int b=0;b<2;++b) {
            uint8_t* p=w.data()+r*stride+b*block;
            for (size_t k=0;k<block;++k) p[k]=uint8_t(next(state));
            const uint16_t scale=uint16_t(0x2400+(next(state)%0x1000));
            std::memcpy(p,&scale,2);
        }
        std::vector<block_q8_K> acts(T*2);
        std::vector<float> input(N);
        for (int t=0;t<T;++t) {
            for (auto& v:input) v=float(int(next(state)%20001)-10000)/1024.0f;
            quantize_row_q8_K_ref(input.data(),acts.data()+t*2,N);
        }
        const auto wb=w; const auto ab=acts;
        const void* ap[T]; float* out[T];
        std::vector<float> expected(T*R), got(T*R,123456.0f);
        for (int t=0;t<T;++t) {
            ap[t]=acts.data()+t*2; out[t]=got.data()+t*R;
            for (int r=0;r<R;++r) {
                if (type==21) ggml_vec_dot_iq3_s_q8_K(N,&expected[t*R+r],0,w.data()+r*stride,0,ap[t],0,1);
                else ggml_vec_dot_iq2_s_q8_K(N,&expected[t*R+r],0,w.data()+r*stride,0,ap[t],0,1);
            }
        }
        for (int variant : {0,1}) for (int nt=1;nt<=T;++nt) {
            std::fill(got.begin(),got.end(),123456.0f);
            cpu::iq256_rows_v(variant,type,w.data(),stride,N,ap,nt,out,0,0);
            for (float x:got) preservation+=bits(x)!=bits(123456.0f);
            cpu::iq256_rows_v(variant,type,w.data(),stride,N,ap,nt,out,R/2,R);
            for (int t=0;t<T;++t) for (int r=0;r<R/2;++r) preservation+=bits(got[t*R+r])!=bits(123456.0f);
            cpu::iq256_rows_v(variant,type,w.data(),stride,N,ap,nt,out,0,R/2);
            for (int t=0;t<T;++t) for (int r=0;r<R;++r) {
                if(t<nt) { ++checked; mismatches+=bits(got[t*R+r])!=bits(expected[t*R+r]); }
                else preservation+=bits(got[t*R+r])!=bits(123456.0f);
            }
        }
        preservation+=w!=wb;
        preservation+=std::memcmp(acts.data(),ab.data(),acts.size()*sizeof(block_q8_K))!=0;
    }
    std::printf("synthetic=1 checked=%llu bit_mismatches=%llu preservation_failures=%llu\n",
        (unsigned long long)checked,(unsigned long long)mismatches,(unsigned long long)preservation);
    return mismatches || preservation ? 1 : 0;
}
