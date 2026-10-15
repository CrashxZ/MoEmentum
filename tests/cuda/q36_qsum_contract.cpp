// Contract test: compare with pinned upstream bytes, not a second local quantizer.
#include "strata/prefill/moe_mmq.hpp"
#include <cuda_runtime.h>
#include <ggml.h>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
void quantize_mmq_q8_1_cuda(const float*,const int32_t*,void*,ggml_type,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,cudaStream_t);
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int main(int argc,char**argv) {
 try {
 bool on=argc>1 && std::atoi(argv[1])==1; int failures=0, changed=0;
 for(int type: {12,13,14,8,21,22}) for(int rows: {1,3,105}) {
 const int cols=768,padded=1024; std::vector<float>x(rows*cols); for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.37f)*.7f+.001f;
 float* dx;void *a,*b; size_t n=strata::prefill::mmq::q8_bytes(rows,cols); ck(cudaMalloc(&dx,x.size()*4));ck(cudaMalloc(&a,n));ck(cudaMalloc(&b,n));ck(cudaMemset(a,0,n));ck(cudaMemset(b,0,n));ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
 quantize_mmq_q8_1_cuda(dx,nullptr,b,(ggml_type)type,cols,cols,rows*cols,rows*cols,padded,rows,1,1,0);
 strata::prefill::mmq::quantize(dx,nullptr,a,type,cols,cols,rows,nullptr);ck(cudaDeviceSynchronize());
 std::vector<unsigned char>aa(n),bb(n);ck(cudaMemcpy(aa.data(),a,n,cudaMemcpyDeviceToHost));ck(cudaMemcpy(bb.data(),b,n,cudaMemcpyDeviceToHost));
 if(on && type==12) {for(int block=0;block<padded/128*rows;++block)for(int j=0;j<4;++j){auto p=block*144+j*4;ggml_fp16_t h;std::memcpy(&h,bb.data()+p,2);int sum=0;for(int k=0;k<32;++k)sum+=(int8_t)bb[block*144+16+j*32+k];h=ggml_fp32_to_fp16(ggml_fp16_to_fp32(h)*sum);if(std::memcmp(bb.data()+p+2,&h,2))++changed;std::memcpy(bb.data()+p+2,&h,2);}}
 if(aa!=bb){std::fprintf(stderr,"FAIL type=%d rows=%d quantized-sum=%d bytes differ\n",type,rows,on);++failures;}else std::printf("PASS type=%d rows=%d quantized-sum=%d\n",type,rows,on);
 cudaFree(a);cudaFree(b);cudaFree(dx);
 }
 if(on && !changed) return 2; return failures?1:0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}
