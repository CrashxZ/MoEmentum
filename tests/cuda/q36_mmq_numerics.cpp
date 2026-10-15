// Standalone diagnostic only. Reuse the EXACT legacy synthetic fixture (seed is argv index).
// Build/link with native_expert_parity's compile/link flags, but a NEW output path.
#define main strata_retained_parity_main
#include "../../src/kernels/native_expert_parity.cpp"
#undef main
#include <stdexcept>

namespace diagnostic {
namespace mmq = strata::prefill::mmq;
void ck(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Dev {
    void* p; explicit Dev(size_t n) { ck(cudaMalloc(&p,n)); ck(cudaMemset(p,0,n)); }
    ~Dev() { cudaFree(p); }
};
float half_at(const uint8_t* p) { ggml_fp16_t h; std::memcpy(&h,p,2); return ggml_fp16_to_fp32(h); }
void scale_min(const uint8_t* s, int j, int& sc, int& mn) {
    if(j<4) { sc=s[j]&63; mn=s[j+4]&63; }
    else { sc=(s[j+4]&15)|((s[j-4]>>6)<<4); mn=(s[j+4]>>4)|((s[j]>>6)<<4); }
}
// Independent scalar Q4_K x MMQ DS4 reference. The asymmetric minimum uses
// half(sum(ORIGINAL x)), NOT half(d)*sum(q). This is not a simple dot(W,d*q).
std::vector<float> reference(const uint8_t* w, int nr, int nc, const std::vector<uint8_t>& aq,
                             bool quantized_sum) {
    std::vector<float> y(NT*nr);
    for(int t=0;t<NT;++t) for(int r=0;r<nr;++r) {
        double acc=0;
        for(int b=0;b<nc/256;++b) {
            const uint8_t* wb=w+(size_t)r*(nc/256)*144+b*144;
            for(int j=0;j<8;++j) {
                int sc,mn; scale_min(wb+4,j,sc,mn);
                int k0=b*256+j*32;
                const uint8_t* ab=aq.data()+((k0/128)*NT+t)*144;
                float da=half_at(ab+(k0%128)/32*4);
                double sa=half_at(ab+(k0%128)/32*4+2);
                int dot=0,sumq=0;
                for(int k=0;k<32;++k) {
                    const int a=(int8_t)ab[16+k0%128+k];
                    const int code=(wb[16+(j/2)*32+k] >> ((j%2)*4))&15;
                    dot+=code*a; sumq+=a;
                }
                if(quantized_sum) sa=(double)da*sumq;
                // SM86 MMA load_tiles_q4_K multiplies each weight super-scale by
                // its 6-bit subscale IN HALF2 before converting to float for accumulation.
                const float ws=ggml_fp16_to_fp32(ggml_fp32_to_fp16(half_at(wb)*sc));
                const float wm=ggml_fp16_to_fp32(ggml_fp32_to_fp16(half_at(wb+2)*mn));
                acc+=(double)ws*da*dot-(double)wm*sa;
            }
        }
        y[t*nr+r]=(float)acc;
    }
    return y;
}
std::vector<float> float_product(const uint8_t* w,int nr,int nc,const std::vector<float>& x) {
    std::vector<float> y(NT*nr), row(nc);
    for(int r=0;r<nr;++r) {
        ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(w+(size_t)r*(nc/256)*144,row.data(),nc);
        for(int t=0;t<NT;++t) { double acc=0; for(int k=0;k<nc;++k) acc+=(double)row[k]*x[t*nc+k]; y[t*nr+r]=(float)acc; }
    }
    return y;
}
std::vector<float> swiglu(const std::vector<float>& gu) {
    std::vector<float> h(NT*FF);
    for(int t=0;t<NT;++t) for(int r=0;r<FF;++r) {
        double g=gu[t*2*FF+r],u=gu[t*2*FF+FF+r];
        h[t*FF+r]=(float)(g/(1+std::exp(-g))*u);
    }
    return h;
}
struct Result { std::vector<float> gpu, exact, qsum, fp; };
Result product(const uint8_t* w,int nr,int nc,const std::vector<float>& x,cudaStream_t s,bool altered_sum=false) {
    const size_t wb=(size_t)nr*(nc/256)*144;
    Dev dw(wb+4096),dx(x.size()*4),dq(mmq::q8_bytes(NT,nc)),dy(NT*nr*4),db(2*4),di(NT*4);
    int bounds[]={0,NT},ids[]={0,1,2};
    ck(cudaMemcpy(dw.p,w,wb,cudaMemcpyHostToDevice)); ck(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(db.p,bounds,sizeof bounds,cudaMemcpyHostToDevice)); ck(cudaMemcpy(di.p,ids,sizeof ids,cudaMemcpyHostToDevice));
    ck(cudaDeviceSynchronize());
    mmq::quantize((float*)dx.p,nullptr,dq.p,12,nc,nc,NT,s);
    ck(cudaStreamSynchronize(s));
    std::vector<uint8_t> aq(mmq::q8_bytes(NT,nc)); ck(cudaMemcpy(aq.data(),dq.p,aq.size(),cudaMemcpyDeviceToHost));
    if(altered_sum) {
        for(int b=0;b<nc/128;++b) for(int t=0;t<NT;++t) for(int j=0;j<4;++j) {
            auto* ab=aq.data()+(b*NT+t)*144; int sq=0; for(int k=0;k<32;++k) sq+=(int8_t)ab[16+j*32+k];
            ggml_fp16_t sum=ggml_fp32_to_fp16(half_at(ab+j*4)*sq); std::memcpy(ab+j*4+2,&sum,2);
        }
        ck(cudaMemcpy(dq.p,aq.data(),aq.size(),cudaMemcpyHostToDevice));
    }
    mmq::Context ctx; mmq::Product p;
    p.w=dw.p; p.type=12; p.w_rows=nr; p.w_cols=nc; p.expert_bytes=wb; p.n=1;
    p.xq=dq.p; p.bounds=(int32_t*)db.p; p.ids=(int32_t*)di.p; p.total_rows=NT; p.max_rows=NT;
    p.dst=(float*)dy.p; p.ld_dst=nr;
    ctx.run(p,s); ck(cudaStreamSynchronize(s));
    Result z; z.gpu.resize(NT*nr); ck(cudaMemcpy(z.gpu.data(),dy.p,z.gpu.size()*4,cudaMemcpyDeviceToHost));
    z.exact=reference(w,nr,nc,aq,false); z.qsum=reference(w,nr,nc,aq,true); z.fp=float_product(w,nr,nc,x);
    return z;
}
void metric(const char* tag,const std::vector<float>& a,const std::vector<float>& b) {
    double maxabs=0; for(size_t i=0;i<a.size();++i) maxabs=std::max(maxabs,std::fabs((double)a[i]-b[i]));
    std::printf("%-48s rel_L1=%.12g max_abs=%.12g\n",tag,rel(a,b),maxabs);
}
void self_test() {
    // All weight codes=2, scale=1, minimum=1. Stored input scale=.5,
    // codes=3, original sum=50. Per subblock result = 2*.5*96 - 50 = 46.
    std::vector<uint8_t> w(144,0),a(NT*2*144,0); ggml_fp16_t one=ggml_fp32_to_fp16(1),d=ggml_fp32_to_fp16(.5),sum=ggml_fp32_to_fp16(50);
    std::memcpy(w.data(),&one,2); std::memcpy(w.data()+2,&one,2);
    for(int j=0;j<4;++j) {w[4+j]=1;w[8+j]=1;} for(int j=0;j<4;++j) w[12+j]=17;
    std::fill(w.begin()+16,w.end(),0x22);
    for(int b=0;b<2*NT;++b) { for(int j=0;j<4;++j) {std::memcpy(a.data()+b*144+j*4,&d,2); std::memcpy(a.data()+b*144+j*4+2,&sum,2);} std::fill(a.begin()+b*144+16,a.begin()+(b+1)*144,3); }
    auto orig=reference(w.data(),1,256,a,false),q=reference(w.data(),1,256,a,true);
    for(int t=0;t<NT;++t) if(orig[t]!=368 || q[t]!=384) throw std::runtime_error("DS4 original-vs-quantized sum self-test failed");
    std::printf("PASS scalar DS4 original-sum/minimum semantic self-test (368 vs 384)\n");
}
}
int main(int argc,char** argv) {
    try {
        diagnostic::self_test();
        if(!q36_geometry) throw std::runtime_error("set STRATA_TEST_Q36=1");
        int seed=argc>1?std::atoi(argv[1]):6; // The retained failing Q4/Q4 is fifth pair => argv index 6.
        cpu::NativeFmt f; std::string err; if(!cpu::native_fmt(12,12,H,FF,f,err)) throw std::runtime_error(err);
        auto blob=synthetic_blob(f,seed);
        std::mt19937 rng(11+seed); std::normal_distribution<float> nd(0,1); std::vector<float>x(NT*H);for(auto&v:x)v=nd(rng);
        cudaStream_t s;diagnostic::ck(cudaStreamCreate(&s));
        auto gu=diagnostic::product(blob.data(),2*FF,H,x,s);
        auto hf=diagnostic::swiglu(gu.fp),he=diagnostic::swiglu(gu.exact),hq=diagnostic::swiglu(gu.qsum);
        // Use actual CUDA SwiGLU for the measured pipeline, not host libm.
        diagnostic::Dev dg(gu.gpu.size()*4),dh(NT*FF*4);diagnostic::ck(cudaMemcpy(dg.p,gu.gpu.data(),gu.gpu.size()*4,cudaMemcpyHostToDevice));
        strata::prefill::mmq::swiglu((float*)dg.p,(float*)dh.p,NT,FF,false,s);diagnostic::ck(cudaStreamSynchronize(s));
        std::vector<float> hg(NT*FF);diagnostic::ck(cudaMemcpy(hg.data(),dh.p,hg.size()*4,cudaMemcpyDeviceToHost));
        auto down=diagnostic::product(blob.data()+f.down_off,H,FF,hg,s);
        auto baseline=diagnostic::float_product(blob.data()+f.down_off,H,FF,hf);
        auto exact_chain=diagnostic::product(blob.data()+f.down_off,H,FF,he,s);
        auto counter=diagnostic::product(blob.data()+f.down_off,H,FF,hq,s);
        auto patched_gu=diagnostic::product(blob.data(),2*FF,H,x,s,true);
        auto patched_down=diagnostic::product(blob.data()+f.down_off,H,FF,diagnostic::swiglu(patched_gu.gpu),s,true);
        using diagnostic::metric;
        std::printf("seed=%d H=%lld FF=%lld NT=%d\n",seed,(long long)H,(long long)FF,NT);
        metric("GU GPU vs exact DS4 semantics",gu.gpu,gu.exact);
        metric("GU GPU vs unrounded FP baseline",gu.gpu,gu.fp);
        metric("GU qsum counterfactual vs FP baseline",gu.qsum,gu.fp);
        metric("SwiGLU GPU vs exact DS4 host chain",hg,he);
        metric("SwiGLU GPU vs unrounded FP baseline",hg,hf);
        metric("down GPU vs exact DS4 same actual input",down.gpu,down.exact);
        metric("down GPU vs unrounded same actual input",down.gpu,down.fp);
        metric("full GPU vs original unrounded baseline",down.gpu,baseline);
        metric("full exact DS4 host chain vs baseline",exact_chain.exact,baseline);
        metric("full GPU vs exact DS4 host chain",down.gpu,exact_chain.exact);
        metric("full qsum counterfactual chain vs baseline",counter.qsum,baseline);
        metric("GPU edited-sum buffers vs baseline",patched_down.gpu,baseline);
        metric("GPU edited-sum GU vs original exact qsum",patched_gu.gpu,gu.qsum);
        // Independent semantic gate: keep the ORIGINAL FP screen separate and unchanged.
        const bool semantic=rel(gu.gpu,gu.exact)<1e-5 && rel(down.gpu,down.exact)<1e-5 && rel(hg,he)<1e-5;
        std::printf("semantic gate 1e-5: %s; retained FP gate 3e-2: %s\n",semantic?"PASS":"FAIL",rel(down.gpu,baseline)<3e-2?"PASS":"FAIL");
        diagnostic::ck(cudaStreamDestroy(s));return semantic?0:1;
    } catch(const std::exception&e) {std::fprintf(stderr,"diagnostic failed: %s\n",e.what());return 1;}
}
