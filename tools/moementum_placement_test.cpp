// CPU seam harness. Production parser/helper/guard/boundary/reset are inserted
// verbatim by test_moementum_placement.py. CUDA stand-ins prove host control
// flow and copy arguments/order, NOT driver behavior, graph replay or numerics.
#include "q36_calibration_map_v4.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>

static std::vector<std::function<void()>> pending;
static std::vector<int> syncs;
static int copies = 0;
constexpr int cudaMemcpyHostToDevice = 1;
int cudaStreamSynchronize(int st) {
    syncs.push_back(st);
    for (auto& f : pending) f();
    pending.clear();
    return 0;
}
int cudaMemcpyAsync(void* dst, const void* src, size_t bytes, int kind, int st) {
    assert(kind == cudaMemcpyHostToDevice && st == 1);
    ++copies;
    pending.emplace_back([=] { std::memcpy(dst, src, bytes); });
    return 0;
}
int cudaMemsetAsync(void* dst, int value, size_t bytes, int st) {
    assert(st==1);
    pending.emplace_back([=] { std::memset(dst,value,bytes); });
    return 0;
}
#define CK(x) assert((x)==0)
struct Geometry {
    int n_expert=256, ssm_conv_channels=1, ssm_d_conv=2;
    int ssm_v_heads=1, ssm_state_size=1, n_embd=1;
};
struct Model { bool hybrid=true; Geometry g; };
struct Layer {
    std::vector<uint8_t> host, device;
    uint8_t *blobs=nullptr, *hblobs=nullptr;
    size_t stride=4;
    struct { size_t bytes=3; } L;
};
struct Engine {
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;
        std::vector<double> score;
        int32_t* slot_of_d=nullptr;
        std::vector<int32_t> device_map;
    };
    struct Sess {
        std::vector<int32_t> live, imgs;
        bool active=false;
        int64_t pos=0;
        int32_t rdelta=0;
    };
    Model model; Model* m=&model;
    int st=1, cst=2, cur=0, n_sess=1;
    bool rows_mode=false;
    std::vector<Sess> sess{1};
    std::vector<Hy> hy{41};
    std::vector<Layer> layers{41};
    float cv=5, state=6, last=7;
    std::vector<float*> conv{&cv}, S{&state};
    float* hlast=&last;
    int64_t pos=4;
    void set_delta(int32_t d) { sess[(size_t)cur].rdelta=d; }
    void select(int s) { assert(s==0); cur=s; }
    const Layer& layer_at(int li) const { return layers[(size_t)li]; }
    Engine() {
        for(size_t li=0;li<41;++li) {
            auto& l=layers[li]; l.host.resize(256*4); l.device.resize(23*4,0xee);
            l.blobs=l.device.data(); l.hblobs=l.host.data();
            for(int e=0;e<256;++e) for(int b=0;b<3;++b) l.host[e*4+b]=(uint8_t)e;
            auto& h=hy[li]; h.slot_of.assign(256,-1); h.score.assign(256,9);
            h.exp_in_slot.resize(23); h.device_map.resize(256,-2); h.slot_of_d=h.device_map.data();
            for(int s=0;s<23;++s) {
                h.exp_in_slot[s]=s+23;
                for(int b=0;b<3;++b) l.device[s*4+b]=(uint8_t)(s+23);
            }
        }
    }
    // EXTRACT_RESET
    // EXTRACT_HELPER
};
struct Serve {
    Engine& E; int n_active=0;
    struct Slot { bool active=false; };
    std::vector<Slot> slots{1};
    int matches=0, picks=0;
    size_t match_len(const Engine::Sess& ss,const std::vector<int32_t>& ids,const std::vector<int32_t>&) {
        ++matches;
        return !ss.live.empty() && ss.live.size()<ids.size() && std::equal(ss.live.begin(),ss.live.end(),ids.begin()) ? ss.live.size() : 0;
    }
    int pick_session(const std::vector<int32_t>&,const std::vector<int32_t>&) { ++picks; return 0; }
    // EXTRACT_GUARD
    size_t boundary() {
        const bool adm=false; const int slot=-1;
        const std::vector<int32_t> ids{1,2,3};
        struct { std::vector<int32_t> imgs; } ir;
        // EXTRACT_BOUNDARY
        return reuse;
    }
};
void check(Engine& e, int first=0) {
    for(size_t li=0;li<41;++li) {
        const auto& h=e.hy[li]; const auto& l=e.layers[li];
        for(int x=0;x<256;++x) {
            assert(h.score[x]==0);
            assert(h.slot_of[x]==(x>=first && x<first+23?x-first:-1));
            assert(h.device_map[x]==h.slot_of[x]);
        }
        for(int s=0;s<23;++s) {
            assert(h.exp_in_slot[s]==first+s);
            for(int b=0;b<3;++b) assert(l.device[s*4+b]==first+s);
            assert(l.device[s*4+3]==0xee); // no stride padding overwritten
        }
    }
}
int main(int argc,char** argv) {
    assert(argc==2 || argc==3);
    const rlimit no_core{0,0}; assert(setrlimit(RLIMIT_CORE,&no_core)==0);
    unsetenv("Q36_INITIAL_PLACEMENT_MAP_V4"); unsetenv("Q36_STATELESS_ADAPTIVE_V2");
    Engine e; Serve s{e}; e.sess[0].live={1,2}; e.sess[0].imgs={9};
    assert(s.boundary()==2 && s.picks==1 && e.cv==5 && copies==0); // unset unchanged
    setenv("Q36_STATELESS_ADAPTIVE_V2","",1); // presence, including empty, opts in
    assert(s.placement_request_allowed(false,false));
    assert(!s.placement_request_allowed(true,false));
    assert(!s.placement_request_allowed(false,true));
    e.n_sess=2; assert(!s.placement_request_allowed(false,false)); e.n_sess=1;
    s.n_active=1; assert(!s.placement_request_allowed(false,false)); s.n_active=0;
    e.sess[0].active=true; assert(!s.placement_request_allowed(false,false)); e.sess[0].active=false;
    s.slots[0].active=true; assert(!s.placement_request_allowed(false,false)); s.slots[0].active=false;
    e.rows_mode=true; assert(!s.placement_request_allowed(false,false)); e.rows_mode=false;
    assert(s.boundary()==0); check(e);
    assert(e.cv==0 && e.state==0 && e.last==0 && e.pos==0 && e.sess[0].pos==0 && e.sess[0].rdelta==0);
    assert(e.sess[0].live.empty() && e.sess[0].imgs.empty() && s.picks==1);
    assert(syncs[0]==1 && syncs[1]==2 && copies==41*24);
    auto* blobs=e.layers[0].blobs; auto* map=e.hy[0].slot_of_d;
    const int before=copies;
    e.sess[0].live={1,2}; e.sess[0].imgs={9}; e.cv=e.state=e.last=8;
    assert(s.boundary()==0); check(e); assert(copies-before==41); // unchanged slots skip blobs
    assert(blobs==e.layers[0].blobs && map==e.hy[0].slot_of_d);
    setenv("Q36_INITIAL_PLACEMENT_MAP_V4",argv[1],1);
    setenv("Q36_EXPECTED_MODEL_SHA256",std::string(64,'a').c_str(),1);
    const int pre_map=copies;
    assert(s.boundary()==0 && e.v4_map_loaded); check(e,100);
    assert(copies-pre_map==41*24);
    e.hy[40].exp_in_slot[0]=50; e.hy[40].score[50]=900;
    e.layers[40].device[0]=50;
    e.sess[0].live={1,2}; e.sess[0].imgs={9}; e.cv=e.state=e.last=8;
    const int pre_repeat=copies;
    assert(s.boundary()==0); check(e,100);
    assert(copies-pre_repeat==42 && e.cv==0 && e.state==0 && e.last==0);
    assert(e.sess[0].live.empty() && e.sess[0].imgs.empty());
    assert(blobs==e.layers[0].blobs && map==e.hy[0].slot_of_d);
    // Validation is a real fatal path before uploads, tested in a child process.
    const pid_t child=fork(); assert(child>=0);
    if(child==0) {
        Engine bad;
        setenv("Q36_EXPECTED_MODEL_SHA256",std::string(64,'b').c_str(),1);
        bad.request_local_placement(); _exit(90);
    }
    int status=0; assert(waitpid(child,&status,0)==child);
    assert(WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT);
    for(int mode=0;mode<3;++mode) {
        const pid_t pid=fork(); assert(pid>=0);
        if(pid==0) {
            Engine bad;
            if(mode==0) bad.hy.pop_back(); // cannot drop inactive MTP
            if(mode==1) bad.hy[40].exp_in_slot.pop_back();
            if(mode==2) bad.model.hybrid=false;
            bad.request_local_placement(); _exit(90);
        }
        assert(waitpid(pid,&status,0)==pid);
        assert(WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT);
    }
    if(argc==3) {
        q36_calibration::CalibrationMap expected; std::string err;
        assert(q36_calibration::load_calibration_map_file(argv[2],expected,err));
        setenv("Q36_INITIAL_PLACEMENT_MAP_V4",argv[2],1);
        setenv("Q36_EXPECTED_MODEL_SHA256",expected.model_sha256.c_str(),1);
        Engine frozen; Serve fs{frozen};
        assert(fs.boundary()==0);
        for(size_t li=0;li<41;++li) {
            const auto& h=frozen.hy[li]; const auto& l=frozen.layers[li];
            std::vector<int32_t> inverse(256,-1);
            for(int sl=0;sl<23;++sl) {
                const int target=expected.layer_targets[li][sl];
                inverse[target]=sl;
                assert(h.exp_in_slot[sl]==target);
                for(int b=0;b<3;++b) assert(l.device[sl*4+b]==target);
            }
            assert(h.slot_of==inverse && h.device_map==inverse);
            for(double score:h.score) assert(score==0);
        }
        std::puts("external pinned frozen map: all 41x23 targets, 41x256 inverse entries and scores verified on CPU stand-ins");
    }
    unsetenv("Q36_STATELESS_ADAPTIVE_V2"); unsetenv("Q36_INITIAL_PLACEMENT_MAP_V4");
    e.n_sess=2; s.n_active=1;
    assert(s.placement_request_allowed(true,true)); // no restriction when unset
    std::puts("actual extracted seams: guards/reset/reuse0/41-layer maps/scores/copy args/order/stable addresses PASS (CPU stand-ins only)");
}
