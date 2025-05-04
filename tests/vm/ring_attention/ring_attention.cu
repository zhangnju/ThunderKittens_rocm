#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

constexpr int SM_COUNT = 148;
constexpr int QO_BLOCK_SIZE = 128; // sequence length must be divisible by this * 2
constexpr int KV_BLOCK_SIZE = 128; // sequence length must be divisible by this
constexpr int HEAD_DIM = 64;

constexpr float softmax_scale = 0.08838834764831843f;          // 1 / sqrt(HEAD_DIM=128)
constexpr float softmax_temp = 1.44269504089f * softmax_scale; // 1 / {sqrt(HEAD_DIM=128) * ln(2)}

using qo_tile = st_bf<QO_BLOCK_SIZE, HEAD_DIM>;
using kv_tile = st_bf<KV_BLOCK_SIZE, HEAD_DIM>;
using a_tile = st_bf<QO_BLOCK_SIZE, KV_BLOCK_SIZE>;

using config = default_config;
struct globals {
    constexpr static int num_devices = 8;

    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using barrier_layout = gl<uint, 1, 1, 1, num_devices>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    
    using qo_layout = gl<bf16, -1, -1, -1, HEAD_DIM, qo_tile>; // Batch, Head, Seq, Dim (full MHA)
    using kv_layout = gl<bf16, -1, -1, -1, HEAD_DIM, kv_tile>;

    instruction_layout instructions;
    gl_array<barrier_layout, num_devices> barriers;
    timing_layout timings;

    qo_layout Q; // local Q sharded on sequence dimension
    gl_array<kv_layout, num_devices> K0s;
    gl_array<kv_layout, num_devices> K1s;
    gl_array<kv_layout, num_devices> V0s;
    gl_array<kv_layout, num_devices> V1s;
    qo_layout O;

    dim3 grid() { return dim3(SM_COUNT); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct RingAttentionOp {
    static constexpr int opcode = 725;
    static constexpr int PIPELINE_STAGES = 2;
    static_assert(config::NUM_CONSUMER_WARPS == 8, "RingAttentionOp only supports 2 consumer warpgroups");

    struct parsed_instruction {
        int B;             // batch index              (in units of 1)
        int H;             // head index               (in units of 1)
        int QO_idx;        // local Q block index      (in units of `QO_BLOCK_SIZE * 2` tokens)
        int num_KV_blocks; // # of KV blocks to handle (in units of `KV_BLOCK_SIZE` tokens)
        int ring_stage; // current ring stage index (0, 1, ..., NUM_DEVS - 1)
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            ?????
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &q_arrived(state<config> &s, int id)            { return s.semaphores()[id]; }
    __device__ static inline semaphore &k_arrived(state<config> &s, int stage)         { return s.semaphores()[2 + PIPELINE_STAGES * 0 + stage]; }
    __device__ static inline semaphore &v_arrived(state<config> &s, int stage)         { return s.semaphores()[2 + PIPELINE_STAGES * 1 + stage]; }
    __device__ static inline semaphore &a_arrived(state<config> &s, int id, int stage) { return s.semaphores()[2 + PIPELINE_STAGES * 1 + stage]; }
    __device__ static inline semaphore &qk_finished(state<config> &s, int stage)       { return s.semaphores()[2 + PIPELINE_STAGES * 2 + stage]; }
    __device__ static inline semaphore &av_finished(state<config> &s, int stage)       { return s.semaphores()[2 + PIPELINE_STAGES * 3 + stage]; }
    __device__ static inline semaphore &qk_unloaded(state<config> &s, int id)          { return s.semaphores()[2 + PIPELINE_STAGES * 4 + id]; }
    __device__ static inline semaphore &o_arrived(state<config> &s, int id)          { return s.semaphores()[2 + PIPELINE_STAGES * 4 + id]; }
    // TODO fix indexing




    __device__ static inline semaphore &att_arrived(state<config> &s, int stage)  { return s.semaphores()[2 + PIPELINE_STAGES * 2 + stage]; }
 
    __device__ static inline int get_q_page(state<config> &s, int id)    { return id; } // use PIDs for now
    __device__ static inline int get_k_page(state<config> &s, int stage) { return 2 + PIPELINE_STAGES * 0 + stage; }
    __device__ static inline int get_v_page(state<config> &s, int stage) { return 2 + PIPELINE_STAGES * 1 + stage; }
    __device__ static inline int get_a_page(state<config> &s, int stage) { return 2 + PIPELINE_STAGES * 2 + stage; }
    __device__ static inline int get_o_page(state<config> &s)            { return 2 + PIPELINE_STAGES * 3; }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            return query; // TODO fix this
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            // TODO
            return 0;
        }
    };

    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            // TODO: Wait & Release unused pages
            // TODO: wait for comms complete
            parsed_instruction inst{s};
            int laneid = warp::laneid();
            if (laneid < 1) { // Load Q for 2 consumer warpgroups
                auto q_page = get_q_page(s, laneid);
                auto &q = *reinterpret_cast<q_tile *>(s.pages[q_page].data);
                wait_page_ready(s, q_page);
                tma::expect(q_arrived(s), q);
                tma::load_async(q, g.Q, {inst.B, inst.H, inst.QO_idx + laneid, 0}, q_arrived(s));
            }
            } else if (laneid == 2) { // Load Ks
                int phasebit = 0;
                for (int i = 0; i < inst.num_kv_blocks i++) {
                    int stage = i % PIPELINE_STAGES;
                    auto k_page = get_k_page(s, stage);
                    auto &k = *reinterpret_cast<kv_tile *>(s.pages[k_page].data);
                    if (i < PIPELINE_STAGES) {
                        wait_page_ready(s, get_k_page(s, stage));
                    }
                    tma::expect(k_arrived(s, stage), kv_tile);
                    tma::load_async(k, g.K0s[inst.dev_idx], {inst.B, inst.H, i, 0}, k_arrived(s, stage));
                    if (i >= PIPELINE_STAGES) {
                        tma::wait(qk_finished(s, stage), phasebit);
                        phasebit ^= 1;
                    }
                }
            } else if (laneid == 3) { // Load Vs
                int phasebit = 0;
                for (int i = 0; i < inst.num_kv_blocks i++) {
                    int stage = i % PIPELINE_STAGES;
                    auto v_page = get_v_page(s, stage);
                    auto &v = *reinterpret_cast<kv_tile *>(s.pages[v_page].data);
                    if (i < PIPELINE_STAGES) {
                        wait_page_ready(s, get_v_page(s, stage));
                    }
                    tma::expect(v_arrived(s, stage), kv_tile);
                    tma::load_async(v, g.V0s[inst.dev_idx], {inst.B, inst.H, i, 0}, v_arrived(s, stage));
                    if (i >= PIPELINE_STAGES) {
                        tma::wait(av_finished(s, stage), phasebit);
                        phasebit ^= 1;
                    }
                }
            }
            // TODO: finish pages
        }
    };

    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();

            // Nothing is ready until the tensor cores are ready
            s.wait_tensor_ready();

            if (laneid < 2) { // Launch Q @ K^T for the 2 consumer warpgroups
                auto q_page = get_q_page(s, laneid);
                auto &q = *reinterpret_cast<qo_tile *>(s.pages[q_page].data);
                auto k_page = get_k_page(s, stage);
                auto &k = *reinterpret_cast<kv_tile *>(s.pages[k_page].data);
                wait(q_arrived(s, laneid), 0);
                
                int phasebit = 0;
                for (int i = 0; i < inst.num_kv_blocks; ++i) {
                    int stage = i % PIPELINE_STAGES;
                    if (i > 0) {
                        wait(qk_unloaded(s, laneid), phasebit);
                    }
                    wait(k_arrived(s, stage), phasebit);
                    auto qk_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, KV_BLOCK_SIZE>>(laneid*KV_BLOCK_SIZE);
                    mm_ABt(qk_accumulator, q, k, qk_finished(s, stage));
                    phasebit ^= 1;
                }
            } else if (laneid < 4) { // Launch ATT @ V for the 2 consumer warpgroups
                auto att_page = get_a_page(s, laneid-2);
                auto &att = *reinterpret_cast<a_tile *>(s.pages[att_page].data);
                auto v_page = get_v_page(s, stage);
                auto &v = *reinterpret_cast<vv_tile *>(s.pages[v_page].data);
                
                int phasebit = 0;
                for (int i = 0; i < inst.num_kv_blocks; ++i) {
                    int stage = i % PIPELINE_STAGES;
                    wait(a_arrived(s, groupid, stage), phasebit);
                    wait(v_arrived(s, stage), phasebit);
                    auto av_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, HEAD_DIM>>(2*KV_BLOCK_SIZE+(laneid-2)*HEAD_DIM);
                    mma_AB(av_accumulator, att, v, av_finished(s, stage));
                    phasebit ^= 1;
                }
            }
            // Finish tensor cores
            // TODO: Wait for last tensor core op to finish. Shouldn't do this here!
            warp::arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
        }
    };

    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {            
            parsed_instruction inst{s};
            int warpid = warpgroup::warpid();
            int groupid = warpgroup::groupid();

            rt_fl<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE> att_fl;
            rt_bf<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE> att_bf;
            rt_bf<QO_BLOCK_SIZE / 4, HEAD_DIM> out;
            col_vec<rt_fl<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE>> max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE>> scaled_max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE>> last_scaled_max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE>> diff_scaled_max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / 4, HEAD_DIM>> norm_vec;

            warp::zero(out);
            warp::neg_infty(max_vec);
            warp::zero(last_scaled_max_vec); // just not +-inf

            auto qk_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, KV_BLOCK_SIZE>>(groupid*KV_BLOCK_SIZE);
            auto av_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, HEAD_DIM>>(2*KV_BLOCK_SIZE + groupid*HEAD_DIM);

            // int o_page = get_o_page(s);
            // s.wait_page_ready(o_page);
            // auto &o = *reinterpret_cast<qo_tile *>(s.pages[o_page].data);
            // auto att_block = *reinterpret_cast<qo_tile *>(s.pages[o_page].data);
            // wait_page_ready(s, o_page);

            int phasebit = 0;
            for (int i = 0; i < inst.num_kv_blocks; ++i) {
                int stage = i % PIPELINE_STAGES;

                // Read in QK^T
                wait(qk_finished(s, stage), phasebit); // wait for mm to finish
                warpgroup::load_async(att_fl, qk_accumulator);
                tensor_load_wait();
                __syncwarp();
                wapgroup::arrive(qk_unloaded(s, groupid));

                // Get maximums and scale by softmax temp
                warp::row_max(max_vec, att_fl, max_vec);
                warp::mul(att_fl, att_fl, softmax_temp);
                warp::mul(scaled_max_vec, max_vec, softmax_temp);

                // Compute softmax numerator
                warp::sub_row(att_fl, att_fl, scaled_max_vec);
                warp::exp2(att_fl, att_fl);

                // Compute normalizer
                warp::sub(diff_scaled_max_vec, last_scaled_max_vec, scaled_max_vec);
                warp::exp2(diff_scaled_max_vec, diff_scaled_max_vec);

                // Normalize and accumulate softmax numerator (A @ V)
                // TODO skip waiting for first i
                warpgroup::load_async(out, av_accumulator);
                int a_page = get_a_page(s, stage);
                auto &a = *reinterpret_cast<a_tile *>(s.pages[a_page].data);
                if (i < PIPELINE_STAGES) {
                    wait_page_ready(s, a_page);
                }
                warpgroup::store(a, att_fl);
                warp::mul_row(out, out, diff_scaled_max_vec_reg);
                consumer::store_async(av_accumulator, out);
                tensor_store_wait();
                __syncwarp();
                warpgroup::arrive(a_arrived(s, groupid, stage));

                // Normalize and accumulate demoniator
                warp::mul(norm_vec, norm_vec, diff_scaled_max_vec_reg);
                warp::row_sum(norm_vec, att_fl, norm_vec);

                wait(av_finished(s, stage), phasebit); // wait for mma to finish
                phasebit ^= 1;
            }

            // Finish
            warpgroup::load_async(out, av_accumulator);
            warpgroup::sync(groupid); // ??
            warp::div_row(out, out, norm_vec);
            warp::log2(L_reg, norm_vec);

            arrive(o_arrived(s, groupid));
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            int laneid = warp::laneid();
            if (laneid < 2) {
                
            })
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(ring_attention, m) {
    m.doc() = "ring attention python module";
    kittens::py::bind_kernel<kvm<config, globals,
        RingAttentionOp<config>
    >>(m, "ring_attention", 
    );
}
