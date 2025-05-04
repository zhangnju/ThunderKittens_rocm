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

using qo_tile = st_bf<QO_BLOCK_SIZE, HEAD_DIM>;
using kv_tile = st_bf<KV_BLOCK_SIZE, HEAD_DIM>;
using a_tile = st_bf<QO_BLOCK_SIZE, KV_BLOCK_SIZE>;

using config = default_config;
struct globals {
    constexpr static int num_devices = 1;

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
        int num_kv_blocks; // # of KV blocks to handle (in units of `KV_BLOCK_SIZE` tokens)
        int ring_stage;    // current ring stage index (0, 1, ..., NUM_DEVS - 1)
        int num_comms;     // number of SMs doing comms
        int num_comps;     // number of instructions per ring stage per device
        int dev_idx;       // current device index     (0, 1, ..., NUM_DEVS - 1)
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            B = instruction[1];
            H = instruction[2];
            QO_idx = instruction[3];
            num_kv_blocks = instruction[4];
            ring_stage = instruction[5];
            num_comms = instruction[6];
            num_comps = instruction[7];
            dev_idx = instruction[8];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &q_arrived(state<config> &s, int id)      { return s.semaphores()[id]; }
    __device__ static inline semaphore &o_arrived(state<config> &s, int id)      { return s.semaphores()[2 + id]; }
    __device__ static inline semaphore &qk_unloaded(state<config> &s, int id)    { return s.semaphores()[4 + id]; }
    __device__ static inline semaphore &av_ready(state<config> &s, int id)       { return s.semaphores()[6 + id]; }
    __device__ static inline semaphore &k_arrived(state<config> &s, int stage)   { return s.semaphores()[8 + PIPELINE_STAGES * 0 + stage]; }
    __device__ static inline semaphore &v_arrived(state<config> &s, int stage)   { return s.semaphores()[8 + PIPELINE_STAGES * 1 + stage]; }
    __device__ static inline semaphore &qk_finished(state<config> &s, int stage) { return s.semaphores()[8 + PIPELINE_STAGES * 2 + stage]; }
    __device__ static inline semaphore &av_finished(state<config> &s, int stage) { return s.semaphores()[8 + PIPELINE_STAGES * 3 + stage]; }

    __device__ static inline int get_q_page(state<config> &s, int id)    { return id; } // use PIDs for now
    __device__ static inline int get_a_page(state<config> &s, int id)    { return 2 + id; }
    __device__ static inline int get_o_page(state<config> &s, int id)    { return 4 + id; }
    __device__ static inline int get_k_page(state<config> &s, int stage) { return 6 + PIPELINE_STAGES * 0 + stage; }
    __device__ static inline int get_v_page(state<config> &s, int stage) { return 6 + PIPELINE_STAGES * 1 + stage; }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int lids[config::NUM_PAGES] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            return lids[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for (int i = 0; i < 2; ++i) {
                init_semaphore(q_arrived(s, i), 1);
                init_semaphore(o_arrived(s, i), 4);
                init_semaphore(qk_unloaded(s, i), 4);
                init_semaphore(av_ready(s, i), 4);
            }
            for (int i = 0; i < PIPELINE_STAGES; ++i) {
                init_semaphore(k_arrived(s, i), 1);
                init_semaphore(v_arrived(s, i), 1);
                init_semaphore(qk_finished(s, i), 2);
                init_semaphore(av_finished(s, i), 2);
            }
            return 2*4 + PIPELINE_STAGES*4;
        }
    };

    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            // TODO: wait for comms complete
            parsed_instruction inst{s};
            int laneid = warp::laneid();
            if (laneid < 2) { // Load Q for the 2 consumer warpgroups
                auto q_page = get_q_page(s, laneid);
                auto &q = *reinterpret_cast<qo_tile *>(s.pages[q_page].data);
                s.wait_page_ready(q_page);
                tma::expect(q_arrived(s, laneid), q);
                tma::load_async(q, g.Q, {inst.B, inst.H, inst.QO_idx + laneid, 0}, q_arrived(s, laneid));
                // printf("Q load start %d\n", laneid);
            } else if (laneid == 2) { // Load Ks
                uint32_t phasebit = 0;
                for (int i = 0; i < inst.num_kv_blocks; i++) {
                    int stage = i % PIPELINE_STAGES;
                    auto k_page = get_k_page(s, stage);
                    auto &k = *reinterpret_cast<kv_tile *>(s.pages[k_page].data);
                    if (i < PIPELINE_STAGES) {
                        s.wait_page_ready(get_k_page(s, stage));
                    } else {
                        wait(qk_finished(s, stage), get_phasebit<0>(phasebit, stage));
                        update_phasebit<0>(phasebit, stage);
                    }
                    tma::expect(k_arrived(s, stage), k);
                    // Todo: vary by ring stage
                    tma::load_async(k, g.K0s[inst.dev_idx], {inst.B, inst.H, i, 0}, k_arrived(s, stage));
                    // printf("K load start %d\n", i);
                }
                for (int i = 0; i < PIPELINE_STAGES; i++) {
                    int stage = (i + inst.num_kv_blocks) % PIPELINE_STAGES;
                    wait(qk_finished(s, stage), get_phasebit<0>(phasebit, stage));
                    // printf("arriving K finished page %d\n", stage);
                    s.finish_page(get_k_page(s, stage), config::NUM_CONSUMER_WARPS);
                    update_phasebit<0>(phasebit, stage);
                }
            } else if (laneid == 3) { // Load Vs
                uint32_t phasebit = 0;
                for (int i = 0; i < inst.num_kv_blocks; i++) {
                    int stage = i % PIPELINE_STAGES;
                    auto v_page = get_v_page(s, stage);
                    auto &v = *reinterpret_cast<kv_tile *>(s.pages[v_page].data);
                    if (i < PIPELINE_STAGES) {
                        s.wait_page_ready(get_v_page(s, stage));
                    } else {
                        wait(av_finished(s, stage), get_phasebit<0>(phasebit, stage));
                        update_phasebit<0>(phasebit, stage);
                    }
                    tma::expect(v_arrived(s, stage), v);
                    tma::load_async(v, g.V0s[inst.dev_idx], {inst.B, inst.H, i, 0}, v_arrived(s, stage));
                    // printf("V load start %d\n", i);
                }
                for (int i = 0; i < PIPELINE_STAGES; i++) {
                    int stage = (i + inst.num_kv_blocks) % PIPELINE_STAGES;
                    wait(av_finished(s, stage), get_phasebit<0>(phasebit, stage));
                    s.finish_page(get_v_page(s, stage), config::NUM_CONSUMER_WARPS);
                    update_phasebit<0>(phasebit, stage);
                }
            } else if (6 + PIPELINE_STAGES*2 <= laneid && laneid < config::NUM_PAGES) { // Finish unused pages
                s.wait_page_ready(laneid);
                s.finish_page(laneid, config::NUM_CONSUMER_WARPS);
                // printf("page %d finished\n", laneid);
            }
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
                wait(q_arrived(s, laneid), 0);
                // printf("Q load done %d\n", laneid);
                
                uint32_t phasebit = 0;
                for (int i = 0; i < inst.num_kv_blocks; ++i) {
                    int stage = i % PIPELINE_STAGES;
                    if (i > 0) {
                        wait(qk_unloaded(s, laneid), get_phasebit<1>(phasebit, laneid));
                        update_phasebit<1>(phasebit, laneid);
                        // printf("QK unload done %d - %d\n", laneid, i - 1);
                    }
                    auto k_page = get_k_page(s, stage);
                    auto &k = *reinterpret_cast<kv_tile *>(s.pages[k_page].data);
                    wait(k_arrived(s, stage), get_phasebit<0>(phasebit, stage));
                    // printf("K load done %d\n", i);
                    // if (laneid == 0) {
                    //     printf("Launching QK %d - %d\n", laneid, i);
                    //     printf("Q:");
                    //     for (int x = 0; x < 128; x++)
                    //         printf("%f ", float(q[x]));
                    //     printf("\n");
                    //     printf("K:");
                    //     for (int x = 0; x < 128; x++)
                    //         printf("%f ", float(k[x]));
                    //     printf("\n");
                    // }


                    auto qk_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, KV_BLOCK_SIZE>>(laneid*KV_BLOCK_SIZE);
                    mm_ABt(qk_accumulator, q, k, qk_finished(s, stage));
                    // printf("qk launched %d - %d\n", laneid, i);
                    update_phasebit<0>(phasebit, stage);
                }
            } else if (laneid < 4) { // Launch ATT @ V for the 2 consumer warpgroups
                auto att_page = get_a_page(s, laneid-2);
                auto &att = *reinterpret_cast<a_tile *>(s.pages[att_page].data);

                uint32_t phasebit = 0;
                for (int i = 0; i < inst.num_kv_blocks; ++i) {
                    int stage = i % PIPELINE_STAGES;
                    auto v_page = get_v_page(s, stage);
                    auto &v = *reinterpret_cast<kv_tile *>(s.pages[v_page].data);
                    wait(v_arrived(s, stage), get_phasebit<0>(phasebit, stage));
                    wait(av_ready(s, laneid-2), get_phasebit<1>(phasebit, laneid-2));
                    
                    // printf("v load done and av ready %d - %d\n", laneid, i);
                    auto av_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, HEAD_DIM>>(2*KV_BLOCK_SIZE+(laneid-2)*HEAD_DIM);
                    mma_AB(av_accumulator, att, v, av_finished(s, stage));
                    update_phasebit<0>(phasebit, stage);
                    update_phasebit<1>(phasebit, laneid-2);
                    // printf("av launched %d - %d\n", laneid, i);
                }
            }
        }
    };

    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {                   
            parsed_instruction inst{s};
            int warpid = warpgroup::warpid();
            int groupid = warpgroup::groupid();

            // constexpr float softmax_scale = 0.08838834764831843f;         // 1 / sqrt(HEAD_DIM=128)
            constexpr float softmax_scale = 0.125;                         // 1 / sqrt(HEAD_DIM=64)
            constexpr float softmax_temp = softmax_scale * 1.44269504089f; // 1 / {sqrt(HEAD_DIM=128) * ln(2)}

            rt_fl<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE> att_fl;
            rt_fl<QO_BLOCK_SIZE / 4, HEAD_DIM> out_fl;
            col_vec<rt_fl<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE>> max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE>> scaled_max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE>> last_scaled_max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / 4, KV_BLOCK_SIZE>> diff_scaled_max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / 4, HEAD_DIM>> norm_vec;

            warp::zero(out_fl);
            warp::neg_infty(max_vec);
            warp::zero(last_scaled_max_vec); // just not +-inf
            warp::zero(norm_vec);

            auto qk_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, KV_BLOCK_SIZE>>(groupid*KV_BLOCK_SIZE);
            auto av_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, HEAD_DIM>>(2*KV_BLOCK_SIZE + groupid*HEAD_DIM);

            uint32_t phasebit = 0;
            for (int i = 0; i < inst.num_kv_blocks; ++i) {
                int stage = i % PIPELINE_STAGES;

                // Read in QK^T
                wait(qk_finished(s, stage), get_phasebit<0>(phasebit, stage)); // wait for mm to finish
                // printf("QK finished %d - %d\n", groupid, i);
                if (i == inst.num_kv_blocks - 1) {
                    s.warp_finish_page(get_q_page(s, groupid), config::NUM_CONSUMER_WARPS / 4);
                }
                warpgroup::load_async(att_fl, qk_accumulator);
                tensor_load_wait();
                __syncwarp();
                warp::arrive(qk_unloaded(s, groupid));


                auto &test = *reinterpret_cast<st_bf<QO_BLOCK_SIZE, KV_BLOCK_SIZE> *>(s.pages[10].data);
                warpgroup::store(test, att_fl);
                for (int x = 0; x < 100; x++) {
                    if (groupid == 0 && warpgroup::laneid() == 0)
                         printf("%f ", float(test[x]));
                }
                if (groupid == 0 && warpgroup::laneid() == 0)
                    printf("\n");



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
                warp::copy(last_scaled_max_vec, scaled_max_vec); // save for next iteration

                // Prepare for AV
                auto att_page = get_a_page(s, groupid);
                auto &att = *reinterpret_cast<a_tile *>(s.pages[att_page].data);
                if (i == 0) {
                    s.wait_page_ready(att_page);
                } else {
                    int prev_stage = (i + PIPELINE_STAGES - 1) % PIPELINE_STAGES;
                    // printf("waiting for av finished %d - %d\n", groupid, prev_stage);
                    wait(av_finished(s, prev_stage), get_phasebit<0>(phasebit, prev_stage)^1); // wait for the previous mma to finish
                    // printf("av finished %d - %d\n", groupid, prev_stage);
                    warpgroup::load_async(out_fl, av_accumulator);
                    tensor_load_wait(); // TODO: is this needed?
                }
                warp::mul_row(out_fl, out_fl, diff_scaled_max_vec); // normalize previous outputs
                warpgroup::store_async(av_accumulator, out_fl);
                warpgroup::store(att, att_fl);
                tensor_store_wait();
                __syncwarp();
                warp::arrive(av_ready(s, groupid));

                // Normalize and accumulate softmax denominator
                warp::mul(norm_vec, norm_vec, diff_scaled_max_vec);
                warp::row_sum(norm_vec, att_fl, norm_vec);

                update_phasebit<0>(phasebit, stage);
            }

            // printf("Arrivedddddd\n");

            // Finish
            int last_stage = (inst.num_kv_blocks - 1) % PIPELINE_STAGES;
            wait(av_finished(s, last_stage), get_phasebit<0>(phasebit, last_stage)^1);
            warpgroup::load_async(out_fl, av_accumulator);
            s.warp_finish_page(get_a_page(s, groupid), config::NUM_CONSUMER_WARPS / 4);
            tensor_load_wait();
            warp::arrive(s.tensor_finished);
            warp::div_row(out_fl, out_fl, norm_vec);

            // printf("Arrivedddddd2\n");

            int out_page = get_o_page(s, groupid);
            auto &out = *reinterpret_cast<qo_tile *>(s.pages[out_page].data);
            s.wait_page_ready(out_page);
            warpgroup::store(out, out_fl);
            warp::arrive(o_arrived(s, groupid));

            // printf("Arrivedddddd3\n");
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();
            if (laneid < 2) {

                // printf("Arrivedddddd4\n");
                int out_page = get_o_page(s, laneid);
                auto &out = *reinterpret_cast<qo_tile *>(s.pages[out_page].data);
                wait(o_arrived(s, laneid), 0);

                // printf("Arrivedddddd5\n");
                tma::store_async(g.O, out, {inst.B, inst.H, inst.QO_idx + laneid, 0});
                tma::store_async_read_wait(); // or wait until read complete

                // printf("Arrivedddddd6\n");
                s.finish_page(out_page, config::NUM_CONSUMER_WARPS);
            }
            // TODO: write to barrier
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(ring_attention, m) {
    m.doc() = "ring attention python module";
    kittens::py::bind_kernel<kvm<config, globals,
        RingAttentionOp<config>
    >>(m, "ring_attention",
        &globals::instructions,
        &globals::barriers,
        &globals::timings,
        &globals::Q,
        &globals::K0s,
        &globals::K1s,
        &globals::V0s,
        &globals::V1s,
        &globals::O
    );
}
