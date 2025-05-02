#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

constexpr int SM_COUNT = 148;

using config = default_config;
struct globals {
    constexpr static int num_devices = 8;
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using barrier_layout = gl<uint, 1, 1, 1, num_devices>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using fp8_matrix = gl<fp8e4m3, 1, 1, -1, -1, st_fp8e4m3<128, 128>>;
    instruction_layout instructions;
    gl_array<barrier_layout, num_devices> barriers;
    timing_layout timings;
    gl_array<fp8_matrix, num_devices> A0s; // A is shared across devices.
    gl_array<fp8_matrix, num_devices> A1s;
    fp8_matrix B;
    fp8_matrix C;
    dim3 grid() { return dim3(SM_COUNT); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct MatmulOp {
    static constexpr int opcode = 725;
    static constexpr int PIPELINE_STAGES = 3;

    struct parsed_instruction {
        int row;
        int col;
        int iters;
        int ring_stage;
        int num_comms;
        int num_comps;
        int dev_idx;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            row = instruction[1];
            col = instruction[2];
            iters = instruction[3];
            ring_stage = instruction[4];
            num_comms = instruction[5];
            num_comps = instruction[6];
            dev_idx = instruction[7];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &inputs_arrived(state<config> &s, int id) {
        return s.semaphores()[id];
    }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int id) {
        return s.semaphores()[id+PIPELINE_STAGES];
    }
    __device__ static inline semaphore &outputs_arrived(state<config> &s, int id) {
        return s.semaphores()[id+PIPELINE_STAGES*2];
    }
    __device__ static inline semaphore &outputs_shared(state<config> &s, int id) {
        return s.semaphores()[id+PIPELINE_STAGES*2+4];
    }
    __device__ static inline int get_a_page(state<config> &s, int stage, int offset) {
        return s.pid(stage*4 + offset);
    }
    __device__ static inline int get_b_page(state<config> &s, int stage, int offset) {
        return s.pid(stage*4 + offset + 2);
    }
    __device__ static inline int get_store_page(state<config> &s, parsed_instruction &inst, int offset) {
        return s.pid(((inst.iters+2)%PIPELINE_STAGES)*4 + offset);
    }
    
    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            parsed_instruction inst{instruction};
            if(query == 0) return 12;
            else return ((query-1)+(inst.iters%PIPELINE_STAGES)*4)%12;
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int i = 0; i < PIPELINE_STAGES; i++) {
                init_semaphore(inputs_arrived(s, i), 1);  // Inputs arrived.
                init_semaphore(inputs_finished(s, i), 4); // Inputs finished.
            }
            for(int i = 0; i < 4; i++) {
                init_semaphore(outputs_arrived(s, i), 1); // outputs arrived.
                init_semaphore(outputs_shared(s, i), 1);  // outputs shared.
            }
            return 2*PIPELINE_STAGES + 8;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            warp::arrive(s.page_finished[s.pid(12)], config::NUM_CONSUMER_WARPS); // Release the unused page immediately.

            parsed_instruction inst{s};
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

            while (inst.ring_stage > 0 && *(volatile int *)&g.barriers[inst.dev_idx][{inst.ring_stage - 1}] < inst.num_comms + inst.num_comps)
                __nanosleep(20);

            int pipeline_stage = 0;
            for(int i = 0; i < inst.iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), 128*128*4);
                if(laneid() < 2) {
                    int a_page = get_a_page(s, pipeline_stage, laneid());
                    if(i < PIPELINE_STAGES) {
                        s.wait_page_ready(a_page);
                    }
                    st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
                    if (inst.ring_stage % 2 == 0)
                        tma::load_async(a, g.A0s[inst.dev_idx], {inst.row+laneid(), i}, inputs_arrived(s, pipeline_stage));
                    else
                        tma::load_async(a, g.A1s[inst.dev_idx], {inst.row+laneid(), i}, inputs_arrived(s, pipeline_stage));
                }
                if(laneid() < 2) {
                    int b_page = get_b_page(s, pipeline_stage, laneid());
                    if(i < PIPELINE_STAGES) {
                        s.wait_page_ready(b_page);
                    }
                    st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
                    tma::load_async(b, g.B, {inst.col+laneid(), i}, inputs_arrived(s, pipeline_stage));
                }
                update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            }
            warp::sync();

            if(laneid() >= 28) {
                for(int i = 0; i < PIPELINE_STAGES-1; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                    wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                    int release_lid = pipeline_stage*4 + laneid() - 28;
                    int release_pid = s.pid(release_lid);
                    arrive(s.page_finished[release_pid], config::NUM_CONSUMER_WARPS);
                }
            }
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            int pipeline_stage = 0;
            wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
            s.wait_tensor_ready();

            if(laneid() < 4) {
                auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(laneid()*128);
                st_fp8e4m3<128, 128> &a = s.pages[get_a_page(s, pipeline_stage, laneid()/2)].template as_st<fp8e4m3>();
                st_fp8e4m3<128, 128> &b = s.pages[get_b_page(s, pipeline_stage, laneid()%2)].template as_st<fp8e4m3>();
                mm<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
            }
            update_phasebit<0>(semaphore_bitfield, pipeline_stage);

            pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage);
            for(int i = 1; i < inst.iters-1; i++, update_phasebit<0>(semaphore_bitfield, pipeline_stage), pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                if(laneid() < 4) {
                    auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(laneid()*128);
                    st_fp8e4m3<128, 128> &a = s.pages[get_a_page(s, pipeline_stage, laneid()/2)].template as_st<fp8e4m3>();
                    st_fp8e4m3<128, 128> &b = s.pages[get_b_page(s, pipeline_stage, laneid()%2)].template as_st<fp8e4m3>();
                    mma<transpose::N, transpose::T>(accumulator, a, b, inputs_finished(s, pipeline_stage));
                }
            }

            wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
            if (laneid() == 0) atomicAdd_system(&g.barriers[inst.dev_idx][{inst.ring_stage}], 1); // Inputs all finished, increase the barrier for next ring comm
            if (laneid() < 4) {
                auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(laneid()*128);
                st_fp8e4m3<128, 128> &a = s.pages[get_a_page(s, pipeline_stage, laneid()/2)].template as_st<fp8e4m3>();
                st_fp8e4m3<128, 128> &b = s.pages[get_b_page(s, pipeline_stage, laneid()%2)].template as_st<fp8e4m3>();
                mma<transpose::N, transpose::T>(accumulator, a, b, outputs_arrived(s, laneid()));
            }
            warp::sync();
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int groupid = warpgroup::groupid();
            wait(outputs_arrived(s, groupid), 0);

            st_fp8e4m3<128, 128> &store_buffer = s.pages[get_store_page(s, inst, groupid)].template as_st<fp8e4m3>();
            auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(groupid*128);
            rt_fl<32, 128> acc_rt;
            rt_fp8e4m3<32, 128> acc_fp8;
            warpgroup::load_async(acc_rt, accumulator);
            warp::copy(acc_fp8, acc_rt);
            tensor_load_wait();
            warp::arrive(s.tensor_finished);
            warpgroup::store(store_buffer, acc_fp8); 
            warpgroup::sync(groupid);
            warpgroup::arrive(outputs_shared(s, groupid));
        }
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            if(laneid() < 4) {
                wait(outputs_shared(s, laneid()), 0);

                int store_page = get_store_page(s, inst, laneid());
                st_fp8e4m3<128, 128> &output = s.pages[store_page].template as_st<fp8e4m3>();
                tma::store_async(g.C, output, {inst.row+laneid()/2, inst.col+laneid()%2});
                tma::store_async_read_wait();
                arrive(s.page_finished[store_page], config::NUM_CONSUMER_WARPS);
            }
            warp::sync();
        }
    };
};

template<typename config=config> struct CommOp {
    static constexpr int opcode = 97;

    struct parsed_instruction {
        int comm_size; // number of chunks each comm will do
        int comm_idx;
        int num_comms;
        int num_comps;
        int num_chunk_cols;
        int dev_idx;
        int prev_dev_idx;
        int next_dev_idx;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            comm_size = instruction[1];
            comm_idx = instruction[2];
            num_comms = instruction[3];
            num_comps = instruction[4];
            num_chunk_cols = instruction[5];
            dev_idx = instruction[6];
            prev_dev_idx = instruction[7];
            next_dev_idx = instruction[8];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &data_arrived(state<config> &s, int idx) {
        return s.semaphores()[idx];
    }
    __device__ static inline semaphore &data_finished(state<config> &s, int idx) {
        return s.semaphores()[idx + config::NUM_PAGES];
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int release_order[config::NUM_PAGES] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            return release_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int i = 0; i < config::NUM_PAGES; i++) {
                init_semaphore(data_arrived(s, i), 1);
                init_semaphore(data_finished(s, i), 1);
                arrive(data_finished(s, i)); // arrive first
            }
            return 2 * config::NUM_PAGES;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();
            constexpr uint32_t membermask = 0xFFFFFFFF >> (32 - config::NUM_PAGES);
            if (laneid < config::NUM_PAGES) {
                int page = s.pid(laneid);
                s.wait_page_ready(page);
                auto &data = reinterpret_cast<st_fp8e4m3<128, 128> &>(s.pages[page]); // use sv_bf as a placeholder for full page
                int phasebit = 0;
                int iters = (inst.comm_size + config::NUM_PAGES - 1) / config::NUM_PAGES;
                for (int ring_stage = 0; ring_stage < globals::num_devices - 1; ++ring_stage) {
                    // Are we ready to move on? == previous device's store done + current device's compute done + next device's load done
                    // TODO: technically, I can put the latter two conditions before the first store
                    while (ring_stage > 0 && 
                           (*(volatile int *)&g.barriers[inst.prev_dev_idx][{ring_stage - 1}] < inst.num_comms + inst.num_comps ||
                            *(volatile int *)&g.barriers[inst.dev_idx     ][{ring_stage - 1}] < inst.num_comms + inst.num_comps ||
                            *(volatile int *)&g.barriers[inst.next_dev_idx][{ring_stage - 1}] < inst.num_comms + inst.num_comps))
                        __nanosleep(20);
                    for (int i = 0; i < iters; ++i) {
                        int local_index = i * config::NUM_PAGES + laneid;
                        if (local_index < inst.comm_size) {
                            int index = inst.comm_idx * inst.comm_size + local_index;
                            int row = index / inst.num_chunk_cols;
                            int col = index % inst.num_chunk_cols;
                            kittens::tma::expect(data_arrived(s, laneid), data);
                            if (ring_stage % 2 == 0)
                                kittens::tma::load_async(data, g.A0s[inst.prev_dev_idx], {row, col}, data_arrived(s, laneid));
                            else
                                kittens::tma::load_async(data, g.A1s[inst.prev_dev_idx], {row, col}, data_arrived(s, laneid));
                            wait(data_arrived(s, laneid), phasebit);
                            phasebit = 1 - phasebit;
                            if (ring_stage % 2 == 0)
                                kittens::tma::store_async(g.A1s[inst.dev_idx], data, {row, col});
                            else
                                kittens::tma::store_async(g.A0s[inst.dev_idx], data, {row, col});
                        }
                        asm volatile("{bar.warp.sync %0;}" ::"n"(membermask));
                        if (laneid == 0) asm volatile("{cp.async.bulk.wait_group 0;}");
                        asm volatile("{bar.warp.sync %0;}" ::"n"(membermask));
                    }
                    if (laneid == 0) {
                        asm volatile("{fence.acq_rel.sys;}");
                        atomicAdd_system(&g.barriers[inst.dev_idx][{ring_stage}], 1); // mark store finished
                    }
                }
                kittens::arrive(s.page_finished[page], config::NUM_CONSUMER_WARPS); 
            }
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) { 
            s.wait_tensor_ready();
            if (laneid() == 0)
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) { }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) { }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(matmul, m) {
    m.doc() = "tp async matmul python module";
    kittens::py::bind_kernel<kvm<config, globals, 
        MatmulOp<config>, 
        CommOp<config>
    >>(m, "matmul",
        &globals::instructions,
        &globals::barriers,
        &globals::timings,
        &globals::A0s,
        &globals::A1s,
        &globals::B,
        &globals::C
    );
}
