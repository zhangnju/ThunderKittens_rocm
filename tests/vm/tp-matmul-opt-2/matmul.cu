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

template<typename config=config> struct CommOp {
    static constexpr int opcode = 97;

    struct parsed_instruction {
        int comm_size; // number of chunks each comm will do
        int comm_idx;
        int num_comms;
        int num_chunk_cols;
        int dev_idx;
        int prev_dev_idx;
        int next_dev_idx;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            comm_size = instruction[1];
            comm_idx = instruction[2];
            num_comms = instruction[3];
            num_chunk_cols = instruction[4];
            dev_idx = instruction[5];
            prev_dev_idx = instruction[6];
            next_dev_idx = instruction[7];
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
                for (int stage = 0; stage < globals::num_devices - 1; ++stage) {
                    // Are we ready to move on? == previous device's store done + current device's compute done + next device's load done
                    // TODO: technically, I can put the latter two conditions before the first store
                    while (stage > 0 && 
                           (*(volatile int *)&g.barriers[inst.prev_dev_idx][{stage - 1}] < inst.num_comms + 1 ||
                            *(volatile int *)&g.barriers[inst.dev_idx     ][{stage - 1}] < inst.num_comms + 1 ||
                            *(volatile int *)&g.barriers[inst.next_dev_idx][{stage - 1}] < inst.num_comms + 1))
                        __nanosleep(20);
                    for (int i = 0; i < iters; ++i) {
                        int local_index = i * config::NUM_PAGES + laneid;
                        if (local_index < inst.comm_size) {
                            int index = inst.comm_idx * inst.comm_size + local_index;
                            int row = index / inst.num_chunk_cols;
                            int col = index % inst.num_chunk_cols;
                            kittens::tma::expect(data_arrived(s, laneid), data);
                            if (stage % 2 == 0)
                                kittens::tma::load_async(data, g.A0s[inst.prev_dev_idx], {row, col}, data_arrived(s, laneid));
                            else
                                kittens::tma::load_async(data, g.A1s[inst.prev_dev_idx], {row, col}, data_arrived(s, laneid));
                            wait(data_arrived(s, laneid), phasebit);
                            phasebit = 1 - phasebit;
                            if (stage % 2 == 0)
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
                        atomicAdd_system(&g.barriers[inst.dev_idx][{stage}], 1); // mark store finished
                        if (inst.comm_idx == 0) atomicAdd_system(&g.barriers[inst.dev_idx][{stage}], 1); // temp -- mark compute complete
                    }
                }
                kittens::arrive(s.page_finished[page], config::NUM_CONSUMER_WARPS); 
            }
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) { }
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
    m.doc() = "matmul python module";
    kittens::py::bind_kernel<kvm<config, globals, 
        // MatmulOp<config>, 
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
