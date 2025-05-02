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
    static constexpr int size_per_iter = config::NUM_PAGES * config::PAGE_SIZE; // 212,992

    struct parsed_instruction {
        int comm_size; // size per stage per SM
        int comm_idx;
        int num_comms;
        int dev_idx;
        int prev_dev_idx;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            comm_size = instruction[1];
            comm_idx = instruction[2];
            num_comms = instruction[3];
            dev_idx = instruction[4];
            prev_dev_idx = instruction[5];
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
            if (laneid < config::NUM_PAGES) {
                int page = s.pid(laneid);
                s.wait_page_ready(page);
                auto &data = reinterpret_cast<st_fp8e4m3<128, 128> &>(s.pages[page]);
                int phasebit = 0;
                int iters = inst.comm_size / size_per_iter;
                for (int stage = 0; stage < globals::num_devices - 1; ++stage) {
                    // load complete (NUM_PAGES * num_comms) + read from compute complete (1) => must wait for NUM_PAGES * num_comms + 1
                    // TODO: technically, we can separate the waits on the above two, and wait for latter in storing stage
                    while (stage > 0 && *(volatile int *)&g.barriers[inst.prev_dev_idx][{stage - 1}] < config::NUM_PAGES * inst.num_comms + 1)
                        __nanosleep(20);

                    // Sanity check: print barrier value:
                    // printf("Barrier value: %d\n", *(volatile int *)&g.barriers[inst.prev_dev_idx][{stage - 1}]);
                    for (int i = 0; i < iters; ++i) {
                        wait(data_finished(s, laneid), phasebit);
                        kittens::tma::expect(data_arrived(s, laneid), data);
                        if (stage % 2 == 0)
                            kittens::tma::load_async(data, g.A0s[inst.prev_dev_idx], 
                                coord<>{inst.comm_idx * inst.comm_size + i * size_per_iter + laneid * config::PAGE_SIZE}, data_arrived(s, laneid));
                        else
                            kittens::tma::load_async(data, g.A1s[inst.prev_dev_idx], 
                                coord<>{inst.comm_idx * inst.comm_size + i * size_per_iter + laneid * config::PAGE_SIZE}, data_arrived(s, laneid));
                        phasebit = (phasebit + 1) % 2;
                    }
                }
            }
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) {}
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {}
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();
            if (laneid < config::NUM_PAGES) {
                int page = s.pid(laneid);
                auto &data = reinterpret_cast<st_fp8e4m3<128, 128> &>(s.pages[page]);
                int phasebit = 0;
                int iters = inst.comm_size / size_per_iter;
                for (int stage = 0; stage < globals::num_devices - 1; ++stage) {
                    for (int i = 0; i < iters; ++i) {
                        wait(data_arrived(s, laneid), phasebit);
                        if (stage % 2 == 0)
                            kittens::tma::store_async(g.A1s[inst.dev_idx], data, 
                                coord<>{inst.comm_idx * inst.comm_size + i * size_per_iter + laneid * config::PAGE_SIZE});
                        else
                            kittens::tma::store_async(g.A0s[inst.dev_idx], data,
                                coord<>{inst.comm_idx * inst.comm_size + i * size_per_iter + laneid * config::PAGE_SIZE});
                        phasebit = (phasebit + 1) % 2;
                        tma::store_async_wait();
                        arrive(data_finished(s, laneid));
                    }
                    // printf("Adding to barrier from dev idx %d, stage %d, lane id %d\n", inst.dev_idx, stage, laneid);
                    atomicAdd(&g.barriers[inst.dev_idx][{stage}], 1); // equals config::NUM_PAGES * num_comms after completion
                    if (laneid == 0 && inst.comm_idx == 0) atomicAdd(&g.barriers[inst.dev_idx][{stage}], 1); // temp
                }
                kittens::arrive(s.page_finished[page], config::NUM_CONSUMER_WARPS);
            }
        }
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
