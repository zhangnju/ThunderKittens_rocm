/**
 * @file
 * @brief An aggregate header of all group (multi-warp) operations defined by ThunderKittens
 */

#pragma once

#include <cuda/pipeline>

#include "../../common/common.cuh"
#include "../../types/types.cuh"
#include "../warp/warp.cuh" // several group memory ops rely on underlying warp-scope ops

// A "warpgroup" is a special group of 4 consecutive warps defined by NVIDIA for certain SM_90+ operations.
#define KITTENS_CHECK_WARPGROUP static_assert(N_WARPS==4, "PTX warpgroup (N_WARPS=4) function called from a non-warpgroup group.");

// WGMMA relies on some template structures that cannot be specialized within the group struct, so we declare them in advance.
#ifdef KITTENS_HOPPER
#include "wgmma/base/base.cuh"
#endif

namespace kittens {
/*
This is meant to be used with a `using group_N = kittens::group<NUM_WORKERS>;` at the start of every kernel.
*/
template<int N_WARPS>
struct group {
static constexpr int GROUP_WARPS = N_WARPS; // This alias produces nice parallelism.
static constexpr int GROUP_THREADS = N_WARPS * kittens::WARP_THREADS; // This alias produces nice parallelism.
__device__ static inline int laneid() { return threadIdx.x % GROUP_THREADS; }
__device__ static inline int warpid() { return laneid() / kittens::WARP_THREADS; }
__device__ static inline int groupid() { return threadIdx.x / GROUP_THREADS; }

__device__ static inline void sync(int id) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}
__device__ static inline void arrive(int id) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}

#include "memory/memory.cuh"
#include "shared/shared.cuh"
#include "register/register.cuh"

#ifdef KITTENS_HOPPER
#include "wgmma/wgmma.cuh"

template<int n_reg> __device__ static inline void increase_registers() {
    static_assert(N_WARPS % 4 == 0, "N_WARPS must be a multiple of 4");
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(n_reg));
}
template<int n_reg> __device__ static inline void decrease_registers() {
    static_assert(N_WARPS % 4 == 0, "N_WARPS must be a multiple of 4");
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(n_reg));
}
__device__ static inline void producer_registers() { decrease_registers<24>(); }
template<int NCWG> __device__ static inline void consumer_registers() { increase_registers<480/NCWG - 8*(NCWG>3) - 224*(NCWG==1)>(); }

#endif

};

using warpgroup = group<4>; // special scope commonly used by SM_90 and later.

namespace grid {
static __device__ uint32_t __grid_barrier__[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// Note that unlike cooperative groups, this does NOT mask the threads, nor do intra-block syncs.
// To achieve the full effect, place barriers on either side, and mask to thread 0 before calling this.
static __device__ inline void sync(int grid_barrier=0) {
    if(laneid() == 0) {
        uint32_t arrival = 1, initial_value, new_value;
        if(blockIdx.x + blockIdx.y + blockIdx.z == 0) {
            arrival = 0x80000000 - (gridDim.x * gridDim.y * gridDim.z - 1);
        }
        asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;" : "=r"(initial_value) : "l"((uint64_t)(&__grid_barrier__[0] + grid_barrier)), "r"(arrival) : "memory");
        do {
            asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(new_value) : "l"((uint64_t)(&__grid_barrier__[0] + grid_barrier)) : "memory");
        } while (((initial_value^new_value) & 0x80000000) == 0);
    }
}

} // namespace grid
} // namespace kittens