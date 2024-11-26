#include "kittens.cuh"

using namespace kittens;
#define NUM_THREADS (kittens::WARP_THREADS)

#define _row 16
#define _col 32

struct micro_globals {
    using x_gl  = gl<bf16, -1, -1, -1, -1, st_bf<_row, _col>>;
    using o_gl  = gl<float, -1, -1, -1, -1, st_fl<_row, _row>>;
    x_gl x;
    o_gl o;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<_row, _col> (&x_s) = al.allocate<st_bf<_row, _col>>(); // bf16 tiles
    st_fl<_row, _row> (&o_s) = al.allocate<st_fl<_row, _row>>(); // float tiles

    // register memory
    rt_bf<_row, _col> x_reg; // bf16 register
    rt_fl <_row, _row> accum_tile;  
    zero(accum_tile);

    // load from HBM to shared
    load(x_s, g.x, {0, 0, 0, 0});
    __syncthreads();

    // load from shared to register
    load(x_reg, x_s);
    __syncthreads();

    // now do the matmul and accumulate to accum_tile
    mma_ABt(accum_tile, x_reg, x_reg, accum_tile); // o = torch.matmul(x, x.transpose(1, 2))
    __syncthreads();

    // store from register to shared
    store(o_s, accum_tile);
    __syncthreads();

    // store from shared to HBM
    store(g.o, o_s, {0, 0, 0, 0});
    __syncthreads();
}

void dispatch_micro( bf16 *d_x, float *d_o ) {
    using x_gl = gl<bf16, -1, -1, -1, -1, st_bf<_row, _col>>;
    using o_gl = gl<float, -1, -1, -1, -1, st_fl<_row, _row>>;
    using globals = micro_globals;
    x_gl  x_arg{d_x, 1, 1, _row, _col};
    o_gl  o_arg{d_o, 1, 1, _row, _row};
    globals g{x_arg, o_arg};
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<1,32,mem_size>>>(g);
    cudaDeviceSynchronize();
}
#include "harness.impl"

