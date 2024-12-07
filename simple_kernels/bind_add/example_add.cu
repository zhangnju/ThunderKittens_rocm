#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define _row 16
#define _col 32

using _gl  = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;

struct micro_globals {
    _gl x, o; // define layout her
    // check with simran, added this
    dim3 grid()  { return dim3(x.batch, x.depth, x.rows); } // number of thread blocks we are launching
    dim3 block() { return dim3(x.cols); } // number of threads in a thread block
    size_t dynamic_shared_memory() { return 224000; } // I added this but kinda sus
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fl<_row, _col> (&x_s) = al.allocate<st_fl<_row, _col>>();
    st_fl<_row, _col> (&o_s) = al.allocate<st_fl<_row, _col>>();

    // register memory 
    rt_fl<_row, _col> x_reg_fl;

    // load from HBM to shared
    load(x_s, g.x, {0, 0, 0, 0});
    __syncthreads();

    // load from shared to register
    load(x_reg_fl, x_s);
    __syncthreads();

    // x (dst) = x (src b) + x (src a)
    add(x_reg_fl, x_reg_fl, x_reg_fl);
    __syncthreads();

    // store from register to shared
    store(o_s, x_reg_fl);
    __syncthreads();

    // store from shared to HBM
    store(g.o, o_s, {0, 0, 0, 0});
    __syncthreads();
}


// old code
// void dispatch_micro( float *d_x, float *d_o ) {
//     using _gl = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
//     using globals = micro_globals;
//     _gl  x_arg{d_x, 1, 1, _row, _col};
//     _gl  o_arg{d_o, 1, 1, _row, _col};
//     globals g{x_arg, o_arg};
//     unsigned long mem_size = 50480; 
//     cudaFuncSetAttribute(
//         micro_tk,
//         cudaFuncAttributeMaxDynamicSharedMemorySize,
//         mem_size
//     );
//     micro_tk<<<1,32,mem_size>>>(g);
//     cudaDeviceSynchronize();
// }


void dispatch_micro(micro_globals g) {
    // MISSING??
    // I THINK WE NEED TO REDO THE LAYOUT!
    // somehow do the step? _gl  x_arg{d_x, 1, 1, _row, _col};
    // i tried to put another TK layout here but it didn't work

    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    // <<< # number thread block launching == grid, # number of threads in a thread block == block, # dynamic shared memory size>>>
    // micro_tk<<<1,32,mem_size>>>(g);/
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    // micro_tk<<<1,32, mem_size>>>(g);
    cudaDeviceSynchronize();
}


PYBIND11_MODULE(simple_tk, m) {
    m.doc() = "simple_tk python module";
    BIND_KERNEL(m, "micro_tk", micro_tk, micro_globals, x, o); // For wrapping kernels directly.
    BIND_FUNCTION(m, "dispatch_micro", dispatch_micro, micro_globals, x, o); // For host functions that wrap the kernel.
}
