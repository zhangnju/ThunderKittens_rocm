#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WORKERS (1) // use 1 warp for this example
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)  // number of threads is num_workers * num_threads_per_worker (32)

// MODEL TODO: SET DIMENSIONS OF INPUTS
#define _row 16 // rows per input matrix x
#define _col 16 // columns per input matrix x

/* 
MODEL TODO: DEFINE GLOBAL MEMORY DESCRIPTORS
gl: indicates global layout
bf16: indicates the data type
dimmensions: {batch, head, depth, width} for global tensors (-1 means runtime dimension, non-negative means compile-time dimension)
st: when loading from global tensor at some {b, h, d, w} index, this is the shape of the tile that will be loaded to shared memory
*/
using x_gl  = gl<bf16, -1, -1, -1, -1,  st<bf16, _row, _col>>;  // input is bfloat16
using o_gl  = gl<float, -1, -1, -1, -1, st<float, _row, _row>>; // output is float32

/*
CREATE GLOBAL MEMORY DESCRIPTORS
CREATE GRID AND BLOCK LAUNCH DIMENSIONS
*/
struct micro_globals {
    x_gl x, y;
    o_gl o;
    dim3 grid()  { return dim3(o.depth, o.batch); } // dimensions we parallelize over (e.g., batches, heads)
    dim3 block() { return dim3(NUM_THREADS); } // number of threads per threadblock
    size_t dynamic_shared_memory() { return 224000; } // I added this but kinda sus
    
};

/*
ACTUAL CUDA KERNEL
1. Define shared memory allocator and tiles
2. Define register memory
3. Load from global to shared memory using {b, h, d, w} indexing
4. Load from shared to register memory
5. Do the work on tiles
6. Store from register to shared memory
7. Store from shared to global memory using {b, h, d, w} indexing
*/
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {
    // get current position in grid
    int head = blockIdx.x;
    int batch = blockIdx.y;

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<_row, _col> (&x_s) = al.allocate<st_bf<_row, _col>>(); // bf16 tiles
    st_bf<_row, _col> (&y_s) = al.allocate<st_bf<_row, _col>>(); // bf16 tiles
    st_fl<_row, _row> (&o_s) = al.allocate<st_fl<_row, _row>>(); // float tiles

    // register memory
    rt_bf<_row, _col> x_reg; // bf16 register
    rt_bf<_row, _col> y_reg; // bf16 register
    rt_fl <_row, _row> accum_tile;  
    zero(accum_tile);

    // load from HBM to shared
    load(x_s, g.x, {batch, head, 0, 0});
    load(y_s, g.y, {batch, head, 0, 0});
    __syncthreads();

    // load from shared to register
    load(x_reg, x_s);
    load(y_reg, y_s);
    __syncthreads();

    // now do the matmul and accumulate to accum_tile
    mma_ABt(accum_tile, x_reg, y_reg, accum_tile); // o = torch.matmul(x, x.transpose(1, 2))
    __syncthreads();

    // store from register to shared
    store(o_s, accum_tile);
    __syncthreads();

    // store from shared to HBM
    store(g.o, o_s, {batch, head, 0, 0});
    __syncthreads();
}

/*
DISPATCH FUNCTION 
*/
void dispatch_micro(micro_globals g) {
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    cudaDeviceSynchronize();
}

/* 
PYTHON BINDINGS
*/
PYBIND11_MODULE(simple_tk, m) {
    m.doc() = "simple_tk python module";
    BIND_KERNEL(m, "micro_tk", micro_tk, micro_globals, x, y, o); // For wrapping kernels directly.
    BIND_FUNCTION(m, "dispatch_micro", dispatch_micro, micro_globals, x, y, o); // For host functions that wrap the kernel.
}
