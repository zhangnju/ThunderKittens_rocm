#include "kittens.cuh"
#include "prototype.cuh"
#include "static_switch.h"
#include <iostream>

constexpr int NUM_CONSUMERS = (2); 
constexpr int NUM_PRODUCERS = (1);

constexpr int NCTA = 2;
constexpr int CLUSTER_DIM = 2;
constexpr int PIPE_DEPTH = 4;

using namespace kittens;

template<int Mb, int Nb, int Kb>
struct matmul_config_t {
    static constexpr int Mb_ = Mb;
    static constexpr int Nb_ = Nb;
    static constexpr int Kb_ = Kb;
};

template <typename Config>
struct matmul_globals {
    using a_tile = st_fl8_e4m3<Config::Mb_, Config::Kb_>;
    using b_tile = st_fl8_e4m3<Config::Nb_/CLUSTER_DIM, Config::Kb_>;
    using d_tile = st_hf<Config::Mb_, Config::Nb_/PIPE_DEPTH>;

    using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<half,    1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;
};

constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS;

template <typename Config>
__device__ static inline int get_iters_per_task(const matmul_globals<Config> &g) {
    return g.a.cols / Config::Kb_;
}

template<typename Config, int SUPER_M=8>
__device__ static inline int2 get_task_idx(const matmul_globals<Config> &g, int task_iter, bool is_consumer) {
    constexpr int CLUSTER_M = 4*Config::Mb_, CLUSTER_N = Config::Nb_;
    int cluster_x = clusterIdx().x, ctarank = cluster_ctarank();
    int task_id = task_iter * (gridDim.x/2) + cluster_x;
    int Rblocks = g.d.rows / CLUSTER_M, Cblocks = g.d.cols / CLUSTER_N;
    int super_rows = (Rblocks/SUPER_M)*SUPER_M,
        final_rows = Rblocks - super_rows,
        super_repeat = SUPER_M*Cblocks;
    if (task_id < super_rows * Cblocks) {
        return { 
            (SUPER_M*(task_id/super_repeat) + task_id%SUPER_M)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()),
            is_consumer ? (task_id%super_repeat)/SUPER_M : 2*((task_id%super_repeat)/SUPER_M) + ctarank
        };
    }
    else if (task_id < Rblocks*Cblocks) {
        int remainder_id = task_id - super_rows*Cblocks;
        return {
            (super_rows + remainder_id%final_rows)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()),
            is_consumer ? remainder_id/final_rows : 2*(remainder_id/final_rows) + ctarank
        };
    }
    else {
        return { -1, -1 };
    }
}

template <typename Config>
__global__ __cluster_dims__(CLUSTER_DIM) __launch_bounds__(NUM_THREADS, 1)
void matmul(const __grid_constant__ matmul_globals<Config> g) {

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int iters_per_task = get_iters_per_task(g);

    using a_tile = matmul_globals<Config>::a_tile;
    using b_tile = matmul_globals<Config>::b_tile;
    using d_tile = matmul_globals<Config>::d_tile;
    
    a_tile (&a_smem)[PIPE_DEPTH][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_DEPTH, NUM_CONSUMERS>();
    b_tile (&b_smem)[PIPE_DEPTH]                = al.allocate<b_tile, PIPE_DEPTH>();
    d_tile (&d_smem)                            = al.allocate<d_tile>();

    tma::cluster::sync();
    auto all_tmem = allocate_tmem<1, NCTA>();
    using d_tmem_t = tmem<float, Config::Mb_, Config::Nb_>;

    __shared__ kittens::semaphore inputs_arrived[PIPE_DEPTH], inputs_finished[PIPE_DEPTH], outputs_arrived, outputs_finished[NUM_CONSUMERS];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) { 
        for(int i = 0; i < PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 2); 
            init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS); 
        }
        init_semaphore(outputs_arrived, 0, 1);
        for(int i = 0; i < NUM_CONSUMERS; i++) {
            init_semaphore(outputs_finished[i], 0, 1);
        }
    }

    tma::cluster::sync();
    
    if(warpgroupid == NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();
        int ctarank = cluster_ctarank(); 
        if(warpgroup::warpid() == 3) {
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx<Config>(g, task_iter, false);
                if(rowcol.x == -1) {
                    for(int idx = 0; idx < (PIPE_DEPTH); idx++) {
                        tma::cluster::wait(inputs_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring));
                        input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                    }
                    if(laneid() == 0) arrive(outputs_arrived);
                    return;
                }
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring));
                    prototype::update_phasebit<1>(bitfield, input_ring);
                    if(task_iter>0 && idx==PIPE_DEPTH-1 && laneid() == 0) arrive(outputs_arrived); 
                    tma::cluster::expect(inputs_arrived[input_ring], 0, a_smem[0][0], a_smem[0][1], b_smem[0]);
                    tma::cluster::load_async(a_smem[input_ring][0], g.a, {(rowcol.x+0), idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    tma::cluster::load_async(a_smem[input_ring][1], g.a, {(rowcol.x+1), idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    tma::cluster::load_async(b_smem[input_ring],    g.b, { rowcol.y,    idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                }
            }
        }
        else if(ctarank == 0 && (warpgroup::warpid() == 0 || warpgroup::warpid() == 1)) { // launch the MMA's
            d_tmem_t d_tmem = all_tmem.subtile<d_tmem_t>(0, warpgroup::warpid()*Config::Nb_);
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx<Config>(g, task_iter, false);
                if(rowcol.x == -1) return;
                tma::cluster::wait(outputs_finished[warpgroup::warpid()], (task_iter+1)%2); // make sure tensor memory is ready to be written to.
                tma::cluster::wait(inputs_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring));
                prototype::update_phasebit<0>(bitfield, input_ring);
                mm2_ABt(d_tmem, a_smem[0][warpgroup::warpid()], b_smem[0], inputs_finished[0]);
                input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                for(int idx = 1; idx < iters_per_task; idx++) {
                    tma::cluster::wait(inputs_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring));
                    prototype::update_phasebit<0>(bitfield, input_ring);
                    mma2_ABt(d_tmem, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                }
            }
        }
    }
    else {
        warpgroup::increase_registers<224>();
        d_tmem_t d_tmem = all_tmem.subtile<d_tmem_t>(0, warpgroupid*Config::Nb_);
        for(int task_iter = 0; true; task_iter++) {
            int2 rowcol = get_task_idx<Config>(g, task_iter, true);
            if(rowcol.x == -1) return;
            kittens::wait(outputs_arrived, task_iter%2);
            rt_hf<Config::Mb_/4, d_tile::cols> d_reg[4];
            if(warpgroupid == 1) group<8>::sync(15);
            #pragma unroll
            for(int i = 0; i < Config::Nb_/d_tile::cols; i++) {
                warpgroup::load_async(d_reg[i], d_tmem.template subtile<tmem<float, Config::Mb_, Config::Nb_/PIPE_DEPTH>>(0, Config::Nb_/PIPE_DEPTH*i));
            }
            tm_load_wait();
            warpgroup::sync(warpgroupid);
            if(warpgroup::laneid() == 0) arrive(outputs_finished[warpgroupid]); // Tensor memory for warpgroup 0 is now free.
            if(warpgroupid == 0) group<8>::sync(15);
            if(warpgroupid == 1) group<8>::sync(14);
            warpgroup::store(d_smem, d_reg[0]);
            warpgroup::sync(warpgroupid);
            if(warpgroup::warpid() == 0) tma::store_async(g.d, d_smem, {rowcol.x, 4*rowcol.y+0});
            #pragma unroll
            for(int i = 1; i < Config::Nb_/d_tile::cols; i++) {
                tma::store_async_read_wait();
                warpgroup::sync(warpgroupid);
                warpgroup::store(d_smem, d_reg[i]);
                warpgroup::sync(warpgroupid);
                if(warpgroup::warpid() == 0) tma::store_async(g.d, d_smem, {rowcol.x, 4*rowcol.y+i});
            }
            tma::store_async_read_wait();
            if(warpgroupid == 0) group<8>::sync(14);
            group<8>::sync(15); // All consumers sync here.
        }
    }
}


constexpr bool NCU = false;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[j * N + k];
            }
            c[i * N + j] = sum;
        }
    }
}

template<int M, int N, int K>
struct runtime_config {
    static constexpr int Mb_ = M;
    static constexpr int Nb_ = N;
    static constexpr int Kb_ = K;
};

template <typename Config>
void dispatch_matmul(fp8e4m3 *d_A, fp8e4m3 *d_B, half *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using globals = matmul_globals<Config>;
    typename globals::a_gl Ag{d_A, nullptr, nullptr, M, K};
    typename globals::b_gl Bg{d_B, nullptr, nullptr, N, K};
    typename globals::d_gl Dg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Dg};
    matmul<<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

template <typename Config>
int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << Config::Mb_ << "x" << Config::Nb_<< "\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    fp8e4m3 *d_A, *d_B;
    half *d_C;
    cudaMalloc(&d_A, M*K*sizeof(fp8e4m3));
    cudaMalloc(&d_B, K*N*sizeof(fp8e4m3));
    cudaMalloc(&d_C, M*N*sizeof(half));

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    fp8e4m3 *h_A_fp8 = new fp8e4m3[M * K];
    fp8e4m3 *h_B_fp8 = new fp8e4m3[K * N];
    for (int i = 0; i < M * K; ++i) h_A_fp8[i] = fp8e4m3(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_fp8[i] = fp8e4m3(h_B[i]);
    for (int i = 0; i < M * K; ++i) h_A[i] = float(h_A_fp8[i]);
    for (int i = 0; i < K * N; ++i) h_B[i] = float(h_B_fp8[i]);

    cudaMemcpy(d_A, h_A_fp8, M*K*sizeof(fp8e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_fp8, K*N*sizeof(fp8e4m3), cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(matmul<Config>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dim3 grid(148, 1);
    dim3 block(NUM_THREADS);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < (NCU ? 1 : 1); i++) { // warmup
        dispatch_matmul<Config>(d_A, d_B, d_C, M, N, K, grid, block);
    }

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = (NCU ? 1 : 5);
    for(int i = 0; i < ITERS; i++) {
        dispatch_matmul<Config>(d_A, d_B, d_C, M, N, K, grid, block);
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;

    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    half *h_C_fp16 = new half[M * N];
    cudaMemcpy(h_C_fp16, d_C, M*N*sizeof(half), cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_C[i] = __half2float(h_C_fp16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if(error > 1.0) { // large because of bf16 vs fp32 numerics
            if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_fp8;
    delete[] h_B_fp8;
    delete[] h_C_fp16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int run_matmul_with_config(size_t M, size_t N, size_t K, int tile_m, int tile_n, int tile_k) {
    int result = -1;
    // TILEM_SWITCH(tile_m, [&]() {
        TILEN_SWITCH(tile_n, [&]() {
            TILEK_SWITCH(tile_k, [&]() {
                result = run_benchmark<runtime_config<128, kTileN, kTileK>>(M, N, K);
            });
        });
    // });
    return result;
}

int main() {
    std::vector<std::tuple<int,int,int>> shapes = {
        // {8192, 8192, 8192},
        {16384, 16384, 16384},
        // {128, 14336, 8192},
        // {256, 14336, 8192},
        // {512, 14336, 8192},
        // {1024, 14336, 8192},
        // {4096, 14336, 8192},
        {8192, 14336, 8192}
    };


    std::vector<std::tuple<int, int, int>> configs = {  
        // {64, 64, 64},
        // {64, 64, 128},
        // {64, 128, 64},
        // {64, 128, 128},
        // {64, 256, 64},
        // {64, 256, 128},
        {128, 64, 64},
        {128, 64, 128},
        {128, 128, 64},
        {128, 128, 128},
        {128, 256, 64},
        {128, 256, 128},
    };

    for (const auto& [tile_m, tile_n, tile_k] : configs) {
        std::cout << "\nTesting configuration: TileM=" << tile_m << " TileN=" << tile_n << " TileK=" << tile_k << std::endl;
        for(const auto& [M, N, K] : shapes) {
            std::cout << "Running with shape: M=" << M << " N=" << N << " K=" << K << std::endl;
            run_matmul_with_config(M, N, K, tile_m, tile_n, tile_k);
        }
    }

    return 0;
}
