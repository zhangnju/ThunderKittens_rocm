#include "cuda_runtime.h"
#include <stdio.h>

__global__ void accumulate_and_transform(float* __restrict__ input_array, 
                                       float* __restrict__ output_array,
                                       int32_t array_size) {
    // partial sum
    float partial_sum = 0.0f;
    
    // calculate global thread index and stride
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < array_size; 
         idx += blockDim.x * gridDim.x) {
        partial_sum += output_array[idx];
    }

    // ensure all threads complete accumulation
    cudaTriggerProgrammaticLaunchCompletion();

    // output array with transformed values
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < array_size; 
         idx += blockDim.x * gridDim.x) {
        output_array[idx] = input_array[idx] + partial_sum;
    }
}

__global__ void transform_with_offset(const float* __restrict__ source_array,
                                    float* __restrict__ dest_array,
                                    const float* __restrict__ offset_array,
                                    int32_t array_size) {
    float offset_sum = 0.0f;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < array_size; 
         idx += blockDim.x * gridDim.x) {
        offset_sum += offset_array[idx];
    }

    // sync grid to ensure offset calculation is complete
    cudaGridDependencySynchronize();

    // apply transformation with offset and constant
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < array_size; 
         idx += blockDim.x * gridDim.x) {
        dest_array[idx] = source_array[idx] + offset_sum + 2.0f;
    }
}

void run_benchmark(int array_size, int num_blocks, int threads_per_block, bool use_pdl) {
    float *d_input_array, *d_intermediate_array, *d_output_array, *d_offset_array;
    
    // Allocate device memory
    cudaMalloc(&d_input_array, array_size * sizeof(float));
    cudaMalloc(&d_intermediate_array, array_size * sizeof(float));
    cudaMalloc(&d_output_array, array_size * sizeof(float));
    cudaMalloc(&d_offset_array, array_size * sizeof(float));

    // Initialize with some data
    float* h_input = new float[array_size];
    float* h_offset = new float[array_size];
    for(int i = 0; i < array_size; i++) {
        h_input[i] = 1.0f;
        h_offset[i] = 0.5f;
    }
    cudaMemcpy(d_input_array, h_input, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset_array, h_offset, array_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t computation_stream;
    cudaStreamCreate(&computation_stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    accumulate_and_transform<<<num_blocks, threads_per_block, 0, computation_stream>>>
        (d_input_array, d_intermediate_array, array_size);
    transform_with_offset<<<num_blocks, threads_per_block, 0, computation_stream>>>
        (d_intermediate_array, d_output_array, d_offset_array, array_size);
    cudaDeviceSynchronize();

    const int num_iterations = 100; // Increased iterations for better timing
    float total_time = 0.0f;

    cudaEventRecord(start, computation_stream);
    
    if (use_pdl) {
        cudaLaunchConfig_t pdl_config = {0};
        pdl_config.gridDim = num_blocks;
        pdl_config.blockDim = threads_per_block;
        pdl_config.dynamicSmemBytes = 0;
        pdl_config.stream = computation_stream;

        cudaLaunchAttribute pdl_attributes[1];
        pdl_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        pdl_attributes[0].val.programmaticStreamSerializationAllowed = 1;
        pdl_config.attrs = pdl_attributes;
        pdl_config.numAttrs = 1;

        for (int i = 0; i < num_iterations; i++) {
            accumulate_and_transform<<<num_blocks, threads_per_block, 0, computation_stream>>>
                (d_input_array, d_intermediate_array, array_size);
            transform_with_offset<<<num_blocks, threads_per_block, 0, computation_stream>>>
                (d_intermediate_array, d_output_array, d_offset_array, array_size);
        }
    } else {
        // Run without PDL - forces complete kernel serialization
        for (int i = 0; i < num_iterations; i++) {
            accumulate_and_transform<<<num_blocks, threads_per_block, 0, computation_stream>>>
                (d_input_array, d_intermediate_array, array_size);
            cudaDeviceSynchronize(); // Force complete serialization
            transform_with_offset<<<num_blocks, threads_per_block, 0, computation_stream>>>
                (d_intermediate_array, d_output_array, d_offset_array, array_size);
            cudaDeviceSynchronize();
        }
    }

    cudaEventRecord(stop, computation_stream);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_time = milliseconds;

    float avg_time = total_time / num_iterations;
    float throughput = (array_size * sizeof(float) * 3) / (avg_time * 1e6);

    printf("Config [%s]: Size=%d KB, Blocks=%d, Threads/Block=%d\n", 
           use_pdl ? "PDL" : "No PDL", 
           (array_size * sizeof(float)) / 1024,
           num_blocks, 
           threads_per_block);
    printf("  Avg time: %.3f ms\n", avg_time);
    printf("  Throughput: %.2f GB/s\n", throughput);

    // Cleanup
    delete[] h_input;
    delete[] h_offset;
    cudaFree(d_input_array);
    cudaFree(d_intermediate_array);
    cudaFree(d_output_array);
    cudaFree(d_offset_array);
    cudaStreamDestroy(computation_stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("\n=== PDL Performance Analysis on Hopper ===\n");
    
    // Test configurations
    const int sizes[] = {
        1024 * 32,      // Small: 128 KB
        1024 * 256,     // Medium: 1 MB
        1024 * 1024,    // Large: 4 MB
        1024 * 4096     // Very Large: 16 MB
    };
    
    const int block_sizes[] = {128, 256, 512};
    const int grid_sizes[] = {8, 16, 32};

    printf("\nTesting different data sizes with fixed grid/block size...\n");
    for (int size : sizes) {
        run_benchmark(size, 16, 256, true);  // With PDL
        run_benchmark(size, 16, 256, false); // Without PDL
        printf("\n");
    }

    printf("\nTesting different grid sizes with fixed data size...\n");
    for (int grid : grid_sizes) {
        run_benchmark(1024 * 1024, grid, 256, true);  // With PDL
        run_benchmark(1024 * 1024, grid, 256, false); // Without PDL
        printf("\n");
    }

    printf("\nTesting different block sizes with fixed data size...\n");
    for (int block : block_sizes) {
        run_benchmark(1024 * 1024, 16, block, true);  // With PDL
        run_benchmark(1024 * 1024, 16, block, false); // Without PDL
        printf("\n");
    }

    return 0;
}