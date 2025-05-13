#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

constexpr int COLS = 128;

struct globals {
    gl<fp8e4m3, 1, 1, -1, -1, st<fp8e4m3, 128/8, COLS>> tile_in_1;
    gl<fp8e4m3, 1, 1, -1, -1, st<fp8e4m3, 128/8, COLS>> tile_in_2;
    gl<fp8e4m3, 1, 1, -1, -1, st<fp8e4m3, 128/8, COLS>> tile_1_copy_out;
    gl<fp8e4m3, 1, 1, -1, -1, st<fp8e4m3, 128/8, COLS>> tile_12_matmul_out;

    int dynamic_shared_memory() { return 226000; }
    dim3 grid()  { return dim3(1); }
    dim3 block() { return dim3(256); }
};

__global__ void test(globals g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    using st_type = st<fp8e4m3, 128, COLS>;
    using st_warp_split_type = st<fp8e4m3, 128/8, COLS>;

    st_type (& smem_tile_list_1)[1] = al.allocate<st_type, 1>();
    st_type (& smem_tile_list_2)[1] = al.allocate<st_type, 1>();
    st_type & smem_tile_1 = smem_tile_list_1[0];
    st_type & smem_tile_2 = smem_tile_list_2[0];

    st_warp_split_type (&smem_warp_split_1)[8] = reinterpret_cast<st_warp_split_type(&)[8]>(smem_tile_1);
    st_warp_split_type (&smem_warp_split_2)[8] = reinterpret_cast<st_warp_split_type(&)[8]>(smem_tile_2);

    warp::load_async(smem_warp_split_1[group<8>::warpid()], g.tile_in_1, {group<8>::warpid(), 0});
    warp::load_async(smem_warp_split_2[group<8>::warpid()], g.tile_in_2, {group<8>::warpid(), 0});

    load_async_wait();
    group<8>::sync(2);

    rt<fp8e4m3, 128/8, COLS> tile_1_copy;
    warp::load(tile_1_copy, smem_warp_split_1[group<8>::warpid()]);

    tensor_allocator<1, 1> tm_alloc{};

    using tt_type = tt<fp8e4m3, 128, COLS>;
    tt_type tile_1_copy_tt = tm_alloc.allocate<tt_type>(0);
    group<8>::store_async(tile_1_copy_tt, tile_1_copy);
    tensor_store_wait();
    group<8>::sync(2);
    group<8>::load_async(tile_1_copy, tile_1_copy_tt);

    warp::store(smem_warp_split_1[group<8>::warpid()], tile_1_copy);
    __syncwarp();
    warp::store(g.tile_1_copy_out, smem_warp_split_1[group<8>::warpid()], {group<8>::warpid(), 0});
    // warp::tma::store_async(g.tile_1_copy_out, smem_warp_split_1[group<8>::warpid()], {group<8>::warpid(), 0});
}

PYBIND11_MODULE(test, m) {
    m.doc() = "test";
    py::bind_kernel<test>(m,
        "test",
        &globals::tile_in_1,
        &globals::tile_in_2,
        &globals::tile_1_copy_out,
        &globals::tile_12_matmul_out
    );
}