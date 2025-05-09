#include "kittens.cuh"
#include "prototype.cuh"
#include <iostream>
#include "pyutils/pyutils.cuh"

using namespace kittens;

using fp8 = fp8e4m3;
template<int rows, int cols> using st_fp8 = st_fp8e4m3<rows, cols>;
template<int rows, int cols> using rt_fp8 = rt_fp8e4m3<rows, cols>;

struct conversion_globals {
    gl<bf16, -1, -1, -1, -1, st_bf<16, 64>> a;
    gl<fp8, -1, -1, -1, -1, st_fp8<16, 64>> b;
};

struct sizes {
    int global_id;
    int num_warps;
    int num_reads;
    int num_reads_per_warp;

    __global__ int get_indices(const __grid_constant__ conversion_globals g, int i) {
        int read_id = num_warps * i + global_id;
        int w = read_id % g.a.w();
        int new_id = read_id / g.a.w();
        int z = new_id % g.a.z();
        new_id = new_id / g.a.z();
        int y = new_id % g.a.y();
        new_id = new_id / g.a.y();
        int x = new_id % g.a.x();
        return {x, y, z, w};
    }
};

__global__ sizes get_sizes(const __grid_constant__ conversion_globals g) {
    int global_id = gridIdx().x *  + warpid();
    int num_warps = gridDim.x * blockDim.x / 32;
    assert(g.a.x() == g.b.x());
    assert(g.a.y() == g.b.y());
    assert(g.a.z() == g.b.z());
    assert(g.a.w() == g.b.w());
    assert(g.a.rows() % 16 == 0);
    assert(g.a.cols() % 64 == 0);

    int num_reads = g.a.x() * g.a.y() * (g.a.rows() / 16) * (g.a.cols() / 64);
    int num_reads_per_warp = (num_reads + num_warps - 1)/ num_warps;

    return {global_id, num_warps, num_reads, num_reads_per_warp};
}

__global__ void pack_bf16_to_fp8(const __grid_constant__ conversion_globals g) {
    sizes s = get_sizes(g);

    st_bf<16, 64> a_st;
    rt_bf<16, 64> a_rt;
    st_fp8<16, 64> b_st;
    rt_fp8<16, 64> b_rt;
    for(int i = 0; i < s.num_reads_per_warp; i++) {
        int x, y, z, w = s.get_indices(g, i);
        warp::load(a_st, g.a, {x, y, z, w});
        warp::load(a_rt, a_st);
        warp::copy(b_rt, a_rt);
        warp::store(b_st, b_rt);
        warp::store(g.b, b_st, {x, y, z, w});
    }
}

__global__ void unpack_fp8_to_bf16(const __grid_constant__ conversion_globals g) {
    sizes s = get_sizes(g);

    st_bf<16, 64> a_st;
    rt_bf<16, 64> a_rt;
    st_fp8<16, 64> b_st;
    rt_fp8<16, 64> b_rt;
    for(int i = 0; i < s.num_reads_per_warp; i++) {
        int x, y, z, w = s.get_indices(g, i);
        warp::load(b_st, g.b, {x, y, z, w});
        warp::load(b_rt, b_st);
        warp::copy(a_rt, b_rt);
        warp::store(a_st, a_rt);
        warp::store(g.a, a_st, {x, y, z, w});
    }
}

PYBIND11_MODULE(pack_unpack_global_fp8, m) {
    py::bind_kernel<pack_bf16_to_fp8>(m, "pack_bf16_to_fp8", &conversion_globals::a, &conversion_globals::b);
    py::bind_kernel<unpack_fp8_to_bf16>(m, "unpack_fp8_to_bf16", &conversion_globals::a, &conversion_globals::b);
}