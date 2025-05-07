
#pragma once

#include "util.cuh"

namespace kittens
{
    namespace prototype
    {
        namespace vm
        {

            template <typename config, typename globals>
            struct BaseOp
            {
                static constexpr int opcode = 0;

                struct controller
                {
                    static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query)
                    {
                        return query;
                    }
                    static __device__ int init_semaphores(const globals &g, state<config> &s)
                    {
                        return 0;
                    }
                };
                struct loader
                {
                    static __device__ void run(const globals &g, state<config> &s)
                    {
                        if (laneid() < config::NUM_PAGES)
                        { // Release all pages, ASAP.
                            auto pid = s.pid(laneid());
                            s.wait_page_ready(pid);
                            s.finish_page(pid, config::NUM_CONSUMER_WARPS);
                        }
                    }
                };
                struct sync_loader
                {
                    static __device__ void run(const globals &g, state<config> &s) {}
                };
                struct prefetcher
                {
                    static __device__ void run(const globals &g, state<config> &s) {}
                };
                struct launcher
                { // launches mma's
                    // launcher does nothing here, since this doesn't use tensor cores.
                    static __device__ void run(const globals &g, state<config> &s)
                    {
#ifdef KITTENS_BLACKWELL
                        s.wait_tensor_ready();
                        if (laneid() == 0)
                            arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
#endif
                    }
                };
                struct consumer
                {
                    static __device__ void run(const globals &g, state<config> &s) {}
                };
                struct storer
                {
                    static __device__ void run(const globals &g, state<config> &s) {}
                };

                struct greg
                {
                    static __device__ void run(const globals &g, state<config> &s) {}
                };

                struct greg2
                {
                    static __device__ void run(const globals &g, state<config> &s) {}
                };
            };

            template <typename config, typename globals>
            struct NoOp : BaseOp<config, globals>
            {
                static constexpr int opcode = 0;
            };

        } // namespace vm
    } // namespace prototype
} // namespace kittens