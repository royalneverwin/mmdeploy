#ifndef TRT_MY_BEV_POOL_KERNEL_HPP
#define TRT_MY_BEV_POOL_KERNEL_HPP
#include <cuda_runtime.h>

#include "common_cuda_helper.hpp"

// CUDA function declarations
void my_bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths, float* out, cudaStream_t stream);

void my_bev_pool_v2_set_zero(int n_points, float* out);
#endif // TRT_BEV_POOL_KERNEL_HPP