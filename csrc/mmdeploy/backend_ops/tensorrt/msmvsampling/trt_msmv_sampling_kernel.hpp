//
// Created by Xinhao Wang on 2024/09/13.
//

#ifndef TRT_MSMV_SAMPLING_KERNEL_HPP
#define TRT_MSMV_SAMPLING_KERNEL_HPP
#include <cuda_runtime.h>

#include "common_cuda_helper.hpp"

// CUDA function declarations
void ms_deformable_im2col_cuda_c2345(
    const float* feat_c2,
    const float* feat_c3,
    const float* feat_c4,
    const float* feat_c5,
    const int h_c2, const int w_c2,
    const int h_c3, const int w_c3,
    const int h_c4, const int w_c4,
    const int h_c5, const int w_c5,
    const float* data_sampling_loc,
    const float* data_attn_weight,
    const int batch_size,
    const int channels,
    const int num_views,
    const int num_query,
    const int num_point,
    float* data_col,
    cudaStream_t stream);

void msmv_sampling_set_zero(int n_points, float* out);
#endif // TRT_MSMV_SAMPLING_KERNEL_HPP