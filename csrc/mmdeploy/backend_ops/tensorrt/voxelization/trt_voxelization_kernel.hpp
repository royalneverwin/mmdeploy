// #ifndef TRT_VOXELIZATION_KERNEL_HPP
// #define TRT_VOXELIZATION_KERNEL_HPP
// #include <cuda_runtime.h>

// #include "common_cuda_helper.hpp"

// // CUDA function declarations
// int HardVoxelizeForwardCUDAKernelLauncher(
//     const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
//     at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
//     const std::vector<float> coors_range, const int max_points,
//     const int max_voxels, const int NDim, cudaStream_t stream);

// #endif // TRT_VOXELIZATION_KERNEL_HPP