// // Copyright (c) OpenMMLab. All rights reserved.
// #include "trt_voxelization.hpp"

// #include <assert.h>
// #include <chrono>
// #include <cstring>
// #include <map>

// #include "pytorch_cpp_helper.hpp"
// #include "pytorch_device_registry.hpp"

// #include "trt_voxelization_kernel.hpp"
// #include "trt_plugin_helper.hpp"
// #include "trt_serialize.hpp"

// namespace mmdeploy {
// namespace {
// static const char *PLUGIN_VERSION{"1"};
// static const char *PLUGIN_NAME{"voxelization"};
// }  // namespace

// TRTVOXELIZATION::TRTVOXELIZATION(const std::string &name, int maxpoints, int maxvoxels) :
//       TRTPluginBase(name),
//       maxPoints(maxpoints),
//       maxVoxels(maxvoxels){}

// TRTVOXELIZATION::TRTVOXELIZATION(const std::string name, const void *data, size_t length)
//     : TRTPluginBase(name) {
//   deserialize_value(&data, &length, &maxPoints);
//   deserialize_value(&data, &length, &maxVoxels);
// }

// nvinfer1::IPluginV2DynamicExt *TRTVOXELIZATION::clone() const TRT_NOEXCEPT {
//   TRTVOXELIZATION *plugin = new TRTVOXELIZATION(mLayerName, maxVoxels, maxPoints);
//   plugin->setPluginNamespace(getPluginNamespace());

//   return plugin;
// }

// nvinfer1::DimsExprs TRTVOXELIZATION::getOutputDimensions(
//     int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
//     nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
//     if (outputIndex == 0)
//     {
//         nvinfer1::DimsExprs dim0{};
//         dim0.nbDims = 3;
//         dim0.d[0] = exprBuilder.constant(maxVoxels);
//         dim0.d[1] = exprBuilder.constant(maxPoints);
//         dim0.d[2] = inputs[0].d[1];
//         return dim0;
//     }
//     if (outputIndex == 1)
//     {
//         nvinfer1::DimsExprs dim1{};
//         dim1.nbDims = 2;
//         dim1.d[0] = exprBuilder.constant(maxVoxels);
//         dim1.d[1] = exprBuilder.constant(3);
//         return dim1;
//     }
//     if (outputIndex == 2)
//     {
//         nvinfer1::DimsExprs dim2{};
//         dim2.nbDims = 1;
//         dim2.d[0] = exprBuilder.constant(maxVoxels);
//         return dim2;
//     }

//     nvinfer1::DimsExprs dim3{};
//     dim3.nbDims = 1;
//     dim3.d[0] = exprBuilder.constant(1);
//     return dim3;
// }

// bool TRTVOXELIZATION::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
//                                                int nbInputs, int nbOutputs) TRT_NOEXCEPT {
//   // input[0] == points->kFLOAT
//   // input[1] == voxel_size->kFLOAT
//   // input[2] == coors_range->kFLOAT
//   // output[0] == voxels->kFLOAT
//   // output[1] == coors->kINT32
//   // output[2] == num_points_per_voxel->kINT32
//   // output[3] == voxel_num->kINT32

//   if (pos == 0) {
//     return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
//             ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
//   } else {
//     return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
//   }
// }

// void TRTVOXELIZATION::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
//                                      const nvinfer1::DynamicPluginTensorDesc *outputs,
//                                      int nbOutputs) TRT_NOEXCEPT {
//   // Validate input arguments

//   ASSERT(nbInputs == 3);
//   ASSERT(nbOutputs == 4);
// }

// size_t TRTVOXELIZATION::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
//                                         const nvinfer1::PluginTensorDesc *outputs,
//                                         int nbOutputs) const TRT_NOEXCEPT {
//   return 0;
// }

// int TRTVOXELIZATION::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
//                             const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
//                             void *const *outputs, void *workSpace,
//                             cudaStream_t stream) TRT_NOEXCEPT {
//     int32_t *voxel_num_data = (*static_cast<at::Tensor *>(outputs[3])).data_ptr<int32_t>();

//     std::vector<float> voxel_size_v(
//         (*(at::Tensor *)inputs[1]).data_ptr<float>(),
//         (*(at::Tensor *)inputs[1]).data_ptr<float>() + (*(at::Tensor *)inputs[1]).numel());
//     std::vector<float> coors_range_v(
//         (*(at::Tensor *)inputs[2]).data_ptr<float>(),
//         (*(at::Tensor *)inputs[2]).data_ptr<float>() + (*(at::Tensor *)inputs[2]).numel());

//     auto data_type = inputDesc[0].type;
//     switch (data_type) {
//         case nvinfer1::DataType::kFLOAT:
//             *voxel_num_data = HardVoxelizeForwardCUDAKernelLauncher(
//             *(at::Tensor *)inputs[0], *(at::Tensor *)outputs[0], *(at::Tensor *)outputs[1], *(at::Tensor *)outputs[2], voxel_size_v,
//             coors_range_v, maxPoints, maxVoxels, 3, stream);
//             break;
//         default:
//             return 1;
//         break;
//     }

//     return 0;
// }

// nvinfer1::DataType TRTVOXELIZATION::getOutputDataType(int index,
//                                                      const nvinfer1::DataType *inputTypes,
//                                                      int nbInputs) const TRT_NOEXCEPT {
//     if (index == 0)
//     {
//         return inputTypes[0];
//     }

//     // if (index == 3)
//     // {
//     //     return nvinfer1::DataType::kINT32
//     // }
//     return nvinfer1::DataType::kINT32;
// }

// // IPluginV2 Methods
// const char *TRTVOXELIZATION::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

// const char *TRTVOXELIZATION::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

// int TRTVOXELIZATION::getNbOutputs() const TRT_NOEXCEPT { return 4; }

// size_t TRTVOXELIZATION::getSerializationSize() const TRT_NOEXCEPT {
//   return serialized_size(maxPoints) + serialized_size(maxVoxels);
// }

// void TRTVOXELIZATION::serialize(void *buffer) const TRT_NOEXCEPT {
//   serialize_value(&buffer, maxPoints);
//   serialize_value(&buffer, maxVoxels);
// }

// ////////////////////// creator /////////////////////////////

// TRTVOXELIZATIONCreator::TRTVOXELIZATIONCreator() {
//   mPluginAttributes = std::vector<nvinfer1::PluginField>(
//       {nvinfer1::PluginField("maxpoints"), nvinfer1::PluginField("maxvoxels")});
//   mFC.nbFields = mPluginAttributes.size();
//   mFC.fields = mPluginAttributes.data();
// }

// const char *TRTVOXELIZATIONCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

// const char *TRTVOXELIZATIONCreator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

// nvinfer1::IPluginV2 *TRTVOXELIZATIONCreator::createPlugin(
//     const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
//   int maxpoints = 0;
//   int maxvoxels = 0;
//   for (int i = 0; i < fc->nbFields; i++) {
//     if (fc->fields[i].data == nullptr) {
//       continue;
//     }
//     std::string field_name(fc->fields[i].name);

//     if (field_name.compare("maxpoints") == 0) {
//       maxpoints = static_cast<const int *>(fc->fields[i].data)[0];
//     }

//     if (field_name.compare("maxvoxels") == 0) {
//       maxvoxels = static_cast<const int *>(fc->fields[i].data)[0];
//     }
//   }
//   ASSERT(maxpoints > 0);
//   ASSERT(maxvoxels > 0);

//   TRTVOXELIZATION *plugin = new TRTVOXELIZATION(name, maxpoints, maxvoxels);
//   plugin->setPluginNamespace(getPluginNamespace());
//   return plugin;
// }

// nvinfer1::IPluginV2 *TRTVOXELIZATIONCreator::deserializePlugin(const char *name,
//                                                               const void *serialData,
//                                                               size_t serialLength) TRT_NOEXCEPT {
//   // This object will be deleted when the network is destroyed, which will
//   // call FCPluginDynamic::destroy()
//   auto plugin = new TRTVOXELIZATION(name, serialData, serialLength);
//   plugin->setPluginNamespace(getPluginNamespace());
//   return plugin;
// }

// REGISTER_TENSORRT_PLUGIN(TRTVOXELIZATIONCreator);
// }  // namespace mmdeploy
