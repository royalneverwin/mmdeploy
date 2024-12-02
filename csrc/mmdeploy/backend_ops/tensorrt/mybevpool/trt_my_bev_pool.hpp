// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_MY_BEV_POOL_HPP
#define TRT_MY_BEV_POOL_HPP
#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

#include "trt_plugin_base.hpp"

namespace mmdeploy {

class TRTBEVPoolV2WithZ : public TRTPluginBase {
 public:
  TRTBEVPoolV2WithZ(const std::string &name, int outWidth, int outHeight, int outZ);

  TRTBEVPoolV2WithZ(const std::string name, const void *data, size_t length);

  TRTBEVPoolV2WithZ() = delete;

  ~TRTBEVPoolV2WithZ() TRT_NOEXCEPT override = default;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                          int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out,
                       int nbOutputs) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const TRT_NOEXCEPT override;

  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
              void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const TRT_NOEXCEPT override;

  // IPluginV2 Methods
  const char *getPluginType() const TRT_NOEXCEPT override;

  const char *getPluginVersion() const TRT_NOEXCEPT override;

  int getNbOutputs() const TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;

  void serialize(void *buffer) const TRT_NOEXCEPT override;

 private:
  int mOutWidth;
  int mOutHeight;
  int mOutZ;
};

class TRTBEVPoolV2WithZCreator : public TRTPluginCreatorBase {
 public:
  TRTBEVPoolV2WithZCreator();

  ~TRTBEVPoolV2WithZCreator() TRT_NOEXCEPT override = default;

  const char *getPluginName() const TRT_NOEXCEPT override;

  const char *getPluginVersion() const TRT_NOEXCEPT override;

  nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      TRT_NOEXCEPT override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                         size_t serialLength) TRT_NOEXCEPT override;
};
}  // namespace mmdeploy
#endif  // 
