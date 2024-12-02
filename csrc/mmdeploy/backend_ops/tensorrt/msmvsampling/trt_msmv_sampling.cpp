//
// Created by Xinhao Wang on 2024/09/13.
//

#include "trt_msmv_sampling.hpp"

#include <assert.h>

#include <chrono>

#include "trt_msmv_sampling_kernel.hpp"
#include "trt_plugin_helper.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"msmv_sampling"};
}  // namespace

TRTMSMVSAMPLING::TRTMSMVSAMPLING(const std::string &name) :
      TRTPluginBase(name){}

TRTMSMVSAMPLING::TRTMSMVSAMPLING(const std::string name, const void *data, size_t length)
    : TRTPluginBase(name) {}

nvinfer1::IPluginV2DynamicExt *TRTMSMVSAMPLING::clone() const TRT_NOEXCEPT {
  TRTMSMVSAMPLING *plugin = new TRTMSMVSAMPLING(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TRTMSMVSAMPLING::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  // input[0] == mlvl_feats[0]
  // input[1] == mlvl_feats[1]
  // input[2] == mlvl_feats[2]
  // input[3] == mlvl_feats[3]
  // input[4] == sample_points_cam
  // input[4] == scale_weights
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = exprBuilder.constant(32);
  ret.d[1] = exprBuilder.constant(900);
  ret.d[2] = exprBuilder.constant(64);
  ret.d[3] = exprBuilder.constant(4);
  return ret;
}

bool TRTMSMVSAMPLING::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
                                               int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  // input[0] == mlvl_feats[0]->kFLOAT
  // input[1] == mlvl_feats[1]->kFLOAT
  // input[2] == mlvl_feats[2]->kFLOAT
  // input[3] == mlvl_feats[3]->kFLOAT
  // input[4] == sample_points_cam->kFLOAT
  // input[5] == scale_weights->kFLOAT
  // output[0] == final->kFLOAT
  if (pos == 0) {
    return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
  }
}

void TRTMSMVSAMPLING::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
                                     const nvinfer1::DynamicPluginTensorDesc *outputs,
                                     int nbOutputs) TRT_NOEXCEPT {
  // Validate input arguments

  ASSERT(nbInputs == 6);
  ASSERT(nbOutputs == 1);
}

size_t TRTMSMVSAMPLING::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                        const nvinfer1::PluginTensorDesc *outputs,
                                        int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int TRTMSMVSAMPLING::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                            const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                            void *const *outputs, void *workSpace,
                            cudaStream_t stream) TRT_NOEXCEPT {
  auto data_type = inputDesc[0].type;
  nvinfer1::Dims feat_dims_2 = inputDesc[0].dims; // bnhwc
  nvinfer1::Dims feat_dims_3 = inputDesc[1].dims; // bnhwc
  nvinfer1::Dims feat_dims_4 = inputDesc[2].dims; // bnhwc
  nvinfer1::Dims feat_dims_5 = inputDesc[3].dims; // bnhwc
  nvinfer1::Dims sampling_loc_dims = inputDesc[4].dims; // bqpc
  nvinfer1::Dims out_dims = outputDesc[0].dims; //bhwc
  int num_points = out_dims.d[0]*out_dims.d[1]*out_dims.d[2]*out_dims.d[3];
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      msmv_sampling_set_zero(num_points, (float *)outputs[0]);
      ms_deformable_im2col_cuda_c2345((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)inputs[3], 
      feat_dims_2.d[2], feat_dims_2.d[3], feat_dims_3.d[2], feat_dims_3.d[3], feat_dims_4.d[2], feat_dims_4.d[3], feat_dims_5.d[2], feat_dims_5.d[3], 
      (float *)inputs[4], (float *)inputs[5], feat_dims_2.d[0], feat_dims_2.d[4], feat_dims_2.d[1], sampling_loc_dims.d[1], sampling_loc_dims.d[2], (float *)outputs[0], stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType TRTMSMVSAMPLING::getOutputDataType(int index,
                                                     const nvinfer1::DataType *inputTypes,
                                                     int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TRTMSMVSAMPLING::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTMSMVSAMPLING::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int TRTMSMVSAMPLING::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t TRTMSMVSAMPLING::getSerializationSize() const TRT_NOEXCEPT {
  return 0;
}

void TRTMSMVSAMPLING::serialize(void *buffer) const TRT_NOEXCEPT {}

////////////////////// creator /////////////////////////////

TRTMSMVSAMPLINGCreator::TRTMSMVSAMPLINGCreator() {
  mPluginAttributes = std::vector<nvinfer1::PluginField>();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TRTMSMVSAMPLINGCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTMSMVSAMPLINGCreator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *TRTMSMVSAMPLINGCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  
  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
  }

  TRTMSMVSAMPLING *plugin = new TRTMSMVSAMPLING(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *TRTMSMVSAMPLINGCreator::deserializePlugin(const char *name,
                                                              const void *serialData,
                                                              size_t serialLength) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TRTMSMVSAMPLING(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTMSMVSAMPLINGCreator);
}  // namespace mmdeploy
