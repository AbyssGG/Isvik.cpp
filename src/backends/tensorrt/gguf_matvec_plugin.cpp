#include "gguf_matvec_plugin.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <string_view>

#include <NvInfer.h>

#include "gguf_matvec_kernel.h"
#include "isvik/core/gguf_quant.h"

namespace isvik::tensorrt_backend::detail {
namespace {

constexpr char kPluginName[] = "IsvikGgufQuantizedMatVec";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "isvik";
constexpr char kTensorTypeField[] = "tensor_type";
constexpr char kInputFeaturesField[] = "input_features";
constexpr char kOutputFeaturesField[] = "output_features";

bool ComputeRowBytes(uint32_t type, uint64_t input_features,
                     uint64_t* row_bytes) {
  const auto block = GgufQuantBlockInfoForType(type);
  if (!block.has_value() || input_features == 0U ||
      input_features % block->elements != 0U) {
    return false;
  }
  const uint64_t block_count = input_features / block->elements;
  if (block_count > UINT64_MAX / block->bytes) return false;
  *row_bytes = block_count * block->bytes;
  return true;
}

class GgufQuantizedMatVecPlugin final : public nvinfer1::IPluginV3,
                                        public nvinfer1::IPluginV3OneCore,
                                        public nvinfer1::IPluginV3OneBuild,
                                        public nvinfer1::IPluginV3OneRuntime {
 public:
  GgufQuantizedMatVecPlugin(uint32_t tensor_type, uint64_t input_features,
                            uint64_t output_features)
      : tensor_type_(tensor_type),
        input_features_(input_features),
        output_features_(output_features) {
    static_cast<void>(ComputeRowBytes(tensor_type_, input_features_, &row_bytes_));
    UpdateSerializedFields();
  }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    switch (type) {
      case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
      case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
      case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
      default:
        return nullptr;
    }
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    try {
      return new GgufQuantizedMatVecPlugin(tensor_type_, input_features_,
                                           output_features_);
    } catch (...) {
      return nullptr;
    }
  }

  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override { return kPluginVersion; }
  const char* getPluginNamespace() const noexcept override { return kPluginNamespace; }
  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t nb_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t nb_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr || nb_outputs != 1 ||
        nb_inputs != 2 || input_types[0] != nvinfer1::DataType::kFLOAT ||
        input_types[1] != nvinfer1::DataType::kINT8) {
      return 1;
    }
    output_types[0] = nvinfer1::DataType::kFLOAT;
    return 0;
  }

  int32_t getOutputShapes(const nvinfer1::DimsExprs* inputs,
                          int32_t nb_inputs,
                          const nvinfer1::DimsExprs* shape_inputs,
                          int32_t nb_shape_inputs,
                          nvinfer1::DimsExprs* outputs,
                          int32_t nb_outputs,
                          nvinfer1::IExprBuilder& expr_builder) noexcept override {
    if (inputs == nullptr || outputs == nullptr || nb_inputs != 2 ||
        nb_shape_inputs != 0 || shape_inputs != nullptr || nb_outputs != 1 ||
        inputs[0].nbDims != 2 || inputs[1].nbDims != 1 ||
        output_features_ > static_cast<uint64_t>(INT32_MAX) ||
        row_bytes_ == 0U || output_features_ > UINT64_MAX / row_bytes_ ||
        inputs[1].d[0] == nullptr) {
      return 1;
    }
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = expr_builder.constant(static_cast<int64_t>(output_features_));
    return outputs[0].d[1] == nullptr ? 1 : 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t nb_inputs, int32_t nb_outputs) noexcept override {
    if (in_out == nullptr || nb_inputs != 2 || nb_outputs != 1 ||
        position < 0 || position >= 3) {
      return false;
    }
    if (in_out[position].desc.format != nvinfer1::TensorFormat::kLINEAR) return false;
    if (position == 0) return in_out[position].desc.type == nvinfer1::DataType::kFLOAT;
    if (position == 1) return in_out[position].desc.type == nvinfer1::DataType::kINT8;
    return in_out[position].desc.type == nvinfer1::DataType::kFLOAT;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                          int32_t nb_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* out,
                          int32_t nb_outputs) noexcept override {
    if (in == nullptr || out == nullptr || nb_inputs != 2 || nb_outputs != 1 ||
        in[0].desc.dims.nbDims != 2 || in[1].desc.dims.nbDims != 1 ||
        out[0].desc.dims.nbDims != 2 ||
        in[0].desc.type != nvinfer1::DataType::kFLOAT ||
        in[1].desc.type != nvinfer1::DataType::kINT8 ||
        out[0].desc.type != nvinfer1::DataType::kFLOAT ||
        in[0].desc.dims.d[1] != static_cast<int32_t>(input_features_) ||
        in[1].desc.dims.d[0] != static_cast<int32_t>(row_bytes_ * output_features_)) {
      return 1;
    }
    return 0;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
                          const nvinfer1::DynamicPluginTensorDesc*,
                          int32_t) const noexcept override {
    return 0U;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* in,
                       int32_t nb_inputs,
                       const nvinfer1::PluginTensorDesc* out,
                       int32_t nb_outputs) noexcept override {
    if (in == nullptr || out == nullptr || nb_inputs != 2 || nb_outputs != 1 ||
        in[0].dims.nbDims != 2 || in[1].dims.nbDims != 1 ||
        out[0].dims.nbDims != 2 ||
        in[0].dims.d[1] != static_cast<int32_t>(input_features_) ||
        in[1].dims.d[0] != static_cast<int32_t>(row_bytes_ * output_features_) ||
        out[0].dims.d[0] != in[0].dims.d[0] ||
        out[0].dims.d[1] != static_cast<int32_t>(output_features_)) {
      return 1;
    }
    return 0;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc*,
                  const void* const* inputs, void* const* outputs,
                  void*, cudaStream_t stream) noexcept override {
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
        input_desc[0].dims.d[0] <= 0 || row_bytes_ == 0U) {
      return 1;
    }
    const cudaError_t status = LaunchGgufQuantizedMatVec(
        tensor_type_, input_features_, output_features_,
        static_cast<uint64_t>(input_desc[0].dims.d[0]),
        static_cast<const float*>(inputs[0]),
        static_cast<const uint8_t*>(inputs[1]),
        static_cast<float*>(outputs[0]), stream);
    return status == cudaSuccess ? 0 : 1;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext*) noexcept override {
    return clone();
  }

  nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
    UpdateSerializedFields();
    return &serialized_fields_;
  }

 private:
  void UpdateSerializedFields() noexcept {
    serialized_type_ = static_cast<int32_t>(tensor_type_);
    serialized_values_[1] = static_cast<int64_t>(input_features_);
    serialized_values_[2] = static_cast<int64_t>(output_features_);
    serialized_fields_data_[0] = {kTensorTypeField, &serialized_type_,
                                   nvinfer1::PluginFieldType::kINT32, 1};
    serialized_fields_data_[1] = {kInputFeaturesField, &serialized_values_[1],
                                   nvinfer1::PluginFieldType::kINT64, 1};
    serialized_fields_data_[2] = {kOutputFeaturesField, &serialized_values_[2],
                                   nvinfer1::PluginFieldType::kINT64, 1};
    serialized_fields_.nbFields = static_cast<int32_t>(serialized_fields_data_.size());
    serialized_fields_.fields = serialized_fields_data_.data();
  }

  uint32_t tensor_type_ = 0U;
  uint64_t input_features_ = 0U;
  uint64_t output_features_ = 0U;
  uint64_t row_bytes_ = 0U;
  int32_t serialized_type_ = 0;
  std::array<int64_t, 3> serialized_values_{};
  std::array<nvinfer1::PluginField, 3> serialized_fields_data_{};
  nvinfer1::PluginFieldCollection serialized_fields_{};
};

class GgufQuantizedMatVecCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  GgufQuantizedMatVecCreator() {
    field_names_[0] = {kTensorTypeField, nullptr, nvinfer1::PluginFieldType::kINT32, 1};
    field_names_[1] = {kInputFeaturesField, nullptr, nvinfer1::PluginFieldType::kINT64, 1};
    field_names_[2] = {kOutputFeaturesField, nullptr, nvinfer1::PluginFieldType::kINT64, 1};
    field_collection_.nbFields = static_cast<int32_t>(field_names_.size());
    field_collection_.fields = field_names_.data();
  }

  nvinfer1::IPluginV3* createPlugin(const char*,
                                   const nvinfer1::PluginFieldCollection* fields,
                                   nvinfer1::TensorRTPhase) noexcept override {
    if (fields == nullptr || fields->fields == nullptr || fields->nbFields != 3) return nullptr;
    uint32_t tensor_type = UINT32_MAX;
    uint64_t input_features = 0U;
    uint64_t output_features = 0U;
    for (int32_t index = 0; index < fields->nbFields; ++index) {
      const nvinfer1::PluginField& field = fields->fields[index];
      if (field.name == nullptr || field.data == nullptr || field.length != 1) return nullptr;
      const std::string_view field_name(field.name);
      if (field_name == kTensorTypeField && field.type == nvinfer1::PluginFieldType::kINT32) {
        const int32_t value = *static_cast<const int32_t*>(field.data);
        if (value < 0) return nullptr;
        tensor_type = static_cast<uint32_t>(value);
      } else if (field_name == kInputFeaturesField &&
                 field.type == nvinfer1::PluginFieldType::kINT64) {
        const int64_t value = *static_cast<const int64_t*>(field.data);
        if (value <= 0) return nullptr;
        input_features = static_cast<uint64_t>(value);
      } else if (field_name == kOutputFeaturesField &&
                 field.type == nvinfer1::PluginFieldType::kINT64) {
        const int64_t value = *static_cast<const int64_t*>(field.data);
        if (value <= 0) return nullptr;
        output_features = static_cast<uint64_t>(value);
      } else {
        return nullptr;
      }
    }
    uint64_t row_bytes = 0U;
    if (tensor_type == UINT32_MAX || !IsGgufTensorEncodingDecodable(tensor_type) ||
        !ComputeRowBytes(tensor_type, input_features, &row_bytes) || row_bytes == 0U ||
        output_features > UINT64_MAX / row_bytes ||
        input_features > static_cast<uint64_t>(INT32_MAX) ||
        output_features > static_cast<uint64_t>(INT32_MAX) ||
        row_bytes * output_features > static_cast<uint64_t>(INT32_MAX)) {
      return nullptr;
    }
    try {
      return new GgufQuantizedMatVecPlugin(tensor_type, input_features, output_features);
    } catch (...) {
      return nullptr;
    }
  }

  nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
    return &field_collection_;
  }

  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override { return kPluginVersion; }
  const char* getPluginNamespace() const noexcept override { return kPluginNamespace; }

 private:
  std::array<nvinfer1::PluginField, 3> field_names_{};
  nvinfer1::PluginFieldCollection field_collection_{};
};

}  // namespace

Status RegisterGgufQuantizedMatVecPlugin() {
  static std::mutex mutex;
  static GgufQuantizedMatVecCreator creator;
  std::lock_guard lock(mutex);
  nvinfer1::IPluginRegistry* registry = ::getPluginRegistry();
  if (registry == nullptr) return Status::Unavailable("TensorRT plugin registry is unavailable");
  if (registry->getCreator(kPluginName, kPluginVersion, kPluginNamespace) != nullptr) {
    return Status();
  }
  if (!registry->registerCreator(creator, kPluginNamespace)) {
    return Status::Unavailable("TensorRT could not register the native GGUF matvec plugin");
  }
  return Status();
}

nvinfer1::IPluginV3* CreateGgufQuantizedMatVecPlugin(
    uint32_t tensor_type, uint64_t input_features,
    uint64_t output_features) noexcept {
  uint64_t row_bytes = 0U;
  if (!IsGgufTensorEncodingDecodable(tensor_type) ||
      !ComputeRowBytes(tensor_type, input_features, &row_bytes) ||
      row_bytes == 0U || output_features == 0U ||
      output_features > UINT64_MAX / row_bytes ||
      input_features > static_cast<uint64_t>(INT32_MAX) ||
      output_features > static_cast<uint64_t>(INT32_MAX) ||
      row_bytes * output_features > static_cast<uint64_t>(INT32_MAX)) {
    return nullptr;
  }
  try {
    return new GgufQuantizedMatVecPlugin(tensor_type, input_features, output_features);
  } catch (...) {
    return nullptr;
  }
}

}  // namespace isvik::tensorrt_backend::detail
