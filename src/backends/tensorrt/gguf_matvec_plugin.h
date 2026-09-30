#ifndef ISVIK_TENSORRT_GGUF_MATVEC_PLUGIN_H_
#define ISVIK_TENSORRT_GGUF_MATVEC_PLUGIN_H_

#include <cstdint>

#include <NvInferRuntime.h>

#include "isvik/core/status.h"

namespace isvik::tensorrt_backend::detail {

[[nodiscard]] Status RegisterGgufQuantizedMatVecPlugin();
[[nodiscard]] nvinfer1::IPluginV3* CreateGgufQuantizedMatVecPlugin(
    uint32_t tensor_type, uint64_t input_features,
    uint64_t output_features) noexcept;

}  // namespace isvik::tensorrt_backend::detail

#endif  // ISVIK_TENSORRT_GGUF_MATVEC_PLUGIN_H_
