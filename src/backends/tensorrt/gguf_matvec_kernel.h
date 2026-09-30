#ifndef ISVIK_TENSORRT_GGUF_MATVEC_KERNEL_H_
#define ISVIK_TENSORRT_GGUF_MATVEC_KERNEL_H_

#include <cstdint>

#include <cuda_runtime_api.h>

namespace isvik::tensorrt_backend::detail {

[[nodiscard]] cudaError_t LaunchGgufQuantizedMatVec(
    uint32_t tensor_type, uint64_t input_features, uint64_t output_features,
    uint64_t batch_size, const float* activations,
    const uint8_t* packed_weights, float* output, cudaStream_t stream);

}  // namespace isvik::tensorrt_backend::detail

#endif  // ISVIK_TENSORRT_GGUF_MATVEC_KERNEL_H_
