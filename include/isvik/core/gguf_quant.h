#ifndef ISVIK_CORE_GGUF_QUANT_H_
#define ISVIK_CORE_GGUF_QUANT_H_

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "isvik/core/status.h"

namespace isvik {

struct GgufQuantBlockInfo {
  uint64_t elements = 0;
  uint64_t bytes = 0;
};

inline constexpr uint64_t kMaxGgufCpuDecodeElements = 16'777'216U;

// Returns block dimensions for the GGML tensor encodings supported by the
// direct Gemma 4 reader. Tensor type IDs are the numeric GGUF/GGML IDs.
[[nodiscard]] std::optional<GgufQuantBlockInfo> GgufQuantBlockInfoForType(
    uint32_t tensor_type);

[[nodiscard]] bool IsGgufTensorEncodingDecodable(uint32_t tensor_type);

// Decodes complete, consecutive blocks into a bounded CPU reference buffer.
// It never converts or writes a GGUF model. Callers should pass a small range
// of blocks rather than an entire large tensor.
[[nodiscard]] StatusOr<std::vector<float>> DecodeGgufTensorBlocks(
    uint32_t tensor_type, std::span<const uint8_t> encoded_blocks);

}  // namespace isvik

#endif  // ISVIK_CORE_GGUF_QUANT_H_
