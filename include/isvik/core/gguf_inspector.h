#ifndef ISVIK_CORE_GGUF_INSPECTOR_H_
#define ISVIK_CORE_GGUF_INSPECTOR_H_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "isvik/core/status.h"

namespace isvik {

struct GgufTensorInfo {
  std::string name;
  std::vector<uint64_t> dimensions;
  uint32_t type = 0;
  uint64_t data_offset = 0;
  uint64_t available_bytes = 0;
  std::optional<uint64_t> payload_bytes;
  uint64_t descriptor_type_position = 0;
  uint64_t descriptor_offset_position = 0;
};

struct GgufGemma4Config {
  uint64_t block_count = 0;
  uint64_t embedding_length = 0;
  uint64_t feed_forward_length = 0;
  uint64_t attention_head_count = 0;
  std::vector<uint64_t> attention_head_count_kv;
  uint64_t expert_count = 0;
  uint64_t expert_used_count = 0;
  uint64_t attention_key_length = 0;
  uint64_t attention_value_length = 0;
  uint64_t sliding_window = 0;
  std::vector<bool> sliding_window_pattern;
  uint64_t shared_kv_layers = 0;
  uint64_t embedding_length_per_layer_input = 0;
  uint64_t attention_key_length_swa = 0;
  uint64_t attention_value_length_swa = 0;
  uint64_t expert_feed_forward_length = 0;
  double rope_frequency_base = 0.0;
  double rope_frequency_base_swa = 0.0;
  double rms_norm_epsilon = 0.0;
  double final_logit_softcapping = 0.0;
};

struct GgufTokenizerConfig {
  std::string model;
  std::string pre_tokenizer;
  std::vector<std::string> tokens;
  std::vector<std::string> merges;
  std::vector<int32_t> token_types;
  uint32_t unknown_token_id = 3;
  uint32_t bos_token_id = 2;
  uint32_t eos_token_id = 1;
  uint32_t padding_token_id = 0;
};

struct GgufInspection {
  uint32_t version = 0;
  uint64_t tensor_count = 0;
  uint64_t metadata_count = 0;
  uint64_t file_size = 0;
  uint32_t alignment = 32;
  uint64_t tensor_data_start = 0;
  uint64_t file_type_value_position = 0;
  uint32_t file_type_value_size = 0;
  std::string architecture;
  std::string name;
  uint64_t parameter_count = 0;
  std::string quantization;
  uint64_t context_length = 0;
  GgufGemma4Config gemma4;
  GgufTokenizerConfig tokenizer;
  std::vector<GgufTensorInfo> tensors;
};

[[nodiscard]] std::string GgufTensorTypeName(uint32_t type);

class GgufInspector {
 public:
  // Reads and validates the GGUF header, metadata encoding, and tensor table.
  // Tensor payloads are not loaded.
  [[nodiscard]] static StatusOr<GgufInspection> Inspect(
      const std::filesystem::path& path);
};

}  // namespace isvik

#endif  // ISVIK_CORE_GGUF_INSPECTOR_H_
