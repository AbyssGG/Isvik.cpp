#ifndef ISVIK_CORE_GGUF_TOKENIZER_H_
#define ISVIK_CORE_GGUF_TOKENIZER_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "isvik/core/gguf_inspector.h"
#include "isvik/core/status.h"

namespace isvik {

// Minimal native BPE tokenizer for the tokenizer.ggml metadata embedded in a
// GGUF file. It has no llama.cpp/ggml runtime dependency.
class GgufTokenizer {
 public:
  [[nodiscard]] static StatusOr<GgufTokenizer> Create(
      const GgufTokenizerConfig& config);

  [[nodiscard]] StatusOr<std::vector<uint32_t>> Encode(
      std::string_view text, bool add_bos = false) const;
  [[nodiscard]] std::string DecodeToken(uint32_t token_id) const;
  [[nodiscard]] std::string Decode(const std::vector<uint32_t>& token_ids) const;

  [[nodiscard]] uint32_t bos_token_id() const { return bos_token_id_; }
  [[nodiscard]] uint32_t eos_token_id() const { return eos_token_id_; }
  [[nodiscard]] uint32_t unknown_token_id() const { return unknown_token_id_; }
  [[nodiscard]] std::size_t vocabulary_size() const { return tokens_.size(); }

 private:
  std::vector<std::string> tokens_;
  std::vector<int32_t> token_types_;
  std::unordered_map<std::string, uint32_t> token_ids_;
  std::unordered_map<std::string, uint32_t> merge_ranks_;
  uint32_t bos_token_id_ = 2;
  uint32_t eos_token_id_ = 1;
  uint32_t unknown_token_id_ = 3;
};

}  // namespace isvik

#endif  // ISVIK_CORE_GGUF_TOKENIZER_H_
