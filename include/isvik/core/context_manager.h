#ifndef ISVIK_CORE_CONTEXT_MANAGER_H_
#define ISVIK_CORE_CONTEXT_MANAGER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "isvik/core/inference_request.h"
#include "isvik/core/status.h"

namespace isvik {

enum class ContextCompressionMode {
  kOff,
  kSlidingWindow,
  kSummarize,
  kHybrid,
};

struct ContextMessage {
  InferenceMessage message;
  bool pinned = false;
  // Messages with the same non-empty retention group are kept or dropped together.
  std::string retention_group_id;
  // Compatibility field for existing tool-call groups.
  std::string tool_group_id;
};

struct ContextOptions {
  ContextCompressionMode compression = ContextCompressionMode::kHybrid;
  uint64_t max_context_tokens = 8192;
  uint64_t reserved_output_tokens = 512;
  uint32_t estimated_bytes_per_token = 4;
};

struct ContextBuildResult {
  std::vector<ContextMessage> messages;
  uint64_t estimated_prompt_tokens = 0;
  uint64_t dropped_messages = 0;
};

class ContextManager {
 public:
  // Estimates tokens from UTF-8 byte length; replace with the model tokenizer when available.
  [[nodiscard]] static StatusOr<ContextBuildResult> BuildPrompt(
      const std::vector<ContextMessage>& messages, const ContextOptions& options);
  [[nodiscard]] static uint64_t EstimateMessageTokens(
      const InferenceMessage& message, uint32_t estimated_bytes_per_token = 4);
};

}  // namespace isvik

#endif  // ISVIK_CORE_CONTEXT_MANAGER_H_
