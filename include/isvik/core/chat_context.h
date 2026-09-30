#ifndef ISVIK_CORE_CHAT_CONTEXT_H_
#define ISVIK_CORE_CHAT_CONTEXT_H_

#include <cstdint>
#include <string_view>
#include <vector>

#include "isvik/core/context_manager.h"
#include "isvik/core/memory_service.h"
#include "isvik/core/status.h"

namespace isvik {

struct ChatContextResult {
  std::vector<InferenceMessage> messages;
  uint64_t estimated_prompt_tokens = 0;
  uint64_t dropped_history_messages = 0;
  uint64_t memories_used = 0;
};

// Adds user-saved notes that match the latest message, then fits the prompt to
// the model budget while retaining the system note and latest exchange.
[[nodiscard]] StatusOr<ChatContextResult> BuildChatContext(
    const std::vector<InferenceMessage>& conversation,
    const MemoryService* memory_service,
    std::string_view session_id,
    bool use_memories,
    const ContextOptions& options);

}  // namespace isvik

#endif  // ISVIK_CORE_CHAT_CONTEXT_H_
