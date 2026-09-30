#ifndef ISVIK_CORE_INFERENCE_REQUEST_H_
#define ISVIK_CORE_INFERENCE_REQUEST_H_

#include <string>
#include <vector>

#include "isvik/core/generation_config.h"
#include "isvik/core/status.h"

namespace isvik {

enum class MessageRole {
  kSystem,
  kUser,
  kAssistant,
  kTool,
};

struct InferenceMessage {
  MessageRole role = MessageRole::kUser;
  std::string content;
  std::string name;
  std::string tool_call_id;
};

struct UnifiedInferenceRequest {
  std::string request_id;
  std::string model_id;
  std::vector<InferenceMessage> messages;
  GenerationConfig generation;
};

Status ValidateInferenceRequest(const UnifiedInferenceRequest& request);

}  // namespace isvik

#endif  // ISVIK_CORE_INFERENCE_REQUEST_H_
