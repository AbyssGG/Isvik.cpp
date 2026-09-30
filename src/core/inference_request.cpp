#include "isvik/core/inference_request.h"

namespace isvik {

Status ValidateInferenceRequest(const UnifiedInferenceRequest& request) {
  if (request.model_id.empty()) {
    return Status::InvalidArgument("model_id cannot be empty");
  }
  if (request.messages.empty()) {
    return Status::InvalidArgument("messages cannot be empty");
  }
  for (const InferenceMessage& message : request.messages) {
    if (message.role == MessageRole::kTool && message.tool_call_id.empty()) {
      return Status::InvalidArgument("tool messages must include tool_call_id");
    }
  }
  return ValidateGenerationConfig(request.generation);
}

}  // namespace isvik
