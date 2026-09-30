#ifndef ISVIK_CORE_INFERENCE_EVENT_H_
#define ISVIK_CORE_INFERENCE_EVENT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "isvik/core/status.h"

namespace isvik {

struct InferenceStarted {
  std::string model_id;
};

struct ContentDelta {
  std::string text;
};

struct ReasoningDelta {
  std::string text;
};

struct ToolCallStarted {
  std::size_t index = 0;
  std::string id;
  std::string name;
};

struct ToolCallDelta {
  std::size_t index = 0;
  std::string arguments_delta;
};

struct UsageUpdated {
  uint64_t input_tokens = 0;
  uint64_t output_tokens = 0;
};

struct InferenceCompleted {
  uint64_t input_tokens = 0;
  uint64_t output_tokens = 0;
  double duration_seconds = 0.0;
  double tokens_per_second = 0.0;
};
struct InferenceCancelled {};

struct InferenceError {
  Status status;
};

using InferenceEventPayload = std::variant<
    InferenceStarted,
    ContentDelta,
    ReasoningDelta,
    ToolCallStarted,
    ToolCallDelta,
    UsageUpdated,
    InferenceCompleted,
    InferenceCancelled,
    InferenceError>;

struct InferenceEvent {
  std::string request_id;
  InferenceEventPayload payload;
};

}  // namespace isvik

#endif  // ISVIK_CORE_INFERENCE_EVENT_H_
