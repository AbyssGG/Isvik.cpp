#ifndef ISVIK_BACKENDS_TENSORRT_GGUF_GENAI_ENGINE_H_
#define ISVIK_BACKENDS_TENSORRT_GGUF_GENAI_ENGINE_H_

#include <functional>
#include <memory>
#include <mutex>
#include <optional>

#include "isvik/core/cancellation.h"
#include "isvik/core/inference_event.h"
#include "isvik/core/inference_request.h"
#include "isvik/core/model_descriptor.h"

namespace isvik::tensorrt_backend {

using GgufInferenceEventHandler = std::function<void(const InferenceEvent&)>;
using GgufRuntimeLogHandler = std::function<void(std::string_view)>;

// Native Gemma 4 GGUF text runtime. GGUF weights remain in their source file
// and quantization; matrix-vector products execute through Isvik's TensorRT
// plugin while tokenization, KV cache and graph orchestration are owned here.
class GgufGenAiEngine {
 public:
  explicit GgufGenAiEngine(GgufRuntimeLogHandler log_handler = {});
  ~GgufGenAiEngine();

  GgufGenAiEngine(const GgufGenAiEngine&) = delete;
  GgufGenAiEngine& operator=(const GgufGenAiEngine&) = delete;

  [[nodiscard]] Status LoadModel(const ModelDescriptor& model, int device_index = 0);
  [[nodiscard]] Status Generate(const UnifiedInferenceRequest& request,
                                const CancellationToken& cancellation,
                                const GgufInferenceEventHandler& handler);
  [[nodiscard]] std::optional<ModelDescriptor> loaded_model() const;
  [[nodiscard]] int loaded_device() const;
  void UnloadModel();

  [[nodiscard]] static Status ValidateModel(const ModelDescriptor& model);

 private:
  struct Impl;
  mutable std::mutex mutex_;
  std::unique_ptr<Impl> impl_;
  std::optional<ModelDescriptor> loaded_model_;
  int loaded_device_ = -1;
};

}  // namespace isvik::tensorrt_backend

#endif  // ISVIK_BACKENDS_TENSORRT_GGUF_GENAI_ENGINE_H_
