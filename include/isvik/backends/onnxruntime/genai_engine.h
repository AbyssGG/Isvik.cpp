#ifndef ISVIK_BACKENDS_ONNXRUNTIME_GENAI_ENGINE_H_
#define ISVIK_BACKENDS_ONNXRUNTIME_GENAI_ENGINE_H_

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "isvik/core/cancellation.h"
#include "isvik/core/inference_event.h"
#include "isvik/core/inference_request.h"
#include "isvik/core/model_descriptor.h"

namespace isvik::onnxruntime_backend {

using InferenceEventHandler = std::function<void(const InferenceEvent&)>;

class GenAiEngine {
 public:
  GenAiEngine();
  ~GenAiEngine();

  GenAiEngine(const GenAiEngine&) = delete;
  GenAiEngine& operator=(const GenAiEngine&) = delete;

  [[nodiscard]] Status LoadModel(const ModelDescriptor& model,
                                 std::string device = "CPU");
  [[nodiscard]] Status Generate(const UnifiedInferenceRequest& request,
                                const CancellationToken& cancellation,
                                const InferenceEventHandler& handler);
  [[nodiscard]] std::optional<ModelDescriptor> loaded_model() const;
  [[nodiscard]] std::string loaded_device() const;
  void UnloadModel();

 private:
  struct Impl;
  mutable std::mutex mutex_;
  std::unique_ptr<Impl> impl_;
  std::optional<ModelDescriptor> loaded_model_;
  std::string loaded_device_;
};

}  // namespace isvik::onnxruntime_backend

#endif  // ISVIK_BACKENDS_ONNXRUNTIME_GENAI_ENGINE_H_
