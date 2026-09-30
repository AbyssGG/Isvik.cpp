#include "isvik/backends/openvino/genai_engine.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>

#if defined(ISVIK_HAS_OPENVINO_GENAI)
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>
#endif
#include "isvik/core/model_catalog.h"

namespace isvik::openvino_backend {

struct GenAiEngine::Impl {
#if defined(ISVIK_HAS_OPENVINO_GENAI)
  std::unique_ptr<ov::genai::LLMPipeline> pipeline;
  std::unique_ptr<ov::genai::VLMPipeline> visual_pipeline;
#endif
};

namespace {

#if defined(ISVIK_HAS_OPENVINO_GENAI)
const char* RoleName(MessageRole role) {
  switch (role) {
    case MessageRole::kSystem: return "system";
    case MessageRole::kUser: return "user";
    case MessageRole::kAssistant: return "assistant";
    case MessageRole::kTool: return "tool";
  }
  return "user";
}

std::string PlainPrompt(const std::vector<InferenceMessage>& messages) {
  if (messages.size() == 1U && messages.front().role == MessageRole::kUser) {
    return messages.front().content;
  }

  std::ostringstream prompt;
  for (const InferenceMessage& message : messages) {
    switch (message.role) {
      case MessageRole::kSystem:
        prompt << "System: ";
        break;
      case MessageRole::kUser:
        prompt << "User: ";
        break;
      case MessageRole::kAssistant:
        prompt << "Assistant: ";
        break;
      case MessageRole::kTool:
        prompt << "Tool: ";
        break;
    }
    prompt << message.content << '\n';
  }
  prompt << "Assistant:";
  return prompt.str();
}

StatusOr<ov::genai::GenerationConfig> MakeGenerationConfig(
    const GenerationConfig& source, bool has_chat_template) {
  if (source.decoding_mode == DecodingMode::kSpeculative) {
    return Status::Unsupported(
        "speculative decoding requires a draft model, which is not configured");
  }

  ov::genai::GenerationConfig config;
  config.max_new_tokens = source.max_tokens.has_value()
      ? static_cast<std::size_t>(*source.max_tokens)
      : 128U;
  config.do_sample = source.decoding_mode == DecodingMode::kSampling;
  if (source.decoding_mode == DecodingMode::kBeamSearch ||
      source.decoding_mode == DecodingMode::kDiverseBeamSearch) {
    config.num_beams = static_cast<std::size_t>(source.num_beams);
  }
  if (source.decoding_mode == DecodingMode::kDiverseBeamSearch) {
    config.num_beam_groups = static_cast<std::size_t>(source.num_beam_groups);
    config.diversity_penalty = source.diversity_penalty;
  }
  if (source.sampling.temperature.has_value()) {
    config.temperature = *source.sampling.temperature;
  }
  if (source.sampling.top_p.has_value()) config.top_p = *source.sampling.top_p;
  if (source.sampling.top_k.has_value()) {
    config.top_k = static_cast<std::size_t>(*source.sampling.top_k);
  }
  if (source.sampling.seed.has_value()) {
    if (*source.sampling.seed > std::numeric_limits<std::size_t>::max()) {
      return Status::InvalidArgument("seed is too large for this platform");
    }
    config.rng_seed = static_cast<std::size_t>(*source.sampling.seed);
  }
  if (source.sampling.presence_penalty.has_value()) {
    config.presence_penalty = *source.sampling.presence_penalty;
  }
  if (source.sampling.frequency_penalty.has_value()) {
    config.frequency_penalty = *source.sampling.frequency_penalty;
  }
  if (source.sampling.repetition_penalty.has_value()) {
    config.repetition_penalty = *source.sampling.repetition_penalty;
  }
  config.stop_strings.insert(source.stop_sequences.begin(), source.stop_sequences.end());
  config.apply_chat_template = has_chat_template;
  return config;
}
#endif

}  // namespace

GenAiEngine::GenAiEngine() : impl_(std::make_unique<Impl>()) {}

GenAiEngine::~GenAiEngine() = default;

Status GenAiEngine::LoadModel(const ModelDescriptor& model, std::string device) {
#if defined(ISVIK_HAS_OPENVINO_GENAI)
  if (model.format != ModelFormat::kOpenVinoIr && model.format != ModelFormat::kGguf) {
    return Status::Unsupported(
        "OpenVINO GenAI requires an OpenVINO IR or supported GGUF model");
  }
  if (device.empty()) return Status::InvalidArgument("device cannot be empty");

  std::lock_guard lock(mutex_);
  try {
    const auto entry = ResolveModelEntry(model.path);
    if (!entry.ok()) return entry.status();
#if defined(_WIN32) && defined(_DEBUG)
    // The bundled 2026.4 Debug runtime aborts inside openvinod.dll while loading
    // this VLM layout. Return a recoverable error instead of losing the UI.
    if (entry.value().filename() == "openvino_language_model.xml") {
      return Status::Unavailable(
          "The OpenVINO 2026.4 Windows Debug runtime cannot load this VLM layout safely. "
          "Build/run the Release configuration to use this model.");
    }
#endif
    const std::filesystem::path model_directory = entry.value().parent_path();
    if (model.format == ModelFormat::kGguf) {
      // OpenVINO GenAI reads supported GGUF topologies natively. The file itself
      // is passed to the runtime; no llama.cpp process or engine is involved.
      auto pipeline = std::make_unique<ov::genai::LLMPipeline>(entry.value(), device);
      impl_->pipeline = std::move(pipeline);
      impl_->visual_pipeline.reset();
    } else if (entry.value().filename() == "openvino_language_model.xml") {
      auto pipeline = std::make_unique<ov::genai::VLMPipeline>(model_directory, device);
      impl_->visual_pipeline = std::move(pipeline);
      impl_->pipeline.reset();
    } else {
      auto pipeline = std::make_unique<ov::genai::LLMPipeline>(model_directory, device);
      impl_->pipeline = std::move(pipeline);
      impl_->visual_pipeline.reset();
    }
    loaded_model_ = model;
    loaded_device_ = std::move(device);
    return Status();
  } catch (const ov::Exception& error) {
    return Status::Unavailable(std::string("OpenVINO model load failed: ") + error.what());
  } catch (const std::exception& error) {
    return Status::Internal(std::string("Unexpected model load error: ") + error.what());
  }
#else
  (void)model;
  (void)device;
  return Status::Unavailable("Isvik was built without the OpenVINO GenAI C++ runtime");
#endif
}

Status GenAiEngine::Generate(const UnifiedInferenceRequest& request,
                             const CancellationToken& cancellation,
                             const InferenceEventHandler& handler) {
#if defined(ISVIK_HAS_OPENVINO_GENAI)
  const Status validation = ValidateInferenceRequest(request);
  if (!validation.ok()) return validation;

  std::lock_guard lock(mutex_);
  if ((!impl_->pipeline && !impl_->visual_pipeline) || !loaded_model_.has_value()) {
    return Status::Unavailable("no OpenVINO model is loaded");
  }
  if (request.model_id != loaded_model_->id) {
    return Status::InvalidArgument("request model_id does not match the loaded model");
  }

  const auto emit = [&handler, &request](InferenceEventPayload payload) {
    if (handler) handler(InferenceEvent{request.request_id, std::move(payload)});
  };

  if (cancellation.IsCancellationRequested()) {
    emit(InferenceCancelled{});
    return Status::Cancelled("generation cancelled before it started");
  }

  try {
    const bool has_chat_template =
        !(impl_->visual_pipeline ? impl_->visual_pipeline->get_tokenizer()
                                 : impl_->pipeline->get_tokenizer()).get_chat_template().empty();
    StatusOr<ov::genai::GenerationConfig> config =
        MakeGenerationConfig(request.generation, has_chat_template);
    if (!config.ok()) return config.status();

    emit(InferenceStarted{loaded_model_->id});
    std::string generated_text;
    const auto streamer = [&](std::string piece) {
      if (cancellation.IsCancellationRequested()) {
        return ov::genai::StreamingStatus::CANCEL;
      }
      if (!piece.empty()) {
        generated_text += piece;
        if (request.generation.stream) emit(ContentDelta{std::move(piece)});
      }
      return ov::genai::StreamingStatus::RUNNING;
    };

    ov::genai::PerfMetrics performance;
    if (has_chat_template) {
      ov::genai::ChatHistory history;
      for (const InferenceMessage& message : request.messages) {
        ov::AnyMap entry{{"role", RoleName(message.role)}, {"content", message.content}};
        if (!message.name.empty()) entry.emplace("name", message.name);
        if (!message.tool_call_id.empty()) {
          entry.emplace("tool_call_id", message.tool_call_id);
        }
        history.push_back(entry);
      }
      if (impl_->visual_pipeline) {
        auto result = impl_->visual_pipeline->generate(history, std::vector<ov::Tensor>{},
                                                       config.value(), streamer);
        performance = result.perf_metrics;
        if (!request.generation.stream && !result.texts.empty()) generated_text = result.texts.front();
      } else {
        auto result = impl_->pipeline->generate(history, config.value(), streamer);
        performance = result.perf_metrics;
        if (!request.generation.stream && !result.texts.empty()) generated_text = result.texts.front();
      }
    } else {
      if (impl_->visual_pipeline) {
        auto result = impl_->visual_pipeline->generate(PlainPrompt(request.messages),
            std::vector<ov::Tensor>{}, config.value(), streamer);
        performance = result.perf_metrics;
        if (!request.generation.stream && !result.texts.empty()) generated_text = result.texts.front();
      } else {
        auto result = impl_->pipeline->generate(PlainPrompt(request.messages), config.value(), streamer);
        performance = result.perf_metrics;
        if (!request.generation.stream && !result.texts.empty()) generated_text = result.texts.front();
      }
    }

    if (cancellation.IsCancellationRequested()) {
      emit(InferenceCancelled{});
      return Status::Cancelled("generation cancelled");
    }
    if (!request.generation.stream) {
      emit(ContentDelta{generated_text});
    }
    const double duration_seconds =
        static_cast<double>(performance.get_generate_duration().mean) / 1000.0;
    const double tokens_per_second = performance.get_throughput().mean;
    emit(InferenceCompleted{
        static_cast<uint64_t>(performance.get_num_input_tokens()),
        static_cast<uint64_t>(performance.get_num_generated_tokens()),
        duration_seconds,
        tokens_per_second});
    return Status();
  } catch (const ov::Exception& error) {
    const Status status = Status::Unavailable(
        std::string("OpenVINO generation failed: ") + error.what());
    emit(InferenceError{status});
    return status;
  } catch (const std::exception& error) {
    const Status status = Status::Internal(
        std::string("Unexpected generation error: ") + error.what());
    emit(InferenceError{status});
    return status;
  }
#else
  (void)request;
  (void)cancellation;
  (void)handler;
  return Status::Unavailable("Isvik was built without the OpenVINO GenAI C++ runtime");
#endif
}

std::optional<ModelDescriptor> GenAiEngine::loaded_model() const {
  std::lock_guard lock(mutex_);
  return loaded_model_;
}

std::string GenAiEngine::loaded_device() const {
  std::lock_guard lock(mutex_);
  return loaded_device_;
}

void GenAiEngine::UnloadModel() {
  std::lock_guard lock(mutex_);
#if defined(ISVIK_HAS_OPENVINO_GENAI)
  impl_->pipeline.reset();
  impl_->visual_pipeline.reset();
#endif
  loaded_model_.reset();
  loaded_device_.clear();
}

}  // namespace isvik::openvino_backend
