#include "isvik/backends/onnxruntime/genai_engine.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <string_view>
#include <system_error>
#include <utility>

#include "isvik/core/model_catalog.h"
#include <nlohmann/json.hpp>
#include "ort_genai.h"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace isvik::onnxruntime_backend {
namespace {

std::string_view RoleName(MessageRole role) {
  switch (role) {
    case MessageRole::kSystem: return "system";
    case MessageRole::kUser: return "user";
    case MessageRole::kAssistant: return "assistant";
    case MessageRole::kTool: return "tool";
  }
  return "user";
}

std::filesystem::path FindModelRoot(const std::filesystem::path& graph_path) {
  std::error_code error;
  auto directory = graph_path.parent_path();
  while (!directory.empty()) {
    if (std::filesystem::is_regular_file(directory / "genai_config.json", error) &&
        std::filesystem::is_regular_file(directory / "tokenizer_config.json", error)) {
      return directory;
    }
    const auto parent = directory.parent_path();
    if (parent == directory) break;
    directory = parent;
  }
  return {};
}

#if defined(ISVIK_HAS_TENSORRT_RTX_EP)
Status RegisterTensorRtRtxProvider() {
  static std::mutex registration_mutex;
  static bool registered = false;
  std::lock_guard lock(registration_mutex);
  if (registered) return Status();

  std::filesystem::path executable_path;
#ifdef _WIN32
  std::wstring buffer(32768U, L'\0');
  const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
  if (length == 0U || length >= buffer.size()) {
    return Status::Unavailable("cannot locate the application folder for the TensorRT-RTX provider");
  }
  buffer.resize(length);
  executable_path = std::filesystem::path(buffer).parent_path();
#else
  std::error_code error;
  executable_path = std::filesystem::current_path(error);
  if (error) return Status::Unavailable("cannot locate the application folder: " + error.message());
#endif

  const auto provider_path = executable_path / "onnxruntime_providers_nv_tensorrt_rtx.dll";
  std::error_code error;
  if (!std::filesystem::is_regular_file(provider_path, error)) {
    return Status::Unavailable("TensorRT-RTX provider library is missing beside the application");
  }
  const std::string provider_path_text = PathUtf8(provider_path);
  OgaRegisterExecutionProviderLibrary("NvTensorRTRTXExecutionProvider", provider_path_text.c_str());
  registered = true;
  return Status();
}

StatusOr<int> TensorRtRtxDeviceIndex(const std::string& device) {
  constexpr std::string_view prefix = "GPU.";
  if (!device.starts_with(prefix)) return 0;
  const std::string_view suffix(device.data() + prefix.size(), device.size() - prefix.size());
  int device_index = 0;
  const auto parsed = std::from_chars(suffix.data(), suffix.data() + suffix.size(), device_index);
  if (suffix.empty() || parsed.ec != std::errc{} || parsed.ptr != suffix.data() + suffix.size() || device_index < 0) {
    return Status::InvalidArgument("TensorRT-RTX device must be GPU followed by a non-negative index");
  }
  return device_index;
}
#endif

std::string FirstWord(std::string text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  text.erase(0U, first);
  const auto end = text.find_first_of(" \t\r\n");
  if (end != std::string::npos) text.resize(end);
  return text;
}

// GPT-OSS emits Harmony control tokens. Keep user-facing channels and suppress
// the analysis channel and token spellings from the transcript.
class HarmonyTextFilter {
 public:
  template <typename Emit>
  void Push(std::string_view fragment, Emit&& emit) {
    pending_.append(fragment);
    while (!pending_.empty()) {
      if (in_header_) {
        const auto message = pending_.find("<|message|>");
        if (message == std::string::npos) return;
        channel_ = FirstWord(pending_.substr(0U, message));
        pending_.erase(0U, message + std::string_view("<|message|>").size());
        in_header_ = false;
        continue;
      }

      const auto marker = pending_.find("<|");
      if (marker == std::string::npos) {
        emit_if_visible(pending_, emit);
        pending_.clear();
        return;
      }
      if (marker > 0U) {
        emit_if_visible(pending_.substr(0U, marker), emit);
        pending_.erase(0U, marker);
      }
      const auto end = pending_.find("|>", 2U);
      if (end == std::string::npos) return;
      const std::string token = pending_.substr(0U, end + 2U);
      pending_.erase(0U, end + 2U);
      if (token == "<|channel|>" || token == "<|meta_sep|>" || token == "<|meta_start|>") {
        in_header_ = true;
      } else if (token == "<|message|>") {
        if (channel_.empty()) channel_ = "final";
      } else if (token == "<|assistant|>") {
        channel_ = "final";
      }
    }
  }

  template <typename Emit>
  void Flush(Emit&& emit) {
    if (!in_header_) emit_if_visible(pending_, emit);
    pending_.clear();
  }

 private:
  template <typename Emit>
  void emit_if_visible(const std::string& text, Emit& emit) const {
    if (!text.empty() && channel_ != "analysis" && channel_ != "justify" &&
        channel_ != "confidence" && channel_ != "summary") {
      emit(text);
    }
  }

  std::string pending_;
  std::string channel_;
  bool in_header_ = false;
};

}  // namespace

struct GenAiEngine::Impl {
  std::unique_ptr<OgaConfig> config;
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaTokenizer> tokenizer;
};

GenAiEngine::GenAiEngine() : impl_(std::make_unique<Impl>()) {}
GenAiEngine::~GenAiEngine() = default;

Status GenAiEngine::LoadModel(const ModelDescriptor& descriptor, std::string device) {
  if (descriptor.format != ModelFormat::kOnnx) {
    return Status::Unsupported("ONNX Runtime GenAI requires an ONNX model with tokenizer assets");
  }
  if (device.empty()) device = "CPU";
  const bool use_tensorrt_rtx = device == "TENSORRT" || device.starts_with("GPU.");
  int tensorrt_rtx_device_id = 0;
  if (use_tensorrt_rtx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP)
    Status registration = RegisterTensorRtRtxProvider();
    if (!registration.ok()) return registration;
    StatusOr<int> device_index = TensorRtRtxDeviceIndex(device);
    if (!device_index.ok()) return device_index.status();
    tensorrt_rtx_device_id = device_index.value();
#else
    return Status::Unsupported("TensorRT-RTX ONNX support is not included in this build");
#endif
  } else if (device != "CPU" && device != "AUTO") {
    return Status::Unsupported("ONNX Runtime GenAI supports CPU or TensorRT-RTX devices in this build");
  }

  const auto resolved = ResolveModelEntry(descriptor.path);
  if (!resolved.ok()) return resolved.status();
  const auto model_root = FindModelRoot(resolved.value());
  if (model_root.empty()) {
    return Status::Unsupported(
        "ONNX Runtime GenAI needs genai_config.json and tokenizer_config.json beside the model assets");
  }
  std::error_code error;
  if (!std::filesystem::is_regular_file(model_root / "tokenizer.json", error)) {
    return Status::Unsupported("ONNX Runtime GenAI needs tokenizer.json beside genai_config.json");
  }

  std::lock_guard lock(mutex_);
  try {
    auto next = std::make_unique<Impl>();
    const std::string root_text = PathUtf8(model_root);
    next->config = OgaConfig::Create(root_text.c_str());
    // The source config targets Transformers.js WebGPU. Remove that provider
    // selection and override only the graph filename in memory; model files are
    // kept in their original directory and external-data shards stay untouched.
    next->config->ClearProviders();
#if defined(ISVIK_HAS_TENSORRT_RTX_EP)
    if (use_tensorrt_rtx) {
      constexpr const char* provider = "NvTensorRtRtx";
      next->config->AppendProvider(provider);
      next->config->AppendProvider("CPUExecutionProvider");
      const std::string device_id = std::to_string(tensorrt_rtx_device_id);
      next->config->SetProviderOption(provider, "device_id", device_id.c_str());
      next->config->SetProviderOption(provider, "nv_op_types_to_exclude",
          "QMoE,MatMulNBits,GroupQueryAttention,SkipSimplifiedLayerNormalization,SimplifiedLayerNormalization");
      next->config->SetProviderOption(provider, "nv_detailed_build_log", "1");
      next->config->SetProviderOption(provider, "enable_cuda_graph", "0");
      std::error_code cache_error;
      const auto cache_directory = std::filesystem::temp_directory_path(cache_error) /
          "Isvik" / "TensorRT-RTX-cache";
      if (!cache_error) {
        std::filesystem::create_directories(cache_directory, cache_error);
        if (!cache_error) {
          const std::string cache_path = PathUtf8(cache_directory);
          next->config->SetProviderOption(provider, "nv_runtime_cache_path", cache_path.c_str());
        }
      }
    }
#endif
    const auto relative_graph = std::filesystem::relative(resolved.value(), model_root, error);
    const auto first_component = relative_graph.begin();
    if (error || relative_graph.empty() || relative_graph.is_absolute() ||
        (first_component != relative_graph.end() && *first_component == "..")) {
      return Status::InvalidArgument("The ONNX graph must be inside its GenAI model directory");
    }
    const nlohmann::json overlay{{"model", {{"decoder", {{"filename", relative_graph.generic_string()}}}}}};
    const std::string overlay_text = overlay.dump();
    next->config->Overlay(overlay_text.c_str());
    next->model = OgaModel::Create(*next->config);
    next->tokenizer = OgaTokenizer::Create(*next->model);
    impl_ = std::move(next);
    loaded_model_ = descriptor;
    loaded_device_ = use_tensorrt_rtx
        ? "GPU." + std::to_string(tensorrt_rtx_device_id) + " (TensorRT-RTX + CPU fallback)"
        : "CPU";
    return Status();
  } catch (const std::exception& error) {
    return Status::Unavailable(std::string("ONNX Runtime GenAI model load failed: ") + error.what());
  }
}

Status GenAiEngine::Generate(const UnifiedInferenceRequest& request,
                             const CancellationToken& cancellation,
                             const InferenceEventHandler& handler) {
  const Status validation = ValidateInferenceRequest(request);
  if (!validation.ok()) return validation;

  std::lock_guard lock(mutex_);
  if (!impl_->model || !impl_->tokenizer || !loaded_model_) {
    return Status::Unavailable("no ONNX Runtime GenAI model is loaded");
  }
  if (request.model_id != loaded_model_->id) {
    return Status::InvalidArgument("request model_id does not match the loaded model");
  }
  if (cancellation.IsCancellationRequested()) {
    if (handler) handler(InferenceEvent{request.request_id, InferenceCancelled{}});
    return Status::Cancelled("generation cancelled before it started");
  }

  const auto emit = [&handler, &request](InferenceEventPayload payload) {
    if (handler) handler(InferenceEvent{request.request_id, std::move(payload)});
  };

  try {
    nlohmann::json messages = nlohmann::json::array();
    for (const auto& message : request.messages) {
      nlohmann::json item{{"role", std::string(RoleName(message.role))},
                          {"content", message.content}};
      if (!message.name.empty()) item["name"] = message.name;
      if (!message.tool_call_id.empty()) item["tool_call_id"] = message.tool_call_id;
      messages.push_back(std::move(item));
    }
    const std::string messages_text = messages.dump();
    const OgaString prompt = impl_->tokenizer->ApplyChatTemplate(
        nullptr, messages_text.c_str(), nullptr, true);
    auto input = OgaSequences::Create();
    impl_->tokenizer->Encode(static_cast<const char*>(prompt), *input);
    const std::size_t input_tokens = input->SequenceCount(0U);
    if (input_tokens == 0U) return Status::InvalidArgument("chat template produced an empty prompt");

    const int output_limit = request.generation.max_tokens.value_or(128);
    if (output_limit <= 0) return Status::InvalidArgument("max_tokens must be greater than zero");
    auto params = OgaGeneratorParams::Create(*impl_->model);
    params->SetSearchOption("max_length", static_cast<double>(input_tokens) + output_limit);
    params->SetSearchOption("batch_size", 1.0);
    const bool sampling = request.generation.decoding_mode == DecodingMode::kSampling;
    params->SetSearchOptionBool("do_sample", sampling);
    if (sampling) {
      params->SetSearchOption("temperature",
          request.generation.sampling.temperature.value_or(1.0F));
      if (request.generation.sampling.top_k) {
        params->SetSearchOption("top_k", *request.generation.sampling.top_k);
      }
      if (request.generation.sampling.top_p) {
        params->SetSearchOption("top_p", *request.generation.sampling.top_p);
      }
    }
    if (request.generation.sampling.repetition_penalty) {
      params->SetSearchOption("repetition_penalty",
          *request.generation.sampling.repetition_penalty);
    }
    if (request.generation.decoding_mode == DecodingMode::kBeamSearch ||
        request.generation.decoding_mode == DecodingMode::kDiverseBeamSearch) {
      params->SetSearchOption("num_beams", request.generation.num_beams);
      if (request.generation.decoding_mode == DecodingMode::kDiverseBeamSearch) {
        params->SetSearchOption("num_beam_groups", request.generation.num_beam_groups);
        params->SetSearchOption("diversity_penalty", request.generation.diversity_penalty);
      }
    }

    auto generator = OgaGenerator::Create(*impl_->model, *params);
    generator->AppendTokenSequences(*input);
    auto stream = OgaTokenizerStream::Create(*impl_->tokenizer);
    HarmonyTextFilter output_filter;
    std::string generated_text;
    const auto started = std::chrono::steady_clock::now();
    emit(InferenceStarted{loaded_model_->id});
    const auto forward_piece = [&emit, &generated_text, &request](const std::string& piece) {
      generated_text += piece;
      if (request.generation.stream) emit(ContentDelta{piece});
    };

    while (!generator->IsDone()) {
      if (cancellation.IsCancellationRequested()) {
        emit(InferenceCancelled{});
        return Status::Cancelled("generation cancelled");
      }
      generator->GenerateNextToken();
      const auto tokens = generator->GetNextTokens();
      for (const std::int32_t token : tokens) {
        const char* decoded = stream->Decode(token);
        if (decoded != nullptr) output_filter.Push(decoded, forward_piece);
      }
    }
    output_filter.Flush(forward_piece);
    if (!request.generation.stream && !generated_text.empty()) {
      emit(ContentDelta{generated_text});
    }

    const double duration = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const std::size_t sequence_tokens = generator->GetSequenceCount(0U);
    const uint64_t output_tokens = sequence_tokens > input_tokens
        ? static_cast<uint64_t>(sequence_tokens - input_tokens) : 0U;
    const double throughput = duration > 0.0
        ? static_cast<double>(output_tokens) / duration : 0.0;
    emit(InferenceCompleted{static_cast<uint64_t>(input_tokens), output_tokens,
                            duration, throughput});
    return Status();
  } catch (const std::exception& error) {
    const Status status = Status::Unavailable(
        std::string("ONNX Runtime generation failed: ") + error.what());
    emit(InferenceError{status});
    return status;
  }
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
  impl_->tokenizer.reset();
  impl_->model.reset();
  impl_->config.reset();
  loaded_model_.reset();
  loaded_device_.clear();
}

}  // namespace isvik::onnxruntime_backend
