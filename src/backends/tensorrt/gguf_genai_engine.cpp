#include "isvik/backends/tensorrt/gguf_genai_engine.h"

#include "isvik/backends/tensorrt/engine.h"
#include "isvik/core/gguf_model_file.h"
#include "isvik/core/gguf_quant.h"
#include "isvik/core/gguf_tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace isvik::tensorrt_backend {
namespace {

const GgufTensorInfo* FindTensor(const GgufInspection& inspection,
                                 std::string_view name) {
  const auto found = std::find_if(inspection.tensors.begin(), inspection.tensors.end(),
      [name](const GgufTensorInfo& tensor) { return tensor.name == name; });
  return found == inspection.tensors.end() ? nullptr : &*found;
}

Status ValidateGemma4(const GgufInspection& inspection) {
  const auto& config = inspection.gemma4;
  if (inspection.architecture != "gemma4") {
    return Status::Unsupported("TensorRT native GGUF chat currently supports Gemma 4 models");
  }
  if (config.block_count == 0U || config.embedding_length == 0U ||
      config.attention_head_count == 0U || config.expert_count == 0U ||
      config.expert_used_count == 0U || config.expert_feed_forward_length == 0U) {
    return Status::InvalidArgument("Gemma 4 GGUF is missing required model metadata");
  }
  if (inspection.tokenizer.tokens.empty() || inspection.tokenizer.merges.empty()) {
    return Status::InvalidArgument("Gemma 4 GGUF has no embedded BPE tokenizer");
  }
  for (const auto& tensor : inspection.tensors) {
    if (!IsGgufTensorEncodingDecodable(tensor.type)) {
      return Status::Unsupported("TensorRT native GGUF decoder does not support " +
                                 GgufTensorTypeName(tensor.type) + " tensor: " + tensor.name);
    }
  }
  const std::vector<std::string> required = {
      "token_embd.weight", "output_norm.weight", "blk.0.attn_q.weight",
      "blk.0.attn_k.weight", "blk.0.attn_output.weight", "blk.0.ffn_gate.weight",
      "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"};
  for (const auto& name : required) {
    if (FindTensor(inspection, name) == nullptr) {
      return Status::InvalidArgument("Gemma 4 GGUF is missing tensor: " + name);
    }
  }
  return Status();
}

std::vector<float> Rms(const std::vector<float>& input, double epsilon,
                       const std::vector<float>* weight = nullptr) {
  double sum = 0.0;
  for (const float value : input) sum += static_cast<double>(value) * value;
  const float scale = static_cast<float>(1.0 / std::sqrt(sum / input.size() + epsilon));
  std::vector<float> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = input[i] * scale * (weight == nullptr ? 1.0F : (*weight)[i]);
  }
  return output;
}

float Gelu(float value) {
  constexpr float k = 0.7978845608028654F;
  return 0.5F * value * (1.0F + std::tanh(k * (value + 0.044715F * value * value * value)));
}

void AddInPlace(std::vector<float>* destination, const std::vector<float>& source,
                float scale = 1.0F) {
  for (std::size_t i = 0; i < destination->size(); ++i) {
    (*destination)[i] += source[i] * scale;
  }
}

void ApplyRope(std::vector<float>* values, uint64_t heads, uint64_t head_size,
               uint64_t rotated_dimensions, uint64_t position, double theta) {
  const uint64_t half = rotated_dimensions / 2U;
  for (uint64_t head = 0; head < heads; ++head) {
    const uint64_t base = head * head_size;
    for (uint64_t index = 0; index < half; ++index) {
      const double frequency = std::pow(theta, -2.0 * static_cast<double>(index) /
                                                  static_cast<double>(rotated_dimensions));
      const float cosine = static_cast<float>(std::cos(position * frequency));
      const float sine = static_cast<float>(std::sin(position * frequency));
      const float first = (*values)[base + index];
      const float second = (*values)[base + index + half];
      (*values)[base + index] = first * cosine - second * sine;
      (*values)[base + index + half] = second * cosine + first * sine;
    }
  }
}

std::string EngineKey(uint32_t type, uint64_t input, uint64_t output) {
  return std::to_string(type) + "-" + std::to_string(input) + "-" +
         std::to_string(output);
}

int ParseDevice(std::string_view value) {
  if (value.starts_with("GPU.")) value.remove_prefix(4U);
  int result = 0;
  for (const char character : value) {
    if (character < '0' || character > '9') return -1;
    result = result * 10 + character - '0';
  }
  return result;
}

std::string ChatPrompt(const std::vector<InferenceMessage>& messages) {
  std::string result;
  for (const auto& message : messages) {
    const char* role = "user";
    if (message.role == MessageRole::kSystem) role = "system";
    else if (message.role == MessageRole::kAssistant) role = "model";
    else if (message.role == MessageRole::kTool) role = "tool";
    result += "<|turn>";
    result += role;
    result += '\n';
    result += message.content;
    result += "<turn|>\n";
  }
  result += "<|turn>model\n<|channel>thought\n<channel|>";
  return result;
}

}  // namespace

struct GgufGenAiEngine::Impl {
  struct LayerCache {
    uint64_t heads = 0;
    uint64_t head_size = 0;
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
  };

  struct MatVecProfile {
    std::chrono::nanoseconds weight_read{};
    std::chrono::nanoseconds engine_lookup{};
    std::chrono::nanoseconds inference{};
    uint64_t calls = 0U;
    uint64_t weight_bytes = 0U;
  };

  explicit Impl(GgufRuntimeLogHandler handler) : log(std::move(handler)) {}

  GgufRuntimeLogHandler log;
  std::shared_ptr<GgufModelFile> file;
  std::optional<GgufTokenizer> tokenizer;
  std::unordered_map<std::string, std::unique_ptr<Engine>> engines;
  std::unordered_map<std::string, std::vector<float>> small_tensors;
  std::vector<LayerCache> cache;
  MatVecProfile profile;
  int device = 0;

  void Log(std::string message) const {
    if (log) log(message);
  }

  StatusOr<const std::vector<float>*> Small(std::string_view name) {
    const std::string key(name);
    if (const auto found = small_tensors.find(key); found != small_tensors.end()) {
      return &found->second;
    }
    const auto* info = FindTensor(file->inspection(), name);
    if (info == nullptr || info->dimensions.empty()) {
      return Status::NotFound("GGUF tensor does not exist: " + key);
    }
    uint64_t elements = 1U;
    for (const uint64_t dimension : info->dimensions) elements *= dimension;
    const auto block = GgufQuantBlockInfoForType(info->type);
    if (!block || elements % block->elements != 0U) {
      return Status::InvalidArgument("invalid GGUF tensor layout: " + key);
    }
    auto decoded = file->ReadDecodedTensorBlocks(name, 0U, elements / block->elements);
    if (!decoded.ok()) return decoded.status();
    auto [position, inserted] = small_tensors.emplace(key, std::move(decoded).value());
    return &position->second;
  }

  StatusOr<Engine*> GetEngine(uint32_t type, uint64_t input, uint64_t output) {
    const std::string key = EngineKey(type, input, output);
    if (const auto found = engines.find(key); found != engines.end()) return found->second.get();
    std::error_code error;
    auto directory = std::filesystem::temp_directory_path(error) / "Isvik" / "TensorRT";
    if (error) return Status::Unavailable("cannot locate TensorRT cache directory");
    std::filesystem::create_directories(directory, error);
    if (error) return Status::Unavailable("cannot create TensorRT cache directory: " + error.message());
    const auto path = directory / ("gguf-matvec-" + key + ".engine");
    if (!std::filesystem::is_regular_file(path, error)) {
      Log("Building TensorRT GGUF kernel " + key);
      BuildOptions options;
      options.device_index = device;
      const auto status = BuildGgufQuantizedMatVecEngine(type, input, output, path, options);
      if (!status.ok()) return status;
    }
    auto engine = std::make_unique<Engine>();
    const auto status = engine->Load(path, device);
    if (!status.ok()) return status;
    Engine* pointer = engine.get();
    engines.emplace(key, std::move(engine));
    return pointer;
  }

  StatusOr<std::vector<float>> MatVec(std::string_view name,
                                      const std::vector<float>& input,
                                      std::optional<uint64_t> slice = std::nullopt) {
    const auto* info = FindTensor(file->inspection(), name);
    if (info == nullptr || info->dimensions.size() < 2U) {
      return Status::NotFound("GGUF matrix does not exist: " + std::string(name));
    }
    const uint64_t width = info->dimensions[0];
    const uint64_t rows = info->dimensions[1];
    if (width != input.size()) {
      return Status::InvalidArgument("GGUF matrix input width mismatch: " + std::string(name));
    }
    const auto block = GgufQuantBlockInfoForType(info->type);
    if (!block || width % block->elements != 0U) {
      return Status::InvalidArgument("GGUF matrix row layout is invalid: " + std::string(name));
    }
    const uint64_t row_bytes = width / block->elements * block->bytes;
    uint64_t offset = 0U;
    if (slice.has_value()) {
      if (info->dimensions.size() != 3U || *slice >= info->dimensions[2]) {
        return Status::InvalidArgument("GGUF expert slice is out of range: " + std::string(name));
      }
      offset = *slice * rows * row_bytes;
    }
    // TensorRT's static compiler can produce an engine much larger than the
    // packed GGUF payload for very tall matrices (notably the vocabulary head).
    // Fixed row chunks keep engine size and persistent GPU memory bounded.
    constexpr uint64_t kMaximumRowsPerEngine = 4096U;
    std::vector<float> output;
    output.reserve(static_cast<std::size_t>(rows));
    for (uint64_t first_row = 0U; first_row < rows;
         first_row += kMaximumRowsPerEngine) {
      const uint64_t current_rows = std::min(kMaximumRowsPerEngine, rows - first_row);
      const auto read_started = std::chrono::steady_clock::now();
      auto packed = file->ReadTensorRange(
          name, offset + first_row * row_bytes, current_rows * row_bytes);
      profile.weight_read += std::chrono::steady_clock::now() - read_started;
      if (!packed.ok()) return packed.status();
      const auto engine_started = std::chrono::steady_clock::now();
      auto engine = GetEngine(info->type, width, current_rows);
      profile.engine_lookup += std::chrono::steady_clock::now() - engine_started;
      if (!engine.ok()) return engine.status();

      Tensor activations;
      activations.name = "activations";
      activations.data_type = TensorDataType::kFloat32;
      activations.dimensions = {1, static_cast<int64_t>(width)};
      activations.bytes.resize(input.size() * sizeof(float));
      std::memcpy(activations.bytes.data(), input.data(), activations.bytes.size());
      Tensor weights;
      weights.name = "packed_weights";
      // TensorRT has signed INT8 I/O; the plugin treats the bytes as the original
      // opaque GGUF payload, so their bit patterns are unchanged.
      weights.data_type = TensorDataType::kInt8;
      weights.dimensions = {static_cast<int64_t>(packed.value().size())};
      weights.bytes = std::move(packed).value();
      const auto inference_started = std::chrono::steady_clock::now();
      auto result = engine.value()->Infer({std::move(activations), std::move(weights)});
      profile.inference += std::chrono::steady_clock::now() - inference_started;
      if (!result.ok()) return result.status();
      if (result.value().size() != 1U ||
          result.value()[0].bytes.size() != current_rows * sizeof(float)) {
        return Status::Internal("TensorRT GGUF matrix output has an unexpected shape");
      }
      const std::size_t old_size = output.size();
      output.resize(old_size + static_cast<std::size_t>(current_rows));
      std::memcpy(output.data() + old_size, result.value()[0].bytes.data(),
                  result.value()[0].bytes.size());
      ++profile.calls;
      profile.weight_bytes += current_rows * row_bytes;
    }
    return output;
  }

  StatusOr<std::vector<float>> Embedding(uint32_t token) {
    const auto* info = FindTensor(file->inspection(), "token_embd.weight");
    if (info == nullptr || info->dimensions.size() != 2U || token >= info->dimensions[1]) {
      return Status::InvalidArgument("token id is outside the GGUF embedding table");
    }
    const auto block = GgufQuantBlockInfoForType(info->type);
    const uint64_t width = info->dimensions[0];
    if (!block || width % block->elements != 0U) {
      return Status::InvalidArgument("GGUF embedding row layout is invalid");
    }
    const uint64_t blocks = width / block->elements;
    return file->ReadDecodedTensorBlocks("token_embd.weight", token * blocks, blocks);
  }

  StatusOr<std::vector<float>> Forward(uint32_t token, uint64_t position,
                                       const CancellationToken& cancellation) {
    const auto& inspection = file->inspection();
    const auto& config = inspection.gemma4;
    auto hidden_result = Embedding(token);
    if (!hidden_result.ok()) return hidden_result.status();
    std::vector<float> hidden = std::move(hidden_result).value();
    const float embedding_scale = std::sqrt(static_cast<float>(config.embedding_length));
    for (float& value : hidden) value *= embedding_scale;

    for (uint64_t layer = 0U; layer < config.block_count; ++layer) {
      if (cancellation.IsCancellationRequested()) return Status::Cancelled("generation cancelled");
      const std::string prefix = "blk." + std::to_string(layer) + ".";
      auto attn_weight = Small(prefix + "attn_norm.weight");
      if (!attn_weight.ok()) return attn_weight.status();
      auto normalized = Rms(hidden, config.rms_norm_epsilon, attn_weight.value());
      auto query = MatVec(prefix + "attn_q.weight", normalized);
      auto key = MatVec(prefix + "attn_k.weight", normalized);
      if (!query.ok()) return query.status();
      if (!key.ok()) return key.status();
      auto value = FindTensor(inspection, prefix + "attn_v.weight") != nullptr
          ? MatVec(prefix + "attn_v.weight", normalized) : key;
      if (!value.ok()) return value.status();
      const uint64_t query_heads = config.attention_head_count;
      const uint64_t head_size = query.value().size() / query_heads;
      const uint64_t key_heads = key.value().size() / head_size;
      auto q_norm = Small(prefix + "attn_q_norm.weight");
      auto k_norm = Small(prefix + "attn_k_norm.weight");
      if (!q_norm.ok()) return q_norm.status();
      if (!k_norm.ok()) return k_norm.status();
      for (uint64_t head = 0; head < query_heads; ++head) {
        std::vector<float> part(query.value().begin() + head * head_size,
                                query.value().begin() + (head + 1U) * head_size);
        part = Rms(part, config.rms_norm_epsilon, q_norm.value());
        std::copy(part.begin(), part.end(), query.value().begin() + head * head_size);
      }
      for (uint64_t head = 0; head < key_heads; ++head) {
        std::vector<float> kpart(key.value().begin() + head * head_size,
                                 key.value().begin() + (head + 1U) * head_size);
        std::vector<float> vpart(value.value().begin() + head * head_size,
                                 value.value().begin() + (head + 1U) * head_size);
        kpart = Rms(kpart, config.rms_norm_epsilon, k_norm.value());
        vpart = Rms(vpart, config.rms_norm_epsilon);
        std::copy(kpart.begin(), kpart.end(), key.value().begin() + head * head_size);
        std::copy(vpart.begin(), vpart.end(), value.value().begin() + head * head_size);
      }
      const bool sliding = layer < config.sliding_window_pattern.size()
          ? config.sliding_window_pattern[static_cast<std::size_t>(layer)] : true;
      const uint64_t rotated = sliding ? head_size : head_size / 4U;
      const double theta = sliding ? config.rope_frequency_base_swa : config.rope_frequency_base;
      ApplyRope(&query.value(), query_heads, head_size, rotated, position, theta);
      ApplyRope(&key.value(), key_heads, head_size, rotated, position, theta);

      LayerCache& layer_cache = cache[static_cast<std::size_t>(layer)];
      layer_cache.heads = key_heads;
      layer_cache.head_size = head_size;
      layer_cache.keys.push_back(std::move(key).value());
      layer_cache.values.push_back(std::move(value).value());
      std::size_t first = 0U;
      if (sliding && config.sliding_window > 0U &&
          layer_cache.keys.size() > config.sliding_window) {
        first = layer_cache.keys.size() - static_cast<std::size_t>(config.sliding_window);
      }
      std::vector<float> attended(query.value().size(), 0.0F);
      std::vector<float> scores(layer_cache.keys.size() - first);
      for (uint64_t qhead = 0; qhead < query_heads; ++qhead) {
        const uint64_t kvhead = qhead * key_heads / query_heads;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t at = first; at < layer_cache.keys.size(); ++at) {
          float score = 0.0F;
          for (uint64_t d = 0; d < head_size; ++d) {
            score += query.value()[qhead * head_size + d] *
                     layer_cache.keys[at][kvhead * head_size + d];
          }
          scores[at - first] = score;
          maximum = std::max(maximum, score);
        }
        double denominator = 0.0;
        for (float& score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (std::size_t at = first; at < layer_cache.values.size(); ++at) {
          const float probability = static_cast<float>(scores[at - first] / denominator);
          for (uint64_t d = 0; d < head_size; ++d) {
            attended[qhead * head_size + d] +=
                probability * layer_cache.values[at][kvhead * head_size + d];
          }
        }
      }
      auto projected = MatVec(prefix + "attn_output.weight", attended);
      if (!projected.ok()) return projected.status();
      auto post_attn = Small(prefix + "post_attention_norm.weight");
      if (!post_attn.ok()) return post_attn.status();
      auto attn_out = Rms(projected.value(), config.rms_norm_epsilon, post_attn.value());
      AddInPlace(&attn_out, hidden);

      auto ffn_norm = Small(prefix + "ffn_norm.weight");
      auto ffn_post_1 = Small(prefix + "post_ffw_norm_1.weight");
      auto ffn_pre_2 = Small(prefix + "pre_ffw_norm_2.weight");
      auto ffn_post_2 = Small(prefix + "post_ffw_norm_2.weight");
      if (!ffn_norm.ok() || !ffn_post_1.ok() || !ffn_pre_2.ok() || !ffn_post_2.ok()) {
        return Status::InvalidArgument("Gemma 4 MoE normalization tensors are incomplete");
      }
      auto shared_input = Rms(attn_out, config.rms_norm_epsilon, ffn_norm.value());
      auto shared_gate = MatVec(prefix + "ffn_gate.weight", shared_input);
      auto shared_up = MatVec(prefix + "ffn_up.weight", shared_input);
      if (!shared_gate.ok()) return shared_gate.status();
      if (!shared_up.ok()) return shared_up.status();
      for (std::size_t i = 0; i < shared_gate.value().size(); ++i) {
        shared_gate.value()[i] = Gelu(shared_gate.value()[i]) * shared_up.value()[i];
      }
      auto shared = MatVec(prefix + "ffn_down.weight", shared_gate.value());
      if (!shared.ok()) return shared.status();
      shared.value() = Rms(shared.value(), config.rms_norm_epsilon, ffn_post_1.value());

      auto expert_input = Rms(attn_out, config.rms_norm_epsilon, ffn_pre_2.value());
      auto router_input = Rms(attn_out, config.rms_norm_epsilon);
      auto router_scale = Small(prefix + "ffn_gate_inp.scale");
      if (!router_scale.ok()) return router_scale.status();
      const float router_normalizer = 1.0F / std::sqrt(static_cast<float>(config.embedding_length));
      for (std::size_t i = 0; i < router_input.size(); ++i) {
        router_input[i] *= router_normalizer * (*router_scale.value())[i];
      }
      auto router = MatVec(prefix + "ffn_gate_inp.weight", router_input);
      if (!router.ok()) return router.status();
      const float router_max = *std::max_element(router.value().begin(), router.value().end());
      double router_sum = 0.0;
      for (float& score : router.value()) {
        score = std::exp(score - router_max);
        router_sum += score;
      }
      for (float& score : router.value()) score = static_cast<float>(score / router_sum);
      std::vector<uint64_t> experts(router.value().size());
      std::iota(experts.begin(), experts.end(), 0U);
      std::partial_sort(experts.begin(), experts.begin() + config.expert_used_count,
                        experts.end(), [&router](uint64_t left, uint64_t right) {
                          return router.value()[left] > router.value()[right];
                        });
      auto expert_scales = Small(prefix + "ffn_down_exps.scale");
      if (!expert_scales.ok()) return expert_scales.status();
      std::vector<float> expert_output(hidden.size(), 0.0F);
      for (uint64_t selected = 0; selected < config.expert_used_count; ++selected) {
        const uint64_t expert = experts[static_cast<std::size_t>(selected)];
        auto gate_up = MatVec(prefix + "ffn_gate_up_exps.weight", expert_input, expert);
        if (!gate_up.ok()) return gate_up.status();
        const std::size_t width = gate_up.value().size() / 2U;
        std::vector<float> activated(width);
        for (std::size_t i = 0; i < width; ++i) {
          activated[i] = Gelu(gate_up.value()[i]) * gate_up.value()[i + width];
        }
        auto down = MatVec(prefix + "ffn_down_exps.weight", activated, expert);
        if (!down.ok()) return down.status();
        const float mixture = router.value()[expert] * (*expert_scales.value())[expert];
        AddInPlace(&expert_output, down.value(), mixture);
      }
      expert_output = Rms(expert_output, config.rms_norm_epsilon, ffn_post_2.value());
      AddInPlace(&shared.value(), expert_output);
      auto post_ffw = Small(prefix + "post_ffw_norm.weight");
      if (!post_ffw.ok()) return post_ffw.status();
      hidden = Rms(shared.value(), config.rms_norm_epsilon, post_ffw.value());
      AddInPlace(&hidden, attn_out);
      if (FindTensor(inspection, prefix + "layer_output_scale.weight") != nullptr) {
        auto scale = Small(prefix + "layer_output_scale.weight");
        if (!scale.ok()) return scale.status();
        for (float& item : hidden) item *= scale.value()->front();
      }
    }
    auto output_norm = Small("output_norm.weight");
    if (!output_norm.ok()) return output_norm.status();
    hidden = Rms(hidden, config.rms_norm_epsilon, output_norm.value());
    auto logits = MatVec("token_embd.weight", hidden);
    if (!logits.ok()) return logits.status();
    if (config.final_logit_softcapping > 0.0) {
      const float cap = static_cast<float>(config.final_logit_softcapping);
      for (float& value : logits.value()) value = std::tanh(value / cap) * cap;
    }
    return logits;
  }

  uint32_t Sample(std::vector<float> logits, const GenerationConfig& generation,
                  std::mt19937_64* random) const {
    if (generation.decoding_mode == DecodingMode::kGreedy ||
        generation.decoding_mode == DecodingMode::kBeamSearch ||
        generation.decoding_mode == DecodingMode::kDiverseBeamSearch) {
      return static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    const float temperature = generation.sampling.temperature.value_or(0.8F);
    for (float& value : logits) value /= temperature;
    std::vector<uint32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0U);
    const std::size_t keep = generation.sampling.top_k.has_value()
        ? std::min<std::size_t>(static_cast<std::size_t>(*generation.sampling.top_k), order.size())
        : order.size();
    if (keep < order.size()) {
      std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                        [&logits](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
      order.resize(keep);
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (const uint32_t id : order) maximum = std::max(maximum, logits[id]);
    std::vector<double> probabilities(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) probabilities[i] = std::exp(logits[order[i]] - maximum);
    if (generation.sampling.top_p.has_value()) {
      std::sort(order.begin(), order.end(), [&logits](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
      probabilities.resize(order.size());
      double total = 0.0;
      for (std::size_t i = 0; i < order.size(); ++i) { probabilities[i] = std::exp(logits[order[i]] - maximum); total += probabilities[i]; }
      double accumulated = 0.0;
      std::size_t count = 0U;
      for (; count < order.size(); ++count) {
        accumulated += probabilities[count] / total;
        if (accumulated >= *generation.sampling.top_p) { ++count; break; }
      }
      order.resize(std::max<std::size_t>(1U, count));
      probabilities.resize(order.size());
    }
    std::discrete_distribution<std::size_t> distribution(probabilities.begin(), probabilities.end());
    return order[distribution(*random)];
  }
};

GgufGenAiEngine::GgufGenAiEngine(GgufRuntimeLogHandler log_handler)
    : impl_(std::make_unique<Impl>(std::move(log_handler))) {}

GgufGenAiEngine::~GgufGenAiEngine() = default;

Status GgufGenAiEngine::ValidateModel(const ModelDescriptor& model) {
  if (model.format != ModelFormat::kGguf) {
    return Status::Unsupported("TensorRT native chat requires a GGUF model");
  }
  const auto inspection = GgufInspector::Inspect(model.path);
  if (!inspection.ok()) return inspection.status();
  return ValidateGemma4(inspection.value());
}

Status GgufGenAiEngine::LoadModel(const ModelDescriptor& model, int device_index) {
  if (device_index < 0) return Status::InvalidArgument("TensorRT device index cannot be negative");
  const auto validation = ValidateModel(model);
  if (!validation.ok()) return validation;
  std::lock_guard lock(mutex_);
  auto file = GgufModelFile::Open(model.path);
  if (!file.ok()) return file.status();
  auto tokenizer = GgufTokenizer::Create(file.value()->inspection().tokenizer);
  if (!tokenizer.ok()) return tokenizer.status();
  const auto devices = EnumerateDevices();
  if (!devices.ok()) return devices.status();
  if (static_cast<std::size_t>(device_index) >= devices.value().size()) {
    return Status::InvalidArgument("selected TensorRT GPU does not exist");
  }
  impl_->file = std::move(file).value();
  impl_->tokenizer = std::move(tokenizer).value();
  impl_->device = device_index;
  impl_->engines.clear();
  impl_->small_tensors.clear();
  impl_->cache.assign(static_cast<std::size_t>(impl_->file->inspection().gemma4.block_count), {});
  loaded_model_ = model;
  loaded_device_ = device_index;
  const auto& config = impl_->file->inspection().gemma4;
  std::ostringstream details;
  details << "TensorRT native GGUF model opened: " << model.display_name
          << "; layers=" << config.block_count
          << "; hidden=" << config.embedding_length
          << "; heads=" << config.attention_head_count
          << "; experts=" << config.expert_count
          << "; active_experts=" << config.expert_used_count
          << "; context=" << impl_->file->inspection().context_length
          << "; GPU." << device_index;
  impl_->Log(details.str());
  return Status();
}

Status GgufGenAiEngine::Generate(const UnifiedInferenceRequest& request,
                                 const CancellationToken& cancellation,
                                 const GgufInferenceEventHandler& handler) {
  const auto request_status = ValidateInferenceRequest(request);
  if (!request_status.ok()) return request_status;
  std::lock_guard lock(mutex_);
  if (!loaded_model_ || !impl_->file || !impl_->tokenizer) {
    return Status::Unavailable("no TensorRT GGUF model is loaded");
  }
  if (request.model_id != loaded_model_->id) {
    return Status::InvalidArgument("request model_id does not match the loaded model");
  }
  const auto emit = [&handler, &request](InferenceEventPayload payload) {
    if (handler) handler(InferenceEvent{request.request_id, std::move(payload)});
  };
  emit(InferenceStarted{loaded_model_->id});
  impl_->cache.assign(impl_->cache.size(), {});
  auto encoded = impl_->tokenizer->Encode(ChatPrompt(request.messages), true);
  if (!encoded.ok()) {
    emit(InferenceError{encoded.status()});
    return encoded.status();
  }
  if (encoded.value().empty()) return Status::InvalidArgument("tokenized prompt is empty");
  impl_->Log("TensorRT prompt tokens: " + std::to_string(encoded.value().size()));
  const auto started = std::chrono::steady_clock::now();
  impl_->profile = {};
  std::vector<float> logits;
  uint64_t position = 0U;
  for (const uint32_t token : encoded.value()) {
    impl_->Log("TensorRT prompt token " + std::to_string(position + 1U) + "/" +
               std::to_string(encoded.value().size()));
    auto result = impl_->Forward(token, position++, cancellation);
    if (!result.ok()) {
      if (result.status().code() == StatusCode::kCancelled) emit(InferenceCancelled{});
      else emit(InferenceError{result.status()});
      return result.status();
    }
    logits = std::move(result).value();
  }
  const int maximum = request.generation.max_tokens.value_or(128);
  std::mt19937_64 random(request.generation.sampling.seed.value_or(std::random_device{}()));
  std::vector<uint32_t> generated;
  std::string decoded;
  for (int index = 0; index < maximum; ++index) {
    if (cancellation.IsCancellationRequested()) {
      emit(InferenceCancelled{});
      return Status::Cancelled("generation cancelled");
    }
    impl_->Log("TensorRT generating token " + std::to_string(index + 1) + "/" +
               std::to_string(maximum));
    const uint32_t token = impl_->Sample(std::move(logits), request.generation, &random);
    if (token == impl_->tokenizer->eos_token_id()) break;
    generated.push_back(token);
    std::string current = impl_->tokenizer->Decode(generated);
    if (current.size() > decoded.size()) {
      const std::string piece = current.substr(decoded.size());
      decoded = std::move(current);
      if (!piece.empty()) emit(ContentDelta{piece});
    }
    bool stopped = false;
    for (const auto& stop : request.generation.stop_sequences) {
      if (!stop.empty() && decoded.ends_with(stop)) { stopped = true; break; }
    }
    if (stopped || index + 1 >= maximum) break;
    auto result = impl_->Forward(token, position++, cancellation);
    if (!result.ok()) {
      if (result.status().code() == StatusCode::kCancelled) emit(InferenceCancelled{});
      else emit(InferenceError{result.status()});
      return result.status();
    }
    logits = std::move(result).value();
    emit(UsageUpdated{static_cast<uint64_t>(encoded.value().size()),
                      static_cast<uint64_t>(generated.size())});
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  if (std::getenv("ISVIK_TENSORRT_PROFILE") != nullptr) {
    const double read_seconds = std::chrono::duration<double>(impl_->profile.weight_read).count();
    const double engine_seconds = std::chrono::duration<double>(impl_->profile.engine_lookup).count();
    const double inference_seconds = std::chrono::duration<double>(impl_->profile.inference).count();
    std::clog << std::fixed << std::setprecision(3)
              << "TensorRT GGUF profile: total=" << seconds << "s, matvec="
              << impl_->profile.calls << ", weights="
              << static_cast<double>(impl_->profile.weight_bytes) / (1024.0 * 1024.0)
              << " MiB, file-read=" << read_seconds << "s, engine=" << engine_seconds
              << "s, infer=" << inference_seconds << "s, other="
              << std::max(0.0, seconds - read_seconds - engine_seconds - inference_seconds)
              << "s\n";
  }
  emit(InferenceCompleted{static_cast<uint64_t>(encoded.value().size()),
                          static_cast<uint64_t>(generated.size()), seconds,
                          seconds > 0.0 ? generated.size() / seconds : 0.0});
  return Status();
}

std::optional<ModelDescriptor> GgufGenAiEngine::loaded_model() const {
  std::lock_guard lock(mutex_);
  return loaded_model_;
}

int GgufGenAiEngine::loaded_device() const {
  std::lock_guard lock(mutex_);
  return loaded_device_;
}

void GgufGenAiEngine::UnloadModel() {
  std::lock_guard lock(mutex_);
  impl_->engines.clear();
  impl_->small_tensors.clear();
  impl_->cache.clear();
  impl_->tokenizer.reset();
  impl_->file.reset();
  loaded_model_.reset();
  loaded_device_ = -1;
}

}  // namespace isvik::tensorrt_backend
