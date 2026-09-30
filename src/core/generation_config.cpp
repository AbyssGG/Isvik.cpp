#include "isvik/core/generation_config.h"

#include <cmath>
#include <initializer_list>
#include <string>

namespace isvik {
namespace {

bool IsFinite(const std::optional<float>& value) {
  return !value.has_value() || std::isfinite(*value);
}

}  // namespace

Status ValidateGenerationConfig(const GenerationConfig& config) {
  if (config.max_tokens.has_value() && *config.max_tokens <= 0) {
    return Status::InvalidArgument("max_tokens must be greater than zero");
  }
  if (config.sampling.top_k.has_value() && *config.sampling.top_k <= 0) {
    return Status::InvalidArgument("top_k must be greater than zero");
  }
  if (config.sampling.temperature.has_value() && *config.sampling.temperature <= 0.0F) {
    return Status::InvalidArgument("temperature must be greater than zero");
  }
  if (config.sampling.top_p.has_value()
      && (*config.sampling.top_p <= 0.0F || *config.sampling.top_p > 1.0F)) {
    return Status::InvalidArgument("top_p must be in the interval (0, 1]");
  }
  if ((config.decoding_mode == DecodingMode::kBeamSearch ||
       config.decoding_mode == DecodingMode::kDiverseBeamSearch) &&
      config.num_beams < 2) {
    return Status::InvalidArgument("beam search requires at least two beams");
  }
  if (config.decoding_mode == DecodingMode::kDiverseBeamSearch &&
      (config.num_beam_groups < 2 || config.num_beam_groups > config.num_beams ||
       config.num_beams % config.num_beam_groups != 0 || config.diversity_penalty < 0.0F)) {
    return Status::InvalidArgument(
        "diverse beam search requires compatible beam groups and a non-negative penalty");
  }

  const std::optional<float> diversity_penalty = config.diversity_penalty;
  const std::initializer_list<const std::optional<float>*> numeric_values = {
      &config.sampling.temperature,
      &config.sampling.top_p,
      &config.sampling.presence_penalty,
      &config.sampling.frequency_penalty,
      &config.sampling.repetition_penalty,
      &diversity_penalty,
  };
  for (const std::optional<float>* value : numeric_values) {
    if (!IsFinite(*value)) {
      return Status::InvalidArgument("generation parameters must be finite numbers");
    }
  }

  for (const std::string& stop_sequence : config.stop_sequences) {
    if (stop_sequence.empty()) {
      return Status::InvalidArgument("stop sequences cannot be empty");
    }
  }
  return Status();
}

}  // namespace isvik
