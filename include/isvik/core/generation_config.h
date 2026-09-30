#ifndef ISVIK_CORE_GENERATION_CONFIG_H_
#define ISVIK_CORE_GENERATION_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "isvik/core/status.h"

namespace isvik {

enum class DecodingMode {
  kGreedy,
  kSampling,
  kBeamSearch,
  kDiverseBeamSearch,
  kSpeculative,
};

struct SamplingConfig {
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<int> top_k;
  std::optional<uint64_t> seed;
  std::optional<float> presence_penalty;
  std::optional<float> frequency_penalty;
  std::optional<float> repetition_penalty;
};

struct GenerationConfig {
  DecodingMode decoding_mode = DecodingMode::kSampling;
  std::optional<int> max_tokens;
  int num_beams = 4;
  int num_beam_groups = 2;
  float diversity_penalty = 0.5F;
  SamplingConfig sampling;
  std::vector<std::string> stop_sequences;
  bool stream = true;
};

Status ValidateGenerationConfig(const GenerationConfig& config);

}  // namespace isvik

#endif  // ISVIK_CORE_GENERATION_CONFIG_H_
