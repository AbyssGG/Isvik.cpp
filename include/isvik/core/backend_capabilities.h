#ifndef ISVIK_CORE_BACKEND_CAPABILITIES_H_
#define ISVIK_CORE_BACKEND_CAPABILITIES_H_

namespace isvik {

enum class BackendType {
  kOpenVino,
  kTensorRt,
  kOnnxRuntime,
  kGgml,
};

struct BackendCapabilities {
  bool supports_cpu = false;
  bool supports_gpu = false;
  bool supports_npu = false;

  bool supports_greedy = false;
  bool supports_sampling = false;
  bool supports_beam_search = false;
  bool supports_speculative = false;

  bool supports_temperature = false;
  bool supports_top_p = false;
  bool supports_top_k = false;

  bool supports_presence_penalty = false;
  bool supports_frequency_penalty = false;
  bool supports_repetition_penalty = false;

  bool supports_fp16 = false;
  bool supports_bf16 = false;
  bool supports_int8 = false;
  bool supports_int4 = false;
};

}  // namespace isvik

#endif  // ISVIK_CORE_BACKEND_CAPABILITIES_H_
