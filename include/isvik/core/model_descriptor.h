#ifndef ISVIK_CORE_MODEL_DESCRIPTOR_H_
#define ISVIK_CORE_MODEL_DESCRIPTOR_H_

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "isvik/core/backend_capabilities.h"

namespace isvik {

enum class ModelFormat {
  kOpenVinoIr,
  kGguf,
  kOnnx,
};

struct ModelDescriptor {
  std::string id;
  std::string display_name;
  ModelFormat format = ModelFormat::kOpenVinoIr;
  std::filesystem::path path;
  std::string architecture;
  std::string parameter_label;
  std::string quantization;
  std::string variant;
  uint64_t context_length = 0;
  uint64_t file_size = 0;
  std::vector<BackendType> compatible_backends;
};

}  // namespace isvik

#endif  // ISVIK_CORE_MODEL_DESCRIPTOR_H_
