#ifndef ISVIK_CORE_MODEL_MANAGER_H_
#define ISVIK_CORE_MODEL_MANAGER_H_

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "isvik/core/model_registry.h"

namespace isvik {

class ModelManager {
 public:
  // Imports a model selected by the local user and inspects its on-disk format.
  [[nodiscard]] Status ImportModel(std::string id,
                                   const std::filesystem::path& path,
                                   std::string display_name = {});
  [[nodiscard]] Status RemoveModel(const std::string& id);
  [[nodiscard]] Status SetDefaultModel(const std::string& id);
  [[nodiscard]] StatusOr<std::string> default_model_id() const;
  [[nodiscard]] StatusOr<ModelDescriptor> FindModel(const std::string& id) const;
  [[nodiscard]] std::vector<ModelDescriptor> ListModels() const;

 private:
  mutable std::mutex mutex_;
  ModelRegistry registry_;
  std::optional<std::string> default_model_id_;
};

}  // namespace isvik

#endif  // ISVIK_CORE_MODEL_MANAGER_H_
