#ifndef ISVIK_CORE_MODEL_REGISTRY_H_
#define ISVIK_CORE_MODEL_REGISTRY_H_

#include <cstddef>
#include <map>
#include <shared_mutex>
#include <string>
#include <vector>

#include "isvik/core/model_descriptor.h"
#include "isvik/core/status.h"

namespace isvik {

class ModelRegistry {
 public:
  Status Register(ModelDescriptor descriptor);
  Status Remove(const std::string& id);
  [[nodiscard]] StatusOr<ModelDescriptor> Find(const std::string& id) const;
  [[nodiscard]] std::vector<ModelDescriptor> List() const;
  [[nodiscard]] std::size_t size() const;

 private:
  mutable std::shared_mutex mutex_;
  std::map<std::string, ModelDescriptor> models_;
};

}  // namespace isvik

#endif  // ISVIK_CORE_MODEL_REGISTRY_H_
