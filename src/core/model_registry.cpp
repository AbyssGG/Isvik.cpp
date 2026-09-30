#include "isvik/core/model_registry.h"

#include <mutex>
#include <utility>

namespace isvik {

Status ModelRegistry::Register(ModelDescriptor descriptor) {
  if (descriptor.id.empty()) {
    return Status::InvalidArgument("model id cannot be empty");
  }
  if (descriptor.display_name.empty()) {
    return Status::InvalidArgument("model display name cannot be empty");
  }
  if (descriptor.path.empty()) {
    return Status::InvalidArgument("model path cannot be empty");
  }

  std::unique_lock lock(mutex_);
  const std::string model_id = descriptor.id;
  const bool inserted = models_.try_emplace(model_id, std::move(descriptor)).second;
  if (!inserted) {
    return Status::AlreadyExists("a model with this id is already registered");
  }
  return Status();
}

Status ModelRegistry::Remove(const std::string& id) {
  std::unique_lock lock(mutex_);
  if (models_.erase(id) == 0U) {
    return Status::NotFound("model id is not registered");
  }
  return Status();
}

StatusOr<ModelDescriptor> ModelRegistry::Find(const std::string& id) const {
  std::shared_lock lock(mutex_);
  const auto iter = models_.find(id);
  if (iter == models_.end()) {
    return Status::NotFound("model id is not registered");
  }
  return iter->second;
}

std::vector<ModelDescriptor> ModelRegistry::List() const {
  std::shared_lock lock(mutex_);
  std::vector<ModelDescriptor> result;
  result.reserve(models_.size());
  for (const auto& [id, descriptor] : models_) {
    static_cast<void>(id);
    result.push_back(descriptor);
  }
  return result;
}

std::size_t ModelRegistry::size() const {
  std::shared_lock lock(mutex_);
  return models_.size();
}

}  // namespace isvik
