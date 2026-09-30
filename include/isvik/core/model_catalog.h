#ifndef ISVIK_CORE_MODEL_CATALOG_H_
#define ISVIK_CORE_MODEL_CATALOG_H_

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "isvik/core/model_descriptor.h"
#include "isvik/core/status.h"

namespace isvik {

// Catalog removal only removes a reference. Model files are never modified.
struct CatalogModel {
  ModelDescriptor model;
  bool available = false;
  bool runnable = false;
  std::string issue;
};

struct ModelScan {
  std::filesystem::path directory;
  std::vector<CatalogModel> models;
  std::vector<std::string> warnings;
};

[[nodiscard]] std::string PathUtf8(const std::filesystem::path& path);
[[nodiscard]] std::filesystem::path PathFromUtf8(const std::string& text);
[[nodiscard]] StatusOr<std::filesystem::path> ResolveModelEntry(
    const std::filesystem::path& path);
[[nodiscard]] StatusOr<ModelScan> ScanModelDirectory(
    const std::filesystem::path& directory);

class ModelCatalog {
 public:
  explicit ModelCatalog(std::filesystem::path storage) : storage_(std::move(storage)) {}
  [[nodiscard]] Status Load();
  [[nodiscard]] Status Save() const;
  void Merge(const ModelScan& scan);
  [[nodiscard]] Status Rename(const std::string& id, std::string name);
  [[nodiscard]] Status Remove(const std::string& id);
  [[nodiscard]] Status SetDefault(const std::string& id);
  [[nodiscard]] const CatalogModel* Find(const std::string& id) const;
  [[nodiscard]] const std::vector<CatalogModel>& models() const { return models_; }
  [[nodiscard]] const std::string& default_id() const { return default_id_; }
  [[nodiscard]] BackendType backend() const { return backend_; }
  void set_backend(BackendType backend) { backend_ = backend; }
  [[nodiscard]] const std::filesystem::path& folder() const { return folder_; }
  void set_folder(std::filesystem::path folder) { folder_ = std::move(folder); }

 private:
  std::filesystem::path storage_;
  std::filesystem::path folder_;
  std::string default_id_;
  BackendType backend_ = BackendType::kOpenVino;
  std::vector<CatalogModel> models_;
};

}  // namespace isvik
#endif
