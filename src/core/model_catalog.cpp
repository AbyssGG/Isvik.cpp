#include "isvik/core/model_catalog.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#include <nlohmann/json.hpp>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include "isvik/core/gguf_inspector.h"
#include "isvik/core/model_manager.h"

namespace isvik {
namespace {
constexpr std::size_t kMaxEntries = 100000;
constexpr std::size_t kMaxModels = 4096;
constexpr std::uintmax_t kMaxCatalogBytes = 4 * 1024 * 1024;

std::string ModelId(const std::filesystem::path& path) {
  uint64_t hash = 14695981039346656037ULL;
  auto key = PathUtf8(path);
#ifdef _WIN32
  for (char& value : key) value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
#endif
  for (unsigned char value : key) {
    hash ^= value;
    hash *= 1099511628211ULL;
  }
  std::ostringstream result;
  result << "model-" << std::hex << hash;
  return result.str();
}

bool FileExists(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error);
}

bool IsGguf(const std::filesystem::path& path) {
  auto extension = PathUtf8(path.extension());
  for (char& value : extension) value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
  return extension == ".gguf";
}

bool IsOnnx(const std::filesystem::path& path) {
  auto extension = PathUtf8(path.extension());
  for (char& value : extension) value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
  return extension == ".onnx";
}

std::filesystem::path ModelMetadataDirectory(const std::filesystem::path& path) {
  auto directory = path.parent_path();
  auto name = PathUtf8(directory.filename());
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  if (IsOnnx(path) && name == "onnx" &&
      FileExists(directory.parent_path() / "config.json")) {
    directory = directory.parent_path();
  }
  return directory;
}

StatusOr<std::filesystem::path> FindOnnxEntry(const std::filesystem::path& directory) {
  std::vector<std::filesystem::path> candidates;
  std::error_code error;
  std::filesystem::recursive_directory_iterator iterator(
      directory, std::filesystem::directory_options::skip_permission_denied, error), end;
  if (error) return Status::Unavailable("Cannot scan model directory: " + error.message());
  std::size_t visited = 0;
  while (iterator != end && visited < kMaxEntries) {
    const auto file = *iterator;
    if (iterator.depth() >= 12 || file.is_symlink(error)) iterator.disable_recursion_pending();
    error.clear();
    if (IsOnnx(file.path()) && file.is_regular_file(error) && !file.is_symlink(error)) {
      candidates.push_back(file.path());
    }
    error.clear();
    iterator.increment(error);
    if (error) error.clear();
    ++visited;
  }
  if (candidates.empty()) return Status::NotFound("No ONNX model file in this directory");
  const auto preferred = std::find_if(candidates.begin(), candidates.end(), [](const auto& path) {
    const auto filename = PathUtf8(path.filename());
    return filename == "model.onnx" || filename == "model_q4f16.onnx";
  });
  if (preferred != candidates.end()) {
    auto canonical = std::filesystem::canonical(*preferred, error);
    if (error) return Status::Unavailable("Cannot resolve ONNX model: " + error.message());
    return canonical;
  }
  if (candidates.size() != 1U) {
    return Status::InvalidArgument("Multiple ONNX files found; select the model .onnx file directly");
  }
  auto canonical = std::filesystem::canonical(candidates.front(), error);
  if (error) return Status::Unavailable("Cannot resolve ONNX model: " + error.message());
  return canonical;
}

CatalogModel Inspect(const std::filesystem::path& path, std::string id,
                     std::string name) {
  CatalogModel entry;
  entry.model.id = std::move(id);
  entry.model.path = path;
  entry.model.format = IsGguf(path) ? ModelFormat::kGguf
      : IsOnnx(path) ? ModelFormat::kOnnx : ModelFormat::kOpenVinoIr;
  entry.model.display_name = std::move(name);
  ModelManager manager;
  const Status status = manager.ImportModel(entry.model.id, path, entry.model.display_name);
  if (!status.ok()) {
    entry.issue = status.message();
    return entry;
  }
  entry.model = manager.FindModel(entry.model.id).value();
  entry.available = true;
  if (entry.model.format == ModelFormat::kGguf) {
    // The bundled GenAI runtime has a native GGUF graph reader for these
    // architectures. Other topologies need an Isvik graph importer; recognizing
    // a GGUF header is not enough to claim that inference is supported.
    if (entry.model.architecture != "llama" && entry.model.architecture != "qwen2" &&
        entry.model.architecture != "qwen3") {
      entry.issue = "No OpenVINO GGUF graph importer for architecture: " +
                    (entry.model.architecture.empty() ? std::string("unknown")
                                                      : entry.model.architecture);
      return entry;
    }
    const auto inspection = GgufInspector::Inspect(path);
    if (!inspection.ok()) {
      entry.issue = inspection.status().message();
      return entry;
    }
    for (const auto& tensor : inspection.value().tensors) {
      // These encodings have matching paths in the bundled GenAI GGUF reader.
      // Other encodings (including Q5_0, IQ3_S and MXFP4) currently fail on load.
      if (tensor.type != 0 && tensor.type != 1 && tensor.type != 2 &&
          tensor.type != 3 && tensor.type != 8 && tensor.type != 12 &&
          tensor.type != 14) {
        entry.issue = "OpenVINO GGUF reader cannot decode " +
                      GgufTensorTypeName(tensor.type) + " tensor: " + tensor.name;
        return entry;
      }
    }
    entry.runnable = true;  // The runtime still validates the graph on load.
    return entry;
  }
  const auto directory = ModelMetadataDirectory(path);
  auto config_path = directory / "config.json";
  std::error_code error;
  if (FileExists(config_path) && std::filesystem::file_size(config_path, error) < 1048576) {
    std::ifstream input(config_path);
    const auto config = nlohmann::json::parse(input, nullptr, false);
    if (config.is_object()) {
      if (config.contains("model_type") && config["model_type"].is_string()) {
        entry.model.architecture = config["model_type"].get<std::string>();
      }
      const auto& text = config.contains("text_config") && config["text_config"].is_object()
          ? config["text_config"] : config;
      if (text.contains("max_position_embeddings") &&
          text["max_position_embeddings"].is_number_unsigned()) {
        entry.model.context_length = text["max_position_embeddings"].get<uint64_t>();
      } else if (config.contains("n_positions") && config["n_positions"].is_number_unsigned()) {
        entry.model.context_length = config["n_positions"].get<uint64_t>();
      }
    }
  }
  if (entry.model.format == ModelFormat::kOnnx) {
    const auto onnx_config_path = directory / "config.json";
    if (FileExists(onnx_config_path) &&
        std::filesystem::file_size(onnx_config_path, error) < 1048576) {
      std::ifstream input(onnx_config_path);
      const auto config = nlohmann::json::parse(input, nullptr, false);
      if (config.is_object() && config.contains("transformers.js_config") &&
          config["transformers.js_config"].is_object() &&
          config["transformers.js_config"].contains("dtype") &&
          config["transformers.js_config"]["dtype"].is_string()) {
        std::string dtype = config["transformers.js_config"]["dtype"].get<std::string>();
        std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char value) {
          return static_cast<char>(std::toupper(value));
        });
        entry.model.quantization = std::move(dtype);
      }
    }
    entry.issue = "ONNX model recognized; chat inference for this ONNX model is not implemented yet.";
    return entry;
  }
  if (entry.model.format == ModelFormat::kOpenVinoIr) {
    const auto quantization_path = directory / "openvino_config.json";
    if (FileExists(quantization_path) &&
        std::filesystem::file_size(quantization_path, error) < 1048576) {
      std::ifstream input(quantization_path);
      const auto metadata = nlohmann::json::parse(input, nullptr, false);
      if (metadata.is_object() && metadata.contains("dtype") && metadata["dtype"].is_string()) {
        std::string dtype = metadata["dtype"].get<std::string>();
        std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char value) {
          return static_cast<char>(std::toupper(value));
        });
        entry.model.quantization = dtype;
        if (metadata.contains("quantization_config") && metadata["quantization_config"].is_object()) {
          const auto& quantization = metadata["quantization_config"];
          if (quantization.contains("quantization_configs") &&
              quantization["quantization_configs"].is_object() &&
              quantization["quantization_configs"].contains("lm_model") &&
              quantization["quantization_configs"]["lm_model"].is_object()) {
            const auto& language_model = quantization["quantization_configs"]["lm_model"];
            if (language_model.contains("group_size") && language_model["group_size"].is_number_integer()) {
              const auto group_size = language_model["group_size"].get<int64_t>();
              if (group_size > 0) entry.model.quantization += " · G" + std::to_string(group_size);
            }
          }
        }
      }
    }
  }
  if (path.filename() == "openvino_language_model.xml") {
    // Count the weights used by a VLM, including its embedding modules.
    uint64_t size = 0;
    std::filesystem::directory_iterator files(directory, error);
    for (const auto& file : files) {
      const auto filename = PathUtf8(file.path().filename());
      if (filename.starts_with("openvino_") &&
          (file.path().extension() == ".xml" || file.path().extension() == ".bin")) {
        const auto bytes = file.file_size(error);
        if (!error && bytes <= UINT64_MAX - size) size += bytes;
        error.clear();
      }
    }
    if (size > 0) entry.model.file_size = size;
  }
  for (const char* filename : {"openvino_tokenizer.xml", "openvino_tokenizer.bin",
                              "openvino_detokenizer.xml", "openvino_detokenizer.bin"}) {
    if (!FileExists(directory / filename)) {
      entry.issue = std::string("Missing tokenizer component: ") + filename;
      return entry;
    }
  }
  if (path.filename() == "openvino_language_model.xml" &&
      !FileExists(directory / "openvino_text_embeddings_model.xml")) {
    entry.issue = "Missing VLM text embeddings";
    return entry;
  }
  entry.runnable = true;  // SDK/device compatibility is checked when loaded.
  return entry;
}

}  // namespace

std::string PathUtf8(const std::filesystem::path& path) {
  const auto text = path.u8string();
  return std::string(text.begin(), text.end());
}

std::filesystem::path PathFromUtf8(const std::string& text) {
  std::u8string utf8;
  utf8.reserve(text.size());
  for (const char value : text) utf8.push_back(static_cast<char8_t>(value));
  return std::filesystem::path(utf8);
}

StatusOr<std::filesystem::path> ResolveModelEntry(const std::filesystem::path& path) {
  std::error_code error;
  const bool is_directory = std::filesystem::is_directory(path, error);
  const auto directory = is_directory ? path : path.parent_path();
  const std::string name = PathUtf8(path.filename());
  if (is_directory || name.starts_with("openvino_")) {
    for (const char* filename : {"openvino_language_model.xml", "openvino_model.xml"}) {
      if (FileExists(directory / filename)) {
        auto entry = std::filesystem::canonical(directory / filename, error);
        if (error) return Status::Unavailable("Cannot resolve model entry: " + error.message());
        return entry;
      }
    }
    if (is_directory) {
      const auto onnx = FindOnnxEntry(directory);
      if (onnx.ok()) return onnx.value();
      if (onnx.status().code() != StatusCode::kNotFound) return onnx.status();
    }
    return Status::NotFound("No supported model entry in this directory");
  }
  if (IsGguf(path) && FileExists(path)) {
    auto canonical = std::filesystem::canonical(path, error);
    if (!error) return canonical;
  }
  if (IsOnnx(path) && FileExists(path)) {
    auto canonical = std::filesystem::canonical(path, error);
    if (!error) return canonical;
  }
  return Status::Unsupported("Select a model directory, an ONNX file, an OpenVINO XML file, or a GGUF file");
}

StatusOr<ModelScan> ScanModelDirectory(const std::filesystem::path& directory) {
  std::error_code error;
  if (!std::filesystem::is_directory(directory, error)) {
    return Status::NotFound("Model directory does not exist or cannot be read");
  }
  ModelScan result;
  result.directory = std::filesystem::canonical(directory, error);
  if (error) return Status::Unavailable(error.message());
  std::unordered_set<std::string> seen;
  std::filesystem::recursive_directory_iterator iterator(
      directory, std::filesystem::directory_options::skip_permission_denied, error), end;
  if (error) return Status::Unavailable(error.message());
  std::size_t visited = 0;
  while (iterator != end) {
    const auto file = *iterator;
    if (iterator.depth() >= 12 || file.is_symlink(error)) iterator.disable_recursion_pending();
    error.clear();
    const auto name = PathUtf8(file.path().filename());
    const bool ir = name == "openvino_model.xml" || name == "openvino_language_model.xml";
    const bool gguf = IsGguf(file.path());
    const bool onnx = IsOnnx(file.path());
    if ((ir || gguf || onnx) && file.is_regular_file(error) && !file.is_symlink(error)) {
      auto resolved = ResolveModelEntry(file.path());
      if (resolved.ok() && seen.insert(PathUtf8(resolved.value())).second) {
        const auto path = resolved.value();
        auto model = Inspect(path, ModelId(path),
            PathUtf8(gguf ? path.stem() : ModelMetadataDirectory(path).filename()));
        result.models.push_back(std::move(model));
      }
    }
    if (++visited >= kMaxEntries || result.models.size() >= kMaxModels) {
      result.warnings.push_back("Scan limit reached. Select a smaller directory.");
      break;
    }
    iterator.increment(error);
    if (error) {
      result.warnings.push_back("Some paths could not be read: " + error.message());
      error.clear();
    }
  }
  std::sort(result.models.begin(), result.models.end(), [](const auto& a, const auto& b) {
    return a.model.display_name < b.model.display_name;
  });
  return result;
}

Status ModelCatalog::Load() {
  std::error_code error;
  if (!std::filesystem::exists(storage_, error)) {
    return error ? Status::Unavailable(error.message()) : Status();
  }
  if (std::filesystem::file_size(storage_, error) > kMaxCatalogBytes || error) {
    return Status::InvalidArgument("Model catalog is too large or cannot be read");
  }
  std::ifstream input(storage_);
  const auto data = nlohmann::json::parse(input, nullptr, false);
  if (!data.is_object() || data.value("version", nlohmann::json()) != 1 ||
      !data.contains("models") || !data["models"].is_array()) {
    return Status::InvalidArgument("Invalid model catalog; the existing file was preserved");
  }
  try {
    std::vector<CatalogModel> models;
    std::unordered_set<std::string> ids, paths;
    if (data["models"].size() > kMaxModels) return Status::InvalidArgument("Too many models");
    for (const auto& item : data["models"]) {
      auto path = PathFromUtf8(item.at("path").get<std::string>());
      auto id = item.at("id").get<std::string>();
      auto name = item.at("name").get<std::string>();
      if (id.empty() || name.empty() || path.empty() || !path.is_absolute() ||
          !ids.insert(id).second || !paths.insert(PathUtf8(path)).second) {
        return Status::InvalidArgument("Invalid or duplicate model catalog entry");
      }
      models.push_back(Inspect(path, std::move(id), std::move(name)));
    }
    auto folder = PathFromUtf8(data.value("folder", std::string()));
    auto default_id = data.value("default", std::string());
    const std::string backend_name = data.value("backend", std::string("openvino"));
    BackendType backend = BackendType::kOpenVino;
    if (backend_name == "tensorrt") {
      backend = BackendType::kTensorRt;
    } else if (backend_name == "onnxruntime") {
      backend = BackendType::kOnnxRuntime;
    } else if (backend_name != "openvino") {
      return Status::InvalidArgument("Invalid backend in model catalog");
    }
    if (!default_id.empty() && !ids.contains(default_id)) default_id.clear();
    models_ = std::move(models);
    folder_ = std::move(folder);
    default_id_ = std::move(default_id);
    backend_ = backend;
  } catch (const nlohmann::json::exception&) {
    return Status::InvalidArgument("Invalid catalog fields; the existing file was preserved");
  }
  return Status();
}

Status ModelCatalog::Save() const {
  if (models_.size() > kMaxModels) return Status::InvalidArgument("Too many models in catalog");
  nlohmann::json data{{"version", 1}, {"folder", PathUtf8(folder_)},
                      {"default", default_id_},
                      {"backend", backend_ == BackendType::kTensorRt ? "tensorrt"
                          : backend_ == BackendType::kOnnxRuntime ? "onnxruntime" : "openvino"},
                      {"models", nlohmann::json::array()}};
  for (const auto& entry : models_) {
    data["models"].push_back({{"id", entry.model.id}, {"name", entry.model.display_name},
                              {"path", PathUtf8(entry.model.path)}});
  }
  const auto serialized = data.dump(2);
  if (serialized.size() > kMaxCatalogBytes) return Status::InvalidArgument("Catalog exceeds size limit");
  std::error_code error;
  if (!storage_.parent_path().empty()) std::filesystem::create_directories(storage_.parent_path(), error);
  if (error) return Status::Unavailable("Cannot create catalog directory: " + error.message());
  auto temporary = storage_;
  temporary += ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output << serialized << '\n';
    output.close();
    if (!output) return Status::Unavailable("Cannot write model catalog");
  }
#ifdef _WIN32
  if (!MoveFileExW(temporary.c_str(), storage_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return Status::Unavailable("Cannot replace model catalog: " + std::to_string(GetLastError()));
  }
#else
  std::filesystem::rename(temporary, storage_, error);
  if (error) return Status::Unavailable("Cannot replace model catalog: " + error.message());
#endif
  return Status();
}

void ModelCatalog::Merge(const ModelScan& scan) {
  if (!scan.directory.empty()) {
    for (auto& entry : models_) {
      const auto relative = entry.model.path.lexically_relative(scan.directory);
      if (!relative.empty() && *relative.begin() != "..") {
        entry = Inspect(entry.model.path, entry.model.id, entry.model.display_name);
      }
    }
  }
  for (const auto& entry : scan.models) {
    auto found = std::find_if(models_.begin(), models_.end(), [&](const auto& current) {
      std::error_code error;
      return current.model.path == entry.model.path || std::filesystem::equivalent(current.model.path, entry.model.path, error);
    });
    if (found == models_.end()) {
      models_.push_back(entry);
    } else {
      const auto id = found->model.id;
      const auto name = found->model.display_name;
      *found = entry;
      found->model.id = id;
      found->model.display_name = name;
    }
  }
}

const CatalogModel* ModelCatalog::Find(const std::string& id) const {
  const auto found = std::find_if(models_.begin(), models_.end(), [&](const auto& entry) {
    return entry.model.id == id;
  });
  return found == models_.end() ? nullptr : &*found;
}

Status ModelCatalog::Rename(const std::string& id, std::string name) {
  const auto start = name.find_first_not_of(" \t\r\n");
  const auto finish = name.find_last_not_of(" \t\r\n");
  if (start == std::string::npos || name.size() > 256) return Status::InvalidArgument("Model name must be 1–256 bytes");
  name = name.substr(start, finish - start + 1);
  for (auto& entry : models_) {
    if (entry.model.id == id) { entry.model.display_name = std::move(name); return Status(); }
  }
  return Status::NotFound("Model is not in the catalog");
}

Status ModelCatalog::Remove(const std::string& id) {
  const auto count = std::erase_if(models_, [&](const auto& entry) { return entry.model.id == id; });
  if (count == 0) return Status::NotFound("Model is not in the catalog");
  if (default_id_ == id) default_id_.clear();
  return Status();
}

Status ModelCatalog::SetDefault(const std::string& id) {
  if (!Find(id)) return Status::NotFound("Model is not in the catalog");
  default_id_ = id;
  return Status();
}

}  // namespace isvik
