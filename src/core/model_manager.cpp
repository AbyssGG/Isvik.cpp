#include "isvik/core/model_manager.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <system_error>
#include <string_view>
#include <utility>

#include "isvik/core/gguf_inspector.h"
#include "isvik/core/gguf_quant.h"

namespace isvik {
namespace {

bool IsValidModelId(const std::string& id) {
  if (id.empty() || id.size() > 64U) return false;
  const auto is_alphanumeric = [](unsigned char value) {
    return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
           (value >= '0' && value <= '9');
  };
  if (!is_alphanumeric(static_cast<unsigned char>(id.front()))) return false;
  return std::all_of(id.begin(), id.end(), [&](char character) {
    const unsigned char value = static_cast<unsigned char>(character);
    return is_alphanumeric(value) || value == '.' || value == '_' || value == '-';
  });
}

std::string LowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return value;
}

std::string Utf8PathComponent(const std::filesystem::path& path) {
  const auto encoded = path.u8string();
  return {encoded.begin(), encoded.end()};
}

std::string ParameterLabel(std::string_view name) {
  static const std::regex pattern(
      R"((?:^|[^A-Za-z0-9])((?:E)?[0-9]+(?:\.[0-9]+)?B(?:-A[0-9]+(?:\.[0-9]+)?B)?)(?:$|[^A-Za-z0-9]))",
      std::regex_constants::icase);
  std::match_results<std::string_view::const_iterator> match;
  if (std::regex_search(name.begin(), name.end(), match, pattern)) {
    std::string label(match[1].first, match[1].second);
    for (char& character : label) {
      if (character == 'b') character = 'B';
      if (character == 'a') character = 'A';
      if (character == 'e') character = 'E';
    }
    return label;
  }
  return {};
}

std::string QuantizationLabel(std::string_view name) {
  static const std::regex pattern(
      R"((?:^|[^A-Za-z0-9])((?:UD-)?(?:IQ|Q)[0-9][A-Za-z0-9_]*)(?:$|[^A-Za-z0-9]))",
      std::regex_constants::icase);
  std::match_results<std::string_view::const_iterator> match;
  if (!std::regex_search(name.begin(), name.end(), match, pattern)) return {};
  std::string label(match[1].first, match[1].second);
  std::transform(label.begin(), label.end(), label.begin(), [](unsigned char character) {
    return static_cast<char>(std::toupper(character));
  });
  return label;
}

std::string OpenVinoPrecisionLabel(std::string_view name) {
  static const std::regex pattern(
      R"((?:^|[^A-Za-z0-9])(INT[248]|FP(?:16|32)|BF16)(?:$|[^A-Za-z0-9]))",
      std::regex_constants::icase);
  std::match_results<std::string_view::const_iterator> match;
  if (!std::regex_search(name.begin(), name.end(), match, pattern)) return {};
  std::string label(match[1].first, match[1].second);
  std::transform(label.begin(), label.end(), label.begin(), [](unsigned char character) {
    return static_cast<char>(std::toupper(character));
  });
  return label;
}

std::string FormatParameterCount(uint64_t count) {
  if (count == 0U) return {};
  const double trillions = static_cast<double>(count) / 1'000'000'000'000.0;
  const double billions = static_cast<double>(count) / 1'000'000'000.0;
  const double millions = static_cast<double>(count) / 1'000'000.0;
  const double value = count >= 1'000'000'000'000ULL ? trillions
      : count >= 1'000'000'000ULL ? billions : millions;
  std::ostringstream label;
  label << std::fixed << std::setprecision(value >= 10.0 ? 0 : 1) << value
        << (count >= 1'000'000'000'000ULL ? "T" : count >= 1'000'000'000ULL ? "B" : "M");
  return label.str();
}

std::string ModelVariant(std::string_view name) {
  std::string token;
  const auto finish = [&token]() {
    std::string result;
    if (token == "it" || token == "instruct") result = "it";
    else if (token == "chat") result = "chat";
    else if (token == "base") result = "base";
    token.clear();
    return result;
  };
  for (const char character : name) {
    if (character == '-' || character == '_' || character == '.' || character == ' ') {
      const std::string result = finish();
      if (!result.empty()) return result;
    } else {
      token.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
  }
  return finish();
}

StatusOr<std::filesystem::path> CanonicalRegularFile(
    const std::filesystem::path& path) {
  if (path.empty()) return Status::InvalidArgument("model path cannot be empty");
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    if (error && error != std::errc::no_such_file_or_directory) {
      return Status::Unavailable("cannot inspect model path: " + error.message());
    }
    return Status::NotFound("model file does not exist");
  }
  std::filesystem::path canonical_path = std::filesystem::canonical(path, error);
  if (error) return Status::Unavailable("cannot resolve model path: " + error.message());
  return canonical_path;
}

StatusOr<uint64_t> FileSize(const std::filesystem::path& path) {
  std::error_code error;
  const uint64_t size = std::filesystem::file_size(path, error);
  if (error) return Status::Unavailable("cannot read model file size: " + error.message());
  return size;
}

StatusOr<uint64_t> OnnxModelFileSize(const std::filesystem::path& path) {
  StatusOr<uint64_t> model_size = FileSize(path);
  if (!model_size.ok()) return model_size.status();
  uint64_t total = model_size.value();
  const std::string prefix = Utf8PathComponent(path.filename()) + "_data";
  std::error_code error;
  std::filesystem::directory_iterator files(path.parent_path(), error), end;
  if (error) return Status::Unavailable("cannot scan ONNX external weights: " + error.message());
  while (files != end) {
    const auto& candidate = *files;
    const std::string name = Utf8PathComponent(candidate.path().filename());
    bool shard_name = name == prefix;
    if (!shard_name && name.starts_with(prefix + "_")) {
      const std::string suffix = name.substr(prefix.size() + 1U);
      shard_name = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), [](unsigned char value) {
        return value >= '0' && value <= '9';
      });
    }
    if (shard_name && candidate.is_regular_file(error)) {
      const StatusOr<uint64_t> shard_size = FileSize(candidate.path());
      if (!shard_size.ok()) return shard_size.status();
      if (shard_size.value() > std::numeric_limits<uint64_t>::max() - total) {
        return Status::InvalidArgument("ONNX model size overflows");
      }
      total += shard_size.value();
    }
    error.clear();
    files.increment(error);
    if (error) return Status::Unavailable("cannot scan ONNX external weights: " + error.message());
  }
  return total;
}

}  // namespace

Status ModelManager::ImportModel(std::string id, const std::filesystem::path& path,
                                 std::string display_name) {
  if (!IsValidModelId(id)) {
    return Status::InvalidArgument(
        "model id must start with a letter or digit and contain only letters, digits, '.', '_' or '-'");
  }
  StatusOr<std::filesystem::path> canonical_path = CanonicalRegularFile(path);
  if (!canonical_path.ok()) return canonical_path.status();

  ModelDescriptor descriptor;
  descriptor.id = std::move(id);
  descriptor.path = std::move(canonical_path).value();
  descriptor.display_name = display_name.empty()
      ? descriptor.path.stem().string()
      : std::move(display_name);
  if (descriptor.display_name.empty()) {
    return Status::InvalidArgument("model display name cannot be empty");
  }
  const std::string extension = LowerAscii(descriptor.path.extension().string());
  std::string source_name;
  if (extension == ".gguf") {
    source_name = Utf8PathComponent(descriptor.path.stem());
  } else if (extension == ".onnx") {
    source_name = Utf8PathComponent(descriptor.path.stem()) + " " +
        Utf8PathComponent(descriptor.path.parent_path().filename());
    if (LowerAscii(descriptor.path.parent_path().filename().string()) == "onnx") {
      source_name += " " + Utf8PathComponent(descriptor.path.parent_path().parent_path().filename());
    }
  } else {
    source_name = Utf8PathComponent(descriptor.path.parent_path().filename());
  }
  descriptor.parameter_label = ParameterLabel(source_name);
  descriptor.variant = ModelVariant(source_name);
  if (extension == ".gguf") {
    StatusOr<GgufInspection> inspection = GgufInspector::Inspect(descriptor.path);
    if (!inspection.ok()) return inspection.status();
    descriptor.format = ModelFormat::kGguf;
    descriptor.architecture = inspection.value().architecture;
    if (descriptor.parameter_label.empty()) {
      descriptor.parameter_label = ParameterLabel(inspection.value().name);
    }
    if (descriptor.parameter_label.empty()) {
      descriptor.parameter_label = FormatParameterCount(inspection.value().parameter_count);
    }
    if (descriptor.variant.empty()) descriptor.variant = ModelVariant(inspection.value().name);
    descriptor.quantization = inspection.value().quantization;
    const std::string filename_quantization = QuantizationLabel(source_name);
    if (!filename_quantization.empty()) descriptor.quantization = filename_quantization;
    bool all_f16 = !inspection.value().tensors.empty();
    for (const auto& tensor : inspection.value().tensors) {
      if (tensor.type != 1U) {
        all_f16 = false;
        break;
      }
    }
    if (all_f16) descriptor.quantization = "F16";
    descriptor.context_length = inspection.value().context_length;
    descriptor.file_size = inspection.value().file_size;
    bool native_tensorrt_compatible = descriptor.architecture == "gemma4" &&
        !inspection.value().tokenizer.tokens.empty() &&
        !inspection.value().tokenizer.merges.empty();
    for (const auto& tensor : inspection.value().tensors) {
      native_tensorrt_compatible = native_tensorrt_compatible &&
          IsGgufTensorEncodingDecodable(tensor.type);
    }
    if (native_tensorrt_compatible) {
      descriptor.compatible_backends.push_back(BackendType::kTensorRt);
    }
  } else if (extension == ".xml") {
    descriptor.quantization = OpenVinoPrecisionLabel(source_name);
    std::filesystem::path weights_path = descriptor.path;
    weights_path.replace_extension(".bin");
    StatusOr<std::filesystem::path> canonical_weights = CanonicalRegularFile(weights_path);
    if (!canonical_weights.ok()) {
      if (canonical_weights.status().code() == StatusCode::kNotFound) {
        return Status::InvalidArgument("OpenVINO IR is missing its sibling .bin weights file");
      }
      return canonical_weights.status();
    }
    StatusOr<uint64_t> xml_size = FileSize(descriptor.path);
    StatusOr<uint64_t> weights_size = FileSize(canonical_weights.value());
    if (!xml_size.ok()) return xml_size.status();
    if (!weights_size.ok()) return weights_size.status();
    if (xml_size.value() > std::numeric_limits<uint64_t>::max() - weights_size.value()) {
      return Status::InvalidArgument("OpenVINO IR model size overflows");
    }
    descriptor.format = ModelFormat::kOpenVinoIr;
    descriptor.file_size = xml_size.value() + weights_size.value();
    descriptor.compatible_backends = {BackendType::kOpenVino};
  } else if (extension == ".onnx") {
    const StatusOr<uint64_t> model_size = OnnxModelFileSize(descriptor.path);
    if (!model_size.ok()) return model_size.status();
    descriptor.format = ModelFormat::kOnnx;
    descriptor.file_size = model_size.value();
    const std::string filename_quantization = QuantizationLabel(source_name);
    if (!filename_quantization.empty()) descriptor.quantization = filename_quantization;
  } else {
    return Status::Unsupported("model format must be ONNX, OpenVINO .xml, or GGUF .gguf");
  }

  std::lock_guard lock(mutex_);
  return registry_.Register(std::move(descriptor));
}

Status ModelManager::RemoveModel(const std::string& id) {
  std::lock_guard lock(mutex_);
  Status status = registry_.Remove(id);
  if (status.ok() && default_model_id_.has_value() && default_model_id_.value() == id) {
    default_model_id_.reset();
  }
  return status;
}

Status ModelManager::SetDefaultModel(const std::string& id) {
  std::lock_guard lock(mutex_);
  StatusOr<ModelDescriptor> model = registry_.Find(id);
  if (!model.ok()) return model.status();
  default_model_id_ = id;
  return Status();
}

StatusOr<std::string> ModelManager::default_model_id() const {
  std::lock_guard lock(mutex_);
  if (!default_model_id_.has_value()) {
    return Status::NotFound("no default model is selected");
  }
  return default_model_id_.value();
}

StatusOr<ModelDescriptor> ModelManager::FindModel(const std::string& id) const {
  std::lock_guard lock(mutex_);
  return registry_.Find(id);
}

std::vector<ModelDescriptor> ModelManager::ListModels() const {
  std::lock_guard lock(mutex_);
  return registry_.List();
}

}  // namespace isvik
