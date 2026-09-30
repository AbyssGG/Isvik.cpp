#include "isvik/core/gguf_inspector.h"
#include "isvik/core/gguf_quant.h"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <string_view>
#include <system_error>

namespace isvik {
namespace {

constexpr uint32_t kMinimumGgufVersion = 2;
constexpr uint32_t kMaximumGgufVersion = 3;
constexpr uint64_t kMaximumMetadataEntries = 1'000'000;
constexpr uint64_t kMaximumMetadataArrayElements = 10'000'000;
constexpr uint32_t kMaximumMetadataArrayDepth = 16;
constexpr uint64_t kMaximumRetainedStringBytes = 4096;
constexpr uint32_t kMaximumTensorDimensions = 4;

enum class MetadataType : uint32_t {
  kUint8 = 0,
  kInt8 = 1,
  kUint16 = 2,
  kInt16 = 3,
  kUint32 = 4,
  kInt32 = 5,
  kFloat32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kUint64 = 10,
  kInt64 = 11,
  kFloat64 = 12,
};

Status Malformed(std::string message) {
  return Status::InvalidArgument("invalid GGUF file: " + std::move(message));
}

class GgufReader {
 public:
  GgufReader(std::ifstream& stream, uint64_t file_size)
      : stream_(stream), file_size_(file_size) {}

  [[nodiscard]] uint64_t position() const { return position_; }
  [[nodiscard]] uint64_t remaining() const { return file_size_ - position_; }

  Status ReadBytes(char* destination, uint64_t size) {
    if (size > remaining()) {
      return Malformed("unexpected end of GGUF file");
    }
    if (size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
      return Malformed("GGUF field is too large to read");
    }
    stream_.read(destination, static_cast<std::streamsize>(size));
    if (!stream_) {
      return Malformed("failed while reading GGUF data");
    }
    position_ += size;
    return Status();
  }

  Status Skip(uint64_t size) {
    if (size > remaining()) {
      return Malformed("GGUF field extends beyond the file");
    }
    if (size > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
      return Malformed("GGUF field offset is too large");
    }
    stream_.seekg(static_cast<std::streamoff>(size), std::ios::cur);
    if (!stream_) {
      return Malformed("failed while skipping GGUF data");
    }
    position_ += size;
    return Status();
  }

  StatusOr<uint8_t> ReadU8() {
    std::array<unsigned char, 1> bytes{};
    Status status = ReadBytes(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!status.ok()) return status;
    return static_cast<uint8_t>(bytes[0]);
  }

  StatusOr<uint32_t> ReadU32() {
    std::array<unsigned char, 4> bytes{};
    Status status = ReadBytes(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!status.ok()) return status;
    uint32_t value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
      value |= static_cast<uint32_t>(bytes[index]) << (index * 8U);
    }
    return value;
  }

  StatusOr<uint64_t> ReadU64() {
    std::array<unsigned char, 8> bytes{};
    Status status = ReadBytes(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!status.ok()) return status;
    uint64_t value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
      value |= static_cast<uint64_t>(bytes[index]) << (index * 8U);
    }
    return value;
  }

  StatusOr<std::string> ReadString(uint64_t maximum_size, bool retain) {
    StatusOr<uint64_t> length = ReadU64();
    if (!length.ok()) return length.status();
    if (length.value() > maximum_size || length.value() > remaining()) {
      return Malformed("GGUF string length is invalid or exceeds its limit");
    }
    if (!retain) {
      Status status = Skip(length.value());
      if (!status.ok()) return status;
      return std::string();
    }
    std::string value(static_cast<std::size_t>(length.value()), '\0');
    Status status = ReadBytes(value.data(), length.value());
    if (!status.ok()) return status;
    return value;
  }

 private:
  std::ifstream& stream_;
  uint64_t file_size_;
  uint64_t position_ = 0;
};

struct MetadataValue {
  std::optional<std::string> string_value;
  std::optional<uint64_t> unsigned_value;
  std::optional<int64_t> signed_value;
  std::optional<double> floating_value;
  std::optional<bool> bool_value;
  std::vector<MetadataValue> array_values;
};

Status ReadMetadataValue(GgufReader& reader, uint32_t raw_type, bool retain,
                         MetadataValue* output, uint32_t depth = 0) {
  if (depth > kMaximumMetadataArrayDepth) {
    return Malformed("metadata array nesting exceeds its limit");
  }
  const MetadataType type = static_cast<MetadataType>(raw_type);
  switch (type) {
    case MetadataType::kUint8: {
      StatusOr<uint8_t> value = reader.ReadU8();
      if (!value.ok()) return value.status();
      if (retain) output->unsigned_value = value.value();
      return Status();
    }
    case MetadataType::kInt8: {
      StatusOr<uint8_t> value = reader.ReadU8();
      if (!value.ok()) return value.status();
      if (retain) output->signed_value = std::bit_cast<int8_t>(value.value());
      return Status();
    }
    case MetadataType::kUint16:
    case MetadataType::kInt16: {
      std::array<unsigned char, 2> bytes{};
      Status status = reader.ReadBytes(reinterpret_cast<char*>(bytes.data()), bytes.size());
      if (!status.ok()) return status;
      const uint16_t value = static_cast<uint16_t>(bytes[0]) |
          static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8U);
      if (retain && type == MetadataType::kUint16) output->unsigned_value = value;
      if (retain && type == MetadataType::kInt16) {
        output->signed_value = std::bit_cast<int16_t>(value);
      }
      return Status();
    }
    case MetadataType::kUint32:
    case MetadataType::kInt32:
    case MetadataType::kFloat32: {
      StatusOr<uint32_t> value = reader.ReadU32();
      if (!value.ok()) return value.status();
      if (retain && type == MetadataType::kUint32) output->unsigned_value = value.value();
      if (retain && type == MetadataType::kInt32) {
        output->signed_value = std::bit_cast<int32_t>(value.value());
      }
      if (retain && type == MetadataType::kFloat32) {
        output->floating_value = std::bit_cast<float>(value.value());
      }
      return Status();
    }
    case MetadataType::kBool: {
      StatusOr<uint8_t> value = reader.ReadU8();
      if (!value.ok()) return value.status();
      if (value.value() > 1U) return Malformed("boolean metadata must be 0 or 1");
      if (retain) output->bool_value = value.value() != 0U;
      return Status();
    }
    case MetadataType::kString: {
      StatusOr<std::string> value = reader.ReadString(
          retain ? kMaximumRetainedStringBytes : std::numeric_limits<uint64_t>::max(),
          retain);
      if (!value.ok()) return value.status();
      if (retain) output->string_value = std::move(value).value();
      return Status();
    }
    case MetadataType::kArray: {
      StatusOr<uint32_t> element_type = reader.ReadU32();
      if (!element_type.ok()) return element_type.status();
      StatusOr<uint64_t> count = reader.ReadU64();
      if (!count.ok()) return count.status();
      if (count.value() > kMaximumMetadataArrayElements ||
          count.value() > reader.remaining()) {
        return Malformed("metadata array length exceeds its limit");
      }
      if (retain) output->array_values.reserve(static_cast<std::size_t>(count.value()));
      for (uint64_t index = 0; index < count.value(); ++index) {
        MetadataValue item;
        Status status = ReadMetadataValue(reader, element_type.value(), retain,
                                          &item, depth + 1);
        if (!status.ok()) return status;
        if (retain) output->array_values.push_back(std::move(item));
      }
      return Status();
    }
    case MetadataType::kUint64:
    case MetadataType::kInt64:
    case MetadataType::kFloat64: {
      StatusOr<uint64_t> value = reader.ReadU64();
      if (!value.ok()) return value.status();
      if (retain && type == MetadataType::kUint64) output->unsigned_value = value.value();
      if (retain && type == MetadataType::kInt64) {
        output->signed_value = std::bit_cast<int64_t>(value.value());
      }
      if (retain && type == MetadataType::kFloat64) {
        output->floating_value = std::bit_cast<double>(value.value());
      }
      return Status();
    }
  }
  return Malformed("unknown metadata value type " + std::to_string(raw_type));
}

bool IsKey(std::string_view actual, std::string_view expected) {
  return actual == expected;
}

std::optional<uint64_t> UnsignedMetadata(const MetadataValue& value) {
  if (value.unsigned_value.has_value()) return value.unsigned_value;
  if (value.signed_value.has_value() && value.signed_value.value() >= 0) {
    return static_cast<uint64_t>(value.signed_value.value());
  }
  return std::nullopt;
}

std::vector<uint64_t> UnsignedMetadataArray(const MetadataValue& value) {
  std::vector<uint64_t> result;
  result.reserve(value.array_values.size());
  for (const auto& item : value.array_values) {
    const auto parsed = UnsignedMetadata(item);
    if (!parsed.has_value()) return {};
    result.push_back(parsed.value());
  }
  return result;
}

std::vector<bool> BooleanMetadataArray(const MetadataValue& value) {
  std::vector<bool> result;
  result.reserve(value.array_values.size());
  for (const auto& item : value.array_values) {
    if (!item.bool_value.has_value()) return {};
    result.push_back(item.bool_value.value());
  }
  return result;
}

std::vector<std::string> StringMetadataArray(const MetadataValue& value) {
  std::vector<std::string> result;
  result.reserve(value.array_values.size());
  for (const auto& item : value.array_values) {
    if (!item.string_value.has_value()) return {};
    result.push_back(item.string_value.value());
  }
  return result;
}

std::vector<int32_t> SignedMetadataArray(const MetadataValue& value) {
  std::vector<int32_t> result;
  result.reserve(value.array_values.size());
  for (const auto& item : value.array_values) {
    const auto parsed = UnsignedMetadata(item);
    if (parsed.has_value() && parsed.value() <= static_cast<uint64_t>(INT32_MAX)) {
      result.push_back(static_cast<int32_t>(parsed.value()));
    } else if (item.signed_value.has_value() && item.signed_value.value() >= INT32_MIN &&
               item.signed_value.value() <= INT32_MAX) {
      result.push_back(static_cast<int32_t>(item.signed_value.value()));
    } else {
      return {};
    }
  }
  return result;
}

double FloatingMetadata(const MetadataValue& value) {
  if (value.floating_value.has_value()) return value.floating_value.value();
  if (value.unsigned_value.has_value()) return static_cast<double>(value.unsigned_value.value());
  if (value.signed_value.has_value()) return static_cast<double>(value.signed_value.value());
  return 0.0;
}

void ReadGemma4Parameter(std::string_view key, const MetadataValue& value,
                         GgufGemma4Config* config) {
  const auto unsigned_value = UnsignedMetadata(value);
  if (key == "gemma4.block_count" && unsigned_value) config->block_count = *unsigned_value;
  else if (key == "gemma4.embedding_length" && unsigned_value) config->embedding_length = *unsigned_value;
  else if (key == "gemma4.feed_forward_length" && unsigned_value) config->feed_forward_length = *unsigned_value;
  else if (key == "gemma4.attention.head_count" && unsigned_value) config->attention_head_count = *unsigned_value;
  else if (key == "gemma4.attention.head_count_kv") config->attention_head_count_kv = UnsignedMetadataArray(value);
  else if (key == "gemma4.expert_count" && unsigned_value) config->expert_count = *unsigned_value;
  else if (key == "gemma4.expert_used_count" && unsigned_value) config->expert_used_count = *unsigned_value;
  else if (key == "gemma4.attention.key_length" && unsigned_value) config->attention_key_length = *unsigned_value;
  else if (key == "gemma4.attention.value_length" && unsigned_value) config->attention_value_length = *unsigned_value;
  else if (key == "gemma4.attention.sliding_window" && unsigned_value) config->sliding_window = *unsigned_value;
  else if (key == "gemma4.attention.sliding_window_pattern") config->sliding_window_pattern = BooleanMetadataArray(value);
  else if (key == "gemma4.attention.shared_kv_layers" && unsigned_value) config->shared_kv_layers = *unsigned_value;
  else if (key == "gemma4.embedding_length_per_layer_input" && unsigned_value) config->embedding_length_per_layer_input = *unsigned_value;
  else if (key == "gemma4.attention.key_length_swa" && unsigned_value) config->attention_key_length_swa = *unsigned_value;
  else if (key == "gemma4.attention.value_length_swa" && unsigned_value) config->attention_value_length_swa = *unsigned_value;
  else if (key == "gemma4.expert_feed_forward_length" && unsigned_value) config->expert_feed_forward_length = *unsigned_value;
  else if (key == "gemma4.rope.freq_base") config->rope_frequency_base = FloatingMetadata(value);
  else if (key == "gemma4.rope.freq_base_swa") config->rope_frequency_base_swa = FloatingMetadata(value);
  else if (key == "gemma4.attention.layer_norm_rms_epsilon") config->rms_norm_epsilon = FloatingMetadata(value);
  else if (key == "gemma4.final_logit_softcapping") config->final_logit_softcapping = FloatingMetadata(value);
}

void ReadTokenizerParameter(std::string_view key, const MetadataValue& value,
                            GgufTokenizerConfig* config) {
  const auto unsigned_value = UnsignedMetadata(value);
  if (key == "tokenizer.ggml.model" && value.string_value) {
    config->model = *value.string_value;
  } else if (key == "tokenizer.ggml.pre" && value.string_value) {
    config->pre_tokenizer = *value.string_value;
  } else if (key == "tokenizer.ggml.tokens") {
    config->tokens = StringMetadataArray(value);
  } else if (key == "tokenizer.ggml.merges") {
    config->merges = StringMetadataArray(value);
  } else if (key == "tokenizer.ggml.token_type") {
    config->token_types = SignedMetadataArray(value);
  } else if (key == "tokenizer.ggml.unknown_token_id" && unsigned_value &&
             *unsigned_value <= UINT32_MAX) {
    config->unknown_token_id = static_cast<uint32_t>(*unsigned_value);
  } else if (key == "tokenizer.ggml.bos_token_id" && unsigned_value &&
             *unsigned_value <= UINT32_MAX) {
    config->bos_token_id = static_cast<uint32_t>(*unsigned_value);
  } else if (key == "tokenizer.ggml.eos_token_id" && unsigned_value &&
             *unsigned_value <= UINT32_MAX) {
    config->eos_token_id = static_cast<uint32_t>(*unsigned_value);
  } else if (key == "tokenizer.ggml.padding_token_id" && unsigned_value &&
             *unsigned_value <= UINT32_MAX) {
    config->padding_token_id = static_cast<uint32_t>(*unsigned_value);
  }
}

std::string QuantizationName(uint64_t file_type) {
  switch (file_type) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "Q4_0";
    case 3: return "Q4_1";
    case 4: return "Q4_1_SOME_F16";
    case 5: return "Q4_2_REMOVED";
    case 6: return "Q4_3_REMOVED";
    case 7: return "Q8_0";
    case 8: return "Q5_0";
    case 9: return "Q5_1";
    case 10: return "Q2_K";
    case 11: return "Q3_K_S";
    case 12: return "Q3_K_M";
    case 13: return "Q3_K_L";
    case 14: return "Q4_K_S";
    case 15: return "Q4_K_M";
    case 16: return "Q5_K_S";
    case 17: return "Q5_K_M";
    case 18: return "Q6_K";
    case 25: return "IQ4_NL";
    case 38: return "MXFP4_MOE";
    default: return "file_type_" + std::to_string(file_type);
  }
}

StatusOr<uint64_t> AlignOffset(uint64_t position, uint32_t alignment) {
  const uint64_t remainder = position % alignment;
  const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
  if (position > std::numeric_limits<uint64_t>::max() - padding) {
    return Malformed("tensor table alignment overflows");
  }
  return position + padding;
}
Status ComputePayloadBytes(const GgufTensorInfo& tensor,
                           std::optional<uint64_t>* payload_bytes) {
  *payload_bytes = std::nullopt;
  const auto block = GgufQuantBlockInfoForType(tensor.type);
  if (!block.has_value()) return Status();
  if (tensor.dimensions.empty() || tensor.dimensions.front() % block->elements != 0U) {
    return Malformed("tensor row size is incompatible with " + GgufTensorTypeName(tensor.type) +
                     ": " + tensor.name);
  }
  uint64_t element_count = 1U;
  for (const uint64_t dimension : tensor.dimensions) {
    if (dimension > std::numeric_limits<uint64_t>::max() / element_count) {
      return Malformed("tensor element count overflows: " + tensor.name);
    }
    element_count *= dimension;
  }
  const uint64_t block_count = element_count / block->elements;
  if (block_count > std::numeric_limits<uint64_t>::max() / block->bytes) {
    return Malformed("tensor payload size overflows: " + tensor.name);
  }
  *payload_bytes = block_count * block->bytes;
  return Status();
}

}  // namespace

std::string GgufTensorTypeName(uint32_t type) {
  switch (type) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "Q4_0";
    case 3: return "Q4_1";
    case 6: return "Q5_0";
    case 7: return "Q5_1";
    case 8: return "Q8_0";
    case 12: return "Q4_K";
    case 13: return "Q5_K";
    case 14: return "Q6_K";
    case 20: return "IQ4_NL";
    case 21: return "IQ3_S";
    case 39: return "MXFP4";
    default: return "tensor_type_" + std::to_string(type);
  }
}

StatusOr<GgufInspection> GgufInspector::Inspect(
    const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    if (error) return Status::Unavailable("cannot inspect model file: " + error.message());
    return Status::NotFound("GGUF model file does not exist");
  }
  const uint64_t file_size = std::filesystem::file_size(path, error);
  if (error) return Status::Unavailable("cannot read model file size: " + error.message());
  if (file_size < 24U) return Malformed("file is shorter than the GGUF header");

  std::ifstream stream(path, std::ios::binary);
  if (!stream) return Status::Unavailable("cannot open GGUF model file");
  GgufReader reader(stream, file_size);
  std::array<char, 4> magic{};
  Status status = reader.ReadBytes(magic.data(), magic.size());
  if (!status.ok()) return status;
  if (magic != std::array<char, 4>{'G', 'G', 'U', 'F'}) {
    return Malformed("magic must be GGUF");
  }

  GgufInspection result;
  result.file_size = file_size;
  StatusOr<uint32_t> version = reader.ReadU32();
  StatusOr<uint64_t> tensors = reader.ReadU64();
  StatusOr<uint64_t> metadata = reader.ReadU64();
  if (!version.ok()) return version.status();
  if (!tensors.ok()) return tensors.status();
  if (!metadata.ok()) return metadata.status();
  if (version.value() < kMinimumGgufVersion || version.value() > kMaximumGgufVersion) {
    return Status::Unsupported("GGUF version is not supported (expected 2 or 3)");
  }
  result.version = version.value();
  result.tensor_count = tensors.value();
  result.metadata_count = metadata.value();
  if (result.tensor_count > kMaximumMetadataEntries ||
      result.tensor_count > reader.remaining() / 24U) {
    return Malformed("tensor count cannot fit in the file");
  }
  if (result.metadata_count > kMaximumMetadataEntries ||
      result.metadata_count > reader.remaining() / 12U) {
    return Malformed("metadata count cannot fit in the file");
  }

  std::optional<uint64_t> file_type;
  std::optional<uint64_t> alignment;
  for (uint64_t index = 0; index < result.metadata_count; ++index) {
    StatusOr<std::string> key = reader.ReadString(65535, true);
    StatusOr<uint32_t> raw_type = reader.ReadU32();
    if (!key.ok()) return key.status();
    if (!raw_type.ok()) return raw_type.status();
    const bool retain_architecture = IsKey(key.value(), "general.architecture");
    const bool retain_name = IsKey(key.value(), "general.name");
    const bool retain_parameter_count = IsKey(key.value(), "general.parameter_count");
    const bool retain_context = key.value().size() > 15U &&
        key.value().ends_with(".context_length");
    const bool retain_file_type = IsKey(key.value(), "general.file_type");
    const bool retain_alignment = IsKey(key.value(), "general.alignment");
    const bool retain_gemma4_parameter = key.value().starts_with("gemma4.");
    const bool retain_tokenizer_parameter = key.value().starts_with("tokenizer.ggml.");
    const uint64_t value_position = reader.position();
    MetadataValue value;
    status = ReadMetadataValue(reader, raw_type.value(),
                              retain_architecture || retain_name || retain_parameter_count || retain_context ||
                                  retain_file_type || retain_alignment || retain_gemma4_parameter ||
                                  retain_tokenizer_parameter,
                              &value);
    if (!status.ok()) return status;
    if (retain_architecture && value.string_value.has_value()) {
      result.architecture = std::move(value.string_value).value();
    } else if (retain_name && value.string_value.has_value()) {
      result.name = std::move(value.string_value).value();
    } else if (retain_parameter_count) {
      const std::optional<uint64_t> parsed = UnsignedMetadata(value);
      if (parsed.has_value()) result.parameter_count = parsed.value();
    } else if (retain_context) {
      const std::optional<uint64_t> parsed = UnsignedMetadata(value);
      if (parsed.has_value()) result.context_length = parsed.value();
    } else if (retain_file_type) {
      file_type = UnsignedMetadata(value);
      result.file_type_value_position = value_position;
      result.file_type_value_size = raw_type.value() == 4U ? 4U : raw_type.value() == 10U ? 8U : 0U;
    } else if (retain_alignment) {
      alignment = UnsignedMetadata(value);
    } else if (retain_gemma4_parameter) {
      ReadGemma4Parameter(key.value(), value, &result.gemma4);
    } else if (retain_tokenizer_parameter) {
      ReadTokenizerParameter(key.value(), value, &result.tokenizer);
    }
  }

  if (alignment.has_value()) {
    if (alignment.value() == 0 ||
        alignment.value() > std::numeric_limits<uint32_t>::max() ||
        alignment.value() % 8U != 0) {
      return Malformed("general.alignment must be a positive multiple of 8");
    }
    result.alignment = static_cast<uint32_t>(alignment.value());
  }
  if (file_type.has_value()) result.quantization = QuantizationName(file_type.value());

  uint64_t largest_tensor_offset = 0;
  result.tensors.reserve(static_cast<std::size_t>(result.tensor_count));
  for (uint64_t tensor_index = 0; tensor_index < result.tensor_count; ++tensor_index) {
    StatusOr<std::string> name = reader.ReadString(64, true);
    StatusOr<uint32_t> dimensions = reader.ReadU32();
    if (!name.ok()) return name.status();
    if (!dimensions.ok()) return dimensions.status();
    if (name.value().empty() || dimensions.value() == 0 ||
        dimensions.value() > kMaximumTensorDimensions) {
      return Malformed("tensor name or dimension count is invalid");
    }
    GgufTensorInfo tensor;
    tensor.name = std::move(name).value();
    tensor.dimensions.reserve(dimensions.value());
    for (uint32_t dimension = 0; dimension < dimensions.value(); ++dimension) {
      StatusOr<uint64_t> size = reader.ReadU64();
      if (!size.ok()) return size.status();
      if (size.value() == 0) return Malformed("tensor dimensions must be nonzero");
      tensor.dimensions.push_back(size.value());
    }
    const uint64_t type_position = reader.position();
    StatusOr<uint32_t> tensor_type = reader.ReadU32();
    const uint64_t offset_position = reader.position();
    StatusOr<uint64_t> tensor_offset = reader.ReadU64();
    if (!tensor_type.ok()) return tensor_type.status();
    if (!tensor_offset.ok()) return tensor_offset.status();
    tensor.type = tensor_type.value();
    tensor.descriptor_type_position = type_position;
    tensor.descriptor_offset_position = offset_position;
    if (tensor_offset.value() % result.alignment != 0) {
      return Malformed("tensor data offset is not aligned");
    }
    largest_tensor_offset = std::max(largest_tensor_offset, tensor_offset.value());
    tensor.data_offset = tensor_offset.value();
    result.tensors.push_back(std::move(tensor));
  }

  StatusOr<uint64_t> tensor_data_start = AlignOffset(reader.position(), result.alignment);
  if (!tensor_data_start.ok()) return tensor_data_start.status();
  if (tensor_data_start.value() > file_size) {
    return Malformed("tensor table extends beyond the file");
  }
  result.tensor_data_start = tensor_data_start.value();
  if (result.tensor_count > 0 &&
      largest_tensor_offset >= file_size - tensor_data_start.value()) {
    return Malformed("tensor offset points outside the tensor data section");
  }
  std::vector<std::size_t> offsets(result.tensors.size());
  std::iota(offsets.begin(), offsets.end(), 0U);
  std::sort(offsets.begin(), offsets.end(), [&result](std::size_t left, std::size_t right) {
    return result.tensors[left].data_offset < result.tensors[right].data_offset;
  });
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    auto& tensor = result.tensors[offsets[index]];
    const uint64_t next = index + 1U < offsets.size()
        ? result.tensors[offsets[index + 1U]].data_offset
        : file_size - tensor_data_start.value();
    if (next <= tensor.data_offset) return Malformed("tensor data offsets overlap");
    tensor.available_bytes = next - tensor.data_offset;
    Status payload_status = ComputePayloadBytes(tensor, &tensor.payload_bytes);
    if (!payload_status.ok()) return payload_status;
    if (tensor.payload_bytes.has_value() && tensor.payload_bytes.value() > tensor.available_bytes) {
      return Malformed("tensor payload is truncated: " + tensor.name);
    }
    tensor.data_offset += tensor_data_start.value();
  }
  if (reader.position() < tensor_data_start.value()) {
    status = reader.Skip(tensor_data_start.value() - reader.position());
    if (!status.ok()) return status;
  }
  // The GGUF specification defines tensor offsets relative to this aligned data block.
  // Known encodings have exact byte ranges validated above; unknown encodings remain
  // inspectable without claiming that their payload size or runtime support is known.
  return result;
}

}  // namespace isvik
