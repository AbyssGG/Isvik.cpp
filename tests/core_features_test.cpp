#include <algorithm>
#include <bit>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "isvik/core/context_manager.h"
#include "isvik/core/gguf_inspector.h"
#include "isvik/core/gguf_model_file.h"
#include "isvik/core/gguf_quant.h"
#include "isvik/core/gguf_tokenizer.h"
#include "isvik/core/model_manager.h"
#include "isvik/core/tool_runtime.h"

namespace isvik {
namespace {

class BinaryWriter {
 public:
  void U8(uint8_t value) { bytes.push_back(static_cast<char>(value)); }
  void U32(uint32_t value) {
    for (uint32_t index = 0; index < 4; ++index) {
      U8(static_cast<uint8_t>((value >> (index * 8U)) & 0xffU));
    }
  }
  void Float32(float value) { U32(std::bit_cast<uint32_t>(value)); }
  void U64(uint64_t value) {
    for (uint32_t index = 0; index < 8; ++index) {
      U8(static_cast<uint8_t>((value >> (index * 8U)) & 0xffU));
    }
  }
  void String(const std::string& value) {
    U64(static_cast<uint64_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  }
  void Align(uint32_t alignment) {
    while (bytes.size() % alignment != 0) U8(0);
  }

  std::vector<char> bytes;
};

struct GgufFixtureOptions {
  uint32_t version = 3;
  std::string magic = "GGUF";
  uint32_t alignment = 32;
  uint32_t tensor_dimensions = 1;
  uint64_t tensor_row_size = 32;
  uint64_t tensor_payload_bytes = 64;
  uint32_t tensor_type = 2;
  uint64_t tensor_offset = 0;
  uint64_t file_type = 15;
  uint64_t context_length = 4096;
  bool include_tensor = false;
  bool include_string_array = false;
  bool include_nested_array = false;
  bool include_large_unretained_string = false;
  bool include_invalid_bool = false;
  bool include_unknown_metadata = false;
};

std::vector<char> MakeGguf(const GgufFixtureOptions& options = {}) {
  BinaryWriter writer;
  writer.bytes.insert(writer.bytes.end(), options.magic.begin(), options.magic.end());
  writer.U32(options.version);
  writer.U64(options.include_tensor ? 1 : 0);
  const uint64_t metadata_count = 4 +
      static_cast<uint64_t>(options.include_string_array) +
      static_cast<uint64_t>(options.include_nested_array) +
      static_cast<uint64_t>(options.include_large_unretained_string) +
      static_cast<uint64_t>(options.include_invalid_bool) +
      static_cast<uint64_t>(options.include_unknown_metadata);
  writer.U64(metadata_count);

  writer.String("general.architecture");
  writer.U32(8);  // STRING
  writer.String("llama");
  writer.String("general.file_type");
  writer.U32(4);  // UINT32
  writer.U32(static_cast<uint32_t>(options.file_type));
  writer.String("llama.context_length");
  writer.U32(10);  // UINT64
  writer.U64(options.context_length);
  writer.String("general.alignment");
  writer.U32(4);  // UINT32
  writer.U32(options.alignment);
  if (options.include_string_array) {
    writer.String("tokenizer.ggml.tokens");
    writer.U32(9);  // ARRAY
    writer.U32(8);  // STRING
    writer.U64(2);
    writer.String("one");
    writer.String("two");
  }
  if (options.include_nested_array) {
    writer.String("test.nested_array");
    writer.U32(9);  // ARRAY
    writer.U32(9);  // ARRAY
    writer.U64(1);
    writer.U32(8);  // STRING
    writer.U64(1);
    writer.String("nested");
  }
  if (options.include_large_unretained_string) {
    writer.String("general.description");
    writer.U32(8);  // STRING
    writer.String(std::string(17 * 1024 * 1024, 'x'));
  }
  if (options.include_invalid_bool) {
    writer.String("test.invalid_bool");
    writer.U32(7);  // BOOL
    writer.U8(2);
  }
  if (options.include_unknown_metadata) {
    writer.String("test.future_type");
    writer.U32(77);
  }

  if (options.include_tensor) {
    writer.String("weight");
    writer.U32(options.tensor_dimensions);
    for (uint32_t index = 0; index < options.tensor_dimensions; ++index) {
      writer.U64(index == 0 ? options.tensor_row_size : 32U);
    }
    writer.U32(options.tensor_type);
    writer.U64(options.tensor_offset);
    writer.Align(options.alignment);
    for (uint64_t index = 0; index < options.tensor_payload_bytes; ++index) {
      writer.U8(static_cast<uint8_t>(index % 251U));
    }
  } else {
    writer.Align(options.alignment);
  }
  return writer.bytes;
}

class ScopedTempFile {
 public:
  explicit ScopedTempFile(std::string extension) {
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
        ("isvik_test_" + std::to_string(ticks) + extension);
  }
  ~ScopedTempFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  const std::filesystem::path& path() const { return path_; }
  void Write(const std::vector<char>& bytes) const {
    std::ofstream stream(path_, std::ios::binary);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream) throw std::runtime_error("failed to write test fixture");
  }
  void WriteText(const std::string& value) const {
    std::ofstream stream(path_, std::ios::binary);
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
    stream.close();
    if (!stream) throw std::runtime_error("failed to write test fixture");
  }

 private:
  std::filesystem::path path_;
};

ContextMessage MakeContextMessage(MessageRole role, std::string content,
                                  bool pinned = false, std::string group = {}) {
  ContextMessage result;
  result.message.role = role;
  result.message.content = std::move(content);
  result.pinned = pinned;
  result.tool_group_id = std::move(group);
  return result;
}

ToolDefinition MakeTool(std::string name, ToolRiskLevel risk,
                        std::function<StatusOr<std::string>(
                            const nlohmann::json&, const CancellationToken&)> handler,
                        nlohmann::json schema = {
                            {"type", "object"},
                            {"properties", nlohmann::json::object()},
                            {"additionalProperties", false}}) {
  ToolDefinition tool;
  tool.metadata.name = std::move(name);
  tool.metadata.description = "test tool";
  tool.metadata.risk = risk;
  tool.metadata.input_schema = std::move(schema);
  tool.handler = std::move(handler);
  return tool;
}

std::vector<char> MakeGemma4MetadataGguf() {
  BinaryWriter writer;
  writer.bytes.insert(writer.bytes.end(), {'G', 'G', 'U', 'F'});
  writer.U32(3U);
  writer.U64(0U);
  writer.U64(9U);

  writer.String("general.architecture"); writer.U32(8U); writer.String("gemma4");
  writer.String("gemma4.block_count"); writer.U32(4U); writer.U32(30U);
  writer.String("gemma4.embedding_length"); writer.U32(4U); writer.U32(2816U);
  writer.String("gemma4.attention.head_count_kv");
  writer.U32(9U); writer.U32(4U); writer.U64(3U);
  writer.U32(8U); writer.U32(4U); writer.U32(8U);
  writer.String("gemma4.attention.sliding_window_pattern");
  writer.U32(9U); writer.U32(7U); writer.U64(3U);
  writer.U8(1U); writer.U8(0U); writer.U8(1U);
  writer.String("gemma4.expert_count"); writer.U32(4U); writer.U32(128U);
  writer.String("gemma4.expert_used_count"); writer.U32(4U); writer.U32(8U);
  writer.String("gemma4.rope.freq_base"); writer.U32(6U); writer.Float32(1000000.0F);
  writer.String("general.alignment"); writer.U32(4U); writer.U32(32U);
  writer.Align(32U);
  return writer.bytes;
}

TEST(GgufInspectorTest, ReadsHeaderAndSelectedMetadata) {
  ScopedTempFile file(".gguf");
  file.Write(MakeGguf());
  const StatusOr<GgufInspection> result = GgufInspector::Inspect(file.path());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result.value().version, 3U);
  EXPECT_EQ(result.value().architecture, "llama");
  EXPECT_EQ(result.value().quantization, "Q4_K_M");
  EXPECT_EQ(result.value().context_length, 4096U);
  EXPECT_EQ(result.value().tensor_count, 0U);
}

TEST(GgufInspectorTest, ReadsGemma4LayerAndMoEConfiguration) {
  ScopedTempFile file(".gguf");
  file.Write(MakeGemma4MetadataGguf());
  const auto result = GgufInspector::Inspect(file.path());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result.value().architecture, "gemma4");
  EXPECT_EQ(result.value().gemma4.block_count, 30U);
  EXPECT_EQ(result.value().gemma4.embedding_length, 2816U);
  EXPECT_EQ(result.value().gemma4.attention_head_count_kv,
            (std::vector<uint64_t>{8U, 4U, 8U}));
  EXPECT_EQ(result.value().gemma4.sliding_window_pattern,
            (std::vector<bool>{true, false, true}));
  EXPECT_EQ(result.value().gemma4.expert_count, 128U);
  EXPECT_EQ(result.value().gemma4.expert_used_count, 8U);
  EXPECT_DOUBLE_EQ(result.value().gemma4.rope_frequency_base, 1000000.0);
}

TEST(GgufInspectorTest, RejectsIncorrectMagic) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.magic = "NOPE";
  file.Write(MakeGguf(options));
  EXPECT_EQ(GgufInspector::Inspect(file.path()).status().code(), StatusCode::kInvalidArgument);
}

TEST(GgufInspectorTest, RejectsUnsupportedVersion) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.version = 4;
  file.Write(MakeGguf(options));
  EXPECT_EQ(GgufInspector::Inspect(file.path()).status().code(), StatusCode::kUnsupported);
}

TEST(GgufInspectorTest, SkipsStringArraysWithoutRetainingTheirValues) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_string_array = true;
  file.Write(MakeGguf(options));
  const StatusOr<GgufInspection> result = GgufInspector::Inspect(file.path());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result.value().metadata_count, 5U);
}

TEST(GgufInspectorTest, RejectsInvalidBooleanMetadata) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_invalid_bool = true;
  file.Write(MakeGguf(options));
  EXPECT_EQ(GgufInspector::Inspect(file.path()).status().code(), StatusCode::kInvalidArgument);
}

TEST(GgufInspectorTest, SkipsNestedMetadataArrays) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_nested_array = true;
  file.Write(MakeGguf(options));
  EXPECT_TRUE(GgufInspector::Inspect(file.path()).ok());
}

TEST(GgufInspectorTest, SkipsLargeUnretainedMetadataStringsWithoutAllocatingThem) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_large_unretained_string = true;
  file.Write(MakeGguf(options));
  EXPECT_TRUE(GgufInspector::Inspect(file.path()).ok());
}

TEST(GgufInspectorTest, RejectsUnknownMetadataTypes) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_unknown_metadata = true;
  file.Write(MakeGguf(options));
  EXPECT_EQ(GgufInspector::Inspect(file.path()).status().code(), StatusCode::kInvalidArgument);
}

TEST(GgufInspectorTest, RejectsMetadataCountThatCannotFit) {
  ScopedTempFile file(".gguf");
  std::vector<char> bytes = MakeGguf();
  std::fill(bytes.begin() + 16, bytes.begin() + 24, static_cast<char>(0xff));
  file.Write(bytes);
  EXPECT_EQ(GgufInspector::Inspect(file.path()).status().code(), StatusCode::kInvalidArgument);
}

TEST(GgufInspectorTest, AcceptsSpecifiedNonPowerOfTwoAlignment) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.alignment = 24;
  file.Write(MakeGguf(options));
  const StatusOr<GgufInspection> result = GgufInspector::Inspect(file.path());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result.value().alignment, 24U);
}

TEST(GgufInspectorTest, RejectsAlignmentNotDivisibleByEight) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.alignment = 7;
  file.Write(MakeGguf(options));
  EXPECT_EQ(GgufInspector::Inspect(file.path()).status().code(), StatusCode::kInvalidArgument);
}

TEST(GgufInspectorTest, MapsGeneralFileTypeValuesFromTheFormat) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.file_type = 7;
  file.Write(MakeGguf(options));
  const StatusOr<GgufInspection> result = GgufInspector::Inspect(file.path());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result.value().quantization, "Q8_0");
}

TEST(GgufInspectorTest, RejectsTruncatedFile) {
  ScopedTempFile file(".gguf");
  std::vector<char> bytes = MakeGguf();
  bytes.pop_back();
  file.Write(bytes);
  EXPECT_EQ(GgufInspector::Inspect(file.path()).status().code(), StatusCode::kInvalidArgument);
}

TEST(GgufInspectorTest, ReportsUnknownQuantizationFileTypeNumerically) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.file_type = 777;
  file.Write(MakeGguf(options));
  const StatusOr<GgufInspection> result = GgufInspector::Inspect(file.path());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result.value().quantization, "file_type_777");
}

TEST(GgufInspectorTest, AcceptsAlignedTensorTableAndPayload) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  file.Write(MakeGguf(options));
  const StatusOr<GgufInspection> result = GgufInspector::Inspect(file.path());
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result.value().tensor_count, 1U);
  ASSERT_EQ(result.value().tensors.size(), 1U);
  EXPECT_EQ(result.value().tensors.front().name, "weight");
  EXPECT_EQ(result.value().tensors.front().dimensions, std::vector<uint64_t>{32U});
  EXPECT_EQ(result.value().tensors.front().data_offset % result.value().alignment, 0U);
  EXPECT_GT(result.value().tensors.front().available_bytes, 0U);
}

TEST(GgufInspectorTest, ComputesPayloadSizesForGemma4Encodings) {
  constexpr std::array<std::array<uint64_t, 3>, 7> cases{{
      {0U, 32U, 128U}, {7U, 32U, 24U}, {8U, 32U, 34U},
      {13U, 256U, 176U}, {20U, 32U, 18U}, {21U, 256U, 110U},
      {39U, 32U, 17U},
  }};
  ScopedTempFile file(".gguf");
  for (const auto& test_case : cases) {
    GgufFixtureOptions options;
    options.include_tensor = true;
    options.tensor_type = static_cast<uint32_t>(test_case[0]);
    options.tensor_row_size = test_case[1];
    options.tensor_payload_bytes = test_case[2];
    file.Write(MakeGguf(options));
    const StatusOr<GgufInspection> result = GgufInspector::Inspect(file.path());
    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_EQ(result.value().tensors.size(), 1U);
    ASSERT_TRUE(result.value().tensors.front().payload_bytes.has_value());
    EXPECT_EQ(result.value().tensors.front().payload_bytes.value(), test_case[2]);
  }
}

TEST(GgufInspectorTest, RejectsTruncatedKnownTensorPayload) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  options.tensor_payload_bytes = 17U;  // Q4_0 needs 18 bytes for 32 elements.
  file.Write(MakeGguf(options));
  const StatusOr<GgufInspection> result = GgufInspector::Inspect(file.path());
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(GgufInspectorTest, LeavesUnknownTensorPayloadSizesUnspecified) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  options.tensor_type = 1000U;
  file.Write(MakeGguf(options));
  const StatusOr<GgufInspection> result = GgufInspector::Inspect(file.path());
  ASSERT_TRUE(result.ok()) << result.status().message();
  ASSERT_EQ(result.value().tensors.size(), 1U);
  EXPECT_FALSE(result.value().tensors.front().payload_bytes.has_value());
}

TEST(GgufInspectorTest, RejectsRowsThatDoNotMatchQuantizationBlockSize) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  options.tensor_row_size = 16U;
  file.Write(MakeGguf(options));
  const StatusOr<GgufInspection> result = GgufInspector::Inspect(file.path());
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}
TEST(GgufInspectorTest, RejectsUnsupportedTensorRank) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  options.tensor_dimensions = 5;
  file.Write(MakeGguf(options));
  EXPECT_EQ(GgufInspector::Inspect(file.path()).status().code(), StatusCode::kInvalidArgument);
}

TEST(GgufInspectorTest, RejectsUnalignedTensorOffset) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  options.tensor_offset = 1;
  file.Write(MakeGguf(options));
  EXPECT_EQ(GgufInspector::Inspect(file.path()).status().code(), StatusCode::kInvalidArgument);
}

TEST(GgufInspectorTest, RejectsTensorOffsetOutsidePayload) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  options.tensor_offset = 4096;
  file.Write(MakeGguf(options));
  EXPECT_EQ(GgufInspector::Inspect(file.path()).status().code(), StatusCode::kInvalidArgument);
}

TEST(GgufModelFileTest, ReadsBoundedRangesFromOriginalTensorPayload) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  options.tensor_payload_bytes = 64U;
  file.Write(MakeGguf(options));

  const auto model = GgufModelFile::Open(file.path());
  ASSERT_TRUE(model.ok()) << model.status().message();
  const auto bytes = model.value()->ReadTensorRange("weight", 4U, 5U);
  ASSERT_TRUE(bytes.ok()) << bytes.status().message();
  EXPECT_EQ(bytes.value(), (std::vector<uint8_t>{4U, 5U, 6U, 7U, 8U}));
  EXPECT_EQ(model.value()->inspection().tensors.front().payload_bytes.value(), 18U);
}

TEST(GgufModelFileTest, RejectsReadsOutsideExactTensorPayload) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  options.tensor_payload_bytes = 64U;
  file.Write(MakeGguf(options));

  const auto model = GgufModelFile::Open(file.path());
  ASSERT_TRUE(model.ok()) << model.status().message();
  EXPECT_EQ(model.value()->ReadTensorRange("weight", 17U, 2U).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(GgufModelFileTest, DoesNotAssumePayloadSizesForUnknownEncodings) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  options.tensor_type = 1000U;
  options.tensor_payload_bytes = 64U;
  file.Write(MakeGguf(options));

  const auto model = GgufModelFile::Open(file.path());
  ASSERT_TRUE(model.ok()) << model.status().message();
  EXPECT_EQ(model.value()->ReadTensorRange("weight", 0U, 1U).status().code(),
            StatusCode::kUnsupported);
}

TEST(GgufQuantDecoderTest, DecodesEveryEncodingUsedByTheGemma4Files) {
  {
    std::vector<uint8_t> block(24U, 0U);
    block[1] = 0x3cU;  // d = 1
    block[3] = 0xbcU;  // m = -1
    block[4] = 1U;     // Set the fifth bit for element 0.
    std::fill(block.begin() + 8, block.end(), 0x11U);
    const auto decoded = DecodeGgufTensorBlocks(7U, block);
    ASSERT_TRUE(decoded.ok()) << decoded.status().message();
    ASSERT_EQ(decoded.value().size(), 32U);
    EXPECT_FLOAT_EQ(decoded.value()[0], 16.0F);
    EXPECT_FLOAT_EQ(decoded.value()[1], 0.0F);
    EXPECT_FLOAT_EQ(decoded.value()[16], 0.0F);
  }
  {
    std::vector<uint8_t> block(34U, 0U);
    block[1] = 0x3cU;  // d = 1
    block[2] = 0xffU;  // -1
    block[3] = 2U;
    const auto decoded = DecodeGgufTensorBlocks(8U, block);
    ASSERT_TRUE(decoded.ok()) << decoded.status().message();
    EXPECT_FLOAT_EQ(decoded.value()[0], -1.0F);
    EXPECT_FLOAT_EQ(decoded.value()[1], 2.0F);
  }
  {
    std::vector<uint8_t> block(176U, 0U);
    block[1] = 0x3cU;  // d = 1
    for (uint32_t index = 4U; index < 8U; ++index) block[index] = 1U;
    for (uint32_t index = 12U; index < 16U; ++index) block[index] = 1U;
    block[16] = 1U;  // Set the fifth bit for element 0.
    std::fill(block.begin() + 48, block.end(), 0x11U);
    const auto decoded = DecodeGgufTensorBlocks(13U, block);
    ASSERT_TRUE(decoded.ok()) << decoded.status().message();
    ASSERT_EQ(decoded.value().size(), 256U);
    EXPECT_FLOAT_EQ(decoded.value()[0], 17.0F);
    EXPECT_FLOAT_EQ(decoded.value()[1], 1.0F);
    EXPECT_FLOAT_EQ(decoded.value()[32], 1.0F);
  }
  {
    std::vector<uint8_t> block(18U, 0U);
    block[1] = 0x40U;  // d = 2
    block[2] = 0x80U;  // low nibble 0, high nibble 8
    const auto decoded = DecodeGgufTensorBlocks(20U, block);
    ASSERT_TRUE(decoded.ok()) << decoded.status().message();
    EXPECT_FLOAT_EQ(decoded.value()[0], -254.0F);
    EXPECT_FLOAT_EQ(decoded.value()[16], 2.0F);
  }
  {
    std::vector<uint8_t> block(110U, 0U);
    block[1] = 0x3cU;  // d = 1, codebook index 0 maps to four +1 values.
    const auto decoded = DecodeGgufTensorBlocks(21U, block);
    ASSERT_TRUE(decoded.ok()) << decoded.status().message();
    ASSERT_EQ(decoded.value().size(), 256U);
    EXPECT_TRUE(std::all_of(decoded.value().begin(), decoded.value().end(),
                            [](float value) { return value == 1.0F; }));
  }
  {
    std::vector<uint8_t> block(17U, 0U);
    block[0] = 128U;  // E8M0/2 scale = 1
    block[1] = 0x91U; // +1 in the low nibble and -1 in the high nibble
    const auto decoded = DecodeGgufTensorBlocks(39U, block);
    ASSERT_TRUE(decoded.ok()) << decoded.status().message();
    EXPECT_FLOAT_EQ(decoded.value()[0], 1.0F);
    EXPECT_FLOAT_EQ(decoded.value()[16], -1.0F);
  }
}

TEST(GgufQuantDecoderTest, RejectsPartialBlocksUnsupportedTypesAndOversizedRanges) {
  EXPECT_EQ(DecodeGgufTensorBlocks(39U, std::vector<uint8_t>(16U)).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(DecodeGgufTensorBlocks(22U, std::vector<uint8_t>(18U)).status().code(),
            StatusCode::kUnsupported);
  const uint64_t block_count = kMaxGgufCpuDecodeElements / 32U + 1U;
  std::vector<uint8_t> bytes(static_cast<size_t>(block_count * 17U));
  EXPECT_EQ(DecodeGgufTensorBlocks(39U, bytes).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(GgufQuantDecoderTest, ReadsAndDecodesRealGemma4SourceFilesWhenConfigured) {
  const char* model_directory = std::getenv("ISVIK_TEST_GEMMA4_GGUF_DIR");
  if (model_directory == nullptr || model_directory[0] == '\0') {
    GTEST_SKIP() << "Set ISVIK_TEST_GEMMA4_GGUF_DIR to the directory containing the two original Gemma 4 GGUF files";
  }

  std::set<uint32_t> observed_types;
  uint32_t verified_files = 0U;
  for (const auto& entry : std::filesystem::directory_iterator(model_directory)) {
    const std::string filename = entry.path().filename().string();
    std::string normalized_filename = filename;
    std::transform(normalized_filename.begin(), normalized_filename.end(),
                   normalized_filename.begin(), [](unsigned char value) {
                     return static_cast<char>(std::tolower(value));
                   });
    const bool is_target_model =
        normalized_filename == "gemma-4-26b-a4b-it-mxfp4_moe.gguf" ||
        normalized_filename == "gemma-4-26b-a4b-it-ud-iq4_nl.gguf";
    if (!entry.is_regular_file() || !is_target_model) {
      continue;
    }
    const auto model = GgufModelFile::Open(entry.path());
    ASSERT_TRUE(model.ok()) << entry.path().string() << ": " << model.status().message();
    for (const GgufTensorInfo& tensor : model.value()->inspection().tensors) {
      const auto block = GgufQuantBlockInfoForType(tensor.type);
      ASSERT_TRUE(block.has_value()) << tensor.name << " has unimplemented type " << tensor.type;
      const auto decoded = model.value()->ReadDecodedTensorBlocks(tensor.name, 0U, 1U);
      ASSERT_TRUE(decoded.ok()) << entry.path().string() << " / " << tensor.name << ": "
                                << decoded.status().message();
      ASSERT_EQ(decoded.value().size(), block->elements);
      ASSERT_TRUE(std::all_of(decoded.value().begin(), decoded.value().end(),
                              [](float value) { return std::isfinite(value); }))
          << entry.path().string() << " / " << tensor.name << " produced a non-finite value";
      observed_types.insert(tensor.type);
    }
    ++verified_files;
  }

  EXPECT_EQ(verified_files, 2U);
  EXPECT_EQ(observed_types, (std::set<uint32_t>{0U, 7U, 8U, 13U, 20U, 21U, 39U}));
}

TEST(GgufTokenizerTest, EncodesAndDecodesEmbeddedGemma4BpeWhenConfigured) {
  const char* model_directory = std::getenv("ISVIK_TEST_GEMMA4_GGUF_DIR");
  if (model_directory == nullptr || model_directory[0] == '\0') {
    GTEST_SKIP() << "Set ISVIK_TEST_GEMMA4_GGUF_DIR to the Gemma 4 GGUF directory";
  }
  const auto inspection = GgufInspector::Inspect(
      std::filesystem::path(model_directory) / "gemma-4-26B-A4B-it-MXFP4_MOE.gguf");
  ASSERT_TRUE(inspection.ok()) << inspection.status().message();
  EXPECT_EQ(inspection.value().tokenizer.tokens.size(), 262144U);
  EXPECT_FALSE(inspection.value().tokenizer.merges.empty());
  const auto tokenizer = GgufTokenizer::Create(inspection.value().tokenizer);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status().message();
  const auto encoded = tokenizer.value().Encode(
      "<bos><|turn>user\n你好 TensorRT<turn|>\n<|turn>model\n", false);
  ASSERT_TRUE(encoded.ok()) << encoded.status().message();
  ASSERT_FALSE(encoded.value().empty());
  EXPECT_EQ(encoded.value().front(), inspection.value().tokenizer.bos_token_id);
  const std::string decoded = tokenizer.value().Decode(encoded.value());
  EXPECT_NE(decoded.find("你好 TensorRT"), std::string::npos);
}

TEST(GgufTokenizerTest, MatchesGemma4ReferenceChatPromptIdsWhenConfigured) {
  const char* model_directory = std::getenv("ISVIK_TEST_GEMMA4_GGUF_DIR");
  if (model_directory == nullptr || std::string_view(model_directory).empty()) {
    GTEST_SKIP() << "Set ISVIK_TEST_GEMMA4_GGUF_DIR to the Gemma 4 GGUF directory";
  }
  const auto inspection = GgufInspector::Inspect(
      std::filesystem::path(model_directory) / "gemma-4-26B-A4B-it-MXFP4_MOE.gguf");
  ASSERT_TRUE(inspection.ok()) << inspection.status().message();
  const auto tokenizer = GgufTokenizer::Create(inspection.value().tokenizer);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status().message();
  const auto encoded = tokenizer.value().Encode(
      "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", true);
  ASSERT_TRUE(encoded.ok()) << encoded.status().message();
  EXPECT_EQ(encoded.value(), (std::vector<uint32_t>{
      2U, 105U, 2364U, 107U, 10979U, 106U, 107U,
      105U, 4368U, 107U, 100U, 45518U, 107U, 101U}));
}

TEST(GgufModelFileTest, ReadsAndDecodesOnlyRequestedOriginalGgufBlocks) {
  ScopedTempFile file(".gguf");
  GgufFixtureOptions options;
  options.include_tensor = true;
  options.tensor_type = 39U;
  options.tensor_payload_bytes = 17U;
  file.Write(MakeGguf(options));

  const auto model = GgufModelFile::Open(file.path());
  ASSERT_TRUE(model.ok()) << model.status().message();
  const auto decoded = model.value()->ReadDecodedTensorBlocks("weight", 0U, 1U);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded.value().size(), 32U);
  EXPECT_EQ(model.value()->ReadDecodedTensorBlocks("weight", 1U, 1U).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(ModelManagerTest, RejectsPathLikeModelIds) {
  ScopedTempFile file(".gguf");
  file.Write(MakeGguf());
  ModelManager manager;
  EXPECT_EQ(manager.ImportModel("../escape", file.path()).code(), StatusCode::kInvalidArgument);
}

TEST(ModelManagerTest, RejectsMissingModelFile) {
  ModelManager manager;
  EXPECT_EQ(manager.ImportModel("missing", "Z:/no/such/model.gguf").code(),
            StatusCode::kNotFound);
}

TEST(ModelManagerTest, ImportsAndPopulatesGgufDescriptor) {
  ScopedTempFile file(".gguf");
  file.Write(MakeGguf());
  ModelManager manager;
  ASSERT_TRUE(manager.ImportModel("local-model", file.path()).ok());
  const StatusOr<ModelDescriptor> model = manager.FindModel("local-model");
  ASSERT_TRUE(model.ok());
  EXPECT_EQ(model.value().format, ModelFormat::kGguf);
  EXPECT_EQ(model.value().architecture, "llama");
  EXPECT_EQ(model.value().context_length, 4096U);
  EXPECT_EQ(model.value().file_size, std::filesystem::file_size(file.path()));
}

TEST(ModelManagerTest, RequiresSiblingWeightsForOpenVinoIr) {
  ScopedTempFile xml(".xml");
  xml.WriteText("<net/>");
  ModelManager manager;
  EXPECT_EQ(manager.ImportModel("ir-model", xml.path()).code(), StatusCode::kInvalidArgument);
}

TEST(ModelManagerTest, ImportsOpenVinoXmlAndBinAsOneModel) {
  ScopedTempFile xml(".xml");
  xml.WriteText("<net/>");
  std::filesystem::path weights = xml.path();
  weights.replace_extension(".bin");
  {
    std::ofstream stream(weights, std::ios::binary);
    const std::string bytes = "weights";
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ModelManager manager;
  const Status status = manager.ImportModel("ir-model", xml.path());
  std::error_code error;
  std::filesystem::remove(weights, error);
  ASSERT_TRUE(status.ok()) << status.message();
  const StatusOr<ModelDescriptor> model = manager.FindModel("ir-model");
  ASSERT_TRUE(model.ok());
  EXPECT_EQ(model.value().format, ModelFormat::kOpenVinoIr);
  EXPECT_EQ(model.value().compatible_backends,
            std::vector<BackendType>{BackendType::kOpenVino});
  EXPECT_EQ(model.value().file_size, 13U);
}

TEST(ModelManagerTest, AllowsOnlyOneRegistrationPerModelId) {
  ScopedTempFile first(".gguf");
  ScopedTempFile second(".gguf");
  first.Write(MakeGguf());
  second.Write(MakeGguf());
  ModelManager manager;
  ASSERT_TRUE(manager.ImportModel("same-id", first.path()).ok());
  EXPECT_EQ(manager.ImportModel("same-id", second.path()).code(), StatusCode::kAlreadyExists);
  EXPECT_EQ(manager.ListModels().size(), 1U);
}

TEST(ModelManagerTest, DefaultMustReferToRegisteredModel) {
  ModelManager manager;
  EXPECT_EQ(manager.SetDefaultModel("missing").code(), StatusCode::kNotFound);
}

TEST(ModelManagerTest, RemovingDefaultModelClearsSelection) {
  ScopedTempFile file(".gguf");
  file.Write(MakeGguf());
  ModelManager manager;
  ASSERT_TRUE(manager.ImportModel("selected", file.path()).ok());
  ASSERT_TRUE(manager.SetDefaultModel("selected").ok());
  ASSERT_TRUE(manager.RemoveModel("selected").ok());
  EXPECT_EQ(manager.default_model_id().status().code(), StatusCode::kNotFound);
}

TEST(ModelManagerTest, RejectsUnsupportedFileExtension) {
  ScopedTempFile file(".bin");
  file.WriteText("not a model");
  ModelManager manager;
  EXPECT_EQ(manager.ImportModel("unsupported", file.path()).code(), StatusCode::kUnsupported);
}

TEST(ContextManagerTest, KeepsAllMessagesWhenTheyFit) {
  std::vector<ContextMessage> messages = {
      MakeContextMessage(MessageRole::kSystem, "system"),
      MakeContextMessage(MessageRole::kUser, "hello")};
  const StatusOr<ContextBuildResult> result =
      ContextManager::BuildPrompt(messages, ContextOptions{});
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().messages.size(), 2U);
  EXPECT_EQ(result.value().dropped_messages, 0U);
}

TEST(ContextManagerTest, RejectsOverflowWhenCompressionIsOff) {
  ContextOptions options;
  options.compression = ContextCompressionMode::kOff;
  options.max_context_tokens = 16;
  options.reserved_output_tokens = 4;
  std::vector<ContextMessage> messages = {
      MakeContextMessage(MessageRole::kUser, std::string(80, 'x'))};
  EXPECT_EQ(ContextManager::BuildPrompt(messages, options).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(ContextManagerTest, KeepsSystemAndLatestMessageWithinSlidingBudget) {
  ContextOptions options;
  options.max_context_tokens = 20;
  options.reserved_output_tokens = 8;
  std::vector<ContextMessage> messages = {
      MakeContextMessage(MessageRole::kSystem, "system"),
      MakeContextMessage(MessageRole::kUser, std::string(80, 'a')),
      MakeContextMessage(MessageRole::kAssistant, "old"),
      MakeContextMessage(MessageRole::kUser, "latest")};
  const StatusOr<ContextBuildResult> result =
      ContextManager::BuildPrompt(messages, options);
  ASSERT_TRUE(result.ok()) << result.status().message();
  ASSERT_EQ(result.value().messages.size(), 2U);
  EXPECT_EQ(result.value().messages[0].message.role, MessageRole::kSystem);
  EXPECT_EQ(result.value().messages[1].message.content, "latest");
}

TEST(ContextManagerTest, KeepsPinnedMessagesDuringCompression) {
  ContextOptions options;
  options.max_context_tokens = 20;
  options.reserved_output_tokens = 8;
  std::vector<ContextMessage> messages = {
      MakeContextMessage(MessageRole::kUser, "pinned", true),
      MakeContextMessage(MessageRole::kAssistant, std::string(80, 'a')),
      MakeContextMessage(MessageRole::kUser, "latest")};
  const StatusOr<ContextBuildResult> result =
      ContextManager::BuildPrompt(messages, options);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result.value().messages.size(), 2U);
  EXPECT_EQ(result.value().messages.front().message.content, "pinned");
}

TEST(ContextManagerTest, KeepsToolCallAndResultGroupTogether) {
  ContextOptions options;
  options.max_context_tokens = 28;
  options.reserved_output_tokens = 8;
  std::vector<ContextMessage> messages = {
      MakeContextMessage(MessageRole::kAssistant, std::string(40, 'x')),
      MakeContextMessage(MessageRole::kAssistant, "tool call", false, "call-1"),
      MakeContextMessage(MessageRole::kTool, "tool result", false, "call-1"),
      MakeContextMessage(MessageRole::kUser, "latest")};
  const StatusOr<ContextBuildResult> result =
      ContextManager::BuildPrompt(messages, options);
  ASSERT_TRUE(result.ok()) << result.status().message();
  ASSERT_EQ(result.value().messages.size(), 3U);
  EXPECT_EQ(result.value().messages[0].message.content, "tool call");
  EXPECT_EQ(result.value().messages[1].message.content, "tool result");
  EXPECT_EQ(result.value().messages[2].message.content, "latest");
}

TEST(ContextManagerTest, ReservesSpaceForGenerationOutput) {
  ContextOptions options;
  options.max_context_tokens = 12;
  options.reserved_output_tokens = 8;
  std::vector<ContextMessage> messages = {
      MakeContextMessage(MessageRole::kUser, "hello world")};
  const StatusOr<ContextBuildResult> result =
      ContextManager::BuildPrompt(messages, options);
  EXPECT_FALSE(result.ok());
}

TEST(ContextManagerTest, ReportsSummarizationAsUnsupportedUntilCompressorExists) {
  ContextOptions options;
  options.compression = ContextCompressionMode::kSummarize;
  options.max_context_tokens = 16;
  options.reserved_output_tokens = 4;
  std::vector<ContextMessage> messages = {
      MakeContextMessage(MessageRole::kUser, std::string(80, 'x'))};
  EXPECT_EQ(ContextManager::BuildPrompt(messages, options).status().code(),
            StatusCode::kUnsupported);
}

TEST(ContextManagerTest, RejectsInvalidTokenBudget) {
  ContextOptions options;
  options.max_context_tokens = 100;
  options.reserved_output_tokens = 100;
  EXPECT_EQ(ContextManager::BuildPrompt({}, options).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(ToolRuntimeTest, RegistersListsAndExecutesReadOnlyTool) {
  ToolRuntime runtime;
  int invocation_count = 0;
  ASSERT_TRUE(runtime.Register(MakeTool(
      "runtime.devices", ToolRiskLevel::kReadOnly,
      [&](const nlohmann::json& args, const CancellationToken&) -> StatusOr<std::string> {
        ++invocation_count;
        return args.at("query").get<std::string>() + " devices";
      }, nlohmann::json{
          {"type", "object"},
          {"properties", {{"query", {{"type", "string"}}}}},
          {"required", {"query"}},
          {"additionalProperties", false}})).ok());
  ASSERT_EQ(runtime.ListTools().size(), 1U);
  ToolExecutionRequest request;
  request.call_id = "call-1";
  request.name = "runtime.devices";
  request.arguments_json = R"({"query":"list"})";
  const StatusOr<std::string> result = runtime.Execute(request);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), "list devices");
  EXPECT_EQ(invocation_count, 1);
}

TEST(ToolRuntimeTest, RejectsDuplicateAndMalformedToolNames) {
  ToolRuntime runtime;
  auto handler = [](const nlohmann::json&, const CancellationToken&) -> StatusOr<std::string> {
    return std::string("ok");
  };
  ASSERT_TRUE(runtime.Register(MakeTool("memory.search", ToolRiskLevel::kReadOnly,
                                        handler)).ok());
  EXPECT_EQ(runtime.Register(MakeTool("memory.search", ToolRiskLevel::kReadOnly,
                                      handler)).code(), StatusCode::kAlreadyExists);
  EXPECT_EQ(runtime.Register(MakeTool("../delete", ToolRiskLevel::kWrite,
                                      handler)).code(), StatusCode::kInvalidArgument);
}

TEST(ToolPermissionPolicyTest, AppliesApprovalByRiskLevel) {
  EXPECT_TRUE(ToolPermissionPolicy::IsAllowed(ToolRiskLevel::kReadOnly, {}));
  EXPECT_FALSE(ToolPermissionPolicy::IsAllowed(ToolRiskLevel::kSensitiveRead, {}));
  EXPECT_TRUE(ToolPermissionPolicy::IsAllowed(
      ToolRiskLevel::kSensitiveRead, ToolApproval{true, false, false}));
  EXPECT_FALSE(ToolPermissionPolicy::IsAllowed(
      ToolRiskLevel::kWrite, ToolApproval{true, false, false}));
  EXPECT_TRUE(ToolPermissionPolicy::IsAllowed(
      ToolRiskLevel::kWrite, ToolApproval{false, true, false}));
  EXPECT_FALSE(ToolPermissionPolicy::IsAllowed(
      ToolRiskLevel::kDestructive, ToolApproval{false, true, false}));
  EXPECT_TRUE(ToolPermissionPolicy::IsAllowed(
      ToolRiskLevel::kDestructive, ToolApproval{false, true, true}));
}

TEST(ToolRuntimeTest, DoesNotCallHandlerWhenApprovalIsMissing) {
  ToolRuntime runtime;
  int invocation_count = 0;
  ASSERT_TRUE(runtime.Register(MakeTool(
      "memory.add", ToolRiskLevel::kWrite,
      [&](const nlohmann::json&, const CancellationToken&) -> StatusOr<std::string> {
        ++invocation_count;
        return std::string("added");
      })).ok());
  ToolExecutionRequest request;
  request.call_id = "write-1";
  request.name = "memory.add";
  EXPECT_EQ(runtime.Execute(request).status().code(), StatusCode::kUnavailable);
  EXPECT_EQ(invocation_count, 0);
}

TEST(ToolRuntimeTest, EnforcesIterationAndArgumentLimits) {
  ToolRuntime runtime(2, 4);
  ASSERT_TRUE(runtime.Register(MakeTool(
      "web.search", ToolRiskLevel::kReadOnly,
      [](const nlohmann::json&, const CancellationToken&) -> StatusOr<std::string> {
        return std::string("ok");
      })).ok());
  ToolExecutionRequest too_late;
  too_late.call_id = "late";
  too_late.name = "web.search";
  too_late.iteration = 3;
  EXPECT_EQ(runtime.Execute(too_late).status().code(), StatusCode::kInvalidArgument);
  ToolExecutionRequest too_large;
  too_large.call_id = "large";
  too_large.name = "web.search";
  too_large.arguments_json = "12345";
  EXPECT_EQ(runtime.Execute(too_large).status().code(), StatusCode::kInvalidArgument);
}

TEST(ToolRuntimeTest, ChecksCancellationBeforeHandlerStarts) {
  ToolRuntime runtime;
  int invocation_count = 0;
  ASSERT_TRUE(runtime.Register(MakeTool(
      "web.fetch", ToolRiskLevel::kReadOnly,
      [&](const nlohmann::json&, const CancellationToken&) -> StatusOr<std::string> {
        ++invocation_count;
        return std::string("ok");
      })).ok());
  CancellationSource source;
  source.Cancel();
  ToolExecutionRequest request;
  request.call_id = "cancelled";
  request.name = "web.fetch";
  request.cancellation = source.token();
  EXPECT_EQ(runtime.Execute(request).status().code(), StatusCode::kCancelled);
  EXPECT_EQ(invocation_count, 0);
}

TEST(ToolRuntimeTest, ReplaysSameCallIdWithoutRepeatingSideEffect) {
  ToolRuntime runtime;
  int invocation_count = 0;
  ASSERT_TRUE(runtime.Register(MakeTool(
      "memory.add", ToolRiskLevel::kWrite,
      [&](const nlohmann::json&, const CancellationToken&) -> StatusOr<std::string> {
        ++invocation_count;
        return std::string("stored");
      })).ok());
  ToolExecutionRequest request;
  request.call_id = "write-2";
  request.name = "memory.add";
  request.arguments_json = "{}";
  request.approval.call_approved = true;
  ASSERT_TRUE(runtime.Execute(request).ok());
  const StatusOr<std::string> replay = runtime.Execute(request);
  ASSERT_TRUE(replay.ok());
  EXPECT_EQ(replay.value(), "stored");
  EXPECT_EQ(invocation_count, 1);
}

TEST(ToolRuntimeTest, RejectsCallIdReuseWithDifferentArguments) {
  ToolRuntime runtime;
  ASSERT_TRUE(runtime.Register(MakeTool(
      "runtime.devices", ToolRiskLevel::kReadOnly,
      [](const nlohmann::json&, const CancellationToken&) -> StatusOr<std::string> {
        return std::string("ok");
      })).ok());
  ToolExecutionRequest request;
  request.call_id = "reused";
  request.name = "runtime.devices";
  request.arguments_json = "{}";
  ASSERT_TRUE(runtime.Execute(request).ok());
  request.arguments_json = R"({"different":true})";
  EXPECT_EQ(runtime.Execute(request).status().code(), StatusCode::kInvalidArgument);
}

TEST(ToolRuntimeTest, RejectsMalformedJsonArguments) {
  ToolRuntime runtime;
  ASSERT_TRUE(runtime.Register(MakeTool(
      "runtime.devices", ToolRiskLevel::kReadOnly,
      [](const nlohmann::json&, const CancellationToken&) -> StatusOr<std::string> {
        return std::string("ok");
      })).ok());
  ToolExecutionRequest request;
  request.call_id = "bad-json";
  request.name = "runtime.devices";
  request.arguments_json = "{";
  EXPECT_EQ(runtime.Execute(request).status().code(), StatusCode::kInvalidArgument);
}

TEST(ToolRuntimeTest, EnforcesRequiredPropertiesAndPropertyTypes) {
  ToolRuntime runtime;
  const nlohmann::json schema{
      {"type", "object"},
      {"properties", {{"query", {{"type", "string"}}}}},
      {"required", {"query"}},
      {"additionalProperties", false}};
  ASSERT_TRUE(runtime.Register(MakeTool(
      "web.search", ToolRiskLevel::kReadOnly,
      [](const nlohmann::json&, const CancellationToken&) -> StatusOr<std::string> {
        return std::string("ok");
      }, schema)).ok());
  ToolExecutionRequest request;
  request.name = "web.search";
  request.call_id = "missing-required";
  EXPECT_EQ(runtime.Execute(request).status().code(), StatusCode::kInvalidArgument);
  request.call_id = "wrong-type";
  request.arguments_json = R"({"query":7})";
  EXPECT_EQ(runtime.Execute(request).status().code(), StatusCode::kInvalidArgument);
  request.call_id = "unknown-property";
  request.arguments_json = R"({"query":"x","extra":true})";
  EXPECT_EQ(runtime.Execute(request).status().code(), StatusCode::kInvalidArgument);
}

TEST(ToolRuntimeTest, RejectsInvalidSchemaAtRegistration) {
  ToolRuntime runtime;
  const nlohmann::json schema{{"type", "object"}, {"required", "not-an-array"}};
  EXPECT_EQ(runtime.Register(MakeTool(
      "web.search", ToolRiskLevel::kReadOnly,
      [](const nlohmann::json&, const CancellationToken&) -> StatusOr<std::string> {
        return std::string("ok");
      }, schema)).code(), StatusCode::kInvalidArgument);
}

TEST(ToolRuntimeTest, EnforcesEnumValuesInToolSchema) {
  ToolRuntime runtime;
  const nlohmann::json schema{
      {"type", "object"},
      {"properties", {{"mode", {{"type", "string"}, {"enum", {"fast", "safe"}}}}}},
      {"required", {"mode"}},
      {"additionalProperties", false}};
  ASSERT_TRUE(runtime.Register(MakeTool(
      "runtime.mode", ToolRiskLevel::kReadOnly,
      [](const nlohmann::json& args, const CancellationToken&) -> StatusOr<std::string> {
        return args.at("mode").get<std::string>();
      }, schema)).ok());
  ToolExecutionRequest request;
  request.name = "runtime.mode";
  request.call_id = "invalid-mode";
  request.arguments_json = R"({"mode":"unknown"})";
  EXPECT_EQ(runtime.Execute(request).status().code(), StatusCode::kInvalidArgument);
  request.call_id = "valid-mode";
  request.arguments_json = R"({"mode":"safe"})";
  const StatusOr<std::string> result = runtime.Execute(request);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), "safe");
}

TEST(ToolRuntimeTest, ConvertsHandlerExceptionsToInternalErrors) {
  ToolRuntime runtime;
  ASSERT_TRUE(runtime.Register(MakeTool(
      "runtime.broken", ToolRiskLevel::kReadOnly,
      [](const nlohmann::json&, const CancellationToken&) -> StatusOr<std::string> {
        throw std::runtime_error("broken");
      })).ok());
  ToolExecutionRequest request;
  request.call_id = "throws";
  request.name = "runtime.broken";
  EXPECT_EQ(runtime.Execute(request).status().code(), StatusCode::kInternal);
}

}  // namespace
}  // namespace isvik
