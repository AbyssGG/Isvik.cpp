#include "isvik/backends/tensorrt/engine.h"
#include "isvik/backends/tensorrt/gguf_genai_engine.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "isvik/core/gguf_quant.h"
#include "isvik/core/gguf_model_file.h"

namespace isvik::tensorrt_backend {
namespace {

void AppendVarint(std::uint64_t value, std::string* output) {
  while (value >= 0x80U) {
    output->push_back(static_cast<char>((value & 0x7fU) | 0x80U));
    value >>= 7U;
  }
  output->push_back(static_cast<char>(value));
}

void AppendVarintField(std::uint32_t field, std::uint64_t value, std::string* output) {
  AppendVarint((static_cast<std::uint64_t>(field) << 3U), output);
  AppendVarint(value, output);
}

void AppendBytesField(std::uint32_t field, std::string_view value, std::string* output) {
  AppendVarint((static_cast<std::uint64_t>(field) << 3U) | 2U, output);
  AppendVarint(value.size(), output);
  output->append(value);
}

std::string TensorShape(bool dynamic_batch) {
  std::string one_dimension;
  if (dynamic_batch) {
    AppendBytesField(2U, "batch", &one_dimension);
  } else {
    AppendVarintField(1U, 1U, &one_dimension);
  }
  std::string two_dimension;
  AppendVarintField(1U, 2U, &two_dimension);
  std::string shape;
  AppendBytesField(1U, one_dimension, &shape);
  AppendBytesField(1U, two_dimension, &shape);
  return shape;
}

std::string ValueInfo(std::string_view name, bool dynamic_batch = false) {
  const std::string shape = TensorShape(dynamic_batch);
  std::string tensor_type;
  AppendVarintField(1U, 1U, &tensor_type);  // TensorProto.DataType.FLOAT
  AppendBytesField(2U, shape, &tensor_type);
  std::string type;
  AppendBytesField(1U, tensor_type, &type);
  std::string value_info;
  AppendBytesField(1U, name, &value_info);
  AppendBytesField(2U, type, &value_info);
  return value_info;
}

std::string AddNode() {
  std::string node;
  AppendBytesField(1U, "left", &node);
  AppendBytesField(1U, "right", &node);
  AppendBytesField(2U, "sum", &node);
  AppendBytesField(3U, "add-two-vectors", &node);
  AppendBytesField(4U, "Add", &node);
  return node;
}

std::string AddOnnxModel(bool dynamic_batch = false) {
  std::string graph;
  AppendBytesField(1U, AddNode(), &graph);
  AppendBytesField(2U, "isvik-tensorrt-smoke", &graph);
  AppendBytesField(11U, ValueInfo("left", dynamic_batch), &graph);
  AppendBytesField(11U, ValueInfo("right", dynamic_batch), &graph);
  AppendBytesField(12U, ValueInfo("sum", dynamic_batch), &graph);

  std::string opset;
  AppendVarintField(2U, 13U, &opset);
  std::string model;
  AppendVarintField(1U, 8U, &model);  // ModelProto.ir_version
  AppendBytesField(2U, "Isvik TensorRT smoke test", &model);
  AppendBytesField(7U, graph, &model);
  AppendBytesField(8U, opset, &model);
  return model;
}

struct TemporaryDirectory {
  std::filesystem::path path;
  bool keep = false;

  TemporaryDirectory() {
    if (const char* requested = std::getenv("ISVIK_TENSORRT_SMOKE_DIR");
        requested != nullptr && requested[0] != '\0') {
      path = std::filesystem::u8path(requested);
      keep = true;
    } else {
      path = std::filesystem::temp_directory_path() /
          ("isvik-tensorrt-smoke-" + std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count()));
    }
  }

  ~TemporaryDirectory() {
    if (keep) return;
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

template <typename T>
std::vector<std::uint8_t> AsBytes(const std::vector<T>& values) {
  std::vector<std::uint8_t> result(values.size() * sizeof(T));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

TEST(TensorRtRuntimeTest, BuildsAnOnnxEngineAndExecutesOnNvidiaGpu) {
  const StatusOr<std::vector<DeviceInfo>> devices = EnumerateDevices();
  if (!devices.ok() || devices.value().empty()) {
    GTEST_SKIP() << "TensorRT GPU is unavailable: "
                 << (devices.ok() ? "no CUDA devices" : devices.status().message());
  }

  TemporaryDirectory temporary;
  std::filesystem::create_directories(temporary.path);
  const auto onnx_path = temporary.path / "add.onnx";
  const auto engine_path = temporary.path / "add.engine";
  {
    std::ofstream output(onnx_path, std::ios::binary);
    const std::string model = AddOnnxModel();
    output.write(model.data(), static_cast<std::streamsize>(model.size()));
    ASSERT_TRUE(output.good());
  }

  const Status build_status = BuildEngineFromOnnx(onnx_path, engine_path);
  ASSERT_TRUE(build_status.ok()) << build_status.message();
  ASSERT_TRUE(std::filesystem::is_regular_file(engine_path));
  ASSERT_GT(std::filesystem::file_size(engine_path), 0U);

  Engine engine;
  const Status load_status = engine.Load(engine_path, devices.value().front().index);
  ASSERT_TRUE(load_status.ok()) << load_status.message();
  const auto tensors = engine.Tensors();
  ASSERT_EQ(tensors.size(), 3U);

  Tensor left{"left", TensorDataType::kFloat32, {1, 2}, AsBytes(std::vector<float>{1.0F, 2.0F})};
  Tensor right{"right", TensorDataType::kFloat32, {1, 2}, AsBytes(std::vector<float>{10.0F, 20.0F})};
  const StatusOr<std::vector<Tensor>> outputs = engine.Infer({left, right});
  ASSERT_TRUE(outputs.ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1U);
  ASSERT_EQ(outputs.value().front().name, "sum");
  ASSERT_EQ(outputs.value().front().dimensions, (std::vector<std::int64_t>{1, 2}));
  ASSERT_EQ(outputs.value().front().bytes.size(), 2U * sizeof(float));
  std::vector<float> values(2U);
  std::memcpy(values.data(), outputs.value().front().bytes.data(), outputs.value().front().bytes.size());
  EXPECT_FLOAT_EQ(values[0], 11.0F);
  EXPECT_FLOAT_EQ(values[1], 22.0F);
}

TEST(TensorRtRuntimeTest, BuildsAndRunsDynamicInputProfiles) {
  const StatusOr<std::vector<DeviceInfo>> devices = EnumerateDevices();
  if (!devices.ok() || devices.value().empty()) {
    GTEST_SKIP() << "TensorRT GPU is unavailable: "
                 << (devices.ok() ? "no CUDA devices" : devices.status().message());
  }

  TemporaryDirectory temporary;
  std::filesystem::create_directories(temporary.path);
  const auto onnx_path = temporary.path / "dynamic-add.onnx";
  const auto engine_path = temporary.path / "dynamic-add.engine";
  {
    std::ofstream output(onnx_path, std::ios::binary);
    const std::string model = AddOnnxModel(true);
    output.write(model.data(), static_cast<std::streamsize>(model.size()));
    ASSERT_TRUE(output.good());
  }

  BuildOptions build_options;
  build_options.dynamic_input_ranges = {
      {"left", {1, 2}, {2, 2}, {4, 2}},
      {"right", {1, 2}, {2, 2}, {4, 2}},
  };
  const Status build_status = BuildEngineFromOnnx(onnx_path, engine_path, build_options);
  ASSERT_TRUE(build_status.ok()) << build_status.message();

  Engine engine;
  const Status load_status = engine.Load(engine_path, devices.value().front().index);
  ASSERT_TRUE(load_status.ok()) << load_status.message();
  Tensor left{"left", TensorDataType::kFloat32, {2, 2},
              AsBytes(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F})};
  Tensor right{"right", TensorDataType::kFloat32, {2, 2},
               AsBytes(std::vector<float>{10.0F, 20.0F, 30.0F, 40.0F})};
  const StatusOr<std::vector<Tensor>> outputs = engine.Infer({left, right});
  ASSERT_TRUE(outputs.ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1U);
  EXPECT_EQ(outputs.value().front().dimensions, (std::vector<std::int64_t>{2, 2}));
  std::vector<float> values(4U);
  std::memcpy(values.data(), outputs.value().front().bytes.data(), outputs.value().front().bytes.size());
  EXPECT_FLOAT_EQ(values[0], 11.0F);
  EXPECT_FLOAT_EQ(values[1], 22.0F);
  EXPECT_FLOAT_EQ(values[2], 33.0F);
  EXPECT_FLOAT_EQ(values[3], 44.0F);
}

TEST(TensorRtRuntimeTest, RunsOriginalMxfp4GgufWeightsWithoutConversion) {
  const StatusOr<std::vector<DeviceInfo>> devices = EnumerateDevices();
  if (!devices.ok() || devices.value().empty()) {
    GTEST_SKIP() << "TensorRT GPU is unavailable: "
                 << (devices.ok() ? "no CUDA devices" : devices.status().message());
  }

  TemporaryDirectory temporary;
  std::filesystem::create_directories(temporary.path);
  const auto engine_path = temporary.path / "gguf-mxfp4-matvec.engine";
  constexpr std::size_t kInputFeatures = 32U;
  constexpr std::size_t kOutputFeatures = 2U;
  constexpr std::size_t kBytesPerRow = 17U;
  std::vector<std::uint8_t> packed_weights(kOutputFeatures * kBytesPerRow, 0x11U);
  for (std::size_t row = 0; row < kOutputFeatures; ++row) {
    packed_weights[row * kBytesPerRow] = 127U;  // E8M0 scale produces 0.5 for code 1.
  }

  BuildOptions build_options;
  build_options.device_index = devices.value().front().index;
  const Status build_status = BuildGgufQuantizedMatVecEngine(
      39U, kInputFeatures, kOutputFeatures, engine_path, build_options);
  ASSERT_TRUE(build_status.ok()) << build_status.message();
  ASSERT_TRUE(std::filesystem::is_regular_file(engine_path));

  Engine engine;
  const Status load_status = engine.Load(engine_path, devices.value().front().index);
  ASSERT_TRUE(load_status.ok()) << load_status.message();
  Tensor activations{"activations", TensorDataType::kFloat32,
                     {1, static_cast<std::int64_t>(kInputFeatures)},
                     AsBytes(std::vector<float>(kInputFeatures, 1.0F))};
  Tensor weights{"packed_weights", TensorDataType::kInt8,
                 {static_cast<std::int64_t>(packed_weights.size())}, packed_weights};
  const StatusOr<std::vector<Tensor>> outputs = engine.Infer({activations, weights});
  ASSERT_TRUE(outputs.ok()) << outputs.status().message();
  ASSERT_EQ(outputs.value().size(), 1U);
  EXPECT_EQ(outputs.value().front().name, "output");
  EXPECT_EQ(outputs.value().front().dimensions, (std::vector<std::int64_t>{1, 2}));
  ASSERT_EQ(outputs.value().front().bytes.size(), 2U * sizeof(float));
  std::vector<float> values(2U);
  std::memcpy(values.data(), outputs.value().front().bytes.data(),
              outputs.value().front().bytes.size());
  EXPECT_FLOAT_EQ(values[0], 16.0F);
  EXPECT_FLOAT_EQ(values[1], 16.0F);
}

TEST(TensorRtRuntimeTest, NativeGgufMatvecMatchesReferenceForGemma4Encodings) {
  const StatusOr<std::vector<DeviceInfo>> devices = EnumerateDevices();
  if (!devices.ok() || devices.value().empty()) {
    GTEST_SKIP() << "TensorRT GPU is unavailable: "
                 << (devices.ok() ? "no CUDA devices" : devices.status().message());
  }

  TemporaryDirectory temporary;
  std::filesystem::create_directories(temporary.path);
  for (const std::uint32_t tensor_type : {0U, 7U, 8U, 13U, 20U, 21U, 39U}) {
    const auto block_info = GgufQuantBlockInfoForType(tensor_type);
    ASSERT_TRUE(block_info.has_value());
    const std::size_t features = static_cast<std::size_t>(
        tensor_type == 13U || tensor_type == 21U ? 256U : 32U);
    const std::size_t block_count = features / static_cast<std::size_t>(block_info->elements);
    std::vector<std::uint8_t> packed_weights(
        block_count * static_cast<std::size_t>(block_info->bytes), 0U);
    if (tensor_type == 0U) {
      packed_weights = AsBytes(std::vector<float>(features, 1.0F));
    } else if (tensor_type == 7U) {
      for (std::size_t block = 0; block < block_count; ++block) {
        const std::size_t offset = block * static_cast<std::size_t>(block_info->bytes);
        packed_weights[offset] = 0x00U;
        packed_weights[offset + 1U] = 0x3cU;  // half scale = 1
        std::fill(packed_weights.begin() + static_cast<std::ptrdiff_t>(offset + 8U),
                  packed_weights.begin() + static_cast<std::ptrdiff_t>(offset + 24U),
                  0x11U);
      }
    } else if (tensor_type == 8U) {
      for (std::size_t block = 0; block < block_count; ++block) {
        const std::size_t offset = block * static_cast<std::size_t>(block_info->bytes);
        packed_weights[offset] = 0x00U;
        packed_weights[offset + 1U] = 0x3cU;  // half scale = 1
        std::fill(packed_weights.begin() + static_cast<std::ptrdiff_t>(offset + 2U),
                  packed_weights.begin() + static_cast<std::ptrdiff_t>(offset + 34U), 1U);
      }
    } else if (tensor_type == 13U) {
      packed_weights[1U] = 0x3cU;  // block scale = 1
      std::fill(packed_weights.begin() + 0U, packed_weights.begin() + 4U, 1U);
      std::fill(packed_weights.begin() + 8U, packed_weights.begin() + 12U, 1U);
      std::fill(packed_weights.begin() + 48U, packed_weights.end(), 0x11U);
    } else if (tensor_type == 20U) {
      packed_weights[1U] = 0x3cU;  // half scale = 1
      std::fill(packed_weights.begin() + 2U, packed_weights.end(), 0x88U);
    } else if (tensor_type == 21U) {
      packed_weights[1U] = 0x3cU;  // half scale = 1; grid code 0 decodes to ones
    } else if (tensor_type == 39U) {
      std::fill(packed_weights.begin(), packed_weights.end(), 0x11U);
      for (std::size_t block = 0; block < block_count; ++block) {
        packed_weights[block * static_cast<std::size_t>(block_info->bytes)] = 127U;
      }
    }

    const StatusOr<std::vector<float>> decoded =
        DecodeGgufTensorBlocks(tensor_type, packed_weights);
    ASSERT_TRUE(decoded.ok()) << "type=" << tensor_type << ": " << decoded.status().message();
    float expected = 0.0F;
    for (const float weight : decoded.value()) expected += weight;

    const auto engine_path = temporary.path /
        ("gguf-type-" + std::to_string(tensor_type) + ".engine");
    BuildOptions build_options;
    build_options.device_index = devices.value().front().index;
    const Status build_status = BuildGgufQuantizedMatVecEngine(
        tensor_type, features, 1U, engine_path, build_options);
    ASSERT_TRUE(build_status.ok()) << "type=" << tensor_type << ": " << build_status.message();

    Engine engine;
    const Status load_status = engine.Load(engine_path, devices.value().front().index);
    ASSERT_TRUE(load_status.ok()) << "type=" << tensor_type << ": " << load_status.message();
    Tensor activations{"activations", TensorDataType::kFloat32,
                       {1, static_cast<std::int64_t>(features)},
                       AsBytes(std::vector<float>(features, 1.0F))};
    Tensor weights{"packed_weights", TensorDataType::kInt8,
                   {static_cast<std::int64_t>(packed_weights.size())}, packed_weights};
    const StatusOr<std::vector<Tensor>> outputs = engine.Infer({activations, weights});
    ASSERT_TRUE(outputs.ok()) << "type=" << tensor_type << ": " << outputs.status().message();
    ASSERT_EQ(outputs.value().size(), 1U);
    ASSERT_EQ(outputs.value().front().bytes.size(), sizeof(float));
    float actual = 0.0F;
    std::memcpy(&actual, outputs.value().front().bytes.data(), sizeof(actual));
    EXPECT_NEAR(actual, expected, std::max(1.0e-5F, std::abs(expected) * 1.0e-5F))
        << "type=" << tensor_type;
  }
}

TEST(TensorRtRuntimeTest, RunsRowsReadDirectlyFromTheOriginalGemma4Gguf) {
  const char* model_directory = std::getenv("ISVIK_TEST_GEMMA4_GGUF_DIR");
  if (model_directory == nullptr || model_directory[0] == '\0') {
    GTEST_SKIP() << "Set ISVIK_TEST_GEMMA4_GGUF_DIR to the directory containing the untouched Gemma 4 GGUF files";
  }
  const StatusOr<std::vector<DeviceInfo>> devices = EnumerateDevices();
  if (!devices.ok() || devices.value().empty()) {
    GTEST_SKIP() << "TensorRT GPU is unavailable: "
                 << (devices.ok() ? "no CUDA devices" : devices.status().message());
  }

  const std::filesystem::path directory = std::filesystem::u8path(model_directory);
  TemporaryDirectory temporary;
  std::filesystem::create_directories(temporary.path);
  std::size_t case_index = 0U;
  const std::array<std::pair<const char*, std::set<std::uint32_t>>, 2> model_cases{{
      {"gemma-4-26B-A4B-it-MXFP4_MOE.gguf", {0U, 7U, 8U, 13U, 39U}},
      {"gemma-4-26B-A4B-it-UD-IQ4_NL.gguf", {0U, 8U, 20U, 21U}},
  }};
  for (const auto& [filename, expected_types] : model_cases) {
    const auto model = GgufModelFile::Open(directory / filename);
    ASSERT_TRUE(model.ok()) << filename << ": " << model.status().message();
    std::set<std::uint32_t> tested_types;
    for (const GgufTensorInfo& tensor : model.value()->inspection().tensors) {
      if (!IsGgufTensorEncodingDecodable(tensor.type) ||
          !tested_types.insert(tensor.type).second || tensor.dimensions.empty()) {
        continue;
      }
      const auto block = GgufQuantBlockInfoForType(tensor.type);
      ASSERT_TRUE(block.has_value());
      const std::uint64_t features = tensor.dimensions.front();
      if (features == 0U || features % block->elements != 0U) continue;
      const std::uint64_t row_bytes = (features / block->elements) * block->bytes;
      if (!tensor.payload_bytes.has_value() || row_bytes > tensor.payload_bytes.value()) continue;

      const StatusOr<std::vector<std::uint8_t>> weight_row =
          model.value()->ReadTensorRange(tensor.name, 0U, row_bytes);
      ASSERT_TRUE(weight_row.ok()) << tensor.name << ": " << weight_row.status().message();
      const StatusOr<std::vector<float>> decoded =
          DecodeGgufTensorBlocks(tensor.type, weight_row.value());
      ASSERT_TRUE(decoded.ok()) << tensor.name << ": " << decoded.status().message();

      std::vector<float> activation(static_cast<std::size_t>(features));
      float expected = 0.0F;
      for (std::size_t index = 0; index < activation.size(); ++index) {
        activation[index] = std::sin(static_cast<float>(index) * 0.013F);
        expected += activation[index] * decoded.value()[index];
      }

      const auto engine_path = temporary.path /
          ("source-row-" + std::to_string(case_index++) + ".engine");
      BuildOptions build_options;
      build_options.device_index = devices.value().front().index;
      const Status build_status = BuildGgufQuantizedMatVecEngine(
          tensor.type, features, 1U, engine_path, build_options);
      ASSERT_TRUE(build_status.ok()) << tensor.name << ": " << build_status.message();

      Engine engine;
      const Status load_status = engine.Load(engine_path, devices.value().front().index);
      ASSERT_TRUE(load_status.ok()) << tensor.name << ": " << load_status.message();
      Tensor activations{"activations", TensorDataType::kFloat32,
                         {1, static_cast<std::int64_t>(features)}, AsBytes(activation)};
      Tensor weights{"packed_weights", TensorDataType::kInt8,
                     {static_cast<std::int64_t>(weight_row.value().size())}, weight_row.value()};
      const StatusOr<std::vector<Tensor>> outputs = engine.Infer({activations, weights});
      ASSERT_TRUE(outputs.ok()) << tensor.name << ": " << outputs.status().message();
      ASSERT_EQ(outputs.value().size(), 1U);
      ASSERT_EQ(outputs.value().front().bytes.size(), sizeof(float));
      float actual = 0.0F;
      std::memcpy(&actual, outputs.value().front().bytes.data(), sizeof(actual));
      EXPECT_NEAR(actual, expected, std::max(2.0e-3F, std::abs(expected) * 2.0e-4F))
          << tensor.name << " (GGUF type " << tensor.type << ')';
    }
    EXPECT_EQ(tested_types, expected_types) << filename;
  }
}

TEST(TensorRtRuntimeTest, OpensOriginalGemma4GgufAsNativeChatSession) {
  const char* model_directory = std::getenv("ISVIK_TEST_GEMMA4_GGUF_DIR");
  if (model_directory == nullptr || std::string_view(model_directory).empty()) {
    GTEST_SKIP() << "Set ISVIK_TEST_GEMMA4_GGUF_DIR to the untouched Gemma 4 GGUF directory";
  }
  const auto devices = EnumerateDevices();
  if (!devices.ok() || devices.value().empty()) {
    GTEST_SKIP() << "TensorRT GPU is unavailable";
  }
  for (const char* filename : {"gemma-4-26B-A4B-it-MXFP4_MOE.gguf",
                               "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf"}) {
    ModelDescriptor descriptor;
    descriptor.id = filename;
    descriptor.display_name = filename;
    descriptor.format = ModelFormat::kGguf;
    descriptor.architecture = "gemma4";
    descriptor.path = std::filesystem::path(model_directory) / filename;
    ASSERT_TRUE(GgufGenAiEngine::ValidateModel(descriptor).ok()) << filename;
    GgufGenAiEngine session;
    const Status status = session.LoadModel(descriptor, devices.value().front().index);
    ASSERT_TRUE(status.ok()) << filename << ": " << status.message();
    ASSERT_TRUE(session.loaded_model().has_value());
    session.UnloadModel();
  }
}

TEST(TensorRtRuntimeTest, GeneratesOneTokenThroughNativeGemma4GraphWhenRequested) {
  const char* enabled = std::getenv("ISVIK_TEST_GEMMA4_FULL_GENERATION");
  const char* model_directory = std::getenv("ISVIK_TEST_GEMMA4_GGUF_DIR");
  if (enabled == nullptr || std::string_view(enabled) != "1" || model_directory == nullptr) {
    GTEST_SKIP() << "Set ISVIK_TEST_GEMMA4_FULL_GENERATION=1 for the long real-model graph test";
  }
  const auto devices = EnumerateDevices();
  if (!devices.ok() || devices.value().empty()) GTEST_SKIP() << "TensorRT GPU is unavailable";
  ModelDescriptor descriptor;
  descriptor.id = "gemma4-native-smoke";
  descriptor.display_name = "Gemma 4 native smoke";
  descriptor.format = ModelFormat::kGguf;
  descriptor.architecture = "gemma4";
  descriptor.path = std::filesystem::path(model_directory) /
      "gemma-4-26B-A4B-it-MXFP4_MOE.gguf";
  GgufGenAiEngine session;
  ASSERT_TRUE(session.LoadModel(descriptor, devices.value().front().index).ok());
  UnifiedInferenceRequest request;
  request.request_id = "native-smoke";
  request.model_id = descriptor.id;
  request.messages.push_back({MessageRole::kUser, "Hi", {}, {}});
  request.generation.max_tokens = 1;
  request.generation.decoding_mode = DecodingMode::kGreedy;
  CancellationSource cancellation;
  bool completed = false;
  std::string generated_text;
  const auto status = session.Generate(request, cancellation.token(),
      [&completed, &generated_text](const InferenceEvent& event) {
        if (const auto* delta = std::get_if<ContentDelta>(&event.payload)) {
          generated_text += delta->text;
        }
        if (std::holds_alternative<InferenceCompleted>(event.payload)) completed = true;
      });
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_TRUE(completed);
  std::clog << "Native Gemma 4 first decoded token: " << generated_text << '\n';
}

}  // namespace
}  // namespace isvik::tensorrt_backend
