#ifndef ISVIK_BACKENDS_TENSORRT_ENGINE_H_
#define ISVIK_BACKENDS_TENSORRT_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "isvik/core/status.h"

namespace isvik::tensorrt_backend {

enum class LogLevel { kInfo, kWarning, kError };
using LogHandler = std::function<void(LogLevel, std::string_view)>;

enum class TensorDataType {
  kFloat32,
  kFloat16,
  kBFloat16,
  kInt8,
  kInt32,
  kInt64,
  kUInt8,
  kBool,
  kUnsupported,
};

enum class TensorMode { kInput, kOutput };

struct DeviceInfo {
  int index = 0;
  std::string name;
  std::uint64_t total_memory_bytes = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
};

struct TensorInfo {
  std::string name;
  TensorMode mode = TensorMode::kInput;
  TensorDataType data_type = TensorDataType::kUnsupported;
  std::vector<std::int64_t> dimensions;
  bool shape_inference_io = false;
};

// Tensor payloads are raw native-endian bytes matching the TensorRT data type.
struct Tensor {
  std::string name;
  TensorDataType data_type = TensorDataType::kUnsupported;
  std::vector<std::int64_t> dimensions;
  std::vector<std::uint8_t> bytes;
};

struct InputShapeRange {
  std::string tensor_name;
  std::vector<std::int64_t> min_dimensions;
  std::vector<std::int64_t> opt_dimensions;
  std::vector<std::int64_t> max_dimensions;
};

struct BuildOptions {
  int device_index = 0;
  std::size_t workspace_memory_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
  bool allow_tf32 = true;
  std::vector<InputShapeRange> dynamic_input_ranges;
};

[[nodiscard]] StatusOr<std::vector<DeviceInfo>> EnumerateDevices();
[[nodiscard]] std::size_t DataTypeSize(TensorDataType data_type);
[[nodiscard]] const char* DataTypeName(TensorDataType data_type);

// Parses an ONNX graph and writes a TensorRT serialized engine. Dynamic inputs
// require an explicit min/opt/max range in BuildOptions.
[[nodiscard]] Status BuildEngineFromOnnx(
    const std::filesystem::path& onnx_path,
    const std::filesystem::path& engine_path,
    const BuildOptions& options = {},
    const LogHandler& log_handler = {});

// Builds a one-operation TensorRT engine whose second input is the packed
// weight payload from a GGUF tensor. The payload stays quantized and is passed
// to the native Isvik CUDA kernel at inference time.
[[nodiscard]] Status BuildGgufQuantizedMatVecEngine(
    uint32_t tensor_type, uint64_t input_features, uint64_t output_features,
    const std::filesystem::path& engine_path,
    const BuildOptions& options = {}, const LogHandler& log_handler = {});

// Executes generic named tensors. This runtime is intentionally model-format
// agnostic; text generation/chat orchestration requires a separate LLM runtime.
class Engine {
 public:
  explicit Engine(LogHandler log_handler = {});
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) noexcept;
  Engine& operator=(Engine&&) noexcept;

  [[nodiscard]] Status Load(const std::filesystem::path& engine_path,
                            int device_index = 0);
  [[nodiscard]] StatusOr<std::vector<Tensor>> Infer(
      const std::vector<Tensor>& inputs);
  [[nodiscard]] std::vector<TensorInfo> Tensors() const;
  [[nodiscard]] std::filesystem::path loaded_path() const;
  [[nodiscard]] int device_index() const;
  void Unload();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace isvik::tensorrt_backend

#endif  // ISVIK_BACKENDS_TENSORRT_ENGINE_H_
