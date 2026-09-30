#include "isvik/backends/tensorrt/engine.h"

#include "gguf_matvec_plugin.h"

#include "isvik/core/gguf_quant.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace isvik::tensorrt_backend {
namespace {

class TrtLogger final : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(LogHandler handler) : handler_(std::move(handler)) {}

  void log(Severity severity, const char* message) noexcept override {
    if (message == nullptr) return;
    LogLevel level = LogLevel::kInfo;
    if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
      level = LogLevel::kError;
    } else if (severity == Severity::kWARNING) {
      level = LogLevel::kWarning;
    } else if (severity > Severity::kINFO) {
      return;
    }
    try {
      if (handler_) {
        handler_(level, message);
      } else {
        std::ostream& output = level == LogLevel::kError ? std::cerr : std::clog;
        output << "[TensorRT] " << message << '\n';
      }
    } catch (...) {
      // Logging must never throw through TensorRT's C++ ABI.
    }
  }

 private:
  LogHandler handler_;
};

template <typename T>
using TrtObject = std::unique_ptr<T>;

std::string Utf8Path(const std::filesystem::path& path) {
  const auto encoded = path.u8string();
  return {encoded.begin(), encoded.end()};
}

Status CudaStatus(cudaError_t result, std::string_view operation) {
  if (result == cudaSuccess) return Status();
  return Status::Unavailable(std::string(operation) + ": " + cudaGetErrorString(result));
}

bool CheckedElementCount(const std::vector<std::int64_t>& dimensions,
                         std::size_t* count) {
  std::size_t result = 1U;
  for (const std::int64_t dimension : dimensions) {
    if (dimension <= 0 || static_cast<std::uint64_t>(dimension) >
                              std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (result > std::numeric_limits<std::size_t>::max() / value) return false;
    result *= value;
  }
  *count = result;
  return true;
}

bool ToDims(const std::vector<std::int64_t>& dimensions, nvinfer1::Dims* result) {
  if (dimensions.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
    return false;
  }
  result->nbDims = static_cast<int32_t>(dimensions.size());
  for (std::size_t index = 0U; index < dimensions.size(); ++index) {
    if (dimensions[index] <= 0 || dimensions[index] > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    result->d[index] = static_cast<int32_t>(dimensions[index]);
  }
  return true;
}

std::vector<std::int64_t> FromDims(const nvinfer1::Dims& dimensions) {
  std::vector<std::int64_t> result;
  if (dimensions.nbDims < 0 || dimensions.nbDims > nvinfer1::Dims::MAX_DIMS) return result;
  result.reserve(static_cast<std::size_t>(dimensions.nbDims));
  for (int32_t index = 0; index < dimensions.nbDims; ++index) {
    result.push_back(dimensions.d[index]);
  }
  return result;
}

TensorDataType FromTrtDataType(nvinfer1::DataType data_type) {
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT: return TensorDataType::kFloat32;
    case nvinfer1::DataType::kHALF: return TensorDataType::kFloat16;
    case nvinfer1::DataType::kBF16: return TensorDataType::kBFloat16;
    case nvinfer1::DataType::kINT8: return TensorDataType::kInt8;
    case nvinfer1::DataType::kINT32: return TensorDataType::kInt32;
    case nvinfer1::DataType::kINT64: return TensorDataType::kInt64;
    case nvinfer1::DataType::kUINT8: return TensorDataType::kUInt8;
    case nvinfer1::DataType::kBOOL: return TensorDataType::kBool;
    default: return TensorDataType::kUnsupported;
  }
}

bool IsDynamic(const nvinfer1::Dims& dimensions) {
  for (int32_t index = 0; index < dimensions.nbDims; ++index) {
    if (dimensions.d[index] < 0) return true;
  }
  return false;
}

Status ValidatePayload(const Tensor& tensor) {
  const std::size_t element_size = DataTypeSize(tensor.data_type);
  std::size_t element_count = 0U;
  if (element_size == 0U || !CheckedElementCount(tensor.dimensions, &element_count) ||
      element_count > std::numeric_limits<std::size_t>::max() / element_size) {
    return Status::InvalidArgument("invalid TensorRT tensor shape or data type: " + tensor.name);
  }
  const std::size_t expected_size = element_count * element_size;
  if (tensor.bytes.size() != expected_size) {
    return Status::InvalidArgument("tensor byte count does not match its shape and type: " +
                                   tensor.name);
  }
  return Status();
}

struct DeviceBuffer {
  void* pointer = nullptr;
  std::size_t capacity = 0U;
  ~DeviceBuffer() {
    if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
  }
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer(std::exchange(other.pointer, nullptr)),
        capacity(std::exchange(other.capacity, 0U)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
      pointer = std::exchange(other.pointer, nullptr);
      capacity = std::exchange(other.capacity, 0U);
    }
    return *this;
  }
};

Status EnsureDeviceBuffer(DeviceBuffer* buffer, std::size_t bytes) {
  if (buffer == nullptr) return Status::InvalidArgument("device buffer is null");
  if (buffer->pointer != nullptr && buffer->capacity >= bytes) return Status();
  void* replacement = nullptr;
  const cudaError_t allocation_status = cudaMalloc(&replacement, bytes);
  if (allocation_status != cudaSuccess) return CudaStatus(allocation_status, "cudaMalloc");
  if (buffer->pointer != nullptr) static_cast<void>(cudaFree(buffer->pointer));
  buffer->pointer = replacement;
  buffer->capacity = bytes;
  return Status();
}

struct PendingStreamWork {
  cudaStream_t stream = nullptr;
  bool pending = false;

  ~PendingStreamWork() {
    if (pending && stream != nullptr) static_cast<void>(cudaStreamSynchronize(stream));
  }
};

Status SaveSerializedEngine(const std::filesystem::path& engine_path,
                            const nvinfer1::IHostMemory& serialized) {
  if (engine_path.empty()) return Status::InvalidArgument("engine output path is empty");
  std::error_code error;
  if (!engine_path.parent_path().empty()) {
    std::filesystem::create_directories(engine_path.parent_path(), error);
    if (error) return Status::Unavailable("cannot create engine directory: " + error.message());
  }
  std::filesystem::path temporary_path = engine_path;
  temporary_path += ".tmp";
  {
    std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
    if (!output) return Status::Unavailable("cannot create TensorRT engine output file");
    output.write(static_cast<const char*>(serialized.data()),
                 static_cast<std::streamsize>(serialized.size()));
    output.close();
    if (!output) return Status::Unavailable("cannot write TensorRT engine output file");
  }
#ifdef _WIN32
  if (!MoveFileExW(temporary_path.c_str(), engine_path.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return Status::Unavailable("cannot replace TensorRT engine output file: " +
                               std::to_string(GetLastError()));
  }
#else
  std::filesystem::rename(temporary_path, engine_path, error);
  if (error) return Status::Unavailable("cannot replace TensorRT engine output file: " +
                                        error.message());
#endif
  return Status();
}

}  // namespace

std::size_t DataTypeSize(TensorDataType data_type) {
  switch (data_type) {
    case TensorDataType::kFloat32:
    case TensorDataType::kInt32: return 4U;
    case TensorDataType::kFloat16:
    case TensorDataType::kBFloat16: return 2U;
    case TensorDataType::kInt8:
    case TensorDataType::kUInt8:
    case TensorDataType::kBool: return 1U;
    case TensorDataType::kInt64: return 8U;
    case TensorDataType::kUnsupported: return 0U;
  }
  return 0U;
}

const char* DataTypeName(TensorDataType data_type) {
  switch (data_type) {
    case TensorDataType::kFloat32: return "float32";
    case TensorDataType::kFloat16: return "float16";
    case TensorDataType::kBFloat16: return "bfloat16";
    case TensorDataType::kInt8: return "int8";
    case TensorDataType::kInt32: return "int32";
    case TensorDataType::kInt64: return "int64";
    case TensorDataType::kUInt8: return "uint8";
    case TensorDataType::kBool: return "bool";
    case TensorDataType::kUnsupported: return "unsupported";
  }
  return "unsupported";
}

StatusOr<std::vector<DeviceInfo>> EnumerateDevices() {
  int count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&count);
  if (count_status != cudaSuccess) return CudaStatus(count_status, "cudaGetDeviceCount");
  std::vector<DeviceInfo> devices;
  devices.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    cudaDeviceProp properties{};
    const cudaError_t status = cudaGetDeviceProperties(&properties, index);
    if (status != cudaSuccess) return CudaStatus(status, "cudaGetDeviceProperties");
    DeviceInfo device;
    device.index = index;
    device.name = properties.name;
    device.total_memory_bytes = properties.totalGlobalMem;
    device.compute_capability_major = properties.major;
    device.compute_capability_minor = properties.minor;
    devices.push_back(std::move(device));
  }
  return devices;
}

Status BuildEngineFromOnnx(const std::filesystem::path& onnx_path,
                           const std::filesystem::path& engine_path,
                           const BuildOptions& options,
                           const LogHandler& log_handler) {
  if (options.device_index < 0) {
    return Status::InvalidArgument("TensorRT device index cannot be negative");
  }
  if (options.workspace_memory_bytes == 0U) {
    return Status::InvalidArgument("TensorRT workspace memory must be greater than zero");
  }
  Status device_status = CudaStatus(cudaSetDevice(options.device_index), "cudaSetDevice");
  if (!device_status.ok()) return device_status;
  std::error_code file_error;
  if (!std::filesystem::is_regular_file(onnx_path, file_error)) {
    return Status::NotFound("ONNX model file does not exist");
  }
  TrtLogger logger(log_handler);
  if (!initLibNvInferPlugins(&logger, "")) {
    return Status::Unavailable("TensorRT could not initialize its built-in plugins");
  }
  TrtObject<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) return Status::Unavailable("TensorRT could not create an ONNX builder");
  TrtObject<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
  if (!network) return Status::Unavailable("TensorRT could not create a network definition");
  TrtObject<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
  if (!parser) return Status::Unavailable("TensorRT could not create its ONNX parser");
  const std::string onnx_utf8_path = Utf8Path(onnx_path);
  if (!parser->parseFromFile(onnx_utf8_path.c_str(),
                             static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    return Status::InvalidArgument("TensorRT could not parse the ONNX model; see TensorRT log output");
  }

  TrtObject<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!config) return Status::Unavailable("TensorRT could not create a builder configuration");
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             options.workspace_memory_bytes);
  if (!options.allow_tf32) config->clearFlag(nvinfer1::BuilderFlag::kTF32);

  std::unordered_map<std::string, const InputShapeRange*> configured_ranges;
  std::unordered_set<std::string> applied_ranges;
  for (const InputShapeRange& range : options.dynamic_input_ranges) {
    if (range.tensor_name.empty() || !configured_ranges.emplace(range.tensor_name, &range).second) {
      return Status::InvalidArgument("TensorRT dynamic input ranges must have unique tensor names");
    }
    if (range.min_dimensions.size() != range.opt_dimensions.size() ||
        range.min_dimensions.size() != range.max_dimensions.size()) {
      return Status::InvalidArgument("min/opt/max TensorRT input shapes must have the same rank");
    }
    for (std::size_t dimension = 0U; dimension < range.min_dimensions.size(); ++dimension) {
      const auto minimum = range.min_dimensions[dimension];
      const auto optimum = range.opt_dimensions[dimension];
      const auto maximum = range.max_dimensions[dimension];
      if (minimum <= 0 || minimum > optimum || optimum > maximum ||
          maximum > std::numeric_limits<int32_t>::max()) {
        return Status::InvalidArgument("TensorRT input dimensions must satisfy 0 < min <= opt <= max");
      }
    }
  }

  bool needs_profile = false;
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  if (profile == nullptr) return Status::Unavailable("TensorRT could not create a shape profile");
  for (int32_t index = 0; index < network->getNbInputs(); ++index) {
    const nvinfer1::ITensor* input = network->getInput(index);
    if (input == nullptr) return Status::Internal("TensorRT ONNX network contains a null input");
    const nvinfer1::Dims dimensions = input->getDimensions();
    if (!IsDynamic(dimensions)) continue;
    needs_profile = true;
    const auto configured = configured_ranges.find(input->getName());
    if (configured == configured_ranges.end()) {
      return Status::InvalidArgument(std::string("dynamic ONNX input needs a shape range: ") +
                                     input->getName());
    }
    if (input->isShapeTensor()) {
      return Status::Unsupported("TensorRT ONNX shape-tensor profile values are not supported yet");
    }
    const InputShapeRange& range = *configured->second;
    applied_ranges.insert(configured->first);
    if (range.min_dimensions.size() != static_cast<std::size_t>(dimensions.nbDims)) {
      return Status::InvalidArgument(std::string("shape range rank does not match ONNX input: ") +
                                     input->getName());
    }
    nvinfer1::Dims minimum{};
    nvinfer1::Dims optimum{};
    nvinfer1::Dims maximum{};
    if (!ToDims(range.min_dimensions, &minimum) || !ToDims(range.opt_dimensions, &optimum) ||
        !ToDims(range.max_dimensions, &maximum) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minimum) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optimum) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maximum)) {
      return Status::InvalidArgument(std::string("TensorRT rejected the shape range for input: ") +
                                     input->getName());
    }
  }
  if (applied_ranges.size() != configured_ranges.size()) {
    return Status::InvalidArgument("TensorRT shape profile names must match dynamic ONNX inputs");
  }
  if (needs_profile && config->addOptimizationProfile(profile) < 0) {
    return Status::InvalidArgument("TensorRT could not add the dynamic-shape optimization profile");
  }

  TrtObject<nvinfer1::IHostMemory> serialized(
      builder->buildSerializedNetwork(*network, *config));
  if (!serialized) return Status::Unavailable("TensorRT failed to build the serialized engine");
  return SaveSerializedEngine(engine_path, *serialized);
}

Status BuildGgufQuantizedMatVecEngine(
    uint32_t tensor_type, uint64_t input_features, uint64_t output_features,
    const std::filesystem::path& engine_path, const BuildOptions& options,
    const LogHandler& log_handler) {
  if (options.device_index < 0) {
    return Status::InvalidArgument("TensorRT device index cannot be negative");
  }
  if (options.workspace_memory_bytes == 0U) {
    return Status::InvalidArgument("TensorRT workspace memory must be greater than zero");
  }
  if (!IsGgufTensorEncodingDecodable(tensor_type)) {
    return Status::Unsupported("TensorRT GGUF matvec does not support this source encoding");
  }
  const auto block = GgufQuantBlockInfoForType(tensor_type);
  if (!block.has_value() || input_features == 0U || output_features == 0U ||
      input_features % block->elements != 0U ||
      input_features > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      output_features > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return Status::InvalidArgument("TensorRT GGUF matvec dimensions do not match the encoding block");
  }
  const uint64_t bytes_per_row = (input_features / block->elements) * block->bytes;
  if (bytes_per_row == 0U || output_features >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / bytes_per_row) {
    return Status::InvalidArgument("TensorRT GGUF packed weight input is too large");
  }
  const uint64_t total_weight_bytes = bytes_per_row * output_features;
  Status device_status = CudaStatus(cudaSetDevice(options.device_index), "cudaSetDevice");
  if (!device_status.ok()) return device_status;
  TrtLogger logger(log_handler);
  if (!initLibNvInferPlugins(&logger, "")) {
    return Status::Unavailable("TensorRT could not initialize its built-in plugins");
  }
  Status plugin_status = detail::RegisterGgufQuantizedMatVecPlugin();
  if (!plugin_status.ok()) return plugin_status;

  TrtObject<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) return Status::Unavailable("TensorRT could not create a builder");
  std::unique_ptr<nvinfer1::IPluginV3> plugin;
  TrtObject<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
  if (!network) return Status::Unavailable("TensorRT could not create a network definition");

  nvinfer1::Dims activation_dims{};
  activation_dims.nbDims = 2;
  activation_dims.d[0] = 1;
  activation_dims.d[1] = static_cast<int32_t>(input_features);
  nvinfer1::ITensor* activations =
      network->addInput("activations", nvinfer1::DataType::kFLOAT, activation_dims);
  nvinfer1::Dims weight_dims{};
  weight_dims.nbDims = 1;
  weight_dims.d[0] = static_cast<int32_t>(total_weight_bytes);
  nvinfer1::ITensor* packed_weights =
      network->addInput("packed_weights", nvinfer1::DataType::kINT8, weight_dims);
  if (activations == nullptr || packed_weights == nullptr) {
    return Status::Unavailable("TensorRT could not create the GGUF matvec inputs");
  }

  plugin.reset(detail::CreateGgufQuantizedMatVecPlugin(
      tensor_type, input_features, output_features));
  if (!plugin) return Status::Unavailable("could not create the GGUF matvec plugin");
  nvinfer1::ITensor* plugin_inputs[]{activations, packed_weights};
  nvinfer1::IPluginV3Layer* layer =
      network->addPluginV3(plugin_inputs, 2, nullptr, 0, *plugin);
  if (layer == nullptr) {
    return Status::InvalidArgument("TensorRT rejected the GGUF quantized matvec plugin");
  }
  layer->setName("isvik_gguf_quantized_matvec");
  nvinfer1::ITensor* result = layer->getOutput(0);
  if (result == nullptr) return Status::Internal("TensorRT GGUF matvec has no output tensor");
  result->setName("output");
  network->markOutput(*result);

  TrtObject<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!config) return Status::Unavailable("TensorRT could not create a builder configuration");
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             options.workspace_memory_bytes);
  TrtObject<nvinfer1::IHostMemory> serialized(
      builder->buildSerializedNetwork(*network, *config));
  if (!serialized) return Status::Unavailable("TensorRT failed to build the GGUF matvec engine");
  return SaveSerializedEngine(engine_path, *serialized);
}

struct Engine::Impl {
  explicit Impl(LogHandler handler) : logger(std::move(handler)) {}

  mutable std::mutex mutex;
  TrtLogger logger;
  TrtObject<nvinfer1::IRuntime> runtime;
  TrtObject<nvinfer1::ICudaEngine> engine;
  TrtObject<nvinfer1::IExecutionContext> context;
  cudaStream_t stream = nullptr;
  int device = 0;
  std::filesystem::path path;
  std::unordered_map<std::string, DeviceBuffer> device_buffers;

  ~Impl() { Reset(); }

  void Reset() {
    context.reset();
    engine.reset();
    runtime.reset();
    if (stream != nullptr || !device_buffers.empty()) {
      static_cast<void>(cudaSetDevice(device));
      device_buffers.clear();
    }
    if (stream != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream));
      stream = nullptr;
    }
    path.clear();
  }

  std::vector<TensorInfo> Describe() const {
    std::vector<TensorInfo> result;
    if (!engine) return result;
    const int32_t count = engine->getNbIOTensors();
    result.reserve(static_cast<std::size_t>(count));
    for (int32_t index = 0; index < count; ++index) {
      const char* name = engine->getIOTensorName(index);
      if (name == nullptr) continue;
      TensorInfo info;
      info.name = name;
      info.mode = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT
                      ? TensorMode::kInput : TensorMode::kOutput;
      info.data_type = FromTrtDataType(engine->getTensorDataType(name));
      info.dimensions = FromDims(engine->getTensorShape(name));
      info.shape_inference_io = engine->isShapeInferenceIO(name);
      result.push_back(std::move(info));
    }
    return result;
  }
};

Engine::Engine(LogHandler log_handler) : impl_(std::make_unique<Impl>(std::move(log_handler))) {}
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

Status Engine::Load(const std::filesystem::path& engine_path, int device_index) {
  if (!impl_) return Status::Internal("TensorRT engine is moved from");
  std::lock_guard lock(impl_->mutex);
  impl_->Reset();
  if (device_index < 0) return Status::InvalidArgument("TensorRT device index cannot be negative");
  std::error_code error;
  if (!std::filesystem::is_regular_file(engine_path, error)) {
    return Status::NotFound("TensorRT engine file does not exist");
  }
  const std::uintmax_t file_size = std::filesystem::file_size(engine_path, error);
  if (error || file_size == 0U ||
      file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()) ||
      file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::InvalidArgument("TensorRT engine file is empty, too large, or unreadable");
  }
  std::vector<char> serialized(static_cast<std::size_t>(file_size));
  std::ifstream input(engine_path, std::ios::binary);
  input.read(serialized.data(), static_cast<std::streamsize>(serialized.size()));
  if (!input) return Status::Unavailable("cannot read TensorRT engine file");

  Status device_status = CudaStatus(cudaSetDevice(device_index), "cudaSetDevice");
  if (!device_status.ok()) return device_status;
  if (!initLibNvInferPlugins(&impl_->logger, "")) {
    return Status::Unavailable("TensorRT could not initialize its built-in plugins");
  }
  Status plugin_status = detail::RegisterGgufQuantizedMatVecPlugin();
  if (!plugin_status.ok()) return plugin_status;
  TrtObject<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(impl_->logger));
  if (!runtime) return Status::Unavailable("TensorRT could not create a runtime");
  TrtObject<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
  if (!engine) {
    return Status::InvalidArgument("TensorRT could not deserialize this engine on the selected device");
  }
  TrtObject<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
  if (!context) return Status::Unavailable("TensorRT could not create an execution context");
  cudaStream_t stream = nullptr;
  const cudaError_t stream_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (stream_status != cudaSuccess) return CudaStatus(stream_status, "cudaStreamCreateWithFlags");

  impl_->runtime = std::move(runtime);
  impl_->engine = std::move(engine);
  impl_->context = std::move(context);
  impl_->stream = stream;
  impl_->device = device_index;
  impl_->path = engine_path;
  return Status();
}

StatusOr<std::vector<Tensor>> Engine::Infer(const std::vector<Tensor>& inputs) {
  if (!impl_) return Status::Internal("TensorRT engine is moved from");
  std::lock_guard lock(impl_->mutex);
  if (!impl_->engine || !impl_->context || impl_->stream == nullptr) {
    return Status::Unavailable("no TensorRT engine is loaded");
  }
  Status status = CudaStatus(cudaSetDevice(impl_->device), "cudaSetDevice");
  if (!status.ok()) return status;
  PendingStreamWork pending_work{impl_->stream};

  std::unordered_map<std::string, const Tensor*> named_inputs;
  for (const Tensor& input : inputs) {
    if (!named_inputs.emplace(input.name, &input).second) {
      return Status::InvalidArgument("duplicate TensorRT input tensor: " + input.name);
    }
  }
  for (const TensorInfo& info : impl_->Describe()) {
    if (info.mode != TensorMode::kInput) continue;
    const auto found = named_inputs.find(info.name);
    if (found == named_inputs.end()) {
      return Status::InvalidArgument("missing TensorRT input tensor: " + info.name);
    }
    const Tensor& tensor = *found->second;
    if (tensor.data_type != info.data_type) {
      return Status::InvalidArgument("TensorRT input data type mismatch: " + info.name);
    }
    status = ValidatePayload(tensor);
    if (!status.ok()) return status;
    if (impl_->engine->getTensorLocation(info.name.c_str()) != nvinfer1::TensorLocation::kDEVICE) {
      return Status::Unsupported("host-resident TensorRT input tensors are not supported: " + info.name);
    }
    if (info.shape_inference_io) {
      return Status::Unsupported("TensorRT shape-inference I/O tensors are not supported by this runner: " +
                                 info.name);
    }
    const nvinfer1::Dims engine_dimensions = impl_->engine->getTensorShape(info.name.c_str());
    if (IsDynamic(engine_dimensions)) {
      nvinfer1::Dims runtime_dimensions{};
      if (!ToDims(tensor.dimensions, &runtime_dimensions) ||
          !impl_->context->setInputShape(info.name.c_str(), runtime_dimensions)) {
        return Status::InvalidArgument("TensorRT rejected runtime input shape: " + info.name);
      }
    } else if (FromDims(engine_dimensions) != tensor.dimensions) {
      return Status::InvalidArgument("TensorRT input shape mismatch: " + info.name);
    }
  }
  for (const auto& [name, tensor] : named_inputs) {
    (void)tensor;
    if (impl_->engine->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kINPUT) {
      return Status::InvalidArgument("unknown TensorRT input tensor: " + name);
    }
  }

  std::vector<Tensor> outputs;
  const std::vector<TensorInfo> infos = impl_->Describe();
  outputs.reserve(infos.size());

  for (const TensorInfo& info : infos) {
    if (info.mode != TensorMode::kOutput) continue;
    if (info.data_type == TensorDataType::kUnsupported) {
      return Status::Unsupported("unsupported TensorRT output type for tensor: " + info.name);
    }
    if (info.shape_inference_io) {
      return Status::Unsupported("TensorRT shape-inference outputs are not supported by this runner: " +
                                 info.name);
    }
    if (impl_->engine->getTensorLocation(info.name.c_str()) != nvinfer1::TensorLocation::kDEVICE) {
      return Status::Unsupported("host-resident TensorRT output tensors are not supported: " + info.name);
    }
    Tensor tensor;
    tensor.name = info.name;
    tensor.data_type = info.data_type;
    tensor.dimensions = FromDims(impl_->context->getTensorShape(info.name.c_str()));
    std::size_t count = 0U;
    const std::size_t element_size = DataTypeSize(tensor.data_type);
    if (element_size == 0U || !CheckedElementCount(tensor.dimensions, &count) ||
        count > std::numeric_limits<std::size_t>::max() / element_size) {
      return Status::Unsupported("TensorRT output shape is unresolved or too large: " + info.name);
    }
    tensor.bytes.resize(count * element_size);
    outputs.push_back(std::move(tensor));
  }

  std::unordered_map<std::string, void*> addresses;
  for (const TensorInfo& info : infos) {
    if (info.mode == TensorMode::kInput) {
      const Tensor& tensor = *named_inputs.at(info.name);
      if (impl_->engine->getTensorLocation(info.name.c_str()) == nvinfer1::TensorLocation::kHOST) {
        addresses.emplace(info.name, const_cast<std::uint8_t*>(tensor.bytes.data()));
      } else {
        DeviceBuffer& buffer = impl_->device_buffers[info.name];
        status = EnsureDeviceBuffer(&buffer, tensor.bytes.size());
        if (!status.ok()) return status;
        void* pointer = buffer.pointer;
        pending_work.pending = true;
        status = CudaStatus(cudaMemcpyAsync(pointer, tensor.bytes.data(),
                                            tensor.bytes.size(),
                                            cudaMemcpyHostToDevice, impl_->stream),
                            "cudaMemcpyAsync(input)");
        if (!status.ok()) return status;
        addresses.emplace(info.name, pointer);
      }
    }
  }
  for (Tensor& tensor : outputs) {
    if (impl_->engine->getTensorLocation(tensor.name.c_str()) == nvinfer1::TensorLocation::kHOST) {
      addresses.emplace(tensor.name, tensor.bytes.data());
    } else {
      DeviceBuffer& buffer = impl_->device_buffers[tensor.name];
      status = EnsureDeviceBuffer(&buffer, tensor.bytes.size());
      if (!status.ok()) return status;
      addresses.emplace(tensor.name, buffer.pointer);
    }
  }
  for (const auto& [name, address] : addresses) {
    if (!impl_->context->setTensorAddress(name.c_str(), address)) {
      return Status::InvalidArgument("TensorRT rejected a tensor address: " + name);
    }
  }
  pending_work.pending = true;
  if (!impl_->context->enqueueV3(impl_->stream)) {
    return Status::Unavailable("TensorRT enqueueV3 failed; see TensorRT log output");
  }
  for (Tensor& tensor : outputs) {
    if (impl_->engine->getTensorLocation(tensor.name.c_str()) != nvinfer1::TensorLocation::kHOST) {
      status = CudaStatus(cudaMemcpyAsync(tensor.bytes.data(), addresses.at(tensor.name),
                                          tensor.bytes.size(), cudaMemcpyDeviceToHost,
                                          impl_->stream),
                          "cudaMemcpyAsync(output)");
      if (!status.ok()) return status;
    }
  }
  status = CudaStatus(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");
  if (!status.ok()) return status;
  pending_work.pending = false;
  return outputs;
}

std::vector<TensorInfo> Engine::Tensors() const {
  if (!impl_) return {};
  std::lock_guard lock(impl_->mutex);
  return impl_->Describe();
}

std::filesystem::path Engine::loaded_path() const {
  if (!impl_) return {};
  std::lock_guard lock(impl_->mutex);
  return impl_->path;
}

int Engine::device_index() const {
  if (!impl_) return -1;
  std::lock_guard lock(impl_->mutex);
  return impl_->device;
}

void Engine::Unload() {
  if (!impl_) return;
  std::lock_guard lock(impl_->mutex);
  impl_->Reset();
}

}  // namespace isvik::tensorrt_backend
