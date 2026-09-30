#include "cli_app.h"
#include "isvik/app/api_server.h"
#include "native_dialog.h"
#include "isvik/core/chat_context.h"
#include "isvik/core/gguf_inspector.h"
#include "isvik/core/memory_service.h"
#include "isvik/core/model_catalog.h"
#include "isvik/i18n/localization.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#include <bcrypt.h>
#endif

#if defined(ISVIK_HAS_OPENVINO_RUNTIME)
#include "isvik/backends/openvino/device_provider.h"
#endif
#if defined(ISVIK_HAS_OPENVINO_GENAI)
#include "isvik/backends/openvino/genai_engine.h"
#endif
#if defined(ISVIK_HAS_TENSORRT)
#include "isvik/backends/tensorrt/engine.h"
#include "isvik/backends/tensorrt/gguf_genai_engine.h"
#endif
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
#include "isvik/backends/onnxruntime/genai_engine.h"
#endif
#include "isvik/core/model_manager.h"

namespace isvik::app {
namespace {

struct Options {
  bool help = false;
  bool list_devices = false;
  bool inspect = false;
  bool run_once = false;
  bool interactive = false;
  bool server = false;
  bool device_explicit = false;
  bool backend_explicit = false;
  std::string server_backend = "openvino";
  std::string server_host = "127.0.0.1";
  std::string api_key;
  int server_port = 1234;
  bool tensorrt_devices = false;
  bool tensorrt_inspect = false;
  bool tensorrt_build = false;
  bool tensorrt_run = false;
  std::string model_path;
  std::string inspect_path;
  std::string tensorrt_onnx_path;
  std::string tensorrt_engine_path;
  std::string tensorrt_output_directory;
  std::vector<std::string> tensorrt_inputs;
  std::vector<std::string> tensorrt_shapes;
  std::vector<std::string> tensorrt_profiles;
  int tensorrt_device = 0;
  uint64_t tensorrt_workspace_mib = 2048U;
  std::string device = "CPU";
  std::string prompt;
  int max_tokens = 128;
  uint64_t context_tokens = 8192U;
  bool use_memories = true;
  GenerationConfig generation;
};

void PrintUsage(std::ostream& output) {
  output <<
      "Isvik.cpp local inference CLI / Isvik.cpp 本地推理命令行\n"
      "\n"
      "Usage / 用法:\n"
      "  Isvik                         Open the desktop GUI / 打开桌面界面\n"
      "  Isvik --help                  Show this help / 显示帮助\n"
      "  Isvik --devices               List OpenVINO devices / 列出 OpenVINO 设备\n"
      "  Isvik --inspect MODEL         Inspect a local model / 检查本地模型\n"
      "  Isvik --trt-devices           List NVIDIA CUDA devices / 列出 NVIDIA CUDA 设备\n"
      "  Isvik --trt-build --onnx FILE --output-engine FILE [--workspace-mib N]\n"
      "                                Build a TensorRT engine / 构建 TensorRT 引擎\n"
      "  Isvik --trt-inspect ENGINE    Inspect a TensorRT engine / 检查 TensorRT 引擎\n"
      "  Isvik --trt-run ENGINE ...    Run a TensorRT engine / 运行 TensorRT 引擎\n"
      "  Isvik --run --model MODEL --prompt TEXT [options]  Run once / 单次推理\n"
      "  Isvik -cli [options]          Start the interactive REPL / 启动交互式终端工作台\n"
      "  Isvik -cli --model MODEL      Start the REPL with a model / 指定模型启动工作台\n"
      "  Isvik --server --model MODEL  Start the local API server / 启动本地 API 服务\n"
      "\n"
      "Options / 选项:\n"
      "  --device DEVICE       CPU/GPU/NPU/AUTO; default CPU / 设备；默认 CPU\n"
      "  --backend NAME        openvino/onnxruntime/tensorrt / 推理后端\n"
      "  --host ADDRESS        API bind address / API 监听地址\n"
      "  --port NUMBER         API port; default 1234 / API 端口；默认 1234\n"
      "  --api-key KEY         Require API authentication / API 鉴权密钥\n"
      "  --max-tokens COUNT    Maximum output tokens / 最大生成词元数\n"
      "  --context-tokens N    Context budget / 上下文长度\n"
      "  --no-memory           Disable saved memories / 不使用已保存记忆\n"
      "  --greedy              Greedy decoding / 贪心解码\n"
      "  --sampling            Sampling with temperature/top-p/top-k/seed / 采样解码\n"
      "  --stop TEXT           Stop sequence; repeatable / 停止序列；可重复\n"
      "  --temperature VALUE   Sampling temperature / 采样温度\n"
      "  --top-p VALUE         Nucleus sampling threshold / 核采样阈值\n"
      "  --top-k COUNT         Top-k sampling limit / Top-k 采样数量\n"
      "  --seed COUNT          Sampling seed / 采样随机种子\n"
      "  --trt-device INDEX    TensorRT CUDA device / TensorRT CUDA 设备编号\n"
      "  --trt-input NAME=FILE Tensor input bytes / 输入张量文件\n"
      "  --trt-shape NAME=DIMS Runtime tensor shape / 张量运行时形状\n"
      "  --workspace-mib N     TensorRT build workspace / TensorRT 构建工作区\n"
      "  --profile NAME=MIN:OPT:MAX  Dynamic ONNX shape profile / ONNX 动态形状配置\n"
      "\n"
      "REPL / 交互命令: -ls, -use N|ID, -stop, -backend, -devices, -info, -import, -rm,\n"
      "                 -new, -history, -system, -lang zh|en, -exit\n"
      "Run Isvik -cli for the persistent model workbench. / 输入 Isvik -cli 启动常驻模型工作台。\n";
}

bool ParseInt(std::string_view text, int* value) {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), *value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool ParseUInt64(std::string_view text, uint64_t* value) {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), *value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool ParseFloat(std::string_view text, float* value) {
  std::string copy(text);
  char* end = nullptr;
  const float parsed = std::strtof(copy.c_str(), &end);
  if (end == copy.c_str() || end != copy.c_str() + copy.size()) return false;
  *value = parsed;
  return true;
}

StatusOr<Options> ParseOptions(int argc, char** argv) {
  Options options;
  options.generation.decoding_mode = DecodingMode::kGreedy;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto value_after = [&](const char* name) -> StatusOr<std::string> {
      if (index + 1 >= argc) {
        return Status::InvalidArgument(std::string("missing value for ") + name);
      }
      ++index;
      return std::string(argv[index]);
    };

    if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else if (argument == "--devices") {
      options.list_devices = true;
    } else if (argument == "--trt-devices") {
      options.tensorrt_devices = true;
    } else if (argument == "--inspect") {
      StatusOr<std::string> value = value_after("--inspect");
      if (!value.ok()) return value.status();
      options.inspect = true;
      options.inspect_path = std::move(value).value();
    } else if (argument == "--trt-inspect") {
      StatusOr<std::string> value = value_after("--trt-inspect");
      if (!value.ok()) return value.status();
      options.tensorrt_inspect = true;
      options.tensorrt_engine_path = std::move(value).value();
    } else if (argument == "--trt-build") {
      options.tensorrt_build = true;
    } else if (argument == "--trt-run") {
      options.tensorrt_run = true;
      if (index + 1 < argc && std::string_view(argv[index + 1]).starts_with('-') == false) {
        options.tensorrt_engine_path = argv[++index];
      }
    } else if (argument == "--onnx") {
      StatusOr<std::string> value = value_after("--onnx");
      if (!value.ok()) return value.status();
      options.tensorrt_onnx_path = std::move(value).value();
    } else if (argument == "--engine") {
      StatusOr<std::string> value = value_after("--engine");
      if (!value.ok()) return value.status();
      options.tensorrt_engine_path = std::move(value).value();
    } else if (argument == "--output-engine") {
      StatusOr<std::string> value = value_after("--output-engine");
      if (!value.ok()) return value.status();
      options.tensorrt_engine_path = std::move(value).value();
    } else if (argument == "--trt-input") {
      StatusOr<std::string> value = value_after("--trt-input");
      if (!value.ok()) return value.status();
      options.tensorrt_inputs.push_back(std::move(value).value());
    } else if (argument == "--trt-shape") {
      StatusOr<std::string> value = value_after("--trt-shape");
      if (!value.ok()) return value.status();
      options.tensorrt_shapes.push_back(std::move(value).value());
    } else if (argument == "--profile") {
      StatusOr<std::string> value = value_after("--profile");
      if (!value.ok()) return value.status();
      options.tensorrt_profiles.push_back(std::move(value).value());
    } else if (argument == "--output-dir") {
      StatusOr<std::string> value = value_after("--output-dir");
      if (!value.ok()) return value.status();
      options.tensorrt_output_directory = std::move(value).value();
    } else if (argument == "--trt-device") {
      StatusOr<std::string> value = value_after("--trt-device");
      if (!value.ok()) return value.status();
      if (!ParseInt(value.value(), &options.tensorrt_device) || options.tensorrt_device < 0) {
        return Status::InvalidArgument("--trt-device must be a non-negative integer");
      }
    } else if (argument == "--workspace-mib") {
      StatusOr<std::string> value = value_after("--workspace-mib");
      if (!value.ok()) return value.status();
      if (!ParseUInt64(value.value(), &options.tensorrt_workspace_mib) ||
          options.tensorrt_workspace_mib == 0U || options.tensorrt_workspace_mib > 1048576U) {
        return Status::InvalidArgument("--workspace-mib must be between 1 and 1048576");
      }
    } else if (argument == "--run") {
      options.run_once = true;
    } else if (argument == "--server") {
      options.server = true;
    } else if (argument == "--backend") {
      StatusOr<std::string> value = value_after("--backend");
      if (!value.ok()) return value.status();
      options.server_backend = std::move(value).value();
      options.backend_explicit = true;
    } else if (argument == "--host") {
      StatusOr<std::string> value = value_after("--host");
      if (!value.ok()) return value.status();
      options.server_host = std::move(value).value();
    } else if (argument == "--port") {
      StatusOr<std::string> value = value_after("--port");
      if (!value.ok()) return value.status();
      if (!ParseInt(value.value(), &options.server_port) || options.server_port < 1 || options.server_port > 65535)
        return Status::InvalidArgument("--port must be between 1 and 65535");
    } else if (argument == "--api-key") {
      StatusOr<std::string> value = value_after("--api-key");
      if (!value.ok()) return value.status();
      options.api_key = std::move(value).value();
    } else if (argument == "-cli" || argument == "--cli" || argument == "--chat") {
      options.interactive = true;
    } else if (argument == "--model") {
      StatusOr<std::string> value = value_after("--model");
      if (!value.ok()) return value.status();
      options.model_path = std::move(value).value();
    } else if (argument == "--device") {
      StatusOr<std::string> value = value_after("--device");
      if (!value.ok()) return value.status();
      options.device = std::move(value).value();
      options.device_explicit = true;
    } else if (argument == "--prompt") {
      StatusOr<std::string> value = value_after("--prompt");
      if (!value.ok()) return value.status();
      options.prompt = std::move(value).value();
    } else if (argument == "--max-tokens") {
      StatusOr<std::string> value = value_after("--max-tokens");
      if (!value.ok()) return value.status();
      if (!ParseInt(value.value(), &options.max_tokens) || options.max_tokens <= 0) {
        return Status::InvalidArgument("--max-tokens must be a positive integer");
      }
    } else if (argument == "--context-tokens") {
      StatusOr<std::string> value = value_after("--context-tokens");
      if (!value.ok()) return value.status();
      if (!ParseUInt64(value.value(), &options.context_tokens) ||
          options.context_tokens == 0U || options.context_tokens > 1000000U) {
        return Status::InvalidArgument("--context-tokens must be between 1 and 1000000");
      }
    } else if (argument == "--no-memory") {
      options.use_memories = false;
    } else if (argument == "--greedy") {
      options.generation.decoding_mode = DecodingMode::kGreedy;
    } else if (argument == "--sampling") {
      options.generation.decoding_mode = DecodingMode::kSampling;
    } else if (argument == "--temperature") {
      StatusOr<std::string> value = value_after("--temperature");
      if (!value.ok()) return value.status();
      float parsed = 0.0F;
      if (!ParseFloat(value.value(), &parsed)) {
        return Status::InvalidArgument("--temperature must be numeric");
      }
      options.generation.sampling.temperature = parsed;
      options.generation.decoding_mode = DecodingMode::kSampling;
    } else if (argument == "--top-p") {
      StatusOr<std::string> value = value_after("--top-p");
      if (!value.ok()) return value.status();
      float parsed = 0.0F;
      if (!ParseFloat(value.value(), &parsed)) {
        return Status::InvalidArgument("--top-p must be numeric");
      }
      options.generation.sampling.top_p = parsed;
      options.generation.decoding_mode = DecodingMode::kSampling;
    } else if (argument == "--top-k") {
      StatusOr<std::string> value = value_after("--top-k");
      if (!value.ok()) return value.status();
      int parsed = 0;
      if (!ParseInt(value.value(), &parsed)) {
        return Status::InvalidArgument("--top-k must be an integer");
      }
      options.generation.sampling.top_k = parsed;
      options.generation.decoding_mode = DecodingMode::kSampling;
    } else if (argument == "--seed") {
      StatusOr<std::string> value = value_after("--seed");
      if (!value.ok()) return value.status();
      uint64_t parsed = 0U;
      if (!ParseUInt64(value.value(), &parsed)) {
        return Status::InvalidArgument("--seed must be a non-negative integer");
      }
      options.generation.sampling.seed = parsed;
      options.generation.decoding_mode = DecodingMode::kSampling;
    } else if (argument == "--stop") {
      StatusOr<std::string> value = value_after("--stop");
      if (!value.ok()) return value.status();
      options.generation.stop_sequences.push_back(std::move(value).value());
    } else {
      return Status::InvalidArgument("unknown option: " + std::string(argument));
    }
  }
  options.generation.max_tokens = options.max_tokens;

  const unsigned int action_count = static_cast<unsigned int>(options.list_devices)
      + static_cast<unsigned int>(options.inspect)
      + static_cast<unsigned int>(options.run_once)
      + static_cast<unsigned int>(options.interactive)
      + static_cast<unsigned int>(options.server)
      + static_cast<unsigned int>(options.tensorrt_devices)
      + static_cast<unsigned int>(options.tensorrt_inspect)
      + static_cast<unsigned int>(options.tensorrt_build)
      + static_cast<unsigned int>(options.tensorrt_run);
  if (action_count > 1U) {
    return Status::InvalidArgument("choose one command: --devices, --inspect, --trt-devices, "
                                   "--trt-inspect, --trt-build, --trt-run, --run, -cli, or --server");
  }
  if (options.tensorrt_build &&
      (options.tensorrt_onnx_path.empty() || options.tensorrt_engine_path.empty())) {
    return Status::InvalidArgument("--trt-build requires --onnx and --output-engine");
  }
  if (options.tensorrt_inspect && options.tensorrt_engine_path.empty()) {
    return Status::InvalidArgument("--trt-inspect requires an engine path");
  }
  if (options.tensorrt_run && (options.tensorrt_engine_path.empty() ||
      options.tensorrt_output_directory.empty() || options.tensorrt_inputs.empty())) {
    return Status::InvalidArgument("--trt-run requires --engine, at least one --trt-input, "
                                   "and --output-dir");
  }
  if (options.server && options.model_path.empty()) {
    return Status::InvalidArgument("--model is required with --server");
  }
  if (options.server && options.server_backend != "openvino" &&
      options.server_backend != "onnxruntime" && options.server_backend != "tensorrt") {
    return Status::InvalidArgument("--backend must be openvino, onnxruntime, or tensorrt");
  }
  if (options.run_once || options.interactive) {
    if (options.run_once && options.model_path.empty())
      return Status::InvalidArgument("--model is required with --run");
    if (options.run_once && options.prompt.empty()) {
      return Status::InvalidArgument("--prompt is required with --run");
    }
  }
  return options;
}

StatusOr<std::pair<std::string, std::string>> ParseTensorAssignment(
    std::string_view assignment, std::string_view option_name) {
  const std::size_t separator = assignment.find('=');
  if (separator == std::string_view::npos || separator == 0U ||
      separator + 1U >= assignment.size()) {
    return Status::InvalidArgument(std::string(option_name) + " expects NAME=VALUE");
  }
  return std::pair<std::string, std::string>{
      std::string(assignment.substr(0U, separator)),
      std::string(assignment.substr(separator + 1U))};
}

StatusOr<std::vector<std::int64_t>> ParseTensorShape(std::string_view value) {
  std::vector<std::int64_t> dimensions;
  std::size_t start = 0U;
  while (start < value.size()) {
    const std::size_t end = value.find(',', start);
    const std::string_view part = value.substr(start, end == std::string_view::npos
                                                        ? value.size() - start : end - start);
    std::int64_t dimension = 0;
    const auto parsed = std::from_chars(part.data(), part.data() + part.size(), dimension);
    if (part.empty() || parsed.ec != std::errc{} || parsed.ptr != part.data() + part.size() ||
        dimension <= 0 || (end != std::string_view::npos && end + 1U == value.size())) {
      return Status::InvalidArgument("TensorRT dimensions must be positive comma-separated integers");
    }
    dimensions.push_back(dimension);
    if (end == std::string_view::npos) break;
    start = end + 1U;
  }
  if (dimensions.empty()) return Status::InvalidArgument("TensorRT shape cannot be empty");
  return dimensions;
}

std::string ShapeLabel(const std::vector<std::int64_t>& dimensions) {
  std::string result = "[";
  for (std::size_t index = 0U; index < dimensions.size(); ++index) {
    if (index != 0U) result += ", ";
    result += std::to_string(dimensions[index]);
  }
  result += "]";
  return result;
}

std::string FormatBytes(uint64_t bytes);

#if defined(ISVIK_HAS_TENSORRT)
int RunTensorRtCommand(const Options& options) {
  using namespace tensorrt_backend;
  if (options.tensorrt_devices) {
    const StatusOr<std::vector<DeviceInfo>> devices = EnumerateDevices();
    if (!devices.ok()) {
      std::cerr << devices.status().message() << '\n';
      return 1;
    }
    for (const DeviceInfo& device : devices.value()) {
      std::cout << device.index << "\t" << device.name << "\tcompute "
                << device.compute_capability_major << '.' << device.compute_capability_minor
                << "\t" << FormatBytes(device.total_memory_bytes) << '\n';
    }
    return 0;
  }
  if (options.tensorrt_build) {
    BuildOptions build_options;
    build_options.device_index = options.tensorrt_device;
    constexpr std::uint64_t kBytesPerMiB = 1024ULL * 1024ULL;
    build_options.workspace_memory_bytes = static_cast<std::size_t>(
        options.tensorrt_workspace_mib * kBytesPerMiB);
    for (const std::string& profile_text : options.tensorrt_profiles) {
      StatusOr<std::pair<std::string, std::string>> parsed =
          ParseTensorAssignment(profile_text, "--profile");
      if (!parsed.ok()) {
        std::cerr << parsed.status().message() << '\n';
        return 2;
      }
      std::array<std::string_view, 3U> shapes{};
      std::string_view remaining(parsed.value().second);
      for (std::size_t index = 0U; index < shapes.size(); ++index) {
        const std::size_t separator = remaining.find(':');
        if (index + 1U < shapes.size() && separator == std::string_view::npos) {
          std::cerr << "--profile expects NAME=MIN:OPT:MAX\n";
          return 2;
        }
        if (index + 1U == shapes.size() && separator != std::string_view::npos) {
          std::cerr << "--profile expects exactly three shapes\n";
          return 2;
        }
        shapes[index] = remaining.substr(0U, separator);
        if (separator != std::string_view::npos) remaining.remove_prefix(separator + 1U);
      }
      InputShapeRange range;
      range.tensor_name = parsed.value().first;
      StatusOr<std::vector<std::int64_t>> minimum = ParseTensorShape(shapes[0]);
      StatusOr<std::vector<std::int64_t>> optimum = ParseTensorShape(shapes[1]);
      StatusOr<std::vector<std::int64_t>> maximum = ParseTensorShape(shapes[2]);
      if (!minimum.ok() || !optimum.ok() || !maximum.ok()) {
        const Status& shape_status = !minimum.ok() ? minimum.status()
            : !optimum.ok() ? optimum.status() : maximum.status();
        std::cerr << shape_status.message() << '\n';
        return 2;
      }
      range.min_dimensions = std::move(minimum).value();
      range.opt_dimensions = std::move(optimum).value();
      range.max_dimensions = std::move(maximum).value();
      build_options.dynamic_input_ranges.push_back(std::move(range));
    }
    const Status status = BuildEngineFromOnnx(
        PathFromUtf8(options.tensorrt_onnx_path),
        PathFromUtf8(options.tensorrt_engine_path), build_options);
    if (!status.ok()) {
      std::cerr << status.message() << '\n';
      return 1;
    }
    std::cout << "TensorRT engine written: " << options.tensorrt_engine_path << '\n';
    return 0;
  }

  Engine engine;
  const Status load_status = engine.Load(PathFromUtf8(options.tensorrt_engine_path),
                                         options.tensorrt_device);
  if (!load_status.ok()) {
    std::cerr << load_status.message() << '\n';
    return 1;
  }
  const std::vector<TensorInfo> tensors = engine.Tensors();
  if (options.tensorrt_inspect) {
    std::cout << "TensorRT engine: " << PathUtf8(engine.loaded_path()) << '\n'
              << "CUDA device index: " << engine.device_index() << '\n';
    for (const TensorInfo& tensor : tensors) {
      std::cout << (tensor.mode == TensorMode::kInput ? "INPUT  " : "OUTPUT ")
                << tensor.name << "\t" << DataTypeName(tensor.data_type) << "\t"
                << ShapeLabel(tensor.dimensions)
                << (tensor.shape_inference_io ? "\tshape-I/O" : "") << '\n';
    }
    return 0;
  }

  std::unordered_map<std::string, std::filesystem::path> input_paths;
  for (const std::string& assignment : options.tensorrt_inputs) {
    StatusOr<std::pair<std::string, std::string>> parsed =
        ParseTensorAssignment(assignment, "--trt-input");
    if (!parsed.ok()) {
      std::cerr << parsed.status().message() << '\n';
      return 2;
    }
    if (!input_paths.emplace(parsed.value().first, PathFromUtf8(parsed.value().second)).second) {
      std::cerr << "duplicate --trt-input tensor: " << parsed.value().first << '\n';
      return 2;
    }
  }
  std::unordered_map<std::string, std::vector<std::int64_t>> input_shapes;
  for (const std::string& assignment : options.tensorrt_shapes) {
    StatusOr<std::pair<std::string, std::string>> parsed =
        ParseTensorAssignment(assignment, "--trt-shape");
    if (!parsed.ok()) {
      std::cerr << parsed.status().message() << '\n';
      return 2;
    }
    StatusOr<std::vector<std::int64_t>> shape = ParseTensorShape(parsed.value().second);
    if (!shape.ok()) {
      std::cerr << shape.status().message() << '\n';
      return 2;
    }
    if (!input_shapes.emplace(parsed.value().first, std::move(shape).value()).second) {
      std::cerr << "duplicate --trt-shape tensor: " << parsed.value().first << '\n';
      return 2;
    }
  }
  for (const auto& [name, shape] : input_shapes) {
    (void)shape;
    if (std::none_of(tensors.begin(), tensors.end(), [&](const TensorInfo& info) {
          return info.mode == TensorMode::kInput && info.name == name;
        })) {
      std::cerr << "unknown TensorRT input tensor shape: " << name << '\n';
      return 2;
    }
  }

  std::vector<Tensor> inputs;
  for (const TensorInfo& info : tensors) {
    if (info.mode != TensorMode::kInput) continue;
    const auto path = input_paths.find(info.name);
    if (path == input_paths.end()) {
      std::cerr << "missing --trt-input " << info.name << "=FILE\n";
      return 2;
    }
    Tensor tensor;
    tensor.name = info.name;
    tensor.data_type = info.data_type;
    tensor.dimensions = info.dimensions;
    if (const auto shape = input_shapes.find(info.name); shape != input_shapes.end()) {
      tensor.dimensions = shape->second;
    }
    if (std::find(tensor.dimensions.begin(), tensor.dimensions.end(), -1) != tensor.dimensions.end()) {
      std::cerr << "dynamic input " << info.name << " needs --trt-shape " << info.name << "=DIMS\n";
      return 2;
    }
    std::ifstream file(path->second, std::ios::binary | std::ios::ate);
    if (!file) {
      std::cerr << "cannot open raw input file: " << PathUtf8(path->second) << '\n';
      return 1;
    }
    const std::streampos end = file.tellg();
    if (end < 0 || static_cast<std::uintmax_t>(end) >
                       static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
      std::cerr << "raw input file is too large: " << PathUtf8(path->second) << '\n';
      return 1;
    }
    tensor.bytes.resize(static_cast<std::size_t>(end));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(tensor.bytes.data()), static_cast<std::streamsize>(tensor.bytes.size()));
    if (!file) {
      std::cerr << "cannot read raw input file: " << PathUtf8(path->second) << '\n';
      return 1;
    }
    inputs.push_back(std::move(tensor));
  }
  for (const auto& [name, path] : input_paths) {
    (void)path;
    if (std::none_of(tensors.begin(), tensors.end(), [&](const TensorInfo& info) {
          return info.mode == TensorMode::kInput && info.name == name;
        })) {
      std::cerr << "unknown TensorRT input tensor: " << name << '\n';
      return 2;
    }
  }

  const StatusOr<std::vector<Tensor>> outputs = engine.Infer(inputs);
  if (!outputs.ok()) {
    std::cerr << outputs.status().message() << '\n';
    return 1;
  }
  const std::filesystem::path output_directory = PathFromUtf8(options.tensorrt_output_directory);
  std::error_code error;
  std::filesystem::create_directories(output_directory, error);
  if (error) {
    std::cerr << "cannot create output directory: " << error.message() << '\n';
    return 1;
  }
  for (const Tensor& output : outputs.value()) {
    std::string filename = output.name;
    for (char& character : filename) {
      if (character == '/' || character == '\\' || character == ':' || character == '*' ||
          character == '?' || character == '"' || character == '<' || character == '>' ||
          character == '|') character = '_';
    }
    const std::filesystem::path output_path = output_directory / (filename + ".bin");
    std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(output.bytes.data()),
               static_cast<std::streamsize>(output.bytes.size()));
    if (!file) {
      std::cerr << "cannot write TensorRT output: " << PathUtf8(output_path) << '\n';
      return 1;
    }
    std::cout << output.name << "\t" << DataTypeName(output.data_type) << "\t"
              << ShapeLabel(output.dimensions) << "\t" << PathUtf8(output_path) << '\n';
  }
  return 0;
}
#endif

std::string FormatBytes(uint64_t bytes) {
  constexpr uint64_t kMiB = 1024U * 1024U;
  const double mib = static_cast<double>(bytes) / static_cast<double>(kMiB);
  return std::to_string(mib) + " MiB";
}

StatusOr<ModelDescriptor> ImportModel(const std::string& path) {
  ModelManager manager;
  const auto model_path = isvik::PathFromUtf8(path);
  const auto resolved = ResolveModelEntry(model_path);
  if (!resolved.ok()) return resolved.status();
  auto metadata_directory = resolved.value().parent_path();
  if (resolved.value().extension() == ".onnx" && metadata_directory.filename() == "onnx" &&
      std::filesystem::is_regular_file(metadata_directory.parent_path() / "config.json")) {
    metadata_directory = metadata_directory.parent_path();
  }
  const Status status = manager.ImportModel("cli-model", resolved.value(), PathUtf8(metadata_directory.filename()));
  if (!status.ok()) return status;
  ModelDescriptor model = manager.FindModel("cli-model").value();
  if (model.format == ModelFormat::kOpenVinoIr || model.format == ModelFormat::kOnnx) {
    const auto directory = metadata_directory;
    std::error_code error;
    const auto config_path = directory / "config.json";
    if (std::filesystem::is_regular_file(config_path, error) &&
        std::filesystem::file_size(config_path, error) < 1024U * 1024U) {
      std::ifstream input(config_path);
      const auto config = nlohmann::json::parse(input, nullptr, false);
      if (config.is_object()) {
        if (config.contains("model_type") && config["model_type"].is_string()) {
          model.architecture = config["model_type"].get<std::string>();
        }
        const auto& text = config.contains("text_config") && config["text_config"].is_object()
            ? config["text_config"] : config;
        if (text.contains("max_position_embeddings") &&
            text["max_position_embeddings"].is_number_unsigned()) {
          model.context_length = text["max_position_embeddings"].get<uint64_t>();
        } else if (config.contains("n_positions") && config["n_positions"].is_number_unsigned()) {
          model.context_length = config["n_positions"].get<uint64_t>();
        }
      }
    }
    const auto quantization_path = directory / "openvino_config.json";
    error.clear();
    if (std::filesystem::is_regular_file(quantization_path, error) &&
        std::filesystem::file_size(quantization_path, error) < 1024U * 1024U) {
      std::ifstream input(quantization_path);
      const auto metadata = nlohmann::json::parse(input, nullptr, false);
      if (metadata.is_object() && metadata.contains("dtype") && metadata["dtype"].is_string()) {
        std::string dtype = metadata["dtype"].get<std::string>();
        std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char value) {
          return static_cast<char>(std::toupper(value));
        });
        model.quantization = dtype;
        if (metadata.contains("quantization_config") && metadata["quantization_config"].is_object()) {
          const auto& quantization = metadata["quantization_config"];
          if (quantization.contains("quantization_configs") &&
              quantization["quantization_configs"].is_object() &&
              quantization["quantization_configs"].contains("lm_model") &&
              quantization["quantization_configs"]["lm_model"].is_object()) {
            const auto& language_model = quantization["quantization_configs"]["lm_model"];
            if (language_model.contains("group_size") && language_model["group_size"].is_number_integer()) {
              const auto group_size = language_model["group_size"].get<int64_t>();
              if (group_size > 0) model.quantization += " · G" + std::to_string(group_size);
            }
          }
        }
      }
    }
  }
  return model;
}

const char* BackendLabel(BackendType backend) {
  switch (backend) {
    case BackendType::kTensorRt: return "TensorRT";
    case BackendType::kOnnxRuntime: return "ONNX Runtime";
    case BackendType::kGgml: return "GGML";
    case BackendType::kOpenVino: return "OpenVINO";
  }
  return "OpenVINO";
}

std::string BackendName(BackendType backend) {
  switch (backend) {
    case BackendType::kTensorRt: return "tensorrt";
    case BackendType::kOnnxRuntime: return "onnxruntime";
    case BackendType::kGgml: return "ggml";
    case BackendType::kOpenVino: return "openvino";
  }
  return "openvino";
}

BackendType BackendTypeFromName(std::string_view backend) {
  if (backend == "tensorrt") return BackendType::kTensorRt;
  if (backend == "onnxruntime") return BackendType::kOnnxRuntime;
  return BackendType::kOpenVino;
}

const char* ModelFormatLabel(ModelFormat format) {
  switch (format) {
    case ModelFormat::kGguf: return "GGUF";
    case ModelFormat::kOnnx: return "ONNX";
    case ModelFormat::kOpenVinoIr: return "OpenVINO IR";
  }
  return "unknown";
}

bool ModelSupportsBackend(const CatalogModel& model, BackendType backend) {
  switch (backend) {
    case BackendType::kOpenVino:
      return model.model.format != ModelFormat::kOnnx && model.runnable;
    case BackendType::kOnnxRuntime:
      return model.model.format == ModelFormat::kOnnx;
    case BackendType::kTensorRt:
      return model.model.format == ModelFormat::kGguf || model.model.format == ModelFormat::kOnnx;
    case BackendType::kGgml:
      return false;
  }
  return false;
}

std::string_view CliText(const LocalizationService& language, std::string_view chinese,
                         std::string_view english) {
  return language.language() == Language::kEnglish ? english : chinese;
}

std::string DecodingLabel(DecodingMode mode, const LocalizationService& language) {
  switch (mode) {
    case DecodingMode::kGreedy: return std::string(CliText(language, "贪心解码", "Greedy"));
    case DecodingMode::kSampling: return std::string(CliText(language, "随机采样", "Sampling"));
    case DecodingMode::kBeamSearch: return std::string(CliText(language, "束搜索", "Beam search"));
    case DecodingMode::kDiverseBeamSearch:
      return std::string(CliText(language, "多样化束搜索", "Diverse beam search"));
    case DecodingMode::kSpeculative:
      return std::string(CliText(language, "推测解码", "Speculative decoding"));
  }
  return std::string(CliText(language, "未知", "Unknown"));
}

std::string OptionalFloatLabel(const std::optional<float>& value) {
  if (!value.has_value()) return "default";
  std::ostringstream output;
  output << std::setprecision(4) << std::defaultfloat << *value;
  return output.str();
}

void PrintCliParameters(const Options& options, std::string_view backend,
                        std::string_view model, std::string_view device,
                        uint64_t context_tokens, bool use_memories,
                        const LocalizationService& language) {
  std::cout << CliText(language, "运行参数：", "Runtime parameters: ")
            << CliText(language, "模型=", "model=") << model << " | backend=" << backend
            << " | " << CliText(language, "设备=", "device=") << device << '\n'
            << CliText(language, "生成参数：", "Generation parameters: ")
            << CliText(language, "解码=", "decoding=")
            << DecodingLabel(options.generation.decoding_mode, language)
            << " | " << CliText(language, "最大输出=", "max-output=") << options.max_tokens
            << " | " << CliText(language, "上下文=", "context=") << context_tokens
            << " | " << CliText(language, "记忆=", "memory=")
            << (use_memories ? CliText(language, "开", "on") : CliText(language, "关", "off"))
            << " | " << CliText(language, "停止序列=", "stop-sequences=")
            << options.generation.stop_sequences.size() << '\n';
  if (options.generation.decoding_mode == DecodingMode::kSampling) {
    std::cout << CliText(language, "采样参数：", "Sampling parameters: ")
              << CliText(language, "温度 temperature=", "temperature=")
              << OptionalFloatLabel(options.generation.sampling.temperature)
              << " | " << CliText(language, "核采样 top-p=", "top-p=")
              << OptionalFloatLabel(options.generation.sampling.top_p)
              << " | top-k=";
    if (options.generation.sampling.top_k.has_value())
      std::cout << *options.generation.sampling.top_k;
    else
      std::cout << "default";
    std::cout << " | " << CliText(language, "随机种子 seed=", "seed=");
    if (options.generation.sampling.seed.has_value())
      std::cout << *options.generation.sampling.seed;
    else
      std::cout << "auto";
    std::cout << '\n';
  }
}

std::vector<std::string> AvailableCliDevices(BackendType backend) {
  std::vector<std::string> devices;
  const auto append = [&devices](std::string value) { devices.push_back(std::move(value)); };
  if (backend == BackendType::kOpenVino) {
#if defined(ISVIK_HAS_OPENVINO_RUNTIME)
    const auto available = openvino_backend::EnumerateDevices();
    if (available.ok()) {
      for (const auto& device : available.value()) append(device.id);
    }
#endif
  } else if (backend == BackendType::kTensorRt) {
#if defined(ISVIK_HAS_TENSORRT)
    const auto available = tensorrt_backend::EnumerateDevices();
    if (available.ok()) {
      for (const auto& device : available.value())
        append("GPU." + std::to_string(device.index));
    }
#endif
  } else {
    append("CPU");
    append("GPU");
  }
  return devices;
}

void PrintCliColumn(std::string_view text, std::size_t width) {
  std::size_t display_width = 0U;
  for (std::size_t index = 0U; index < text.size();) {
    const unsigned char lead = static_cast<unsigned char>(text[index]);
    std::size_t sequence_length = 1U;
    if ((lead & 0xE0U) == 0xC0U) sequence_length = 2U;
    else if ((lead & 0xF0U) == 0xE0U) sequence_length = 3U;
    else if ((lead & 0xF8U) == 0xF0U) sequence_length = 4U;
    display_width += sequence_length > 1U ? 2U : 1U;
    index += std::min(sequence_length, text.size() - index);
  }
  std::cout << text;
  if (display_width < width) std::cout << std::string(width - display_width, ' ');
}

std::string SupportedDevices(const CatalogModel& entry,
                             const std::vector<std::string>& openvino_devices,
                             const std::vector<std::string>& tensorrt_devices) {
  if (!entry.available) return "-";
  std::string devices;
  std::vector<std::string> seen;
  const auto append = [&devices, &seen](const std::string& value) {
    if (std::find(seen.begin(), seen.end(), value) != seen.end()) return;
    seen.push_back(value);
    if (!devices.empty()) devices += ", ";
    devices += value;
  };
  if (ModelSupportsBackend(entry, BackendType::kOpenVino)) {
    for (const std::string& device : openvino_devices) append(device);
  }
  if (ModelSupportsBackend(entry, BackendType::kTensorRt)) {
    for (const std::string& device : tensorrt_devices) append(device);
  }
  if (entry.model.format == ModelFormat::kOnnx) append("CPU");
  if (devices.empty()) devices = "-";
  return devices;
}

void PrintCliModelList(const ModelCatalog& catalog, const LocalizationService& language) {
  const std::vector<std::string> openvino_devices =
      AvailableCliDevices(BackendType::kOpenVino);
  const std::vector<std::string> tensorrt_devices =
      AvailableCliDevices(BackendType::kTensorRt);
  std::cout << '\n' << CliText(language, "已导入模型：", "Imported Models:") << '\n'
            << "  " << CliText(language, "序号  ", "Index ");
  PrintCliColumn(CliText(language, "模型 ID", "Model ID"), 36U);
  std::cout << CliText(language, "大小    ", "Size    ")
            << CliText(language, "支持设备", "Supported Devices") << '\n';
  if (catalog.models().empty()) {
    std::cout << "  " << CliText(language, "（暂无模型；使用 -import <路径> 导入）",
                                  "(No models; use -import <path> to add one)") << "\n\n";
    return;
  }
  for (std::size_t index = 0U; index < catalog.models().size(); ++index) {
    const CatalogModel& entry = catalog.models()[index];
    std::cout << "  ";
    PrintCliColumn("[" + std::to_string(index + 1U) + "]", 6U);
    PrintCliColumn(entry.model.display_name, 36U);
    if (entry.model.file_size == 0U) {
      std::cout << "-       ";
    } else {
      std::cout << std::fixed << std::setprecision(1) << std::setw(5)
                << (static_cast<double>(entry.model.file_size) / 1000000000.0)
                << " GB" << std::defaultfloat;
    }
    std::cout << "  " << SupportedDevices(entry, openvino_devices, tensorrt_devices);
    if (entry.model.id == catalog.default_id())
      std::cout << "  " << CliText(language, "[默认]", "[DEFAULT]");
    std::cout << '\n';
    if (!entry.available && !entry.issue.empty()) std::cout << "       " << entry.issue << '\n';
  }
  std::cout << '\n';
}

const CatalogModel* FindCatalogModel(const ModelCatalog& catalog, std::string_view selector) {
  for (const CatalogModel& entry : catalog.models()) {
    if (entry.model.id == selector) return &entry;
  }
  std::size_t number = 0U;
  const auto parsed = std::from_chars(selector.data(), selector.data() + selector.size(), number);
  if (parsed.ec == std::errc{} && parsed.ptr == selector.data() + selector.size() &&
      number > 0U && number <= catalog.models().size()) {
    return &catalog.models()[number - 1U];
  }
  return nullptr;
}

std::filesystem::path CliLanguagePath() {
  return CatalogStoragePath().parent_path() / "cli-language.txt";
}

void LoadCliLanguage(LocalizationService& language) {
  std::ifstream input(CliLanguagePath(), std::ios::binary);
  std::string preference;
  input >> preference;
  if (preference == "en") language.SetLanguage(Language::kEnglish);
  else if (preference == "zh") language.SetLanguage(Language::kChineseSimplified);
}

Status SaveCliLanguage(Language language) {
  const std::filesystem::path path = CliLanguagePath();
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) return Status::Unavailable("Cannot create CLI settings folder: " + error.message());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << (language == Language::kEnglish ? "en\n" : "zh\n");
  if (!output) return Status::Unavailable("Cannot save CLI language preference");
  return Status();
}

std::string GenerateCliApiKey() {
  std::array<unsigned char, 32> bytes{};
#if defined(_WIN32)
  if (BCryptGenRandom(nullptr, bytes.data(), static_cast<ULONG>(bytes.size()),
                      BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0) return {};
#else
  try {
    std::random_device random;
    for (unsigned char& byte : bytes) byte = static_cast<unsigned char>(random());
  } catch (...) {
    return {};
  }
#endif
  constexpr char hex[] = "0123456789abcdef";
  std::string key = "isvik_";
  key.reserve(70U);
  for (const unsigned char byte : bytes) {
    key.push_back(hex[byte >> 4U]);
    key.push_back(hex[byte & 0x0fU]);
  }
  return key;
}

int RunApiServerCommand(const Options& options) {
  StatusOr<ModelDescriptor> imported = ImportModel(options.model_path);
  if (!imported.ok()) {
    std::cerr << imported.status().message() << '\n';
    return 1;
  }
  const ModelDescriptor model = std::move(imported).value();
  ApiServerConfig config;
  config.host = options.server_host;
  config.port = options.server_port;
  config.api_key = options.api_key;

  if (options.server_backend == "openvino") {
#if defined(ISVIK_HAS_OPENVINO_GENAI)
    if (model.format == ModelFormat::kOnnx) {
      std::cerr << "OpenVINO server backend accepts OpenVINO IR (.xml) and supported GGUF models, but not ONNX.\n";
      return 2;
    }
    config.backend = "OpenVINO GenAI";
    openvino_backend::GenAiEngine engine;
    std::cout << "Loading " << model.display_name << " on " << options.device << "...\n";
    const Status loaded = engine.LoadModel(model, options.device);
    if (!loaded.ok()) {
      std::cerr << loaded.message() << '\n';
      return 1;
    }
    return RunApiServer(config, model,
        [&engine](const UnifiedInferenceRequest& request, const CancellationToken& cancellation,
                  const std::function<void(const InferenceEvent&)>& handler) {
          return engine.Generate(request, cancellation, handler);
        });
#else
    std::cerr << "This build does not include OpenVINO GenAI. Rebuild with OpenVINO GenAI enabled.\n";
    return 1;
#endif
  }

  if (options.server_backend == "onnxruntime") {
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
    if (model.format != ModelFormat::kOnnx) {
      std::cerr << "The onnxruntime backend requires an ONNX model directory or .onnx file.\n";
      return 2;
    }
    config.backend = "ONNX Runtime GenAI";
    onnxruntime_backend::GenAiEngine engine;
    std::cout << "Loading " << model.display_name << " on " << options.device << "...\n";
    const Status loaded = engine.LoadModel(model, options.device);
    if (!loaded.ok()) {
      std::cerr << loaded.message() << '\n';
      return 1;
    }
    return RunApiServer(config, model,
        [&engine](const UnifiedInferenceRequest& request, const CancellationToken& cancellation,
                  const std::function<void(const InferenceEvent&)>& handler) {
          return engine.Generate(request, cancellation, handler);
        });
#else
    std::cerr << "This build does not include ONNX Runtime GenAI.\n";
    return 1;
#endif
  }

  if (model.format == ModelFormat::kGguf) {
#if defined(ISVIK_HAS_TENSORRT)
    int device_index = 0;
    if (options.device_explicit) {
      std::string_view device = options.device;
      if (device == "GPU") {
        device_index = 0;
      } else if (device.starts_with("GPU.") &&
                 ParseInt(device.substr(4U), &device_index) && device_index >= 0) {
      } else {
        std::cerr << "TensorRT GGUF uses --device GPU or GPU.INDEX (for example GPU.0).\n";
        return 2;
      }
    }
    config.backend = "TensorRT native GGUF";
    tensorrt_backend::GgufGenAiEngine engine(
        [](std::string_view message) { std::cout << "[TensorRT] " << message << std::endl; });
    std::cout << "Loading " << model.display_name << " on GPU." << device_index << "...\n";
    const Status loaded = engine.LoadModel(model, device_index);
    if (!loaded.ok()) {
      std::cerr << loaded.message() << '\n';
      return 1;
    }
    return RunApiServer(config, model,
        [&engine](const UnifiedInferenceRequest& request, const CancellationToken& cancellation,
                  const std::function<void(const InferenceEvent&)>& handler) {
          return engine.Generate(request, cancellation, handler);
        });
#else
    std::cerr << "This build does not include the TensorRT native GGUF runtime.\n";
    return 1;
#endif
  }

  if (model.format == ModelFormat::kOnnx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP) && defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
    if (options.device_explicit && options.device == "CPU") {
      std::cerr << "The TensorRT backend requires a GPU device; use --device GPU.0.\n";
      return 2;
    }
    const std::string device = options.device_explicit ? options.device : "GPU.0";
    config.backend = "TensorRT-RTX ONNX";
    onnxruntime_backend::GenAiEngine engine;
    std::cout << "Loading " << model.display_name << " through TensorRT-RTX on " << device << "...\n";
    const Status loaded = engine.LoadModel(model, device);
    if (!loaded.ok()) {
      std::cerr << loaded.message() << '\n';
      return 1;
    }
    return RunApiServer(config, model,
        [&engine](const UnifiedInferenceRequest& request, const CancellationToken& cancellation,
                  const std::function<void(const InferenceEvent&)>& handler) {
          return engine.Generate(request, cancellation, handler);
        });
#else
    std::cerr << "TensorRT ONNX serving requires both TensorRT-RTX and ONNX Runtime GenAI in this build.\n";
    return 1;
#endif
  }

  std::cerr << "TensorRT server supports native GGUF and ONNX models only.\n";
  return 2;
}

void PrintModel(const ModelDescriptor& model) {
  std::cout << "ID: " << model.id << '\n'
            << "Name: " << model.display_name << '\n'
            << "Format: " << (model.format == ModelFormat::kOpenVinoIr ? "OpenVINO IR"
                : model.format == ModelFormat::kGguf ? "GGUF" : "ONNX") << '\n'
            << "Path: " << PathUtf8(model.path) << '\n'
            << "Size: " << FormatBytes(model.file_size) << '\n';
  if (!model.architecture.empty()) std::cout << "Architecture: " << model.architecture << '\n';
  if (!model.parameter_label.empty()) std::cout << "Parameters: " << model.parameter_label << '\n';
  if (!model.quantization.empty()) std::cout << "Quantization: " << model.quantization << '\n';
  if (!model.variant.empty()) std::cout << "Variant: " << model.variant << '\n';
  if (model.context_length != 0U) std::cout << "Context: " << model.context_length << '\n';
  if (model.format == ModelFormat::kGguf) {
    const auto layout = GgufInspector::Inspect(model.path);
    if (layout.ok()) {
      std::map<uint32_t, std::size_t> counts;
      for (const auto& tensor : layout.value().tensors) ++counts[tensor.type];
      std::cout << "Tensors: " << layout.value().tensor_count << '\n';
      std::cout << "Tensor encodings:";
      for (const auto& [type, count] : counts) {
        std::cout << ' ' << GgufTensorTypeName(type) << '=' << count;
      }
      std::cout << '\n';
    }
  }
}

InferenceMessage UserMessage(std::string content) {
  InferenceMessage message;
  message.role = MessageRole::kUser;
  message.content = std::move(content);
  return message;
}

std::string NewRequestId() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return "cli-" + std::to_string(now);
}

using CliGenerateFunction = std::function<Status(
    const UnifiedInferenceRequest&, const CancellationToken&,
    const std::function<void(const InferenceEvent&)>&)>;

StatusOr<std::string> RunRequest(const CliGenerateFunction& generate,
                                 const ModelDescriptor& model,
                                 std::vector<InferenceMessage> messages,
                                 const GenerationConfig& generation,
                                 const LocalizationService* cli_language = nullptr) {
  UnifiedInferenceRequest request;
  request.request_id = NewRequestId();
  request.model_id = model.id;
  request.messages = std::move(messages);
  request.generation = generation;
  CancellationSource cancellation;
  std::string response;
  bool cli_footer_printed = false;
  const Status status = generate(
      request, cancellation.token(), [&response, &cli_footer_printed, cli_language](const InferenceEvent& event) {
        if (const auto* delta = std::get_if<ContentDelta>(&event.payload)) {
          response += delta->text;
          std::cout << delta->text << std::flush;
        } else if (const auto* metrics = std::get_if<InferenceCompleted>(&event.payload)) {
          if (cli_language != nullptr) {
            if (!response.empty() && response.back() != '\n') std::cout << '\n';
            const auto previous_flags = std::cout.flags();
            const auto previous_precision = std::cout.precision();
            std::cout << CliText(*cli_language, "[统计] 输入 ", "[STATS] Input ")
                      << metrics->input_tokens
                      << CliText(*cli_language, " tokens · 输出 ", " tokens · Output ")
                      << metrics->output_tokens
                      << CliText(*cli_language, " tokens · 用时 ", " tokens · Time ")
                      << std::fixed << std::setprecision(2) << metrics->duration_seconds
                      << CliText(*cli_language, " s · 速度 ", " s · Speed ")
                      << std::setprecision(2) << metrics->tokens_per_second
                      << " tok/s\n";
            std::cout.flags(previous_flags);
            std::cout.precision(previous_precision);
            cli_footer_printed = true;
            return;
          }
          std::cout << "\n[stats] prompt=" << metrics->input_tokens
                    << " tokens; output=" << metrics->output_tokens
                    << " tokens; time=" << std::fixed << std::setprecision(2)
                    << metrics->duration_seconds << " s; speed="
                    << metrics->tokens_per_second << " tokens/s\n";
        } else if (std::holds_alternative<InferenceCancelled>(event.payload)) {
          std::cout << '\n' << (cli_language == nullptr ? "Generation cancelled." :
                      CliText(*cli_language, "生成已取消。", "Generation cancelled.")) << '\n';
          cli_footer_printed = cli_language != nullptr;
        } else if (const auto* error = std::get_if<InferenceError>(&event.payload)) {
          std::cerr << "\n" << error->status.message() << '\n';
          cli_footer_printed = cli_language != nullptr;
        }
      });
  if (cli_language != nullptr && !cli_footer_printed &&
      (response.empty() || response.back() != '\n')) std::cout << '\n';
  if (!status.ok()) return status;
  return response;
}

StatusOr<ChatContextResult> PrepareCliContext(
    const std::vector<InferenceMessage>& conversation,
    const ModelDescriptor& model,
    uint64_t context_tokens,
    int max_tokens,
    const MemoryService* memories,
    std::string_view session_id,
    bool use_memories) {
  ContextOptions options;
  options.max_context_tokens = context_tokens;
  if (model.context_length > 0U) {
    options.max_context_tokens = std::min<uint64_t>(options.max_context_tokens, model.context_length);
  }
  options.reserved_output_tokens = static_cast<uint64_t>(max_tokens);
  options.compression = ContextCompressionMode::kHybrid;
  if (options.reserved_output_tokens >= options.max_context_tokens) {
    return Status::InvalidArgument("context limit must exceed --max-tokens and the model context limit");
  }
  return BuildChatContext(conversation, memories, session_id, use_memories, options);
}

void PrintContextStatus(const ChatContextResult& context, uint64_t context_limit) {
  std::cout << "Context: about " << context.estimated_prompt_tokens << "/" << context_limit
            << " prompt tokens, " << context.dropped_history_messages
            << " older message(s) removed, " << context.memories_used << " memory note(s) used.\n";
}

void PrintMemories(const std::vector<MemoryEntry>& entries) {
  if (entries.empty()) {
    std::cout << "No saved memories. Use /remember TEXT to save one.\n";
    return;
  }
  for (const MemoryEntry& entry : entries) {
    std::cout << (entry.pinned ? "[pinned] " : "")
              << (entry.scope == MemoryScope::kSession ? "[session] " : "[saved] ")
              << entry.id << "\n  " << entry.content << '\n';
    if (!entry.tags.empty()) {
      std::cout << "  tags: ";
      for (std::size_t index = 0U; index < entry.tags.size(); ++index) {
        if (index != 0U) std::cout << ", ";
        std::cout << entry.tags[index];
      }
      std::cout << '\n';
    }
  }
}

struct MemorySessionGuard {
  MemoryService* service = nullptr;
  std::string session;
  ~MemorySessionGuard() { if (service != nullptr) static_cast<void>(service->CloseSession(session)); }
};

class CliEngineRuntime {
 public:
  Status Load(const ModelDescriptor& model, std::string_view backend,
              const std::string& device) {
    Unload();
    if (backend == "openvino") {
      if (model.format == ModelFormat::kOnnx)
        return Status::Unsupported("OpenVINO requires an IR or supported GGUF model");
#if defined(ISVIK_HAS_OPENVINO_GENAI)
      openvino_ = std::make_unique<openvino_backend::GenAiEngine>();
      Status status = openvino_->LoadModel(model, device);
      if (!status.ok()) { Unload(); return status; }
#else
      return Status::Unavailable("This build does not include OpenVINO GenAI");
#endif
    } else if (backend == "onnxruntime") {
      if (model.format != ModelFormat::kOnnx)
        return Status::Unsupported("ONNX Runtime requires an ONNX model");
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
      onnxruntime_ = std::make_unique<onnxruntime_backend::GenAiEngine>();
      Status status = onnxruntime_->LoadModel(model, device);
      if (!status.ok()) { Unload(); return status; }
#else
      return Status::Unavailable("This build does not include ONNX Runtime GenAI");
#endif
    } else if (backend == "tensorrt" && model.format == ModelFormat::kGguf) {
#if defined(ISVIK_HAS_TENSORRT)
      int device_index = 0;
      std::string_view value(device);
      if (value == "GPU") {
        device_index = 0;
      } else if (!value.starts_with("GPU.") ||
                 !ParseInt(value.substr(4U), &device_index) || device_index < 0) {
        return Status::InvalidArgument("TensorRT GGUF requires GPU or GPU.INDEX");
      }
      tensorrt_gguf_ = std::make_unique<tensorrt_backend::GgufGenAiEngine>(
          [](std::string_view message) { std::cout << "[TensorRT] " << message << '\n'; });
      Status status = tensorrt_gguf_->LoadModel(model, device_index);
      if (!status.ok()) { Unload(); return status; }
#else
      return Status::Unavailable("This build does not include the TensorRT GGUF runtime");
#endif
    } else if (backend == "tensorrt" && model.format == ModelFormat::kOnnx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP) && defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
      onnxruntime_ = std::make_unique<onnxruntime_backend::GenAiEngine>();
      Status status = onnxruntime_->LoadModel(model, device);
      if (!status.ok()) { Unload(); return status; }
#else
      return Status::Unavailable("TensorRT ONNX requires TensorRT-RTX and ONNX Runtime GenAI");
#endif
    } else {
      return Status::Unsupported("This backend does not support the selected model format");
    }
    model_ = model;
    backend_ = std::string(backend);
    device_ = device;
    return Status();
  }

  Status Generate(const UnifiedInferenceRequest& request,
                  const CancellationToken& cancellation,
                  const std::function<void(const InferenceEvent&)>& handler) {
#if defined(ISVIK_HAS_OPENVINO_GENAI)
    if (openvino_) return openvino_->Generate(request, cancellation, handler);
#endif
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
    if (onnxruntime_) return onnxruntime_->Generate(request, cancellation, handler);
#endif
#if defined(ISVIK_HAS_TENSORRT)
    if (tensorrt_gguf_) return tensorrt_gguf_->Generate(request, cancellation, handler);
#endif
    return Status::Unavailable("No model is loaded");
  }

  void Unload() {
#if defined(ISVIK_HAS_OPENVINO_GENAI)
    if (openvino_) openvino_->UnloadModel();
    openvino_.reset();
#endif
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
    if (onnxruntime_) onnxruntime_->UnloadModel();
    onnxruntime_.reset();
#endif
#if defined(ISVIK_HAS_TENSORRT)
    if (tensorrt_gguf_) tensorrt_gguf_->UnloadModel();
    tensorrt_gguf_.reset();
#endif
    model_.reset();
    backend_.clear();
    device_.clear();
  }

  [[nodiscard]] bool loaded() const { return model_.has_value(); }
  [[nodiscard]] const std::optional<ModelDescriptor>& model() const { return model_; }
  [[nodiscard]] const std::string& backend() const { return backend_; }
  [[nodiscard]] const std::string& device() const { return device_; }

 private:
#if defined(ISVIK_HAS_OPENVINO_GENAI)
  std::unique_ptr<openvino_backend::GenAiEngine> openvino_;
#endif
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
  std::unique_ptr<onnxruntime_backend::GenAiEngine> onnxruntime_;
#endif
#if defined(ISVIK_HAS_TENSORRT)
  std::unique_ptr<tensorrt_backend::GgufGenAiEngine> tensorrt_gguf_;
#endif
  std::optional<ModelDescriptor> model_;
  std::string backend_;
  std::string device_;
};

int RunInteractiveWorkbench(Options options) {
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  LocalizationService language;
  LoadCliLanguage(language);
  ModelCatalog catalog(CatalogStoragePath());
  const Status catalog_status = catalog.Load();
  if (!catalog_status.ok()) {
    std::cerr << catalog_status.message() << '\n';
    return 1;
  }

  std::string backend = options.backend_explicit ? options.server_backend
                                                  : BackendName(catalog.backend());
  if (backend != "openvino" && backend != "onnxruntime" && backend != "tensorrt") {
    std::cerr << "--backend must be openvino, onnxruntime, or tensorrt.\n";
    return 2;
  }
  if (options.backend_explicit) catalog.set_backend(BackendTypeFromName(backend));
  std::string device = options.device_explicit ? options.device
      : backend == "tensorrt" ? "GPU.0" : "CPU";
  CliEngineRuntime engine;
  std::vector<InferenceMessage> history;
  std::string system_prompt;
  uint64_t context_tokens = options.context_tokens;
  bool use_memories = options.use_memories;
  std::unique_ptr<MemoryService> memories;
  const std::string memory_session = "cli-" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  auto opened_memories = MemoryService::Open(MemoryStoragePath());
  if (opened_memories.ok()) {
    memories = std::move(opened_memories).value();
    static_cast<void>(memories->SetSessionMemoryEnabled(memory_session, true));
  }
  MemorySessionGuard memory_session_guard{memories.get(), memory_session};
  CliGenerateFunction generate = [&engine](const UnifiedInferenceRequest& request,
      const CancellationToken& cancellation,
      const std::function<void(const InferenceEvent&)>& handler) {
    return engine.Generate(request, cancellation, handler);
  };

  const auto sync_system_prompt = [&]() {
    std::erase_if(history, [](const InferenceMessage& message) {
      return message.role == MessageRole::kSystem;
    });
    if (!system_prompt.empty()) {
      history.insert(history.begin(), {MessageRole::kSystem, system_prompt, {}, {}});
    }
  };
  const auto choose_device = [&]() {
    if (options.device_explicit) return options.device;
    return backend == "tensorrt" ? std::string("GPU.0") : std::string("CPU");
  };
  const auto activate_model = [&](const ModelDescriptor& model) -> Status {
    engine.Unload();
    history.clear();
    sync_system_prompt();
    device = choose_device();
    if (backend == "openvino") {
      std::cout << "[INFO] " << CliText(language, "正在编译模型到 ", "Compiling model to ")
                << device << " ... " << std::flush;
    } else {
      std::cout << "[INFO] " << CliText(language, "正在加载模型到 ", "Loading model to ")
                << device << " ... " << std::flush;
    }
    const auto load_started = std::chrono::steady_clock::now();
    const Status loaded = engine.Load(model, backend, device);
    if (!loaded.ok()) {
      std::cout << CliText(language, "失败", "Failed") << '\n';
      return loaded;
    }
    static_cast<void>(catalog.SetDefault(model.id));
    static_cast<void>(catalog.Save());
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - load_started).count();
    std::cout << CliText(language, "完成", "Done") << " (" << elapsed << " ms)\n";
    return Status();
  };

  const std::string startup_model = options.model_path.empty()
      ? std::string(CliText(language, "（无）", "(none)"))
      : PathUtf8(PathFromUtf8(options.model_path).filename());
  const std::string device_policy = options.device_explicit ? options.device : "AUTO";
  std::cout << "  ___  ____  _   _ ___ _  __\n"
            << " |_ _|/ ___|| | | |_ _| |/ /\n"
            << "  | | \\___ \\| | | || || ' /\n"
            << "  | |  ___) | |_| || || . \\\n"
            << " |___||____/ \\___/|___|_|\\_\\\n"
            << "ISVIK 0.1.0  "
            << CliText(language, "本地 AI 工作台", "Local AI Workbench") << '\n'
            << CliText(language,
                "本地优先 · 感知真实设备 · 不伪造运行状态",
                "Local-first · Real device detection · Honest runtime status") << '\n'
            << CliText(language, "模型：", "Model: ") << startup_model << "  |  "
            << CliText(language, "设备策略：", "Device policy: ") << device_policy << '\n'
            << CliText(language,
                "加载模型后直接输入问题开始对话；输入 -ls 查看模型，再输入编号或 -use <编号> 选择。",
                "Load a model, then type a prompt to chat. Use -ls to list models and enter a number or -use <number> to select one.")
            << "\n\n";
  PrintCliParameters(options, backend, startup_model, device_policy,
                     context_tokens, use_memories, language);
  std::cout << '\n';

  if (!options.model_path.empty()) {
    StatusOr<ModelDescriptor> imported = ImportModel(options.model_path);
    if (!imported.ok()) {
      std::cerr << imported.status().message() << '\n';
      return 1;
    }
    for (const CatalogModel& entry : catalog.models()) {
      std::error_code error;
      if (std::filesystem::equivalent(entry.model.path, imported.value().path, error) && !error) {
        imported.value().display_name = entry.model.display_name;
        if (!options.backend_explicit) backend = BackendName(catalog.backend());
        break;
      }
    }
    const Status loaded = activate_model(imported.value());
    if (!loaded.ok()) {
      std::cerr << loaded.message() << '\n';
      return 1;
    }
  }

  std::string line;
  while (true) {
    const std::string model_label = engine.loaded()
        ? engine.model()->display_name : std::string(CliText(language, "未选择模型", "No Model"));
    const std::string prompt_device = engine.loaded() || options.device_explicit ? device : "AUTO";
    std::cout << "isvik[" << model_label << '@' << prompt_device << "]> " << std::flush;
    if (!std::getline(std::cin, line)) break;
    const auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) continue;
    const auto last = line.find_last_not_of(" \t\r\n");
    line = line.substr(first, last - first + 1U);

    const auto separator = line.find_first_of(" \t");
    const std::string command = line.substr(0U, separator);
    std::string argument = separator == std::string::npos ? std::string{} : line.substr(separator + 1U);
    const auto arg_first = argument.find_first_not_of(" \t");
    if (arg_first == std::string::npos) argument.clear();
    else argument.erase(0U, arg_first);
    const auto arg_last = argument.find_last_not_of(" \t");
    if (!argument.empty()) argument.erase(arg_last + 1U);
    if (argument.size() >= 2U &&
        ((argument.front() == '"' && argument.back() == '"') ||
         (argument.front() == '\'' && argument.back() == '\''))) {
      argument = argument.substr(1U, argument.size() - 2U);
    }

    if (command == "-exit" || command == "-quit" || command == "/exit" || command == "quit") {
      break;
    }
    if (command == "-help" || command == "/help" || command == "help") {
      std::cout << CliText(language,
          "工作台命令 / Workbench commands:\n"
          "  -ls, -models             列出模型 / List models\n"
          "  -use <编号|ID>           加载或切换模型 / Load or switch model\n"
          "  -stop, -unload           卸载模型 / Unload model\n"
          "  -backend <名称>          切换后端 / Switch backend\n"
          "  -devices, -doctor        列出设备 / List devices\n"
          "  -info [ID]               查看模型信息 / Show model info\n"
          "  -params                  显示当前运行参数 / Show runtime parameters\n"
          "  -import <路径>           导入模型或目录 / Import model or folder\n"
          "  -rm <编号|ID>            从模型库移除 / Remove from catalog\n"
          "  -new, /new               新建对话 / Start a new chat\n"
          "  -history, /history       查看本次对话 / Show chat history\n"
          "  -system [文本|clear]     查看或设置系统提示词 / Get or set system prompt\n"
          "  -key <list|create|show|revoke>  管理 API 密钥 / Manage API keys\n"
          "  -serve                   前台启动本地 API / Run the local API server\n"
          "  -lang <zh|en>            切换语言 / Change language\n"
          "  -exit                    退出 / Quit\n",
          "Workbench commands / 工作台命令:\n"
          "  -ls, -models             List models / 列出模型\n"
          "  -use <number|ID>         Load or switch model / 加载或切换模型\n"
          "  -stop, -unload           Unload model / 卸载模型\n"
          "  -backend <name>          Switch backend / 切换后端\n"
          "  -devices, -doctor        List devices / 列出设备\n"
          "  -info [ID]               Show model info / 查看模型信息\n"
          "  -params                  Show runtime parameters / 显示当前运行参数\n"
          "  -import <path>           Import model or folder / 导入模型或目录\n"
          "  -rm <number|ID>          Remove from catalog / 从模型库移除\n"
          "  -new, /new               Start a new chat / 新建对话\n"
          "  -history, /history       Show chat history / 查看本次对话\n"
          "  -system [text|clear]     Get or set system prompt / 查看或设置系统提示词\n"
          "  -key <list|create|show|revoke>  Manage API keys / 管理 API 密钥\n"
          "  -serve                   Run the local API server / 前台启动本地 API\n"
          "  -lang <zh|en>            Change language / 切换语言\n"
          "  -exit                    Quit / 退出\n");
      continue;
    }
    if (command == "-ls" || command == "-models" || command == "-list") {
      PrintCliModelList(catalog, language);
      continue;
    }
    if (command == "-params" || command == "-settings") {
      const std::string current_model = engine.loaded()
          ? engine.model()->display_name : std::string(CliText(language, "（无）", "(none)"));
      const std::string current_device = engine.loaded() || options.device_explicit
          ? device : "AUTO";
      PrintCliParameters(options, backend, current_model, current_device,
                         context_tokens, use_memories, language);
      continue;
    }
    if (command == "-lang") {
      if (argument == "zh") language.SetLanguage(Language::kChineseSimplified);
      else if (argument == "en") language.SetLanguage(Language::kEnglish);
      else {
        std::cout << CliText(language, "用法：-lang zh 或 -lang en\n",
                             "Usage: -lang zh or -lang en\n");
        continue;
      }
      const Status saved = SaveCliLanguage(language.language());
      std::cout << CliText(language, "界面语言已切换为中文。\n", "CLI language set to English.\n");
      if (!saved.ok()) std::cerr << saved.message() << '\n';
      continue;
    }
    if (command == "-backend") {
      if (argument.empty()) {
        std::cout << CliText(language, "当前推理后端：", "Current inference backend: ")
                  << backend << '\n';
        continue;
      }
      if (argument != "openvino" && argument != "onnxruntime" && argument != "tensorrt") {
        std::cout << CliText(language,
            "后端必须是 openvino、onnxruntime 或 tensorrt。\n",
            "Backend must be openvino, onnxruntime, or tensorrt.\n");
        continue;
      }
      backend = argument;
      options.backend_explicit = true;
      catalog.set_backend(backend == "tensorrt" ? BackendType::kTensorRt
          : backend == "onnxruntime" ? BackendType::kOnnxRuntime : BackendType::kOpenVino);
      const Status saved = catalog.Save();
      if (!saved.ok()) std::cerr << saved.message() << '\n';
      if (engine.model()) {
        const ModelDescriptor active_model = *engine.model();
        const Status loaded = activate_model(active_model);
        if (!loaded.ok()) std::cerr << loaded.message() << '\n';
      } else {
        device = choose_device();
        std::cout << CliText(language, "后端已切换为 ", "Backend switched to ")
                  << backend << '\n';
      }
      continue;
    }
    if (command == "-devices" || command == "-doctor") {
      bool found_devices = false;
#if defined(ISVIK_HAS_OPENVINO_RUNTIME)
      const auto openvino_devices = openvino_backend::EnumerateDevices();
      if (openvino_devices.ok()) {
        std::cout << "OpenVINO:\n";
        for (const auto& item : openvino_devices.value())
          std::cout << "  " << item.id << "\t" << item.full_name << '\n';
        found_devices = true;
      } else {
        std::cerr << openvino_devices.status().message() << '\n';
      }
#endif
#if defined(ISVIK_HAS_TENSORRT)
      const auto cuda_devices = tensorrt_backend::EnumerateDevices();
      if (cuda_devices.ok()) {
        std::cout << "TensorRT / CUDA:\n";
        for (const auto& item : cuda_devices.value())
          std::cout << "  GPU." << item.index << "\t" << item.name << '\n';
        found_devices = true;
      } else if (!found_devices) {
        std::cerr << cuda_devices.status().message() << '\n';
      }
#endif
      if (!found_devices) std::cout << CliText(language, "此构建没有可用设备枚举后端。\n",
                                               "No device enumeration backend is available.\n");
      continue;
    }
    if (command == "-info") {
      const CatalogModel* entry = argument.empty()
          ? nullptr : FindCatalogModel(catalog, argument);
      if (!argument.empty() && entry == nullptr) {
        std::cout << CliText(language, "模型编号或 ID 不存在。\n", "Unknown model number or ID.\n");
      } else if (entry != nullptr) {
        PrintModel(entry->model);
        if (!entry->issue.empty()) std::cout << entry->issue << '\n';
      } else if (engine.model()) {
        PrintModel(*engine.model());
        std::cout << "Backend: " << engine.backend() << "\nDevice: " << engine.device() << '\n';
      } else {
        std::cout << CliText(language, "当前没有加载模型。\n", "No model is loaded.\n");
      }
      continue;
    }
    if (command == "-use" || (separator == std::string::npos &&
        FindCatalogModel(catalog, command) != nullptr)) {
      const std::string_view key = command == "-use" ? std::string_view(argument)
                                                       : std::string_view(command);
      const CatalogModel* entry = FindCatalogModel(catalog, key);
      if (entry == nullptr) {
        std::cout << CliText(language, "请提供有效模型编号或 ID。\n",
                             "Enter a valid model number or ID.\n");
      } else if (!entry->available) {
        std::cout << CliText(language, "模型不可用：", "Model unavailable: ")
                  << entry->issue << '\n';
      } else if (!ModelSupportsBackend(*entry, BackendTypeFromName(backend))) {
        std::cout << CliText(language,
            "此模型格式与当前后端不兼容。请先运行 -backend openvino|onnxruntime|tensorrt。\n",
            "This model format is incompatible with the current backend. Switch with -backend openvino|onnxruntime|tensorrt.\n");
      } else {
        if (!options.backend_explicit) backend = BackendName(catalog.backend());
        const Status loaded = activate_model(entry->model);
        if (!loaded.ok()) std::cerr << loaded.message() << '\n';
        else if (!entry->runnable && !entry->issue.empty())
          std::cout << entry->issue << '\n';
      }
      continue;
    }
    if (command == "-stop" || command == "-unload") {
      engine.Unload();
      history.clear();
      sync_system_prompt();
      std::cout << CliText(language, "模型已卸载。\n", "Model unloaded.\n");
      continue;
    }
    if (command == "-new" || command == "/new" || command == "/reset") {
      history.clear();
      sync_system_prompt();
      std::cout << CliText(language, "已开启新对话。\n", "Started a new chat.\n");
      continue;
    }
    if (command == "-history" || command == "/history") {
      if (history.empty()) {
        std::cout << CliText(language, "当前对话没有消息。\n", "This chat has no messages yet.\n");
      } else {
        for (const InferenceMessage& message : history) {
          const char* role = message.role == MessageRole::kSystem ? "System" :
              message.role == MessageRole::kUser ? "You" : "Isvik";
          std::cout << role << "> " << message.content << '\n';
        }
      }
      continue;
    }
    if (command == "-system" || command == "/system") {
      if (argument.empty()) {
        std::cout << CliText(language, "系统提示词：", "System prompt: ")
                  << (system_prompt.empty() ? CliText(language, "（未设置）", "(not set)")
                                            : std::string_view(system_prompt)) << '\n';
      } else {
        system_prompt = argument == "clear" ? std::string{} : argument;
        sync_system_prompt();
        std::cout << CliText(language, "系统提示词已更新。\n", "System prompt updated.\n");
      }
      continue;
    }
    if (command == "-key") {
      const auto action_separator = argument.find_first_of(" \t");
      const std::string action = argument.substr(0U, action_separator);
      std::string name = action_separator == std::string::npos
          ? std::string{} : argument.substr(action_separator + 1U);
      const auto name_first = name.find_first_not_of(" \t");
      if (name_first == std::string::npos) name.clear();
      else name.erase(0U, name_first);
      const auto name_last = name.find_last_not_of(" \t");
      if (!name.empty()) name.erase(name_last + 1U);
      if (name.size() >= 2U && name.front() == '"' && name.back() == '"')
        name = name.substr(1U, name.size() - 2U);
      if (action == "list") {
        const auto keys = LoadApiKeyVault();
        if (!keys.ok()) std::cerr << keys.status().message() << '\n';
        else if (keys.value().empty())
          std::cout << CliText(language, "密钥库为空。\n", "No saved API keys.\n");
        else for (const ApiKeyEntry& entry : keys.value()) std::cout << entry.name << '\n';
      } else if (action == "create") {
        if (name.empty()) {
          std::cout << CliText(language, "用法：-key create <名称>\n",
                               "Usage: -key create <name>\n");
          continue;
        }
        const std::string key = GenerateCliApiKey();
        if (key.empty()) {
          std::cerr << CliText(language, "生成随机密钥失败。\n",
                               "Could not generate a random API key.\n");
          continue;
        }
        const Status saved = SaveApiKey(name, key);
        if (!saved.ok()) std::cerr << saved.message() << '\n';
        else std::cout << CliText(language, "密钥已保存，请现在复制：\n",
                                  "Key saved. Copy it now:\n") << key << '\n';
      } else if (action == "show") {
        if (name.empty()) {
          std::cout << CliText(language, "用法：-key show <名称>\n",
                               "Usage: -key show <name>\n");
          continue;
        }
        const auto keys = LoadApiKeyVault();
        if (!keys.ok()) std::cerr << keys.status().message() << '\n';
        else {
          const auto found = std::find_if(keys.value().begin(), keys.value().end(),
              [&name](const ApiKeyEntry& entry) { return entry.name == name; });
          if (found == keys.value().end())
            std::cout << CliText(language, "找不到该密钥。\n", "API key not found.\n");
          else std::cout << found->value << '\n';
        }
      } else if (action == "revoke") {
        if (name.empty()) {
          std::cout << CliText(language, "用法：-key revoke <名称>\n",
                               "Usage: -key revoke <name>\n");
          continue;
        }
        const Status removed = DeleteApiKey(name);
        if (!removed.ok()) std::cerr << removed.message() << '\n';
        else std::cout << CliText(language, "密钥已撤销。\n", "API key revoked.\n");
      } else {
        std::cout << CliText(language,
            "用法：-key list | -key create <名称> | -key show <名称> | -key revoke <名称>\n",
            "Usage: -key list | -key create <name> | -key show <name> | -key revoke <name>\n");
      }
      continue;
    }
    if (command == "-serve") {
      if (!engine.model()) {
        std::cout << CliText(language, "请先加载模型，再运行 -serve。\n",
                             "Load a model before starting the API server.\n");
        continue;
      }
      options.model_path = PathUtf8(engine.model()->path);
      options.server_backend = engine.backend();
      options.device = engine.device();
      options.device_explicit = true;
      options.server = true;
      engine.Unload();
      return RunApiServerCommand(options);
    }
    if (command == "-import") {
      if (argument.empty()) {
        std::cout << CliText(language, "用法：-import <模型文件或目录>\n",
                             "Usage: -import <model file or folder>\n");
        continue;
      }
      const std::filesystem::path path = PathFromUtf8(argument);
      StatusOr<ModelScan> scan = std::filesystem::is_directory(path)
          ? ScanModelDirectory(path)
          : StatusOr<ModelScan>(Status::NotFound(""));
      if (!scan.ok()) {
        if (!std::filesystem::is_directory(path)) {
          StatusOr<ModelDescriptor> imported = ImportModel(argument);
          if (!imported.ok()) {
            std::cerr << imported.status().message() << '\n';
            continue;
          }
          ModelScan single;
          single.directory = imported.value().path.parent_path();
          single.models.push_back({std::move(imported).value(), true, true, {}});
          catalog.Merge(single);
          scan = std::move(single);
        } else {
          std::cerr << scan.status().message() << '\n';
          continue;
        }
      } else {
        catalog.Merge(scan.value());
      }
      const Status saved = catalog.Save();
      if (!saved.ok()) std::cerr << saved.message() << '\n';
      else std::cout << CliText(language, "模型已导入。\n", "Model import complete.\n");
      if (scan.ok()) for (const std::string& warning : scan.value().warnings)
        std::cerr << warning << '\n';
      PrintCliModelList(catalog, language);
      continue;
    }
    if (command == "-rm") {
      const CatalogModel* entry = FindCatalogModel(catalog, argument);
      if (entry == nullptr) {
        std::cout << CliText(language, "用法：-rm <编号|ID>\n", "Usage: -rm <number|ID>\n");
        continue;
      }
      const std::string id = entry->model.id;
      const bool removing_active = engine.model() && engine.model()->id == id;
      if (removing_active) { engine.Unload(); history.clear(); }
      const Status removed = catalog.Remove(id);
      if (!removed.ok()) std::cerr << removed.message() << '\n';
      else {
        const Status saved = catalog.Save();
        if (!saved.ok()) std::cerr << saved.message() << '\n';
        else std::cout << CliText(language, "模型已从库中移除（文件未删除）。\n",
                                  "Model removed from catalog (files were not deleted).\n");
      }
      continue;
    }
    if (command == "-memory" || command == "/memory") {
      if (!memories) { std::cout << CliText(language, "记忆库不可用。\n", "Memory store unavailable.\n"); continue; }
      if (argument == "on" || argument == "off") {
        use_memories = argument == "on";
        std::cout << CliText(language, use_memories ? "已启用记忆。\n" : "已停用记忆。\n",
                             use_memories ? "Memories enabled.\n" : "Memories disabled.\n");
      } else {
        const auto entries = argument.empty() ? memories->List(memory_session, 100U)
                                                : memories->Search(argument, memory_session, 100U);
        if (!entries.ok()) std::cerr << entries.status().message() << '\n';
        else PrintMemories(entries.value());
      }
      continue;
    }
    if (command == "-remember" || command == "/remember") {
      if (!memories || argument.empty()) {
        std::cout << CliText(language, "用法：-remember <要保存的内容>\n",
                             "Usage: -remember <text to remember>\n");
      } else {
        MemoryDraft draft;
        draft.content = argument;
        const auto added = memories->Add(draft);
        if (!added.ok()) std::cerr << added.status().message() << '\n';
        else std::cout << CliText(language, "记忆已保存：", "Memory saved: ") << added.value().id << '\n';
      }
      continue;
    }
    if (command == "-forget" || command == "/forget") {
      if (!memories || argument.empty()) {
        std::cout << CliText(language, "用法：-forget <记忆ID>\n", "Usage: -forget <memory ID>\n");
      } else {
        const Status removed = memories->Remove(argument, memory_session);
        if (!removed.ok()) std::cerr << removed.message() << '\n';
        else std::cout << CliText(language, "记忆已删除。\n", "Memory removed.\n");
      }
      continue;
    }
    if (command == "-context" || command == "/context") {
      if (!argument.empty()) {
        uint64_t requested = 0U;
        if (!ParseUInt64(argument, &requested) || requested == 0U || requested > 1000000U)
          std::cout << CliText(language, "用法：-context <1..1000000>\n",
                               "Usage: -context <1..1000000>\n");
        else {
          context_tokens = requested;
          std::cout << CliText(language, "上下文预算已更新。\n", "Context budget updated.\n");
        }
      } else {
        std::cout << CliText(language, "上下文预算：", "Context budget: ")
                  << context_tokens << " tokens\n";
      }
      continue;
    }

    if (!engine.loaded()) {
      std::cout << CliText(language, "请先用 -ls 查看模型，再输入编号或 -use <编号|ID>。\n",
                           "Load a model first: use -ls, then enter its number or -use <number|ID>.\n");
      continue;
    }
    history.push_back(UserMessage(line));
    const auto context = PrepareCliContext(history, *engine.model(), context_tokens,
        options.max_tokens, memories.get(), memory_session, use_memories);
    if (!context.ok()) {
      std::cerr << context.status().message() << '\n';
      history.pop_back();
      continue;
    }
    const auto response = RunRequest(generate, *engine.model(), context.value().messages,
                                     options.generation, &language);
    if (!response.ok()) {
      std::cerr << response.status().message() << '\n';
      history.pop_back();
      continue;
    }
    history.push_back({MessageRole::kAssistant, std::move(response).value(), {}, {}});
  }
  engine.Unload();
  return 0;
}
}  // namespace

void ConfigureOpenVinoRuntime() {
#if defined(ISVIK_HAS_OPENVINO_GENAI) && defined(_WIN32)
  std::filesystem::path root(ISVIK_OPENVINO_GENAI_ROOT);
#if defined(_DEBUG)
  constexpr wchar_t kConfiguration[] = L"Debug";
#else
  constexpr wchar_t kConfiguration[] = L"Release";
#endif
  std::vector<wchar_t> executable_path(32768U, L'\0');
  const DWORD executable_length = GetModuleFileNameW(
      nullptr, executable_path.data(), static_cast<DWORD>(executable_path.size()));
  if (executable_length > 0U && executable_length < executable_path.size()) {
    const std::filesystem::path executable(
        std::wstring(executable_path.data(), executable_length));
    const std::filesystem::path bundled_root = executable.parent_path() / "openvino_bundle";
    const std::filesystem::path bundled_plugins = bundled_root / "runtime" / "bin" / "intel64"
        / kConfiguration;
    if (std::filesystem::exists(bundled_plugins)) root = bundled_root;
  }
  const std::filesystem::path release_bin = root / "runtime" / "bin" / "intel64" / "Release";
  const std::filesystem::path debug_bin = root / "runtime" / "bin" / "intel64" / "Debug";
  const std::filesystem::path tbb_bin = root / "runtime" / "3rdparty" / "tbb" / "bin";
  (void)_putenv_s("INTEL_OPENVINO_DIR", root.string().c_str());
  const char* current_path = std::getenv("PATH");
  std::string updated_path = tbb_bin.string() + ";" + release_bin.string() + ";"
      + debug_bin.string();
  if (current_path != nullptr && current_path[0] != '\0') {
    updated_path += ";";
    updated_path += current_path;
  }
  (void)_putenv_s("PATH", updated_path.c_str());
  const std::string library_paths = tbb_bin.string() + ";" + release_bin.string() + ";"
      + debug_bin.string();
  (void)_putenv_s("OPENVINO_LIB_PATHS", library_paths.c_str());
#endif
}

int RunCli(int argc, char** argv) {
  StatusOr<Options> parsed = ParseOptions(argc, argv);
  if (!parsed.ok()) {
    std::cerr << parsed.status().message() << "\n\n";
    PrintUsage(std::cerr);
    return 2;
  }
  Options options = std::move(parsed).value();
  if (options.help) {
    PrintUsage(std::cout);
    return 0;
  }

  if (options.list_devices) {
#if defined(ISVIK_HAS_OPENVINO_RUNTIME)
    const StatusOr<std::vector<openvino_backend::DeviceInfo>> devices =
        openvino_backend::EnumerateDevices();
    if (!devices.ok()) {
      std::cerr << devices.status().message() << '\n';
      return 1;
    }
    for (const auto& device : devices.value()) {
      std::cout << device.id << "\t" << device.full_name << '\n';
    }
    return 0;
#else
    std::cerr << "Isvik was built without the OpenVINO Runtime backend.\n";
    return 1;
#endif
  }

  if (options.tensorrt_devices || options.tensorrt_inspect ||
      options.tensorrt_build || options.tensorrt_run) {
#if defined(ISVIK_HAS_TENSORRT)
    return RunTensorRtCommand(options);
#else
    std::cerr << "Isvik was built without the optional TensorRT backend. "
                 "Configure with -DISVIK_ENABLE_TENSORRT=ON and a TensorRT SDK.\n";
    return 1;
#endif
  }

  if (options.inspect) {
    StatusOr<ModelDescriptor> model = ImportModel(options.inspect_path);
    if (!model.ok()) {
      std::cerr << model.status().message() << '\n';
      return 1;
    }
    PrintModel(model.value());
    if (model.value().format == ModelFormat::kGguf) {
      const auto inspection = GgufInspector::Inspect(model.value().path);
      if (inspection.ok() && inspection.value().architecture == "gemma4") {
        const auto& config = inspection.value().gemma4;
        std::cout << "Gemma4 layers: " << config.block_count
                  << " · hidden: " << config.embedding_length
                  << " · FFN: " << config.feed_forward_length << '\n';
        std::cout << "Experts: " << config.expert_count << " · active: "
                  << config.expert_used_count << " · expert FFN: "
                  << config.expert_feed_forward_length << '\n';
        std::cout << "KV heads by layer:";
        for (const uint64_t heads : config.attention_head_count_kv) {
          std::cout << ' ' << heads;
        }
        std::cout << '\n' << "Sliding-window layers:";
        for (std::size_t index = 0; index < config.sliding_window_pattern.size(); ++index) {
          if (config.sliding_window_pattern[index]) std::cout << ' ' << index;
        }
        std::cout << '\n';
      }
    }
    return 0;
  }

  if (options.server) return RunApiServerCommand(options);
  if (options.interactive) return RunInteractiveWorkbench(std::move(options));

  if (!options.run_once && !options.interactive) {
    PrintUsage(std::cout);
    return argc == 1 ? 0 : 2;
  }

#if defined(ISVIK_HAS_OPENVINO_GENAI) || defined(ISVIK_HAS_ONNXRUNTIME_GENAI) || defined(ISVIK_HAS_TENSORRT)
  StatusOr<ModelDescriptor> model = ImportModel(options.model_path);
  if (!model.ok()) {
    std::cerr << model.status().message() << '\n';
    return 1;
  }
  PrintModel(model.value());

  if (options.backend_explicit && options.server_backend != "openvino" &&
      options.server_backend != "onnxruntime" && options.server_backend != "tensorrt") {
    std::cerr << "--backend must be openvino, onnxruntime, or tensorrt.\n";
    return 2;
  }
  const std::string backend = options.server_backend;
  if (backend == "tensorrt" && !options.device_explicit) options.device = "GPU.0";
  CliGenerateFunction generate;
#if defined(ISVIK_HAS_OPENVINO_GENAI)
  std::unique_ptr<openvino_backend::GenAiEngine> openvino_engine;
#endif
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
  std::unique_ptr<onnxruntime_backend::GenAiEngine> onnxruntime_engine;
#endif
#if defined(ISVIK_HAS_TENSORRT)
  std::unique_ptr<tensorrt_backend::GgufGenAiEngine> tensorrt_gguf_engine;
#endif

  if (backend == "openvino") {
    if (model.value().format == ModelFormat::kOnnx) {
      std::cerr << "OpenVINO CLI backend requires OpenVINO IR (.xml) or a supported GGUF model.\n";
      return 2;
    }
#if defined(ISVIK_HAS_OPENVINO_GENAI)
    openvino_engine = std::make_unique<openvino_backend::GenAiEngine>();
    std::cout << "Loading with OpenVINO on " << options.device << "..." << std::flush;
    const Status loaded = openvino_engine->LoadModel(model.value(), options.device);
    if (!loaded.ok()) { std::cerr << "\n" << loaded.message() << '\n'; return 1; }
    generate = [engine = openvino_engine.get()](const UnifiedInferenceRequest& request,
        const CancellationToken& cancellation,
        const std::function<void(const InferenceEvent&)>& handler) {
      return engine->Generate(request, cancellation, handler);
    };
#else
    std::cerr << "This build does not include OpenVINO GenAI.\n";
    return 1;
#endif
  } else if (backend == "onnxruntime") {
    if (model.value().format != ModelFormat::kOnnx) {
      std::cerr << "ONNX Runtime CLI backend requires an ONNX model directory or .onnx file.\n";
      return 2;
    }
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
    onnxruntime_engine = std::make_unique<onnxruntime_backend::GenAiEngine>();
    std::cout << "Loading with ONNX Runtime on " << options.device << "..." << std::flush;
    const Status loaded = onnxruntime_engine->LoadModel(model.value(), options.device);
    if (!loaded.ok()) { std::cerr << "\n" << loaded.message() << '\n'; return 1; }
    generate = [engine = onnxruntime_engine.get()](const UnifiedInferenceRequest& request,
        const CancellationToken& cancellation,
        const std::function<void(const InferenceEvent&)>& handler) {
      return engine->Generate(request, cancellation, handler);
    };
#else
    std::cerr << "This build does not include ONNX Runtime GenAI.\n";
    return 1;
#endif
  } else if (backend == "tensorrt" && model.value().format == ModelFormat::kGguf) {
#if defined(ISVIK_HAS_TENSORRT)
    int device_index = 0;
    std::string_view device = options.device;
    if (device == "GPU") {
      device_index = 0;
    } else if (device.starts_with("GPU.") && ParseInt(device.substr(4U), &device_index) &&
               device_index >= 0) {
    } else {
      std::cerr << "TensorRT GGUF requires GPU or GPU.INDEX (for example GPU.0).\n";
      return 2;
    }
    tensorrt_gguf_engine = std::make_unique<tensorrt_backend::GgufGenAiEngine>(
        [](std::string_view message) { std::cout << "[TensorRT] " << message << std::endl; });
    std::cout << "Loading with TensorRT on GPU." << device_index << "..." << std::flush;
    const Status loaded = tensorrt_gguf_engine->LoadModel(model.value(), device_index);
    if (!loaded.ok()) { std::cerr << "\n" << loaded.message() << '\n'; return 1; }
    generate = [engine = tensorrt_gguf_engine.get()](const UnifiedInferenceRequest& request,
        const CancellationToken& cancellation,
        const std::function<void(const InferenceEvent&)>& handler) {
      return engine->Generate(request, cancellation, handler);
    };
#else
    std::cerr << "This build does not include the TensorRT GGUF runtime.\n";
    return 1;
#endif
  } else if (backend == "tensorrt" && model.value().format == ModelFormat::kOnnx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP) && defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
    if (options.device == "CPU") options.device = "GPU.0";
    onnxruntime_engine = std::make_unique<onnxruntime_backend::GenAiEngine>();
    std::cout << "Loading ONNX through TensorRT-RTX on " << options.device << "..." << std::flush;
    const Status loaded = onnxruntime_engine->LoadModel(model.value(), options.device);
    if (!loaded.ok()) { std::cerr << "\n" << loaded.message() << '\n'; return 1; }
    generate = [engine = onnxruntime_engine.get()](const UnifiedInferenceRequest& request,
        const CancellationToken& cancellation,
        const std::function<void(const InferenceEvent&)>& handler) {
      return engine->Generate(request, cancellation, handler);
    };
#else
    std::cerr << "TensorRT ONNX chat requires TensorRT-RTX and ONNX Runtime GenAI in this build.\n";
    return 1;
#endif
  } else {
    std::cerr << "TensorRT CLI supports GGUF and ONNX models; this model format is not supported.\n";
    return 2;
  }
  std::cout << " ready\n";

  std::unique_ptr<MemoryService> memories;
  std::string memory_session = "cli-" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  auto opened_memories = MemoryService::Open(MemoryStoragePath());
  if (opened_memories.ok()) {
    memories = std::move(opened_memories).value();
    static_cast<void>(memories->SetSessionMemoryEnabled(memory_session, true));
  } else {
    std::cerr << "Memory service unavailable; inference will continue without saved notes: "
              << opened_memories.status().message() << '\n';
  }
  MemorySessionGuard memory_session_guard{memories.get(), memory_session};
  const uint64_t effective_context_tokens = model.value().context_length == 0U
      ? options.context_tokens
      : std::min<uint64_t>(options.context_tokens, model.value().context_length);

  if (options.run_once) {
    const auto context = PrepareCliContext(
        {UserMessage(options.prompt)}, model.value(), options.context_tokens,
        options.max_tokens, memories.get(), memory_session, options.use_memories);
    if (!context.ok()) {
      std::cerr << context.status().message() << '\n';
      return 1;
    }
    PrintContextStatus(context.value(), effective_context_tokens);
    const StatusOr<std::string> status = RunRequest(
        generate, model.value(), context.value().messages, options.generation);
    if (!status.ok()) {
      std::cerr << status.status().message() << '\n';
      return 1;
    }
    return 0;
  }

  std::vector<InferenceMessage> history;
  bool use_memories = options.use_memories;
  bool session_memory_enabled = true;
  uint64_t context_tokens = options.context_tokens;
  ChatContextResult last_context;
  bool has_last_context = false;
  std::cout << "Chat started / 对话已开始. Commands: /new, /history, /memory, /context, /exit.\n";
  std::string line;
  while (std::cout << "You> " && std::getline(std::cin, line)) {
    if (line == "/exit" || line == "/quit" || line == "-exit") break;
    if (line == "/new" || line == "-new" || line == "/reset") {
      history.clear();
      std::cout << "Conversation cleared.\n";
      continue;
    }
    if (line == "/history" || line == "-history") {
      if (history.empty()) {
        std::cout << "No messages in this conversation yet.\n";
      } else {
        for (const InferenceMessage& message : history) {
          std::cout << (message.role == MessageRole::kUser ? "You> " : "Isvik> ")
                    << message.content << '\n';
        }
      }
      continue;
    }
    if (line == "/help" || line == "-help") {
      std::cout << "/new clears this chat; /history prints this session; /memory views saved notes; "
                   "/context shows the context budget; /exit returns to the terminal.\n";
      continue;
    }
    if (line == "/context" || line.starts_with("/context ")) {
      const std::string argument = line.size() > 8U ? line.substr(8U) : std::string{};
      if (argument.empty()) {
        if (has_last_context) PrintContextStatus(last_context, effective_context_tokens);
        else std::cout << "Context budget: " << std::min<uint64_t>(context_tokens, model.value().context_length == 0U ? context_tokens : model.value().context_length) << " tokens; prompt estimates use UTF-8 byte length.\n";
      } else {
        uint64_t parsed = 0U;
        if (!ParseUInt64(argument, &parsed) || parsed == 0U || parsed > 1000000U) {
          std::cerr << "Usage: /context [1..1000000]\n";
        } else {
          context_tokens = parsed;
          const uint64_t limit = model.value().context_length == 0U ? context_tokens
              : std::min<uint64_t>(context_tokens, model.value().context_length);
          std::cout << "Context budget set to " << limit << " tokens for this session.\n";
        }
      }
      continue;
    }
    if (line == "/memory on" || line == "/memory off") {
      use_memories = line == "/memory on";
      std::cout << (use_memories ? "Saved memories will be considered when relevant.\n"
                                 : "Saved memories will not be added to prompts.\n");
      continue;
    }
    if (line == "/session-memory on" || line == "/session-memory off") {
      session_memory_enabled = line == "/session-memory on";
      if (memories) {
        const Status status = memories->SetSessionMemoryEnabled(memory_session, session_memory_enabled);
        if (!status.ok()) std::cerr << status.message() << '\n';
      }
      std::cout << (session_memory_enabled ? "Session-only memories enabled.\n"
                                           : "Session-only memories hidden until re-enabled.\n");
      continue;
    }
    if (line == "/memory" || line.starts_with("/memory ")) {
      const std::string query = line.size() > 8U ? line.substr(8U) : std::string{};
      if (query == "on" || query == "off") continue;
      if (!memories) { std::cerr << "Memory database is unavailable.\n"; continue; }
      const auto entries = query.empty() ? memories->List(memory_session, 100U)
                                         : memories->Search(query, memory_session, 100U);
      if (!entries.ok()) std::cerr << entries.status().message() << '\n';
      else PrintMemories(entries.value());
      continue;
    }
    if (line == "/remember" || line.starts_with("/remember ")) {
      if (!memories) { std::cerr << "Memory database is unavailable.\n"; continue; }
      std::string content = line.size() > 10U ? line.substr(10U) : std::string{};
      MemoryDraft draft;
      if (content.starts_with("--session ")) {
        draft.scope = MemoryScope::kSession;
        draft.session_id = memory_session;
        content.erase(0U, 10U);
        if (!session_memory_enabled) {
          std::cerr << "Enable session memories first with /session-memory on.\n";
          continue;
        }
      }
      if (content.find_first_not_of(" \t\r\n") == std::string::npos) {
        std::cerr << "Usage: /remember [--session] TEXT\n";
        continue;
      }
      draft.content = std::move(content);
      const auto added = memories->Add(draft);
      if (!added.ok()) std::cerr << added.status().message() << '\n';
      else std::cout << "Memory saved as " << added.value().id << ".\n";
      continue;
    }
    if (line == "/forget" || line.starts_with("/forget ")) {
      if (!memories) { std::cerr << "Memory database is unavailable.\n"; continue; }
      const std::string id = line.size() > 8U ? line.substr(8U) : std::string{};
      if (id.empty()) { std::cerr << "Usage: /forget MEMORY_ID\n"; continue; }
      const Status status = memories->Remove(id, memory_session);
      if (!status.ok()) std::cerr << status.message() << '\n';
      else std::cout << "Memory removed.\n";
      continue;
    }
    if (line.empty()) continue;
    history.push_back(UserMessage(line));
    const auto context = PrepareCliContext(history, model.value(), context_tokens,
        options.max_tokens, memories.get(), memory_session, use_memories);
    if (!context.ok()) {
      std::cerr << context.status().message() << '\n';
      history.pop_back();
      continue;
    }
    last_context = context.value();
    has_last_context = true;
    PrintContextStatus(last_context, model.value().context_length == 0U ? context_tokens
        : std::min<uint64_t>(context_tokens, model.value().context_length));
    StatusOr<std::string> response = RunRequest(
        generate, model.value(), last_context.messages, options.generation);
    if (!response.ok()) {
      std::cerr << response.status().message() << '\n';
      history.pop_back();
      continue;
    }
    InferenceMessage assistant;
    assistant.role = MessageRole::kAssistant;
    assistant.content = std::move(response).value();
    history.push_back(std::move(assistant));
  }
  return 0;
#else
  std::cerr << "This build does not include an interactive inference runtime. "
               "Enable OpenVINO GenAI, ONNX Runtime GenAI, or TensorRT GGUF.\n";
  return 1;
#endif
}

}  // namespace isvik::app
