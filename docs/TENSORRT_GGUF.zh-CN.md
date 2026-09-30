# 使用 TensorRT 运行 GGUF 模型

[English](TENSORRT_GGUF.md) | [返回 Isvik.cpp 首页](../README.zh-CN.md)

此功能仍处于实验阶段。Isvik.cpp 主要为 OpenVINO 开发；NVIDIA TensorRT 是可选后端。它可以在 NVIDIA GPU 上直接运行**受支持的 Gemma 4 文本 GGUF 模型**。程序读取原始 GGUF 文件，并通过自定义 TensorRT 插件执行量化矩阵向量乘法。TensorRT 自身不负责解析 GGUF；文件读取、分词、模型解码循环和 CUDA 内核由 Isvik 实现。这条路径不链接或启动 llama.cpp。

## 环境要求与限制

- 构建时启用 `ISVIK_ENABLE_TENSORRT=ON`，并安装兼容的 CUDA Toolkit 和 TensorRT SDK。Windows 预设会请求启用 TensorRT；找不到 SDK 时，CMake 会跳过该后端。
- 选择 NVIDIA GPU，例如 `GPU.0`。
- 使用内嵌 BPE 分词器、包含所需张量的 Gemma 4 文本 GGUF。
- 当前原生解码器接受的 GGUF 张量编码为 `F32`、`F16`、`Q5_1`、`Q8_0`、`Q5_K`、`IQ4_NL`、`IQ3_S` 和 `MXFP4`。能识别量化块布局不代表能执行；完整判断以源码中的 [`IsGgufTensorEncodingDecodable`](../src/core/gguf_quant.cpp) 为准。
- 模型架构、编码或分词器元数据不符合要求时，加载前的校验会返回错误。

这条原生 GGUF 路径与 [TensorRT 的 ONNX 引擎构建](../src/backends/tensorrt/engine.cpp)、ONNX Runtime GenAI 使用的 TensorRT-RTX 执行提供程序是不同的入口。

## 运行 GGUF 模型

1. 按照[构建指南](BUILDING.md)构建包含 TensorRT 的程序。
2. 检查模型：

   ```powershell
   ./out/build/windows-msvc-release/Release/Isvik.exe --inspect "<Gemma-4-GGUF-路径>"
   ```

3. 在 TensorRT GPU 上运行：

   ```powershell
   ./out/build/windows-msvc-release/Release/Isvik.exe --run --model "<Gemma-4-GGUF-路径>" --backend tensorrt --device GPU.0 --prompt "你好" --max-tokens 64
   ```

`--inspect` 会报告文件元数据。真正加载时，程序还会检查架构、分词器、必需张量和编码。

## 按源码阅读实现

| 阶段 | 源码 | 作用 |
| --- | --- | --- |
| 解析文件 | [`gguf_inspector.cpp`](../src/core/gguf_inspector.cpp) | 读取 GGUF 元数据、张量位置、分词器元数据和 Gemma 4 参数。 |
| 读取权重 | [`gguf_model_file.cpp`](../src/core/gguf_model_file.cpp) | 以只读方式打开原文件，尝试内存映射，并按需读取张量区间。 |
| 分词与生成 | [`gguf_tokenizer.cpp`](../src/core/gguf_tokenizer.cpp)、[`gguf_genai_engine.cpp`](../src/backends/tensorrt/gguf_genai_engine.cpp) | 组织提示词，使用内嵌 BPE 分词器，运行 Gemma 4 解码循环，并输出文本及用量数据。 |
| 解码小张量 | [`gguf_quant.cpp`](../src/core/gguf_quant.cpp) | 定义量化块大小，解码嵌入向量和 C++ 解码循环使用的小张量。 |
| 构建 TensorRT 运算 | [`engine.cpp`](../src/backends/tensorrt/engine.cpp) | 构建或加载包含 `activations`、`packed_weights` 输入和自定义插件层的引擎。 |
| 执行量化乘法 | [`gguf_matvec_plugin.cpp`](../src/backends/tensorrt/gguf_matvec_plugin.cpp)、[`gguf_matvec_kernel.cu`](../src/backends/tensorrt/gguf_matvec_kernel.cu) | 注册 `IPluginV3`，在 CUDA 内核中读取原始量化块并计算浮点结果。 |

### 保留 GGUF 原始权重

[`GgufGenAiEngine::Impl::MatVec`](../src/backends/tensorrt/gguf_genai_engine.cpp) 从模型文件中按行读取压缩权重，并把相同的字节作为 TensorRT 的 `INT8` 输入传给插件。插件按照原来的 GGUF 张量类型解释这些字节。`INT8` 在这里只承载字节，没有再次量化模型。

```cpp
auto packed = file->ReadTensorRange(
    name, offset + first_row * row_bytes, current_rows * row_bytes);
auto engine = GetEngine(info->type, width, current_rows);
weights.bytes = std::move(packed).value();
auto result = engine.value()->Infer({std::move(activations), std::move(weights)});
```

[`BuildGgufQuantizedMatVecEngine`](../src/backends/tensorrt/engine.cpp) 为这个运算创建 TensorRT 网络：激活值是浮点输入，压缩权重是字节输入，再加入 Isvik 的 `IPluginV3` 层。

```cpp
nvinfer1::ITensor* packed_weights =
    network->addInput("packed_weights", nvinfer1::DataType::kINT8, weight_dims);
nvinfer1::ITensor* plugin_inputs[]{activations, packed_weights};
network->addPluginV3(plugin_inputs, 2, nullptr, 0, *plugin);
```

[CUDA 内核](../src/backends/tensorrt/gguf_matvec_kernel.cu)在计算矩阵向量结果时解码 GGUF 量化块。[插件实现](../src/backends/tensorrt/gguf_matvec_plugin.cpp)会在构建和加载引擎时保存张量类型与矩阵维度。

### 完成 Gemma 4 解码循环

[`gguf_genai_engine.cpp`](../src/backends/tensorrt/gguf_genai_engine.cpp)在 C++ 中维护 KV 缓存，并执行注意力、归一化、专家路由和采样；矩阵向量运算调用 TensorRT 插件。这样可以直接使用原格式 GGUF，但多次读取权重以及主机与 GPU 之间的数据传输可能让生成速度较慢。

程序根据张量类型和矩阵维度缓存引擎，缓存目录是系统临时目录下的 `Isvik/TensorRT`。对较大的矩阵，每个引擎最多处理 4,096 行，以控制引擎大小和长期占用的 GPU 显存。这些 `.engine` 文件只是运行时缓存，GGUF 原文件不会被改写。

## 查看耗时

运行前设置 `ISVIK_TENSORRT_PROFILE`：

```powershell
$env:ISVIK_TENSORRT_PROFILE = "1"
./out/build/windows-msvc-release/Release/Isvik.exe --run --model "<Gemma-4-GGUF-路径>" --backend tensorrt --device GPU.0 --prompt "你好" --max-tokens 64
```

程序会输出总耗时、矩阵向量调用次数、读取的权重字节数，以及文件读取、引擎查找和推理各阶段耗时。可用这些数据定位当前设备上的主要耗时。
