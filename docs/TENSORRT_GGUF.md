# Run GGUF models with TensorRT

[简体中文](TENSORRT_GGUF.zh-CN.md) | [Isvik.cpp home](../README.md)

Isvik.cpp has a native path for **supported Gemma 4 text GGUF models** on NVIDIA GPUs. It reads the original GGUF file and uses a custom TensorRT plugin for quantized matrix-vector multiplication. TensorRT itself does not parse GGUF; Isvik supplies the file reader, tokenizer, model loop, and CUDA kernel. This path does not link or launch llama.cpp.

## Requirements and limits

- Build Isvik.cpp with `ISVIK_ENABLE_TENSORRT=ON`, a CUDA Toolkit, and a compatible TensorRT SDK. The supplied Windows preset requests TensorRT, but CMake omits the backend when its SDK is unavailable.
- Use an NVIDIA GPU device such as `GPU.0`.
- Use a Gemma 4 text GGUF file with an embedded BPE tokenizer and the required model tensors.
- The native decoder currently accepts these GGUF tensor encodings: `F32`, `F16`, `Q5_1`, `Q8_0`, `Q5_K`, `IQ4_NL`, `IQ3_S`, and `MXFP4`. A type with a known block layout is not necessarily executable; the complete decoder list is in [`IsGgufTensorEncodingDecodable`](../src/core/gguf_quant.cpp).
- Other architectures, unsupported tensor encodings, and missing tokenizer metadata are rejected during validation.

The native GGUF path is separate from [TensorRT's ONNX engine builder](../src/backends/tensorrt/engine.cpp) and the TensorRT-RTX execution provider used with ONNX Runtime GenAI.

## Try a GGUF model

1. [Build the application](BUILDING.md) with TensorRT enabled.
2. Inspect the model:

   ```powershell
   ./out/build/windows-msvc-release/Release/Isvik.exe --inspect "<path-to-gemma-4.gguf>"
   ```

3. Run a prompt on a TensorRT GPU:

   ```powershell
   ./out/build/windows-msvc-release/Release/Isvik.exe --run --model "<path-to-gemma-4.gguf>" --backend tensorrt --device GPU.0 --prompt "Hello" --max-tokens 64
   ```

The inspection command reports the file's metadata. The actual load also checks the architecture, tokenizer, required tensors, and their encodings.

## Follow the source code

| Stage | Source | What it does |
| --- | --- | --- |
| Inspect | [`gguf_inspector.cpp`](../src/core/gguf_inspector.cpp) | Parses GGUF metadata, tensor locations, tokenizer metadata, and Gemma 4 settings. |
| Read weights | [`gguf_model_file.cpp`](../src/core/gguf_model_file.cpp) | Opens the original file read-only, attempts memory mapping, and reads requested tensor ranges. |
| Tokenize and decode | [`gguf_tokenizer.cpp`](../src/core/gguf_tokenizer.cpp), [`gguf_genai_engine.cpp`](../src/backends/tensorrt/gguf_genai_engine.cpp) | Builds the prompt, tokenizes with embedded BPE data, runs the Gemma 4 loop, samples tokens, and emits text and usage events. |
| Decode small tensors | [`gguf_quant.cpp`](../src/core/gguf_quant.cpp) | Defines block sizes and decodes the embeddings and smaller tensors used by the C++ model loop. |
| Build and run a TensorRT operation | [`engine.cpp`](../src/backends/tensorrt/engine.cpp) | Builds or loads an engine with `activations` and `packed_weights` inputs and a custom plugin layer. |
| Execute the plugin | [`gguf_matvec_plugin.cpp`](../src/backends/tensorrt/gguf_matvec_plugin.cpp), [`gguf_matvec_kernel.cu`](../src/backends/tensorrt/gguf_matvec_kernel.cu) | Registers `IPluginV3`, reads the packed GGUF blocks inside a CUDA kernel, and produces a floating-point result. |

### Keep the original GGUF weights

[`GgufGenAiEngine::Impl::MatVec`](../src/backends/tensorrt/gguf_genai_engine.cpp) reads packed rows directly from the model file. It passes their byte patterns as TensorRT `INT8` input; the plugin interprets them according to the original GGUF tensor type. This `INT8` input is an opaque byte carrier, not an extra quantization step.

```cpp
auto packed = file->ReadTensorRange(
    name, offset + first_row * row_bytes, current_rows * row_bytes);
auto engine = GetEngine(info->type, width, current_rows);
weights.bytes = std::move(packed).value();
auto result = engine.value()->Infer({std::move(activations), std::move(weights)});
```

[`BuildGgufQuantizedMatVecEngine`](../src/backends/tensorrt/engine.cpp) creates the operation's TensorRT network with a float activation input and a packed-weight input. It adds Isvik's `IPluginV3` layer instead of converting the GGUF model to ONNX.

```cpp
nvinfer1::ITensor* packed_weights =
    network->addInput("packed_weights", nvinfer1::DataType::kINT8, weight_dims);
nvinfer1::ITensor* plugin_inputs[]{activations, packed_weights};
network->addPluginV3(plugin_inputs, 2, nullptr, 0, *plugin);
```

The [CUDA kernel](../src/backends/tensorrt/gguf_matvec_kernel.cu) decodes weights from their GGUF block representation while computing each matrix-vector result. The [TensorRT plugin](../src/backends/tensorrt/gguf_matvec_plugin.cpp) carries the tensor type and matrix dimensions through engine build and load.

### Complete the Gemma 4 decoding loop

[`gguf_genai_engine.cpp`](../src/backends/tensorrt/gguf_genai_engine.cpp) keeps the KV cache, attention, normalization, expert routing, and token sampling in C++. Matrix-vector calls use the TensorRT plugin. This division makes the GGUF file usable without changing its format, but frequent weight reads and host-to-GPU transfers can make generation slow.

Engines are cached by tensor type and matrix dimensions under the system temporary directory's `Isvik/TensorRT` folder. Large matrices are split into groups of at most 4,096 rows to limit engine size and persistent GPU memory. These `.engine` files are generated runtime caches; the source GGUF file remains unchanged.

## Inspect runtime costs

Set `ISVIK_TENSORRT_PROFILE` before running the command:

```powershell
$env:ISVIK_TENSORRT_PROFILE = "1"
./out/build/windows-msvc-release/Release/Isvik.exe --run --model "<path-to-gemma-4.gguf>" --backend tensorrt --device GPU.0 --prompt "Hello" --max-tokens 64
```

The runtime prints total time, matrix-vector call count, weight bytes read, file-read time, engine lookup time, and inference time. Use these values to identify which stage dominates on your machine.
