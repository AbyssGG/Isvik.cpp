# Isvik.cpp

<p align="center">
  <img src="https://raw.githubusercontent.com/AbyssGG/Isvik.cpp/main/resources/isvik-cpp-banner.png" alt="Isvik.cpp logo on a dark banner" width="100%">
</p>

<p align="center">
  <a href="https://github.com/AbyssGG/Isvik" title="Original Isvik project">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AbyssGG/Isvik.cpp/main/resources/isvik-lockup.png">
      <img src="https://raw.githubusercontent.com/AbyssGG/Isvik.cpp/main/resources/isvik-lockup-black.png" alt="Original Isvik logo" width="240">
    </picture>
  </a><br>
  <sub>Original Isvik logo · © 2026 AbyssGG</sub>
</p>

<p align="center">
  <strong>Built for OpenVINO and local AI on Intel AI PCs.</strong><br>
  C++20 · OpenVINO GenAI · Optional ONNX support · Desktop chat · Interactive CLI · Local APIs
</p>

<p align="center">
  <a href="CMakeLists.txt"><img src="https://img.shields.io/badge/version-0.1.0-1a73e8?style=flat-square" alt="Isvik.cpp source version 0.1.0"></a>
  <a href="docs/MODEL_SUPPORT.md"><img src="https://img.shields.io/badge/OpenVINO%20GenAI-2026.4.0.0-0071C5?style=flat-square&amp;logo=intel&amp;logoColor=white" alt="OpenVINO GenAI 2026.4.0.0"></a>
  <a href="docs/MODEL_SUPPORT.md"><img src="https://img.shields.io/badge/ONNX%20Runtime%20GenAI-0.17.0-005CED?style=flat-square" alt="Optional ONNX Runtime GenAI 0.17.0"></a>
  <a href="docs/BUILDING.md"><img src="https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&amp;logo=cplusplus&amp;logoColor=white" alt="C++20"></a>
  <a href="docs/TENSORRT_GGUF.md"><img src="https://img.shields.io/badge/NVIDIA-TensorRT%20GGUF%20experimental-76B900?style=flat-square&amp;logo=nvidia&amp;logoColor=white" alt="Experimental NVIDIA TensorRT GGUF support"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-2E7D32?style=flat-square" alt="Apache 2.0 license"></a>
  <a href="https://github.com/AbyssGG/Isvik.cpp/actions/workflows/ci.yml"><img src="https://github.com/AbyssGG/Isvik.cpp/actions/workflows/ci.yml/badge.svg" alt="Build status"></a>
</p>

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">简体中文</a>
</p>

## What is Isvik.cpp?

Isvik.cpp is a C++20 local AI runtime designed specifically to make OpenVINO language-model inference accessible on Intel AI PCs. It provides a desktop chat interface, a persistent bilingual CLI, a model library, saved memories, and a local API server. Inference runs on your hardware without a cloud account. Compatible ONNX Runtime GenAI model packages are supported through an optional backend; NVIDIA TensorRT is an experimental integration.

## Built for OpenVINO on Intel AI PCs

The project began with a practical goal: make OpenVINO language-model inference easy to use from a desktop app and a real terminal, then expose the same local model through an API. OpenVINO GenAI is the primary path for OpenVINO IR models and the GGUF models supported by the installed runtime. You can select CPU, Intel GPU, or NPU when the model and device support that combination.

The model library, chat, CLI, and API share the same runtime services. This lets you inspect a model, choose a device, and use it in the interface that fits your workflow. See [model support](docs/MODEL_SUPPORT.md) for the exact compatibility limits.

## Supported backends

| Backend | Role | Model input | Where it runs |
| --- | --- | --- | --- |
| OpenVINO GenAI | Primary runtime | OpenVINO IR and selected GGUF models | Supported CPU, GPU, or NPU devices |
| ONNX Runtime GenAI | Optional | Compatible ONNX text-generation packages | Providers available in the build |
| NVIDIA TensorRT | Experimental | Supported Gemma 4 GGUF models through Isvik's native plugin | NVIDIA GPU |
| TensorRT-RTX provider | Optional | Compatible ONNX Runtime GenAI packages | Supported NVIDIA GPU, with configured fallback |

Model recognition does not guarantee inference support. Architecture, tensor encoding, tokenizer assets, installed SDKs, and the selected device all matter. The TensorRT GGUF implementation has limited model coverage and known performance costs; it is still being developed. See [model support](docs/MODEL_SUPPORT.md).

## Versions

**Isvik.cpp is at source version 0.1.0**, as set in [CMake](CMakeLists.txt) and the Windows executable resource. This identifies the current source; no binary GitHub release has been published yet.

| Component | Version used by this repository |
| --- | --- |
| OpenVINO GenAI and its bundled runtime (Windows) | `2026.4.0.0` |
| ONNX Runtime GenAI (Windows) | `0.17.0` |
| ONNX Runtime (Windows) | `1.26.0` |
| Slint C++ | `1.18.1` |
| TensorRT-RTX execution provider ABI (optional, Windows) | `0.4.2` |
| TensorRT SDK and CUDA Toolkit (optional) | Versions from the local installation; no fixed patch version |

The pinned package values come from the [CMake integration files](cmake/). A backend is included only when its required SDK is available in the selected build.

## Get started

### Build on Windows

Install Visual Studio 2026 with C++ support, CMake 3.21 or later, and Git. From the repository root, run:

```powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --target Isvik --parallel
./out/build/windows-msvc-release/Release/Isvik.exe
```

The project is named Isvik.cpp; the executable is named `Isvik.exe`. The Windows preset enables optional backends when their SDKs are available. See the [build guide](docs/BUILDING.md) for other configurations.

### Build on Linux

**Linux support has not been validated.** The following build commands are provisional; the desktop app, OpenVINO inference, and optional backends have not been confirmed to work on Linux.

Use a C++20 GCC toolchain and Ninja:

```sh
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release --parallel
```

The supplied Linux preset disables OpenVINO by default. Configure optional backends with their SDKs before using them.

## Use the interactive CLI

On Windows, start the terminal workbench directly:

```powershell
./out/build/windows-msvc-release/Release/Isvik.exe -cli
```

The prompt stays open for chat and model management. Enter `-ls` to list imported models, a model number to select one, and `-params` to inspect the active settings. Each completed reply reports input tokens, output tokens, elapsed time, and tokens per second. Use `-lang zh` or `-lang en` to switch language, and `-exit` to quit. Run `Isvik.exe --help` for all commands.

To run a single prompt:

```powershell
./out/build/windows-msvc-release/Release/Isvik.exe --run --model "<path-to-model>" --device CPU --prompt "Explain local inference." --max-tokens 64
```

Choose a backend and device supported by your model. The [model support guide](docs/MODEL_SUPPORT.md) explains the available combinations.

## Start the local API

```powershell
./out/build/windows-msvc-release/Release/Isvik.exe --server --model "<path-to-model>" --backend openvino
```

The default address is `http://127.0.0.1:1234`. The server exposes Isvik-native endpoints, OpenAI Chat Completions, and Anthropic Messages. See the [API guide](docs/API.md) for endpoints, streaming, and authentication.

## How TensorRT runs GGUF

The experimental native TensorRT path reads a supported Gemma 4 GGUF file in place. Isvik parses its metadata, uses its embedded BPE tokenizer, and reads packed tensor blocks from the original file. A custom TensorRT `IPluginV3` passes those packed bytes to CUDA matrix-vector kernels. Isvik's C++ runtime handles the remaining decoding steps, including attention and the KV cache.

This path does not require a GGUF-to-ONNX conversion. TensorRT engines for the custom operation are cached separately; the GGUF model file is not rewritten. Current support is limited to Gemma 4 text models with supported tensor encodings, and this implementation can be slow because it repeatedly reads weights and moves data between host and GPU.

Read the [TensorRT GGUF source guide](docs/TENSORRT_GGUF.md) for the exact source files, data flow, supported encodings, and a run command.

## Documentation

| Guide | Description |
| --- | --- |
| [Build Isvik.cpp](docs/BUILDING.md) | Requirements, presets, and optional backends |
| [Model support](docs/MODEL_SUPPORT.md) | Formats, backends, and limitations |
| [TensorRT GGUF source guide](docs/TENSORRT_GGUF.md) | How the native GGUF path works in source code |
| [API guide](docs/API.md) | Isvik, OpenAI, and Anthropic endpoints |
| [Architecture](docs/ARCHITECTURE.md) | Application, core, and backend layers |
| [Contributing](CONTRIBUTING.md) | How to contribute |
| [Documentation style](docs/STYLE_GUIDE.md) | Writing and C++ style conventions |

See the [code of conduct](CODE_OF_CONDUCT.md), [security policy](SECURITY.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

## License

Isvik.cpp is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies retain their own license terms.

The Isvik.cpp logo and wordmark carry a `© 2026 AbyssGG` notice. The Apache-2.0 license does not grant trademark rights to the project name or logo. See the [branding guidance](BRANDING.md) before reusing them.
