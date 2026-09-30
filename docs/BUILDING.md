# Build Isvik.cpp

This guide describes how to configure, build, and test Isvik.cpp from source.

## Requirements

- CMake 3.21 or later.
- A C++20 compiler.
- Git.
- Network access during the first configure so CMake can download pinned dependencies.

On Windows, use Visual Studio with the Desktop development with C++ workload. The supplied Windows presets use the Visual Studio 2026 generator. On Linux, use GCC and Ninja.

## Configure and build on Windows

Run these commands from the repository root:

~~~powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --target Isvik --parallel
~~~

The Debug preset is `windows-msvc-debug`. Build the `Isvik` target to compile the desktop application.

## Configure and build on Linux

Linux support has not been validated. The commands below are provisional, and the desktop app and inference backends have not been confirmed to run on Linux.

~~~sh
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release --parallel
~~~

The Linux preset disables OpenVINO by default. Enable optional backends only when their SDKs and runtime dependencies are installed.

## Run the tests

~~~powershell
ctest --preset windows-msvc-release
~~~

~~~sh
ctest --preset linux-gcc-release
~~~

CTest reports each failed test and its output. Some Windows TensorRT tests require a supported NVIDIA GPU, CUDA Toolkit, and TensorRT SDK.

## Configure optional backends

CMake options control optional integrations:

- `ISVIK_ENABLE_OPENVINO` enables the OpenVINO runtime integration.
- `ISVIK_ENABLE_OPENVINO_GENAI` enables OpenVINO GenAI inference.
- `ISVIK_ENABLE_ONNXRUNTIME_GENAI` enables ONNX Runtime GenAI inference.
- `ISVIK_ENABLE_TENSORRT` enables TensorRT when its SDK and CUDA Toolkit are available.
- `ISVIK_ENABLE_TENSORRT_RTX_EP` enables the TensorRT-RTX execution provider when its dependencies are available.

CMake skips an optional backend when its required SDK is unavailable. The application can still build without those optional runtimes.

## Build outputs

CMake writes generated files under `out/build/&lt;preset-name&gt;`. On Windows, the application executable is `Release/Isvik.exe` or `Debug/Isvik.exe` under the selected build directory. Install files go to the configured install prefix.
