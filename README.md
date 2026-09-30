# Isvik.cpp

Isvik.cpp is a C++20 desktop application and local runtime for running language models on your own hardware. It includes a chat interface, a command-line interface, model management, persistent memory, and an HTTP API.

## Features

- Run supported OpenVINO IR and GGUF models with OpenVINO GenAI.
- Run compatible ONNX Runtime GenAI models.
- Use the optional NVIDIA TensorRT backend for ONNX engine execution and native Gemma 4 GGUF inference.
- Select CPU, GPU, or NPU devices when the selected backend supports them.
- Manage local models, conversation context, saved memories, and decoding settings.
- Serve Isvik-native, OpenAI Chat Completions, and Anthropic Messages APIs.
- Run without a cloud account or remote inference service.

Model support depends on the model architecture, tokenizer files, quantization, backend, and installed runtime. See [model support](docs/MODEL_SUPPORT.md) for details.

## Get started

### Requirements

- CMake 3.21 or later.
- A C++20 compiler.
- Git.
- Internet access for the first CMake configure unless dependencies are available locally.
- Windows with Visual Studio 2026, or Linux x86-64 with GCC and Ninja for the supplied presets.

CMake downloads pinned third-party packages and verifies their hashes. OpenVINO, ONNX Runtime GenAI, and TensorRT are optional outside the supplied Windows presets.

### Build on Windows

Run these commands from the repository root:

~~~powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --target Isvik --parallel
~~~

Launch the desktop application:

~~~powershell
.\out\build\windows-msvc-release\Release\Isvik.exe
~~~

The CMake project is named Isvik.cpp. The desktop executable is named Isvik.exe.

### Build on Linux

Use a recent GCC toolchain with C++20 support:

~~~sh
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release --parallel
~~~

The Linux preset disables OpenVINO by default. Configure the OpenVINO integration separately when its SDK is available.

## Run a model

Use a model path supported by the selected backend. For example:

~~~powershell
.\out\build\windows-msvc-release\Release\Isvik.exe --run --model "D:\Models\openvino-model" --device CPU --prompt "Explain local inference." --max-tokens 64
~~~

Start the bilingual, persistent CLI workbench. It stays in the console as a real REPL (no desktop window), with an ASCII banner, a compact imported-model table, and a `isvik[model@device]>` prompt. Chat, switch models, and manage the catalog without leaving the prompt:

~~~powershell
.\out\build\windows-msvc-release\Release\Isvik.exe -cli
~~~

On Windows, `Isvik -cli` (or `Isvik --cli`) starts this workbench directly. At the prompt, `-ls` prints imported models with size and supported devices. Enter a model number to load or switch, or run `-use NUMBER|ID`. Each completed reply reports input/output tokens, elapsed time, and generation speed in tok/s. Use `-params` to show the active runtime and generation parameters; `-info [ID]` shows model metadata. Other commands include `-stop`, `-backend openvino|onnxruntime|tensorrt`, `-devices`, `-import PATH`, `-rm NUMBER|ID`, `-new`, `-history`, `-system TEXT`, `-key list|create|show|revoke`, and `-serve`. Use `-lang en` or `-lang zh` to change and save the CLI language. `/new`, `/history`, `/memory`, and `/context` are also accepted. Use `-exit` to quit. To start with a specific model, use `Isvik -cli --model "D:\Models\openvino-model" --device CPU`. See `Isvik.exe --help` for bilingual command-line options.

## Start the API server

Start one loaded model as a local service:

~~~powershell
.\out\build\windows-msvc-release\Release\Isvik.exe --server --model "D:\Models\openvino-model" --backend openvino
~~~

The default address is `http://127.0.0.1:1234`. Isvik.cpp supports Isvik-native endpoints, OpenAI Chat Completions, and Anthropic Messages. See the [API guide](docs/API.md) for request formats, streaming, and authentication.

## Run tests

Configure and build the test targets, then run CTest:

~~~powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --parallel
ctest --preset windows-msvc-release
~~~

For Linux, use the `linux-gcc-release` preset with the same build and test commands.

## Documentation

- [Build guide](docs/BUILDING.md)
- [Model support](docs/MODEL_SUPPORT.md)
- [API guide](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Contribution guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Documentation style guide](docs/STYLE_GUIDE.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

Isvik.cpp is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE). Third-party dependencies have separate notices and license terms.
