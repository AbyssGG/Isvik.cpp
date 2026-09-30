# Architecture

Isvik.cpp separates the desktop application, inference engines, and shared model services.

## Application layer

The `Isvik` executable provides the Slint desktop interface, the command-line interface, and the optional HTTP API server. The API server and CLI use the same inference engine interfaces as the desktop chat.

## Core layer

The core library contains backend-neutral request and event types, model catalog and registry services, context budgeting, memory storage, GGUF inspection, and the tool runtime.

The application stores its model catalog and memory database in the user's local application data directory. Model weights remain in their original locations.

## Backend layer

- OpenVINO GenAI loads supported OpenVINO IR and GGUF models.
- ONNX Runtime GenAI loads compatible ONNX text-generation packages.
- TensorRT supports named-tensor execution and the optional native Gemma 4 GGUF generation path.

The [TensorRT GGUF source guide](TENSORRT_GGUF.md) traces the native GGUF reader, plugin, CUDA kernel, and decoding loop.

CMake controls optional backend integrations. A build can omit runtimes that are unavailable on the host.

## Request flow

The UI, CLI, and API translate user input into a `UnifiedInferenceRequest`. A backend emits `InferenceEvent` values for generated text, reasoning, metrics, cancellation, or errors. The application layer renders or serializes those events for the caller.

The shared request types let each frontend use the same model runtime without coupling API protocol details to backend implementations.
