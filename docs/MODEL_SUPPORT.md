# Model support

Model compatibility depends on the format, architecture, tokenizer assets, quantization, runtime, and device.

## OpenVINO GenAI

OpenVINO GenAI loads OpenVINO IR models and selected GGUF models. GGUF support is limited to the architectures and tensor encodings implemented by the installed OpenVINO runtime.

OpenVINO IR model directories must include the XML graph, matching weights, and tokenizer assets required by the model. Visual-language model layouts use the OpenVINO GenAI visual-language pipeline. The current chat interface accepts text input.

## ONNX Runtime GenAI

ONNX Runtime GenAI loads compatible ONNX model packages that include the required GenAI configuration and tokenizer files. An arbitrary ONNX graph is not necessarily a text-generation model.

The TensorRT-RTX execution provider can run supported graph portions for compatible ONNX models. Unsupported operations may use the configured CPU fallback.

## TensorRT

The TensorRT backend provides named-tensor execution for supported ONNX networks. TensorRT engine files are specific to the TensorRT version, GPU architecture, and build options used to create them.

The optional native TensorRT GGUF runtime reads supported Gemma 4 GGUF weights without rewriting the model. Runtime support depends on the model architecture and tensor encodings. Native GGUF inference can require substantial GPU memory and can be slower than optimized engines.

## Check a model

Use the CLI to inspect a model before loading it:

~~~powershell
Isvik.exe --inspect "D:\Models\model"
~~~

The model library reports whether a model is recognized and whether the selected backend can load it. Recognition does not guarantee that the model graph, tokenizer, or quantization is supported.
