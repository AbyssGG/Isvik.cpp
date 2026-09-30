# Third-party notices for Isvik.cpp

## Slint

Isvik uses the Slint C++ UI framework, version 1.18.1, under the Slint Royalty-free Desktop, Mobile, and Web Applications License 2.0. The standard `AboutSlint` disclosure is available from Settings & about through the **Third-party licenses** control and is collapsed by default. The workspace has no attribution footer or default Slint badge. Slint's full license and third-party dependency notices are installed under the application's `licenses` directory.

License details: [Slint licensing](https://slint.dev/terms-and-conditions).

## nlohmann/json

Isvik uses nlohmann/json 3.12.0 under the MIT License. CMake fetches the official release archive and verifies its SHA-256 digest during configuration.

License details: [nlohmann/json license](https://github.com/nlohmann/json/blob/v3.12.0/LICENSE.MIT).

## SQLite

Isvik uses SQLite 3.53.4 from the official amalgamation archive. SQLite is in the public domain. CMake fetches the official source archive and verifies its SHA-256 digest before compiling it into the application.

Source and download information: [SQLite download page](https://www.sqlite.org/download.html).

## OpenVINO Runtime and GenAI

On Windows, Isvik uses the official OpenVINO Runtime and GenAI C++ SDK 2026.4.0.0 package. CMake downloads the x64 package from Intel's OpenVINO distribution and verifies its SHA-256 digest. The SDK license, GenAI license, and bundled third-party notices are installed with the application when GenAI is enabled.

The package's license and notice files are in its `docs/licensing` directory. The OpenVINO tokenizers and oneTBB notices are also included from their SDK directories.

License and SDK information: [OpenVINO GenAI documentation](https://docs.openvino.ai/2026/openvino-workflow/generative/inference-with-genai.html), [official GenAI releases](https://github.com/openvinotoolkit/openvino.genai/releases).

## GGML quantization reference data

Isvik's bounded GGUF CPU reference decoder uses the IQ3_S codebook adapted from `ggml-org/ggml`, along with independently implemented decoding paths informed by its GGUF quantization layouts. No GGML or llama.cpp library is linked or loaded at runtime.

Source: [ggml-common.h](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h) and [ggml-quants.c](https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c).

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## cpp-httplib

Isvik uses cpp-httplib v0.58.0, a header-only HTTP server library distributed under the MIT License. CMake fetches the versioned upstream archive and verifies its SHA-256 digest. Its license is installed with the application.

Source: https://github.com/yhirose/cpp-httplib/tree/v0.58.0.
## GoogleTest

Isvik.cpp fetches GoogleTest v1.18.0 for test builds under the BSD 3-Clause License. GoogleTest is not included in the installed application.

License: https://github.com/google/googletest/blob/v1.18.0/LICENSE.