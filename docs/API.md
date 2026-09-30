# API guide

Isvik.cpp can serve one loaded model over HTTP. The server provides native Isvik endpoints, OpenAI-compatible Chat Completions, and an Anthropic-compatible Messages endpoint.

## Start the server

~~~powershell
Isvik.exe --server --model "D:\Models\openvino-model" --backend openvino
~~~

The default address is `http://127.0.0.1:1234`. Use `--backend onnxruntime` for a compatible ONNX Runtime GenAI package. Use `--backend tensorrt` for a supported native GGUF model or a compatible ONNX model with TensorRT-RTX enabled.

To allow connections from another machine, bind to an external address and set an API key:

~~~powershell
Isvik.exe --server --model "D:\Models\openvino-model" --backend openvino --host 0.0.0.0 --port 1234 --api-key "replace-with-a-secret"
~~~

The server refuses non-loopback binding without an API key. The server uses plain HTTP. Use a trusted network or terminate TLS with a reverse proxy.

## Native Isvik API

### Check server status

~~~sh
curl http://127.0.0.1:1234/api/v1/health
~~~

### List the loaded model

~~~sh
curl http://127.0.0.1:1234/api/v1/models
~~~

### Generate a response

Send a JSON object with a `messages` array. The optional `generation` object accepts settings such as `max_tokens`, `temperature`, `top_p`, and `stop`.

~~~sh
curl http://127.0.0.1:1234/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"generation":{"max_tokens":64}}'
~~~

Set `generation.stream` to `true` to receive newline-delimited JSON events.

## OpenAI-compatible API

The server supports `GET /v1/models` and `POST /v1/chat/completions`. Configure OpenAI clients to use `http://127.0.0.1:1234/v1` as the base URL. The model ID `cli-model` and the model display name are accepted.

~~~python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="local",
)
response = client.chat.completions.create(
    model="cli-model",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
~~~

Set `stream=True` to receive server-sent events. Function tools are supported through an API-layer compatibility adapter: Isvik adds the tool schemas to the model prompt, then converts a correctly formatted model tool call into the OpenAI response shape. Tool-enabled streams are buffered until the model finishes so the server can distinguish a tool call from normal text. The client remains responsible for executing the call and sending its result in the next request; Isvik does not execute client tools on the host.

```python
response = client.chat.completions.create(
    model="cli-model",
    messages=[{"role": "user", "content": "What is 12 times 13?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }],
)
```

## Anthropic-compatible API

The server supports `POST /v1/messages`. Configure Anthropic clients to use `http://127.0.0.1:1234` as the base URL. Anthropic requests must include `max_tokens`.

~~~python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:1234",
    api_key="local",
)
response = client.messages.create(
    model="cli-model",
    max_tokens=64,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)
~~~

Set `stream=True` to receive server-sent events. Anthropic `tools`, `tool_choice`, `tool_use`, and `tool_result` blocks are translated by the same API-layer adapter. As with the OpenAI endpoint, tool execution stays with the client.

## Authentication

When `--api-key` is set, send the key in an `Authorization: Bearer` header or an `x-api-key` header. Local loopback connections do not require a key unless you configure one.

## Request behavior

- The server keeps one model loaded per process.
- Clients send the conversation history with each request. The API does not persist chat history or add saved memories automatically.
- Generation endpoints support streaming.
- The server accepts text input and tool definitions. Tool selection is prompt-guided because the inference engines do not expose native tool calling; models must follow the adapter's output format for the response to become a structured tool call. The server does not execute client-provided tools. Image input is not implemented.
- The request body limit is 4 MiB.
