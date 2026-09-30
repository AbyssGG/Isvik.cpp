#include "isvik/app/api_server.h"

#define NOMINMAX
#define CPPHTTPLIB_THREAD_POOL_COUNT 4
#include "httplib.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <string_view>
#include <vector>
#include <utility>

#include "isvik/core/model_catalog.h"

namespace isvik::app {
namespace {
using Json = nlohmann::json;
enum class Protocol { kIsvik, kOpenAi, kAnthropic };

struct Usage {
  uint64_t input = 0U;
  uint64_t output = 0U;
  double seconds = 0.0;
  double tokens_per_second = 0.0;
};
struct Result {
  Status status;
  std::string text;
  std::string reasoning;
  struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
  };
  std::vector<ToolCall> tool_calls;
  Usage usage;
  bool cancelled = false;
};
struct ApiTool {
  std::string name;
  std::string description;
  Json input_schema;
};
struct ParsedApiRequest {
  UnifiedInferenceRequest inference;
  std::vector<ApiTool> tools;
  bool tool_mode = false;
  bool tool_choice_required = false;
  std::string required_tool_name;
};
struct Stream {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<std::string> frames;
  bool done = false;
  CancellationSource cancellation;
  std::jthread worker;
};
std::atomic<uint64_t> request_sequence{0U};

std::string NewId(std::string prefix) {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  return prefix + std::to_string(ticks) + "-" +
      std::to_string(request_sequence.fetch_add(1U, std::memory_order_relaxed));
}
int64_t UnixSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
Protocol ProtocolFor(std::string_view path) {
  if (path == "/v1/messages") return Protocol::kAnthropic;
  if (path.starts_with("/v1/")) return Protocol::kOpenAi;
  return Protocol::kIsvik;
}
int HttpStatus(const Status& status) {
  switch (status.code()) {
    case StatusCode::kCancelled: return 499;
    case StatusCode::kInvalidArgument: return 400;
    case StatusCode::kNotFound: return 404;
    case StatusCode::kAlreadyExists: return 409;
    case StatusCode::kUnsupported: return 501;
    case StatusCode::kUnavailable: return 503;
    case StatusCode::kInternal: return 500;
    case StatusCode::kOk: return 200;
  }
  return 500;
}
std::string CodeName(StatusCode code) {
  switch (code) {
    case StatusCode::kCancelled: return "cancelled";
    case StatusCode::kInvalidArgument: return "invalid_argument";
    case StatusCode::kNotFound: return "not_found";
    case StatusCode::kAlreadyExists: return "already_exists";
    case StatusCode::kUnsupported: return "unsupported";
    case StatusCode::kUnavailable: return "unavailable";
    case StatusCode::kInternal: return "internal";
    case StatusCode::kOk: return "ok";
  }
  return "internal";
}
void WriteError(httplib::Response& response, Protocol protocol, int status,
                std::string message, std::string code = "invalid_request_error") {
  response.status = status;
  Json body;
  if (protocol == Protocol::kAnthropic) {
    body = {{"type", "error"}, {"error", {{"type", code}, {"message", message}}}};
  } else if (protocol == Protocol::kOpenAi) {
    body = {{"error", {{"message", message}, {"type", code}, {"param", nullptr}, {"code", nullptr}}}};
  } else {
    body = {{"error", {{"code", code}, {"message", message}}}};
  }
  response.set_content(body.dump(), "application/json; charset=utf-8");
}
void WriteError(httplib::Response& response, Protocol protocol, const Status& status) {
  WriteError(response, protocol, HttpStatus(status), status.message(), CodeName(status.code()));
}
void WriteJson(httplib::Response& response, const Json& body, int status = 200) {
  response.status = status;
  response.set_content(body.dump(), "application/json; charset=utf-8");
}
bool EqualSecret(std::string_view left, std::string_view right) {
  const size_t count = std::max(left.size(), right.size());
  unsigned char difference = static_cast<unsigned char>(left.size() ^ right.size());
  for (size_t i = 0U; i < count; ++i) {
    const unsigned char a = i < left.size() ? static_cast<unsigned char>(left[i]) : 0U;
    const unsigned char b = i < right.size() ? static_cast<unsigned char>(right[i]) : 0U;
    difference = static_cast<unsigned char>(difference | static_cast<unsigned char>(a ^ b));
  }
  return difference == 0U;
}
bool Authorized(const httplib::Request& request, const ApiServerConfig& config) {
  if (config.api_key.empty()) return true;
  std::string token = request.get_header_value("x-api-key");
  const std::string auth = request.get_header_value("Authorization");
  if (auth.starts_with("Bearer ")) token = auth.substr(7U);
  return EqualSecret(token, config.api_key);
}
StatusOr<std::string> ContentText(const Json& content) {
  if (content.is_string()) return content.get<std::string>();
  if (!content.is_array()) return Status::InvalidArgument("message content must be text or text blocks");
  std::string result;
  for (const Json& block : content) {
    if (!block.is_object() || !block.contains("type") || !block["type"].is_string())
      return Status::InvalidArgument("content blocks require a string type");
    const std::string type = block["type"].get<std::string>();
    if (type != "text" && type != "input_text" && type != "output_text")
      return Status::Unsupported("this server currently accepts text-only content blocks");
    if (!block.contains("text") || !block["text"].is_string())
      return Status::InvalidArgument("text blocks require a string text field");
    result += block["text"].get<std::string>();
  }
  return result;
}
Json ToolArgumentsFromHistory(const Json& value) {
  if (value.is_string()) {
    try { return Json::parse(value.get<std::string>()); }
    catch (const Json::exception&) { return value.get<std::string>(); }
  }
  return value;
}
void AppendHistoryToolCall(std::string& content, std::string id,
                           std::string name, Json arguments) {
  if (!content.empty()) content.push_back('\n');
  content += "<isvik_tool_history>";
  content += Json{{"id", std::move(id)}, {"name", std::move(name)},
                  {"arguments", std::move(arguments)}}.dump();
  content += "</isvik_tool_history>";
}
Status AppendAnthropicContent(const Json& content, const std::string& role,
                             std::string& output) {
  if (content.is_string()) {
    output = content.get<std::string>();
    return {};
  }
  if (!content.is_array()) return Status::InvalidArgument("message content must be text or content blocks");
  for (const Json& block : content) {
    if (!block.is_object() || !block.contains("type") || !block["type"].is_string())
      return Status::InvalidArgument("content blocks require a string type");
    const std::string type = block["type"].get<std::string>();
    if (type == "text") {
      if (!block.contains("text") || !block["text"].is_string())
        return Status::InvalidArgument("text blocks require a string text field");
      output += block["text"].get<std::string>();
    } else if (type == "tool_use" && role == "assistant") {
      if (!block.contains("id") || !block["id"].is_string() ||
          !block.contains("name") || !block["name"].is_string() ||
          !block.contains("input") || !block["input"].is_object())
        return Status::InvalidArgument("tool_use blocks require string id/name and object input");
      AppendHistoryToolCall(output, block["id"].get<std::string>(),
                            block["name"].get<std::string>(), block["input"]);
    } else if (type == "tool_result" && role == "user") {
      if (!block.contains("tool_use_id") || !block["tool_use_id"].is_string())
        return Status::InvalidArgument("tool_result blocks require a string tool_use_id");
      Json result_content;
      if (block.contains("content")) {
        if (block["content"].is_string()) result_content = block["content"];
        else {
          StatusOr<std::string> text = ContentText(block["content"]);
          if (!text.ok()) return text.status();
          result_content = std::move(text).value();
        }
      } else result_content = "";
      if (!output.empty()) output.push_back('\n');
      Json result{{"id", block["tool_use_id"]}, {"content", std::move(result_content)}};
      if (block.contains("is_error") && block["is_error"].is_boolean())
        result["is_error"] = block["is_error"];
      output += "<isvik_tool_result>" + result.dump() + "</isvik_tool_result>";
    } else {
      return Status::Unsupported("this server currently accepts text and tool-use content blocks");
    }
  }
  return {};
}
Status AppendOpenAiToolCalls(const Json& item, std::string& content) {
  if (item.contains("tool_calls")) {
    if (!item["tool_calls"].is_array()) return Status::InvalidArgument("tool_calls must be an array");
    for (const Json& call : item["tool_calls"]) {
      if (!call.is_object() || !call.contains("function") || !call["function"].is_object() ||
          !call["function"].contains("name") || !call["function"]["name"].is_string() ||
          !call["function"].contains("arguments"))
        return Status::InvalidArgument("tool calls require function name and arguments");
      const std::string id = call.contains("id") && call["id"].is_string()
          ? call["id"].get<std::string>() : NewId("call_");
      AppendHistoryToolCall(content, id, call["function"]["name"].get<std::string>(),
          ToolArgumentsFromHistory(call["function"]["arguments"]));
    }
  }
  if (item.contains("function_call")) {
    const Json& call = item["function_call"];
    if (!call.is_object() || !call.contains("name") || !call["name"].is_string() ||
        !call.contains("arguments"))
      return Status::InvalidArgument("function_call requires name and arguments");
    AppendHistoryToolCall(content, NewId("call_"), call["name"].get<std::string>(),
                          ToolArgumentsFromHistory(call["arguments"]));
  }
  return {};
}
Status AppendMessage(const Json& item, Protocol protocol,
                     std::vector<InferenceMessage>& messages) {
  if (!item.is_object() || !item.contains("role") || !item["role"].is_string())
    return Status::InvalidArgument("each message requires a string role");
  const std::string role = item["role"].get<std::string>();
  InferenceMessage message;
  if (role == "system" || role == "developer") message.role = MessageRole::kSystem;
  else if (role == "user") message.role = MessageRole::kUser;
  else if (role == "assistant") message.role = MessageRole::kAssistant;
  else if (role == "tool") message.role = MessageRole::kTool;
  else return Status::InvalidArgument("role must be system, developer, user, assistant, or tool");
  if (item.contains("content") && !(item["content"].is_null() && role == "assistant" &&
                                     (item.contains("tool_calls") || item.contains("function_call")))) {
    if (protocol == Protocol::kAnthropic) {
      const Status status = AppendAnthropicContent(item["content"], role, message.content);
      if (!status.ok()) return status;
    } else {
      StatusOr<std::string> text = ContentText(item["content"]);
      if (!text.ok()) return text.status();
      message.content = std::move(text).value();
    }
  } else if (message.role != MessageRole::kAssistant ||
             (!item.contains("tool_calls") && !item.contains("function_call"))) {
    return Status::InvalidArgument("message content is required");
  }
  if (protocol != Protocol::kAnthropic && role == "assistant") {
    const Status status = AppendOpenAiToolCalls(item, message.content);
    if (!status.ok()) return status;
  } else if (protocol == Protocol::kAnthropic && role == "assistant" && item.contains("tool_calls")) {
    return Status::InvalidArgument("Anthropic assistant messages must use tool_use content blocks");
  }
  if (item.contains("name") && item["name"].is_string()) message.name = item["name"].get<std::string>();
  if (item.contains("tool_call_id") && item["tool_call_id"].is_string())
    message.tool_call_id = item["tool_call_id"].get<std::string>();
  messages.push_back(std::move(message));
  return {};
}
Status AppendSystem(const Json& value, std::vector<InferenceMessage>& messages) {
  StatusOr<std::string> text = ContentText(value);
  if (!text.ok()) return text.status();
  messages.push_back({MessageRole::kSystem, std::move(text).value(), {}, {}});
  return {};
}
StatusOr<std::vector<ApiTool>> ParseTools(const Json& body, Protocol protocol) {
  std::vector<ApiTool> tools;
  if (!body.contains("tools") || body["tools"].is_null()) return tools;
  if (!body["tools"].is_array()) return Status::InvalidArgument("tools must be an array");
  if (body["tools"].size() > 64U) return Status::InvalidArgument("at most 64 tools are allowed");
  size_t serialized_bytes = 0U;
  for (const Json& item : body["tools"]) {
    if (!item.is_object()) return Status::InvalidArgument("each tool must be an object");
    ApiTool tool;
    const Json* definition = &item;
    if (protocol == Protocol::kAnthropic) {
      if (!item.contains("name") || !item["name"].is_string())
        return Status::InvalidArgument("Anthropic tools require a string name");
      tool.name = item["name"].get<std::string>();
      if (item.contains("description") && item["description"].is_string())
        tool.description = item["description"].get<std::string>();
      if (!item.contains("input_schema") || !item["input_schema"].is_object())
        return Status::InvalidArgument("Anthropic tools require an object input_schema");
      tool.input_schema = item["input_schema"];
    } else {
      if (!item.contains("type") || !item["type"].is_string() || item["type"] != "function")
        return Status::Unsupported("only function tools are supported");
      if (!item.contains("function") || !item["function"].is_object())
        return Status::InvalidArgument("function tools require a function object");
      definition = &item["function"];
      if (!definition->contains("name") || !(*definition)["name"].is_string())
        return Status::InvalidArgument("function tools require a string name");
      tool.name = (*definition)["name"].get<std::string>();
      if (definition->contains("description") && (*definition)["description"].is_string())
        tool.description = (*definition)["description"].get<std::string>();
      if (!definition->contains("parameters") || !(*definition)["parameters"].is_object())
        return Status::InvalidArgument("function tools require an object parameters schema");
      tool.input_schema = (*definition)["parameters"];
    }
    if (tool.name.empty() || tool.name.size() > 128U)
      return Status::InvalidArgument("tool names must contain 1 to 128 bytes");
    if (tool.input_schema.value("type", std::string{}) != "object")
      return Status::InvalidArgument("tool parameter schemas must have type 'object'");
    if (std::any_of(tools.begin(), tools.end(), [&tool](const ApiTool& existing) {
          return existing.name == tool.name;
        }))
      return Status::InvalidArgument("tool names must be unique");
    serialized_bytes += tool.name.size() + tool.description.size() + tool.input_schema.dump().size();
    if (serialized_bytes > 512U * 1024U)
      return Status::InvalidArgument("tool definitions may not exceed 512 KiB in total");
    tools.push_back(std::move(tool));
  }
  return tools;
}
Status ConfigureToolChoice(const Json& body, Protocol protocol,
                           ParsedApiRequest& parsed) {
  if (parsed.tools.empty()) {
    if (body.contains("tool_choice") && !body["tool_choice"].is_null() &&
        body["tool_choice"] != "none" && body["tool_choice"] != "auto")
      return Status::InvalidArgument("tool_choice requires at least one tool definition");
    return {};
  }
  if (!body.contains("tool_choice") || body["tool_choice"].is_null()) {
    parsed.tool_mode = true;
    return {};
  }
  const Json& choice = body["tool_choice"];
  if (protocol == Protocol::kAnthropic) {
    if (!choice.is_object() || !choice.contains("type") || !choice["type"].is_string())
      return Status::InvalidArgument("Anthropic tool_choice must be an object with a type");
    const std::string type = choice["type"].get<std::string>();
    if (type == "none") return {};
    if (type == "auto") { parsed.tool_mode = true; return {}; }
    if (type == "any") {
      parsed.tool_mode = true;
      parsed.tool_choice_required = true;
      return {};
    }
    if (type == "tool" && choice.contains("name") && choice["name"].is_string()) {
      parsed.required_tool_name = choice["name"].get<std::string>();
      parsed.tool_mode = true;
      parsed.tool_choice_required = true;
    } else return Status::InvalidArgument("unsupported Anthropic tool_choice type");
  } else if (choice.is_string()) {
    const std::string type = choice.get<std::string>();
    if (type == "none") return {};
    if (type == "auto") parsed.tool_mode = true;
    else if (type == "required") {
      parsed.tool_mode = true;
      parsed.tool_choice_required = true;
    } else return Status::InvalidArgument("tool_choice must be auto, none, required, or a function choice");
  } else if (choice.is_object() && choice.value("type", std::string{}) == "function" &&
             choice.contains("function") && choice["function"].is_object() &&
             choice["function"].contains("name") && choice["function"]["name"].is_string()) {
    parsed.required_tool_name = choice["function"]["name"].get<std::string>();
    parsed.tool_mode = true;
    parsed.tool_choice_required = true;
  } else return Status::InvalidArgument("unsupported tool_choice value");

  if (!parsed.required_tool_name.empty() &&
      std::none_of(parsed.tools.begin(), parsed.tools.end(), [&parsed](const ApiTool& tool) {
        return tool.name == parsed.required_tool_name;
      }))
    return Status::InvalidArgument("tool_choice names a tool that was not provided");
  return {};
}
std::string ToolPrompt(const ParsedApiRequest& parsed) {
  Json definitions = Json::array();
  for (const ApiTool& tool : parsed.tools) {
    definitions.push_back({{"name", tool.name}, {"description", tool.description},
                           {"input_schema", tool.input_schema}});
  }
  std::string prompt =
      "You are connected to a tool-calling API adapter. The available client tools are described in JSON below. "
      "Do not claim to execute a tool yourself; the client executes calls and sends the result in a later message. "
      "Previous tool calls and results in the conversation may appear inside <isvik_tool_history> and "
      "<isvik_tool_result> tags. If answering requires a tool, output exactly one call using this format, "
      "with no markdown fences: <isvik_tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}"
      "</isvik_tool_call>. Arguments must be valid JSON and match the selected tool schema. "
      "Otherwise answer normally in plain text. Tool definitions: ";
  if (parsed.tool_choice_required) {
    if (parsed.required_tool_name.empty()) prompt += "A tool call is required. ";
    else prompt += "You must call the tool named '" + parsed.required_tool_name + "'. ";
  }
  prompt += definitions.dump();
  return prompt;
}
StatusOr<int> PositiveInt(const Json& value, std::string_view field, int maximum) {
  if (!value.is_number_integer()) return Status::InvalidArgument(std::string(field) + " must be an integer");
  int64_t parsed = 0;
  try { parsed = value.get<int64_t>(); }
  catch (const Json::exception&) { return Status::InvalidArgument(std::string(field) + " is out of range"); }
  if (parsed < 1 || parsed > maximum)
    return Status::InvalidArgument(std::string(field) + " must be between 1 and " + std::to_string(maximum));
  return static_cast<int>(parsed);
}
Status ReadFloat(const Json& object, std::string_view key, std::optional<float>& output) {
  if (!object.contains(key)) return {};
  if (!object[key].is_number()) return Status::InvalidArgument(std::string(key) + " must be numeric");
  const double value = object[key].get<double>();
  if (!std::isfinite(value) || value < -100.0 || value > 100.0)
    return Status::InvalidArgument(std::string(key) + " must be finite and within [-100, 100]");
  output = static_cast<float>(value);
  return {};
}
Status ReadStops(const Json& object, std::string_view key, std::vector<std::string>& output) {
  if (!object.contains(key)) return {};
  const Json& value = object[key];
  if (value.is_string()) output.push_back(value.get<std::string>());
  else if (value.is_array()) {
    if (value.size() > 16U) return Status::InvalidArgument("at most 16 stop sequences are allowed");
    for (const Json& item : value) {
      if (!item.is_string()) return Status::InvalidArgument(std::string(key) + " must contain strings");
      output.push_back(item.get<std::string>());
    }
  } else return Status::InvalidArgument(std::string(key) + " must be a string or string array");
  for (const auto& stop : output) {
    if (stop.empty() || stop.size() > 256U)
      return Status::InvalidArgument("stop sequences must contain 1 to 256 UTF-8 bytes");
  }
  return {};
}

Status ReadGeneration(const Json& body, Protocol protocol, GenerationConfig& config) {
  const Json* settings = &body;
  if (protocol == Protocol::kIsvik && body.contains("generation")) {
    if (!body["generation"].is_object()) return Status::InvalidArgument("generation must be an object");
    settings = &body["generation"];
  }
  config.decoding_mode = DecodingMode::kSampling;
  const char* max_key = "max_tokens";
  if (protocol == Protocol::kOpenAi && settings->contains("max_completion_tokens"))
    max_key = "max_completion_tokens";
  if (settings->contains(max_key)) {
    StatusOr<int> max = PositiveInt((*settings)[max_key], max_key, 32768);
    if (!max.ok()) return max.status();
    config.max_tokens = max.value();
  } else if (protocol == Protocol::kAnthropic) {
    return Status::InvalidArgument("Anthropic requests require max_tokens");
  }

  Status status = ReadFloat(*settings, "temperature", config.sampling.temperature);
  if (!status.ok()) return status;
  status = ReadFloat(*settings, "top_p", config.sampling.top_p);
  if (!status.ok()) return status;
  status = ReadFloat(*settings, "presence_penalty", config.sampling.presence_penalty);
  if (!status.ok()) return status;
  status = ReadFloat(*settings, "frequency_penalty", config.sampling.frequency_penalty);
  if (!status.ok()) return status;
  status = ReadFloat(*settings, "repetition_penalty", config.sampling.repetition_penalty);
  if (!status.ok()) return status;
  if (settings->contains("top_k")) {
    StatusOr<int> top = PositiveInt((*settings)["top_k"], "top_k", 1000000);
    if (!top.ok()) return top.status();
    config.sampling.top_k = top.value();
  }
  if (settings->contains("seed")) {
    if (!(*settings)["seed"].is_number_integer()) return Status::InvalidArgument("seed must be a non-negative integer");
    try {
      const int64_t seed = (*settings)["seed"].get<int64_t>();
      if (seed < 0) return Status::InvalidArgument("seed must be a non-negative integer");
      config.sampling.seed = static_cast<uint64_t>(seed);
    } catch (const Json::exception&) { return Status::InvalidArgument("seed is out of range"); }
  }
  if (settings->contains("stop") && settings->contains("stop_sequences"))
    return Status::InvalidArgument("provide either stop or stop_sequences, not both");
  status = settings->contains("stop_sequences")
      ? ReadStops(*settings, "stop_sequences", config.stop_sequences)
      : ReadStops(*settings, "stop", config.stop_sequences);
  if (!status.ok()) return status;

  if (protocol == Protocol::kIsvik && settings->contains("decoding_mode")) {
    if (!(*settings)["decoding_mode"].is_string()) return Status::InvalidArgument("decoding_mode must be a string");
    const std::string mode = (*settings)["decoding_mode"].get<std::string>();
    if (mode == "greedy") config.decoding_mode = DecodingMode::kGreedy;
    else if (mode == "sampling") config.decoding_mode = DecodingMode::kSampling;
    else if (mode == "beam_search") config.decoding_mode = DecodingMode::kBeamSearch;
    else if (mode == "diverse_beam_search") config.decoding_mode = DecodingMode::kDiverseBeamSearch;
    else return Status::InvalidArgument("unsupported decoding_mode");
  } else if (config.sampling.temperature && *config.sampling.temperature == 0.0F) {
    config.sampling.temperature.reset();
    config.decoding_mode = DecodingMode::kGreedy;
  }
  if (settings->contains("num_beams")) {
    StatusOr<int> n = PositiveInt((*settings)["num_beams"], "num_beams", 128);
    if (!n.ok()) return n.status();
    config.num_beams = n.value();
  }
  if (settings->contains("num_beam_groups")) {
    StatusOr<int> n = PositiveInt((*settings)["num_beam_groups"], "num_beam_groups", 128);
    if (!n.ok()) return n.status();
    config.num_beam_groups = n.value();
  }
  if (settings->contains("diversity_penalty")) {
    if (!(*settings)["diversity_penalty"].is_number()) return Status::InvalidArgument("diversity_penalty must be numeric");
    config.diversity_penalty = (*settings)["diversity_penalty"].get<float>();
  }
  if (body.contains("stream")) {
    if (!body["stream"].is_boolean()) return Status::InvalidArgument("stream must be a boolean");
    config.stream = body["stream"].get<bool>();
  } else config.stream = false;
  return ValidateGenerationConfig(config);
}
StatusOr<ParsedApiRequest> ParseRequest(const Json& body, Protocol protocol,
                                         const ModelDescriptor& model) {
  if (!body.is_object()) return Status::InvalidArgument("request body must be a JSON object");
  std::string model_id = model.id;
  if (body.contains("model")) {
    if (!body["model"].is_string()) return Status::InvalidArgument("model must be a string");
    model_id = body["model"].get<std::string>();
  }
  if (model_id != model.id && model_id != model.display_name)
    return Status::NotFound("model '" + model_id + "' is not loaded; this server exposes '" + model.display_name + "'");
  ParsedApiRequest parsed;
  StatusOr<std::vector<ApiTool>> tools = ParseTools(body, protocol);
  if (!tools.ok()) return tools.status();
  parsed.tools = std::move(tools).value();
  const Status choice_status = ConfigureToolChoice(body, protocol, parsed);
  if (!choice_status.ok()) return choice_status;
  UnifiedInferenceRequest& request = parsed.inference;
  request.request_id = NewId(protocol == Protocol::kAnthropic ? "msg_" : "req_");
  request.model_id = model.id;
  if (protocol == Protocol::kAnthropic && body.contains("system")) {
    Status status = AppendSystem(body["system"], request.messages);
    if (!status.ok()) return status;
  }
  if (!body.contains("messages") || !body["messages"].is_array())
    return Status::InvalidArgument("messages must be an array");
  if (body["messages"].empty() || body["messages"].size() > 256U)
    return Status::InvalidArgument("messages must contain between 1 and 256 entries");
  for (const Json& item : body["messages"]) {
    const Status status = AppendMessage(item, protocol, request.messages);
    if (!status.ok()) return status;
  }
  if (parsed.tool_mode) {
    InferenceMessage tool_prompt{MessageRole::kSystem, ToolPrompt(parsed), {}, {}};
    auto position = request.messages.begin();
    while (position != request.messages.end() && position->role == MessageRole::kSystem) ++position;
    request.messages.insert(position, std::move(tool_prompt));
  }
  const Status generation = ReadGeneration(body, protocol, request.generation);
  if (!generation.ok()) return generation;
  const Status validation = ValidateInferenceRequest(request);
  if (!validation.ok()) return validation;
  return parsed;
}

Json UsageBody(const Usage& usage) {
  return {{"input_tokens", usage.input}, {"output_tokens", usage.output},
          {"total_tokens", usage.input + usage.output}};
}
Result GenerateOnce(const UnifiedInferenceRequest& request,
                    const CancellationSource& cancellation, const ApiGenerate& generate) {
  Result result;
  std::optional<Status> event_error;
  result.status = generate(request, cancellation.token(), [&result, &event_error](const InferenceEvent& event) {
    if (const auto* delta = std::get_if<ContentDelta>(&event.payload)) result.text += delta->text;
    else if (const auto* reasoning = std::get_if<ReasoningDelta>(&event.payload)) result.reasoning += reasoning->text;
    else if (const auto* usage = std::get_if<InferenceCompleted>(&event.payload)) {
      result.usage = {usage->input_tokens, usage->output_tokens,
                      usage->duration_seconds, usage->tokens_per_second};
    } else if (const auto* error = std::get_if<InferenceError>(&event.payload)) event_error = error->status;
    else if (std::holds_alternative<InferenceCancelled>(event.payload)) result.cancelled = true;
  });
  if (result.status.ok() && event_error) result.status = *event_error;
  return result;
}
const ApiTool* FindTool(const ParsedApiRequest& request, std::string_view name) {
  const auto found = std::find_if(request.tools.begin(), request.tools.end(), [name](const ApiTool& tool) {
    return tool.name == name;
  });
  return found == request.tools.end() ? nullptr : &*found;
}
std::string TrimAscii(std::string value) {
  const auto whitespace = [](unsigned char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'; };
  while (!value.empty() && whitespace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
  while (!value.empty() && whitespace(static_cast<unsigned char>(value.back()))) value.pop_back();
  return value;
}
bool ParseToolCallObject(const Json& value, const ParsedApiRequest& request,
                         Result::ToolCall& output) {
  if (!value.is_object()) return false;
  const Json* function = &value;
  if (value.contains("function") && value["function"].is_object()) function = &value["function"];
  if (!function->contains("name") || !(*function)["name"].is_string()) return false;
  output.name = (*function)["name"].get<std::string>();
  if (!FindTool(request, output.name)) return false;
  if (request.tool_choice_required && !request.required_tool_name.empty() &&
      output.name != request.required_tool_name) return false;
  const char* argument_key = function->contains("arguments") ? "arguments"
      : function->contains("args") ? "args"
      : function->contains("input") ? "input" : nullptr;
  if (argument_key == nullptr) return false;
  const Json& arguments = (*function)[argument_key];
  Json parsed_arguments;
  if (arguments.is_string()) {
    try { parsed_arguments = Json::parse(arguments.get<std::string>()); }
    catch (const Json::exception&) { return false; }
  } else parsed_arguments = arguments;
  if (!parsed_arguments.is_object()) return false;
  output.arguments = parsed_arguments.dump();
  output.id = value.contains("id") && value["id"].is_string()
      ? value["id"].get<std::string>() : NewId("call_");
  return true;
}
void ParseModelToolCalls(Result& result, const ParsedApiRequest& request) {
  if (!request.tool_mode) return;
  constexpr std::string_view start = "<isvik_tool_call>";
  constexpr std::string_view end = "</isvik_tool_call>";
  const std::string original = result.text;
  size_t cursor = 0U;
  std::string visible;
  while (true) {
    const size_t begin = result.text.find(start, cursor);
    if (begin == std::string::npos) break;
    const size_t close = result.text.find(end, begin + start.size());
    if (close == std::string::npos) { result.tool_calls.clear(); return; }
    const std::string prefix = result.text.substr(cursor, begin - cursor);
    visible += prefix;
    Json payload;
    try { payload = Json::parse(result.text.substr(begin + start.size(), close - begin - start.size())); }
    catch (const Json::exception&) { result.tool_calls.clear(); return; }
    if (payload.is_object() && payload.contains("tool_calls") && payload["tool_calls"].is_array()) {
      for (const Json& item : payload["tool_calls"]) {
        Result::ToolCall call;
        if (!ParseToolCallObject(item, request, call) || result.tool_calls.size() >= 16U) {
          result.tool_calls.clear();
          return;
        }
        result.tool_calls.push_back(std::move(call));
      }
    } else {
      Result::ToolCall call;
      if (!ParseToolCallObject(payload, request, call) || result.tool_calls.size() >= 16U) {
        result.tool_calls.clear();
        return;
      }
      result.tool_calls.push_back(std::move(call));
    }
    cursor = close + end.size();
  }
  if (!result.tool_calls.empty()) {
    visible += result.text.substr(cursor);
    result.text = TrimAscii(std::move(visible));
    return;
  }
  // Some instruction-tuned models follow the JSON schema but omit the adapter tags.
  try {
    const Json payload = Json::parse(TrimAscii(original));
    if (payload.is_object() && payload.contains("tool_calls") && payload["tool_calls"].is_array()) {
      for (const Json& item : payload["tool_calls"]) {
        Result::ToolCall call;
        if (!ParseToolCallObject(item, request, call) || result.tool_calls.size() >= 16U) {
          result.tool_calls.clear();
          return;
        }
        result.tool_calls.push_back(std::move(call));
      }
    } else if (payload.is_object() && (payload.contains("arguments") ||
                                       payload.contains("args") || payload.contains("input"))) {
      Result::ToolCall call;
      if (ParseToolCallObject(payload, request, call)) result.tool_calls.push_back(std::move(call));
    }
    if (!result.tool_calls.empty()) result.text.clear();
  } catch (const Json::exception&) {
    // Keep ordinary model prose unchanged when it is not a complete tool-call object.
  }
}
Json OpenAiMessage(const Result& result) {
  Json message{{"role", "assistant"}, {"content", result.text}};
  if (!result.reasoning.empty()) message["reasoning_content"] = result.reasoning;
  if (!result.tool_calls.empty()) {
    message["content"] = nullptr;
    message["tool_calls"] = Json::array();
    for (const Result::ToolCall& call : result.tool_calls) {
      message["tool_calls"].push_back({{"id", call.id}, {"type", "function"},
          {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
    }
  }
  return message;
}
Json OpenAiResponse(const Result& result, std::string_view id, std::string_view model) {
  return {{"id", id}, {"object", "chat.completion"}, {"created", UnixSeconds()}, {"model", model},
      {"choices", Json::array({{{"index", 0}, {"message", OpenAiMessage(result)},
                                  {"finish_reason", result.tool_calls.empty() ? "stop" : "tool_calls"}}})},
      {"usage", {{"prompt_tokens", result.usage.input}, {"completion_tokens", result.usage.output},
                  {"total_tokens", result.usage.input + result.usage.output}}}};
}
Json AnthropicResponse(const Result& result, std::string_view id, std::string_view model, int max_tokens) {
  const bool limit = result.usage.output >= static_cast<uint64_t>(max_tokens);
  Json content = Json::array();
  if (!result.text.empty() || result.tool_calls.empty())
    content.push_back({{"type", "text"}, {"text", result.text}});
  for (const Result::ToolCall& call : result.tool_calls) {
    Json arguments;
    try { arguments = Json::parse(call.arguments); }
    catch (const Json::exception&) { arguments = Json::object(); }
    content.push_back({{"type", "tool_use"}, {"id", call.id}, {"name", call.name},
                       {"input", std::move(arguments)}});
  }
  return {{"id", id}, {"type", "message"}, {"role", "assistant"}, {"model", model},
      {"content", std::move(content)},
      {"stop_reason", !result.tool_calls.empty() ? "tool_use" : limit ? "max_tokens" : "end_turn"},
      {"stop_sequence", nullptr},
      {"usage", {{"input_tokens", result.usage.input}, {"output_tokens", result.usage.output}}}};
}
std::string Sse(std::string_view event, const Json& data) {
  std::string frame;
  if (!event.empty()) frame = "event: " + std::string(event) + "\n";
  return frame + "data: " + data.dump() + "\n\n";
}
void Queue(const std::shared_ptr<Stream>& stream, std::string frame) {
  {
    std::lock_guard lock(stream->mutex);
    if (stream->done) return;
    stream->frames.push_back(std::move(frame));
  }
  stream->changed.notify_one();
}
Json OpenAiChunk(std::string_view id, std::string_view model, Json delta,
                 Json finish = nullptr) {
  return {{"id", id}, {"object", "chat.completion.chunk"}, {"created", UnixSeconds()},
      {"model", model}, {"choices", Json::array({{{"index", 0}, {"delta", std::move(delta)},
                                                      {"finish_reason", std::move(finish)}}})}};
}
void QueueStart(const std::shared_ptr<Stream>& stream, Protocol protocol,
                std::string_view id, std::string_view model,
                const UnifiedInferenceRequest& request, bool defer_anthropic_content) {
  if (protocol == Protocol::kOpenAi) {
    Queue(stream, Sse("", OpenAiChunk(id, model, {{"role", "assistant"}})));
  } else if (protocol == Protocol::kAnthropic) {
    Queue(stream, Sse("message_start", {{"type", "message_start"},
        {"message", {{"id", id}, {"type", "message"}, {"role", "assistant"}, {"model", model},
                      {"content", Json::array()}, {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                      {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}}}}));
    if (!defer_anthropic_content)
      Queue(stream, Sse("content_block_start", {{"type", "content_block_start"}, {"index", 0},
          {"content_block", {{"type", "text"}, {"text", ""}}}}));
  } else {
    Queue(stream, Json{{"type", "inference.started"}, {"request_id", request.request_id},
                       {"model_id", request.model_id}}.dump() + "\n");
  }
}
void QueueFailure(const std::shared_ptr<Stream>& stream, Protocol protocol,
                  const Status& status, std::string_view request_id) {
  if (protocol == Protocol::kAnthropic) {
    Queue(stream, Sse("error", {{"type", "error"}, {"error", {{"type", CodeName(status.code()),
                                                                       {"message", status.message()}}}}}));
  } else if (protocol == Protocol::kOpenAi) {
    Queue(stream, Sse("", {{"error", {{"message", status.message()}, {"type", CodeName(status.code())},
                            {"param", nullptr}, {"code", nullptr}}}}));
    Queue(stream, "data: [DONE]\n\n");
  } else {
    Queue(stream, Json{{"type", "error"}, {"request_id", request_id},
        {"error", {{"code", CodeName(status.code())}, {"message", status.message()}}}}.dump() + "\n");
  }
}
void QueueFinish(const std::shared_ptr<Stream>& stream, Protocol protocol,
                 std::string_view id, std::string_view model,
                 std::string_view request_id, const Result& result,
                 int max_tokens, bool include_usage, bool deferred_content) {
  if (protocol == Protocol::kOpenAi) {
    if (deferred_content) {
      if (result.tool_calls.empty()) {
        if (!result.text.empty())
          Queue(stream, Sse("", OpenAiChunk(id, model, {{"content", result.text}})));
      } else {
        for (size_t i = 0U; i < result.tool_calls.size(); ++i) {
          const Result::ToolCall& call = result.tool_calls[i];
          Queue(stream, Sse("", OpenAiChunk(id, model, {{"tool_calls", Json::array({{{"index", i},
              {"id", call.id}, {"type", "function"},
              {"function", {{"name", call.name}, {"arguments", call.arguments}}}}})}})));
        }
      }
    }
    Queue(stream, Sse("", OpenAiChunk(id, model, Json::object(),
        result.tool_calls.empty() ? "stop" : "tool_calls")));
    if (include_usage) {
      Queue(stream, Sse("", {{"id", id}, {"object", "chat.completion.chunk"},
          {"created", UnixSeconds()}, {"model", model}, {"choices", Json::array()},
          {"usage", {{"prompt_tokens", result.usage.input}, {"completion_tokens", result.usage.output},
                      {"total_tokens", result.usage.input + result.usage.output}}}}));
    }
    Queue(stream, "data: [DONE]\n\n");
  } else if (protocol == Protocol::kAnthropic) {
    std::string stop_reason = "end_turn";
    if (deferred_content) {
      size_t index = 0U;
      if (!result.text.empty() || result.tool_calls.empty()) {
        Queue(stream, Sse("content_block_start", {{"type", "content_block_start"}, {"index", index},
            {"content_block", {{"type", "text"}, {"text", ""}}}}));
        if (!result.text.empty())
          Queue(stream, Sse("content_block_delta", {{"type", "content_block_delta"}, {"index", index},
              {"delta", {{"type", "text_delta"}, {"text", result.text}}}}));
        Queue(stream, Sse("content_block_stop", {{"type", "content_block_stop"}, {"index", index}}));
        ++index;
      }
      for (const Result::ToolCall& call : result.tool_calls) {
        Json arguments;
        try { arguments = Json::parse(call.arguments); }
        catch (const Json::exception&) { arguments = Json::object(); }
        Queue(stream, Sse("content_block_start", {{"type", "content_block_start"}, {"index", index},
            {"content_block", {{"type", "tool_use"}, {"id", call.id},
                                {"name", call.name}, {"input", Json::object()}}}}));
        Queue(stream, Sse("content_block_delta", {{"type", "content_block_delta"}, {"index", index},
            {"delta", {{"type", "input_json_delta"}, {"partial_json", arguments.dump()}}}}));
        Queue(stream, Sse("content_block_stop", {{"type", "content_block_stop"}, {"index", index}}));
        ++index;
      }
    } else {
      Queue(stream, Sse("content_block_stop", {{"type", "content_block_stop"}, {"index", 0}}));
    }
    const bool limit = result.usage.output >= static_cast<uint64_t>(max_tokens);
    if (!result.tool_calls.empty()) stop_reason = "tool_use";
    else if (limit) stop_reason = "max_tokens";
    Queue(stream, Sse("message_delta", {{"type", "message_delta"},
        {"delta", {{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
        {"usage", {{"output_tokens", result.usage.output}}}}));
    Queue(stream, Sse("message_stop", {{"type", "message_stop"}}));
  } else {
    if (deferred_content) {
      if (result.tool_calls.empty()) {
        if (!result.text.empty())
          Queue(stream, Json{{"type", "content.delta"}, {"request_id", request_id},
                             {"text", result.text}}.dump() + "\n");
      } else {
        for (const Result::ToolCall& call : result.tool_calls) {
          Json arguments;
          try { arguments = Json::parse(call.arguments); }
          catch (const Json::exception&) { arguments = Json::object(); }
          Queue(stream, Json{{"type", "tool_call"}, {"request_id", request_id},
              {"id", call.id}, {"name", call.name}, {"arguments", std::move(arguments)}}.dump() + "\n");
        }
      }
    }
    Queue(stream, Json{{"type", "inference.completed"}, {"request_id", request_id},
        {"usage", UsageBody(result.usage)}, {"duration_seconds", result.usage.seconds},
        {"tokens_per_second", result.usage.tokens_per_second}}.dump() + "\n");
  }
}

void StreamResponse(httplib::Response& response, Protocol protocol,
                    const ParsedApiRequest& api_request, const ModelDescriptor& model,
                    const ApiGenerate& generate, bool include_usage) {
  const UnifiedInferenceRequest& request = api_request.inference;
  const bool defer_content = api_request.tool_mode;
  auto stream = std::make_shared<Stream>();
  const std::string id = protocol == Protocol::kAnthropic ? request.request_id
      : NewId(protocol == Protocol::kOpenAi ? "chatcmpl-" : "isvik-");
  const int max_tokens = request.generation.max_tokens.value_or(128);
  QueueStart(stream, protocol, id, model.display_name, request, defer_content);
  stream->worker = std::jthread([stream, protocol, id, api_request, model_name = model.display_name,
                                  generate, include_usage, max_tokens, defer_content] {
    const UnifiedInferenceRequest& request = api_request.inference;
    Result result;
    std::optional<Status> event_error;
    try {
      result.status = generate(request, stream->cancellation.token(),
          [stream, protocol, id, model_name, request_id = request.request_id,
           defer_content, &result, &event_error](const InferenceEvent& event) {
            if (const auto* delta = std::get_if<ContentDelta>(&event.payload)) {
              result.text += delta->text;
              if (defer_content) return;
              if (protocol == Protocol::kOpenAi)
                Queue(stream, Sse("", OpenAiChunk(id, model_name, {{"content", delta->text}})));
              else if (protocol == Protocol::kAnthropic)
                Queue(stream, Sse("content_block_delta", {{"type", "content_block_delta"},
                    {"index", 0}, {"delta", {{"type", "text_delta"}, {"text", delta->text}}}}));
              else Queue(stream, Json{{"type", "content.delta"}, {"request_id", request_id},
                                      {"text", delta->text}}.dump() + "\n");
            } else if (const auto* thought = std::get_if<ReasoningDelta>(&event.payload)) {
              result.reasoning += thought->text;
              if (protocol == Protocol::kOpenAi)
                Queue(stream, Sse("", OpenAiChunk(id, model_name, {{"reasoning_content", thought->text}})));
              else if (protocol == Protocol::kIsvik)
                Queue(stream, Json{{"type", "reasoning.delta"}, {"request_id", request_id},
                                   {"text", thought->text}}.dump() + "\n");
            } else if (const auto* metrics = std::get_if<InferenceCompleted>(&event.payload)) {
              result.usage = {metrics->input_tokens, metrics->output_tokens,
                              metrics->duration_seconds, metrics->tokens_per_second};
            } else if (const auto* error = std::get_if<InferenceError>(&event.payload)) {
              event_error = error->status;
            } else if (std::holds_alternative<InferenceCancelled>(event.payload)) {
              result.cancelled = true;
            }
          });
    } catch (const std::exception& error) {
      result.status = Status::Internal(std::string("generation failed: ") + error.what());
    }
    if (result.status.ok() && event_error) result.status = *event_error;
    if (result.status.ok() && result.cancelled) result.status = Status::Cancelled("generation cancelled");
    if (result.status.ok()) ParseModelToolCalls(result, api_request);
    if (!result.status.ok()) QueueFailure(stream, protocol, result.status, request.request_id);
    else QueueFinish(stream, protocol, id, model_name, request.request_id,
                     result, max_tokens, include_usage, defer_content);
    {
      std::lock_guard lock(stream->mutex);
      stream->done = true;
    }
    stream->changed.notify_all();
  });

  response.status = 200;
  response.set_header("Cache-Control", "no-cache, no-transform");
  response.set_header("X-Accel-Buffering", "no");
  const std::string mime = protocol == Protocol::kIsvik
      ? "application/x-ndjson" : "text/event-stream";
  response.set_chunked_content_provider(mime,
      [stream, protocol](size_t, httplib::DataSink& sink) {
        std::string frame;
        {
          std::unique_lock lock(stream->mutex);
          if (!stream->changed.wait_for(lock, std::chrono::seconds(15), [&stream] {
                return stream->done || !stream->frames.empty();
              })) {
            frame = protocol == Protocol::kIsvik ? "{\"type\":\"ping\"}\n" : ": keep-alive\n\n";
          } else if (!stream->frames.empty()) {
            frame = std::move(stream->frames.front());
            stream->frames.pop_front();
          } else if (stream->done) {
            sink.done();
            return true;
          }
        }
        if (!sink.write(frame.data(), frame.size())) {
          stream->cancellation.Cancel();
          return false;
        }
        return true;
      },
      [stream](bool success) {
        if (!success) stream->cancellation.Cancel();
        stream->changed.notify_all();
        if (stream->worker.joinable()) stream->worker.join();
      });
}
void Handle(const httplib::Request& request, httplib::Response& response,
            Protocol protocol, const ModelDescriptor& model,
            const ApiServerConfig& config, const ApiGenerate& generate) {
  try {
    const Json body = Json::parse(request.body);
    StatusOr<ParsedApiRequest> parsed = ParseRequest(body, protocol, model);
    if (!parsed.ok()) {
      WriteError(response, protocol, parsed.status());
      return;
    }
    ParsedApiRequest api_request = std::move(parsed).value();
    UnifiedInferenceRequest& inference = api_request.inference;
    bool include_usage = false;
    if (protocol == Protocol::kOpenAi && body.contains("stream_options") &&
        body["stream_options"].is_object() && body["stream_options"].contains("include_usage")) {
      include_usage = body["stream_options"]["include_usage"].is_boolean() &&
          body["stream_options"]["include_usage"].get<bool>();
    }
    if (inference.generation.stream) {
      StreamResponse(response, protocol, api_request, model, generate, include_usage);
      return;
    }
    CancellationSource cancellation;
    Result result = GenerateOnce(inference, cancellation, generate);
    if (result.status.ok()) ParseModelToolCalls(result, api_request);
    if (!result.status.ok()) {
      WriteError(response, protocol, result.status);
    } else if (protocol == Protocol::kOpenAi) {
      WriteJson(response, OpenAiResponse(result, NewId("chatcmpl-"), model.display_name));
    } else if (protocol == Protocol::kAnthropic) {
      WriteJson(response, AnthropicResponse(result, inference.request_id, model.display_name,
          inference.generation.max_tokens.value_or(128)));
    } else {
      Json body{{"id", inference.request_id}, {"object", "isvik.inference"},
          {"model", model.display_name}, {"backend", config.backend}, {"content", result.text},
          {"reasoning", result.reasoning}, {"usage", UsageBody(result.usage)},
          {"duration_seconds", result.usage.seconds}, {"tokens_per_second", result.usage.tokens_per_second}};
      if (!result.tool_calls.empty()) {
        body["finish_reason"] = "tool_calls";
        body["tool_calls"] = Json::array();
        for (const Result::ToolCall& call : result.tool_calls) {
          Json arguments;
          try { arguments = Json::parse(call.arguments); }
          catch (const Json::exception&) { arguments = Json::object(); }
          body["tool_calls"].push_back({{"id", call.id}, {"name", call.name},
                                        {"arguments", std::move(arguments)}});
        }
      } else body["finish_reason"] = "stop";
      WriteJson(response, body);
    }
  } catch (const Json::exception& error) {
    WriteError(response, protocol, 400, std::string("invalid JSON request: ") + error.what());
  } catch (const std::exception& error) {
    WriteError(response, protocol, 500, std::string("request failed: ") + error.what(), "internal_error");
  }
}

std::string ModelFormatName(ModelFormat format) {
  if (format == ModelFormat::kOnnx) return "onnx";
  if (format == ModelFormat::kGguf) return "gguf";
  return "openvino_ir";
}
Json NativeModel(const ModelDescriptor& model, const ApiServerConfig& config) {
  return {{"id", model.id}, {"name", model.display_name}, {"format", ModelFormatName(model.format)},
      {"architecture", model.architecture}, {"parameters", model.parameter_label},
      {"quantization", model.quantization}, {"context_length", model.context_length},
      {"backend", config.backend}};
}
Json OpenAiModels(const ModelDescriptor& model) {
  return {{"object", "list"}, {"data", Json::array({{{"id", model.id}, {"object", "model"},
                                                          {"created", 0}, {"owned_by", "isvik"}}})}};
}
bool IsLoopback(std::string_view host) {
  return host == "localhost" || host == "::1" || host == "[::1]" ||
      host == "127.0.0.1" || host.starts_with("127.");
}

}  // namespace

struct ApiServer::Impl {
  ApiServerConfig config;
  ModelDescriptor model;
  ApiGenerate generate;
  std::unique_ptr<httplib::Server> server;
  std::thread listener;
};

ApiServer::ApiServer() = default;
ApiServer::~ApiServer() { Stop(); }

Status ApiServer::Start(const ApiServerConfig& config,
                        const ModelDescriptor& model, ApiGenerate generate) {
  if (IsRunning()) return Status::AlreadyExists("API server is already running");
  Stop();
  if (config.port < 1 || config.port > 65535) {
    return Status::InvalidArgument("API port must be between 1 and 65535");
  }
  if (!IsLoopback(config.host) && config.api_key.empty()) {
    return Status::InvalidArgument(
        "Binding outside loopback requires an API key. HTTP is unencrypted; use a trusted network or a TLS reverse proxy.");
  }

  impl_ = std::make_unique<Impl>();
  impl_->config = config;
  impl_->model = model;
  impl_->generate = std::move(generate);
  impl_->server = std::make_unique<httplib::Server>();
  auto& server = *impl_->server;
  server.new_task_queue = [] { return new httplib::ThreadPool(4U, 16U, 128U); };
  server.set_read_timeout(300);
  server.set_write_timeout(300);
  server.set_keep_alive_timeout(10);
  server.set_payload_max_length(4U * 1024U * 1024U);

  const auto* const context = impl_.get();
  server.set_pre_routing_handler([context](const httplib::Request& request, httplib::Response& response) {
    if (request.method == "OPTIONS") {
      response.status = 204;
      return httplib::Server::HandlerResponse::Handled;
    }
    if (!Authorized(request, context->config)) {
      WriteError(response, ProtocolFor(request.path), 401, "missing or invalid API key", "authentication_error");
      return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });
  server.set_error_handler([](const httplib::Request& request, httplib::Response& response) {
    if (response.status == 404)
      WriteError(response, ProtocolFor(request.path), 404, "endpoint not found", "not_found");
  });
  server.set_exception_handler([](const httplib::Request& request,
                                  httplib::Response& response, std::exception_ptr) {
    WriteError(response, ProtocolFor(request.path), 500, "internal server error", "internal_error");
  });
  server.set_logger([](const httplib::Request& request, const httplib::Response& response) {
    std::cout << "[API] " << request.method << ' ' << request.path << " -> "
              << response.status << std::endl;
  });

  server.Get("/", [context](const httplib::Request&, httplib::Response& response) {
    WriteJson(response, {{"name", "Isvik API"}, {"version", "v1"}, {"model", context->model.display_name},
        {"backend", context->config.backend}, {"endpoints", Json::array({"/api/v1/health", "/api/v1/models",
         "/api/v1/inference", "/v1/models", "/v1/chat/completions", "/v1/messages"})}});
  });
  server.Get("/api/v1/health", [context](const httplib::Request&, httplib::Response& response) {
    WriteJson(response, {{"status", "ok"}, {"model", context->model.display_name}, {"backend", context->config.backend}});
  });
  server.Get("/api/v1/models", [context](const httplib::Request&, httplib::Response& response) {
    WriteJson(response, {{"object", "list"}, {"data", Json::array({NativeModel(context->model, context->config)})}});
  });
  server.Get("/v1/models", [context](const httplib::Request&, httplib::Response& response) {
    WriteJson(response, OpenAiModels(context->model));
  });
  const auto native = [context](const httplib::Request& request, httplib::Response& response) {
    Handle(request, response, Protocol::kIsvik, context->model, context->config, context->generate);
  };
  server.Post("/api/v1/inference", native);
  server.Post("/api/v1/chat/completions", native);
  server.Post("/v1/chat/completions", [context](const httplib::Request& request,
                                                  httplib::Response& response) {
    Handle(request, response, Protocol::kOpenAi, context->model, context->config, context->generate);
  });
  server.Post("/v1/messages", [context](const httplib::Request& request,
                                          httplib::Response& response) {
    Handle(request, response, Protocol::kAnthropic, context->model, context->config, context->generate);
  });

  if (!server.bind_to_port(config.host, config.port)) {
    impl_.reset();
    return Status::Unavailable("Could not bind API server at " + config.host + ":" + std::to_string(config.port));
  }

  std::cout << "Isvik API server at http://" << config.host << ':' << config.port
            << " · model=" << model.display_name << " · backend=" << config.backend << std::endl;
  if (!config.api_key.empty()) std::cout << "API key authentication enabled." << std::endl;
  impl_->listener = std::thread([context] {
    if (!context->server->listen_after_bind())
      std::cerr << "Isvik API listener stopped unexpectedly.\n";
  });

  const auto startup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!server.is_running() && std::chrono::steady_clock::now() < startup_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!server.is_running()) {
    Stop();
    return Status::Unavailable("API server could not start listening on the selected address");
  }
  return Status();
}

void ApiServer::Stop() {
  if (!impl_) return;
  if (impl_->server) impl_->server->stop();
  if (impl_->listener.joinable()) impl_->listener.join();
  impl_.reset();
}

void ApiServer::Wait() {
  if (impl_ && impl_->listener.joinable()) impl_->listener.join();
}

bool ApiServer::IsRunning() const {
  return impl_ && impl_->server && impl_->server->is_running();
}

int RunApiServer(const ApiServerConfig& config, const ModelDescriptor& model,
                 const ApiGenerate& generate) {
  ApiServer server;
  const Status status = server.Start(config, model, generate);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return status.code() == StatusCode::kInvalidArgument ? 2 : 1;
  }
  server.Wait();
  return 0;
}

}  // namespace isvik::app
