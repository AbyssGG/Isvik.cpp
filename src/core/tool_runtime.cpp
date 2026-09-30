#include "isvik/core/tool_runtime.h"

#include <algorithm>
#include <array>
#include <exception>
#include <stdexcept>
#include <utility>

namespace isvik {
namespace {

bool IsValidToolName(const std::string& name) {
  if (name.empty() || name.size() > 128U) return false;
  const auto is_letter = [](unsigned char value) {
    return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z');
  };
  const auto is_alphanumeric = [&](unsigned char value) {
    return is_letter(value) || (value >= '0' && value <= '9');
  };
  if (!is_letter(static_cast<unsigned char>(name.front()))) return false;
  for (const char character : name) {
    const unsigned char value = static_cast<unsigned char>(character);
    if (!is_alphanumeric(value) && value != '.' && value != '_' && value != '-') {
      return false;
    }
  }
  return true;
}

bool IsValidCallId(const std::string& call_id) {
  if (call_id.empty() || call_id.size() > 128U) return false;
  for (const char character : call_id) {
    const unsigned char value = static_cast<unsigned char>(character);
    const bool alphanumeric = (value >= 'a' && value <= 'z') ||
        (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9');
    if (!alphanumeric && value != '-' && value != '_' && value != '.' && value != ':') {
      return false;
    }
  }
  return true;
}

bool IsSupportedSchemaType(const std::string& type) {
  constexpr std::array<const char*, 7> kTypes = {
      "object", "array", "string", "integer", "number", "boolean", "null"};
  return std::any_of(kTypes.begin(), kTypes.end(), [&](const char* candidate) {
    return type == candidate;
  });
}

Status ValidateSchemaDefinition(const nlohmann::json& schema, uint32_t depth = 0) {
  if (depth > 8 || !schema.is_object()) {
    return Status::InvalidArgument("tool input schema must be a shallow JSON object");
  }
  if (schema.contains("type")) {
    if (!schema["type"].is_string() ||
        !IsSupportedSchemaType(schema["type"].get<std::string>())) {
      return Status::InvalidArgument("tool input schema contains an unsupported type");
    }
  }
  if (schema.contains("properties")) {
    if (!schema["properties"].is_object()) {
      return Status::InvalidArgument("tool schema properties must be an object");
    }
    for (const auto& [name, property_schema] : schema["properties"].items()) {
      static_cast<void>(name);
      Status status = ValidateSchemaDefinition(property_schema, depth + 1);
      if (!status.ok()) return status;
    }
  }
  if (schema.contains("items")) {
    Status status = ValidateSchemaDefinition(schema["items"], depth + 1);
    if (!status.ok()) return status;
  }
  if (schema.contains("required")) {
    if (!schema["required"].is_array()) {
      return Status::InvalidArgument("tool schema required must be an array");
    }
    for (const nlohmann::json& name : schema["required"]) {
      if (!name.is_string()) {
        return Status::InvalidArgument("tool schema required entries must be strings");
      }
    }
  }
  if (schema.contains("additionalProperties") &&
      !schema["additionalProperties"].is_boolean()) {
    return Status::InvalidArgument("tool schema additionalProperties must be boolean");
  }
  if (schema.contains("enum") && !schema["enum"].is_array()) {
    return Status::InvalidArgument("tool schema enum must be an array");
  }
  return Status();
}

bool MatchesSchemaType(const nlohmann::json& value, const std::string& type) {
  if (type == "object") return value.is_object();
  if (type == "array") return value.is_array();
  if (type == "string") return value.is_string();
  if (type == "integer") return value.is_number_integer();
  if (type == "number") return value.is_number();
  if (type == "boolean") return value.is_boolean();
  if (type == "null") return value.is_null();
  return false;
}

Status ValidateAgainstSchema(const nlohmann::json& value,
                             const nlohmann::json& schema,
                             const std::string& path, uint32_t depth = 0) {
  if (depth > 16) {
    return Status::InvalidArgument("tool arguments exceed the schema nesting limit");
  }
  if (schema.contains("type") &&
      !MatchesSchemaType(value, schema["type"].get<std::string>())) {
    return Status::InvalidArgument("tool argument has the wrong type at " + path);
  }
  if (schema.contains("enum") &&
      std::find(schema["enum"].begin(), schema["enum"].end(), value) ==
          schema["enum"].end()) {
    return Status::InvalidArgument("tool argument is outside its enum at " + path);
  }
  if (value.is_object()) {
    const nlohmann::json properties = schema.value("properties", nlohmann::json::object());
    const nlohmann::json required = schema.value("required", nlohmann::json::array());
    for (const nlohmann::json& required_name : required) {
      const std::string name = required_name.get<std::string>();
      if (!value.contains(name)) {
        return Status::InvalidArgument("missing required tool argument: " + name);
      }
    }
    const bool allow_additional = schema.value("additionalProperties", true);
    for (const auto& [name, property_value] : value.items()) {
      if (!properties.contains(name)) {
        if (!allow_additional) {
          return Status::InvalidArgument("unknown tool argument: " + name);
        }
        continue;
      }
      Status status = ValidateAgainstSchema(property_value, properties[name],
                                            path + "." + name, depth + 1);
      if (!status.ok()) return status;
    }
  } else if (value.is_array() && schema.contains("items")) {
    std::size_t index = 0;
    for (const nlohmann::json& item : value) {
      Status status = ValidateAgainstSchema(item, schema["items"],
                                            path + "[" + std::to_string(index) + "]",
                                            depth + 1);
      if (!status.ok()) return status;
      ++index;
    }
  }
  return Status();
}

std::string Fingerprint(const ToolExecutionRequest& request) {
  std::string value;
  value.reserve(request.name.size() + request.arguments_json.size() + 1U);
  value.append(request.name);
  value.push_back('\0');
  value.append(request.arguments_json);
  return value;
}

}  // namespace

bool ToolPermissionPolicy::IsAllowed(ToolRiskLevel risk,
                                     const ToolApproval& approval) {
  switch (risk) {
    case ToolRiskLevel::kReadOnly:
      return true;
    case ToolRiskLevel::kSensitiveRead:
      return approval.session_approved;
    case ToolRiskLevel::kWrite:
      return approval.call_approved;
    case ToolRiskLevel::kDestructive:
      return approval.call_approved && approval.destructive_approved;
  }
  return false;
}

ToolRuntime::ToolRuntime(uint32_t max_iterations, std::size_t max_argument_bytes)
    : max_iterations_(max_iterations), max_argument_bytes_(max_argument_bytes) {}

Status ToolRuntime::Register(ToolDefinition tool) {
  if (!IsValidToolName(tool.metadata.name)) {
    return Status::InvalidArgument("tool name contains invalid characters");
  }
  if (tool.metadata.description.empty()) {
    return Status::InvalidArgument("tool description cannot be empty");
  }
  if (!tool.handler) return Status::InvalidArgument("tool handler cannot be empty");
  if (tool.metadata.risk != ToolRiskLevel::kReadOnly &&
      tool.metadata.risk != ToolRiskLevel::kSensitiveRead &&
      tool.metadata.risk != ToolRiskLevel::kWrite &&
      tool.metadata.risk != ToolRiskLevel::kDestructive) {
    return Status::InvalidArgument("tool risk level is invalid");
  }
  if (!tool.metadata.input_schema.is_object() ||
      !tool.metadata.input_schema.contains("type") ||
      !tool.metadata.input_schema["type"].is_string() ||
      tool.metadata.input_schema["type"].get<std::string>() != "object") {
    return Status::InvalidArgument("tool input schema root must have type object");
  }
  Status schema_status = ValidateSchemaDefinition(tool.metadata.input_schema);
  if (!schema_status.ok()) return schema_status;

  std::lock_guard lock(mutex_);
  const std::string name = tool.metadata.name;
  if (!tools_.try_emplace(name, std::move(tool)).second) {
    return Status::AlreadyExists("tool is already registered");
  }
  return Status();
}

std::vector<ToolMetadata> ToolRuntime::ListTools() const {
  std::lock_guard lock(mutex_);
  std::vector<ToolMetadata> result;
  result.reserve(tools_.size());
  for (const auto& [name, tool] : tools_) {
    static_cast<void>(name);
    result.push_back(tool.metadata);
  }
  return result;
}

StatusOr<std::string> ToolRuntime::Execute(const ToolExecutionRequest& request) {
  if (!IsValidCallId(request.call_id)) {
    return Status::InvalidArgument("tool call id contains invalid characters or length");
  }
  if (!IsValidToolName(request.name)) {
    return Status::InvalidArgument("tool name contains invalid characters");
  }
  if (request.arguments_json.size() > max_argument_bytes_) {
    return Status::InvalidArgument("tool arguments exceed the configured byte limit");
  }
  if (max_iterations_ == 0 || request.iteration == 0 ||
      request.iteration > max_iterations_) {
    return Status::InvalidArgument("tool iteration limit was exceeded");
  }
  if (request.cancellation.IsCancellationRequested()) {
    return Status::Cancelled("tool execution was cancelled before it started");
  }

  nlohmann::json arguments;
  try {
    const auto callback = [](int depth, nlohmann::json::parse_event_t,
                             nlohmann::json&) {
      if (depth > 32) throw std::runtime_error("JSON nesting exceeds the parser limit");
      return true;
    };
    arguments = nlohmann::json::parse(request.arguments_json, callback);
  } catch (const std::exception& exception) {
    return Status::InvalidArgument(std::string("tool arguments are not valid JSON: ") +
                                   exception.what());
  }
  if (!arguments.is_object()) {
    return Status::InvalidArgument("tool arguments must be a JSON object");
  }

  ToolDefinition tool;
  const std::string fingerprint = Fingerprint(request);
  {
    std::lock_guard lock(mutex_);
    const auto found = tools_.find(request.name);
    if (found == tools_.end()) return Status::NotFound("tool is not registered");
    if (!ToolPermissionPolicy::IsAllowed(found->second.metadata.risk, request.approval)) {
      return Status::Unavailable("tool execution requires additional approval");
    }
    const auto cached = completed_calls_.find(request.call_id);
    if (cached != completed_calls_.end()) {
      if (cached->second.fingerprint != fingerprint) {
        return Status::InvalidArgument("tool call id was reused with different input");
      }
      if (!cached->second.status.ok()) return cached->second.status;
      return cached->second.result;
    }
    if (in_flight_calls_.contains(request.call_id)) {
      return Status::AlreadyExists("tool call id is currently executing");
    }
    tool = found->second;
    Status validation = ValidateAgainstSchema(arguments, tool.metadata.input_schema, "$", 0);
    if (!validation.ok()) return validation;
    in_flight_calls_.insert(request.call_id);
  }

  StatusOr<std::string> result = Status::Internal("tool handler did not return a result");
  try {
    result = tool.handler(arguments, request.cancellation);
  } catch (const std::exception& exception) {
    result = Status::Internal(std::string("tool handler threw an exception: ") +
                              exception.what());
  } catch (...) {
    result = Status::Internal("tool handler threw an unknown exception");
  }

  {
    std::lock_guard lock(mutex_);
    in_flight_calls_.erase(request.call_id);
    CachedExecution cached;
    cached.fingerprint = fingerprint;
    if (result.ok()) {
      cached.result = result.value();
    } else {
      cached.status = result.status();
    }
    completed_calls_.insert_or_assign(request.call_id, std::move(cached));
    completed_call_order_.push_back(request.call_id);
    constexpr std::size_t kMaximumCachedCalls = 4096;
    while (completed_call_order_.size() > kMaximumCachedCalls) {
      completed_calls_.erase(completed_call_order_.front());
      completed_call_order_.pop_front();
    }
  }
  return result;
}

}  // namespace isvik
