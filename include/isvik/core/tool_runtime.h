#ifndef ISVIK_CORE_TOOL_RUNTIME_H_
#define ISVIK_CORE_TOOL_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "isvik/core/cancellation.h"
#include "isvik/core/status.h"

namespace isvik {

enum class ToolRiskLevel {
  kReadOnly,
  kSensitiveRead,
  kWrite,
  kDestructive,
};

struct ToolMetadata {
  std::string name;
  std::string description;
  ToolRiskLevel risk = ToolRiskLevel::kReadOnly;
  nlohmann::json input_schema = {
      {"type", "object"}, {"properties", nlohmann::json::object()},
      {"additionalProperties", false}};
};

struct ToolDefinition {
  ToolMetadata metadata;
  std::function<StatusOr<std::string>(const nlohmann::json&,
                                      const CancellationToken&)> handler;
};

struct ToolApproval {
  bool session_approved = false;
  bool call_approved = false;
  bool destructive_approved = false;
};

struct ToolExecutionRequest {
  std::string call_id;
  std::string name;
  std::string arguments_json = "{}";
  uint32_t iteration = 1;
  ToolApproval approval;
  CancellationToken cancellation;
};

class ToolPermissionPolicy {
 public:
  [[nodiscard]] static bool IsAllowed(ToolRiskLevel risk,
                                      const ToolApproval& approval);
};

class ToolRuntime {
 public:
  explicit ToolRuntime(uint32_t max_iterations = 8,
                       std::size_t max_argument_bytes = 64 * 1024);

  [[nodiscard]] Status Register(ToolDefinition tool);
  [[nodiscard]] std::vector<ToolMetadata> ListTools() const;
  [[nodiscard]] StatusOr<std::string> Execute(
      const ToolExecutionRequest& request);

 private:
  struct CachedExecution {
    std::string fingerprint;
    Status status;
    std::string result;
  };

  uint32_t max_iterations_;
  std::size_t max_argument_bytes_;
  mutable std::mutex mutex_;
  std::map<std::string, ToolDefinition> tools_;
  std::map<std::string, CachedExecution> completed_calls_;
  std::deque<std::string> completed_call_order_;
  std::set<std::string> in_flight_calls_;
};

}  // namespace isvik

#endif  // ISVIK_CORE_TOOL_RUNTIME_H_
