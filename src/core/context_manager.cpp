#include "isvik/core/context_manager.h"

#include <algorithm>
#include <limits>
#include <map>
#include <utility>

namespace isvik {
namespace {

uint64_t SaturatingAdd(uint64_t left, uint64_t right) {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    return std::numeric_limits<uint64_t>::max();
  }
  return left + right;
}

struct MessageUnit {
  std::vector<std::size_t> indices;
  uint64_t token_cost = 0;
  std::size_t last_index = 0;
  bool required = false;
};

}  // namespace

uint64_t ContextManager::EstimateMessageTokens(const InferenceMessage& message,
                                              uint32_t estimated_bytes_per_token) {
  if (estimated_bytes_per_token == 0) return std::numeric_limits<uint64_t>::max();
  const uint64_t content_bytes = static_cast<uint64_t>(message.content.size());
  const uint64_t content_tokens = content_bytes / estimated_bytes_per_token +
      (content_bytes % estimated_bytes_per_token == 0 ? 0U : 1U);
  return SaturatingAdd(content_tokens, 4U);  // Role and message-boundary allowance.
}

StatusOr<ContextBuildResult> ContextManager::BuildPrompt(
    const std::vector<ContextMessage>& messages, const ContextOptions& options) {
  if (options.max_context_tokens == 0 || options.estimated_bytes_per_token == 0) {
    return Status::InvalidArgument("context token limits must be positive");
  }
  if (options.reserved_output_tokens >= options.max_context_tokens) {
    return Status::InvalidArgument(
        "reserved output tokens must be smaller than the context limit");
  }
  if (messages.empty()) return ContextBuildResult{};

  const uint64_t prompt_budget =
      options.max_context_tokens - options.reserved_output_tokens;
  std::vector<MessageUnit> units;
  std::map<std::string, std::size_t> grouped_units;
  uint64_t total_tokens = 0;
  for (std::size_t index = 0; index < messages.size(); ++index) {
    const ContextMessage& context_message = messages[index];
    const std::string& group_id = context_message.retention_group_id.empty()
        ? context_message.tool_group_id : context_message.retention_group_id;
    std::size_t unit_index = 0;
    if (group_id.empty()) {
      unit_index = units.size();
      units.emplace_back();
    } else {
      const auto existing = grouped_units.find(group_id);
      if (existing == grouped_units.end()) {
        unit_index = units.size();
        grouped_units.emplace(group_id, unit_index);
        units.emplace_back();
      } else {
        unit_index = existing->second;
      }
    }
    MessageUnit& unit = units[unit_index];
    unit.indices.push_back(index);
    unit.last_index = index;
    unit.required = unit.required || context_message.pinned ||
        context_message.message.role == MessageRole::kSystem;
    const uint64_t message_tokens = EstimateMessageTokens(
        context_message.message, options.estimated_bytes_per_token);
    unit.token_cost = SaturatingAdd(unit.token_cost, message_tokens);
    total_tokens = SaturatingAdd(total_tokens, message_tokens);
  }

  const std::size_t latest_message_unit = std::max_element(
      units.begin(), units.end(), [](const MessageUnit& left, const MessageUnit& right) {
        return left.last_index < right.last_index;
      }) - units.begin();
  units[latest_message_unit].required = true;

  if (total_tokens <= prompt_budget) {
    ContextBuildResult result;
    result.messages = messages;
    result.estimated_prompt_tokens = total_tokens;
    return result;
  }
  if (options.compression == ContextCompressionMode::kOff) {
    return Status::InvalidArgument("context exceeds budget while compression is disabled");
  }
  if (options.compression == ContextCompressionMode::kSummarize) {
    return Status::Unsupported("summarization requires an inference-backed compressor");
  }

  std::vector<bool> selected(units.size(), false);
  uint64_t selected_tokens = 0;
  for (std::size_t index = 0; index < units.size(); ++index) {
    if (!units[index].required) continue;
    if (units[index].token_cost > prompt_budget - selected_tokens) {
      return Status::InvalidArgument(
          "system, pinned, or latest tool context exceeds the prompt budget");
    }
    selected[index] = true;
    selected_tokens += units[index].token_cost;
  }

  std::vector<std::size_t> priority(units.size());
  for (std::size_t index = 0; index < units.size(); ++index) priority[index] = index;
  std::sort(priority.begin(), priority.end(), [&](std::size_t left, std::size_t right) {
    return units[left].last_index > units[right].last_index;
  });
  for (const std::size_t index : priority) {
    if (selected[index]) continue;
    if (units[index].token_cost <= prompt_budget - selected_tokens) {
      selected[index] = true;
      selected_tokens += units[index].token_cost;
    }
  }

  std::vector<bool> keep_message(messages.size(), false);
  for (std::size_t unit_index = 0; unit_index < units.size(); ++unit_index) {
    if (!selected[unit_index]) continue;
    for (const std::size_t message_index : units[unit_index].indices) {
      keep_message[message_index] = true;
    }
  }
  ContextBuildResult result;
  result.messages.reserve(messages.size());
  for (std::size_t index = 0; index < messages.size(); ++index) {
    if (keep_message[index]) result.messages.push_back(messages[index]);
  }
  result.estimated_prompt_tokens = selected_tokens;
  result.dropped_messages = static_cast<uint64_t>(messages.size() - result.messages.size());
  return result;
}

}  // namespace isvik
