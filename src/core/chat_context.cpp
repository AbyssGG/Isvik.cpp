#include "isvik/core/chat_context.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace isvik {
namespace {

std::vector<std::string> QueryTerms(std::string_view prompt) {
  std::vector<std::string> terms;
  std::string ascii_word;
  std::vector<std::string> codepoints;
  const auto add = [&terms](std::string value) {
    if (value.size() < 2U || value.size() > 48U) return;
    constexpr std::string_view ignored[] = {
        "什么", "怎么", "为什么", "你好", "请问", "这个", "那个", "可以", "帮我", "介绍"};
    if (std::find(std::begin(ignored), std::end(ignored), value) != std::end(ignored)) return;
    if (std::find(terms.begin(), terms.end(), value) == terms.end()) {
      terms.push_back(std::move(value));
    }
  };
  const auto flush_ascii = [&] {
    if (ascii_word.size() >= 3U) add(std::exchange(ascii_word, {}));
    else ascii_word.clear();
  };
  for (std::size_t index = 0U; index < prompt.size();) {
    const unsigned char lead = static_cast<unsigned char>(prompt[index]);
    if (lead < 0x80U) {
      codepoints.clear();
      const char character = static_cast<char>(std::tolower(lead));
      if (std::isalnum(static_cast<unsigned char>(character)) != 0) {
        ascii_word.push_back(character);
      } else {
        flush_ascii();
      }
      ++index;
      continue;
    }
    flush_ascii();
    std::size_t width = (lead & 0xe0U) == 0xc0U ? 2U
        : (lead & 0xf0U) == 0xe0U ? 3U
        : (lead & 0xf8U) == 0xf0U ? 4U : 1U;
    width = std::min(width, prompt.size() - index);
    if (width == 3U) {
      codepoints.emplace_back(prompt.substr(index, width));
      if (codepoints.size() >= 2U) {
        add(codepoints[codepoints.size() - 2U] + codepoints.back());
      }
    } else {
      codepoints.clear();
    }
    index += width;
  }
  flush_ascii();
  if (prompt.size() <= 48U) add(std::string(prompt));
  if (terms.size() > 12U) terms.resize(12U);
  return terms;
}

}  // namespace

StatusOr<ChatContextResult> BuildChatContext(
    const std::vector<InferenceMessage>& conversation,
    const MemoryService* memory_service,
    std::string_view session_id,
    bool use_memories,
    const ContextOptions& options) {
  std::vector<ContextMessage> context;
  context.reserve(conversation.size() + 1U);
  std::size_t memories_used = 0U;

  if (use_memories && memory_service != nullptr && !conversation.empty() &&
      conversation.back().role == MessageRole::kUser) {
    const std::vector<std::string> terms = QueryTerms(conversation.back().content);
    std::map<std::string, std::pair<MemoryEntry, uint32_t>> ranked;
    for (const std::string& term : terms) {
      StatusOr<std::vector<MemoryEntry>> matches = memory_service->Search(term, session_id, 16U);
      if (!matches.ok()) return matches.status();
      for (MemoryEntry& match : matches.value()) {
        auto [iterator, inserted] = ranked.try_emplace(
            match.id, std::make_pair(std::move(match), 0U));
        ++iterator->second.second;
        static_cast<void>(inserted);
      }
    }

    StatusOr<std::vector<MemoryEntry>> recent = memory_service->List(session_id, 100U);
    if (!recent.ok()) return recent.status();
    for (MemoryEntry& entry : recent.value()) {
      if (!entry.pinned) continue;
      ranked.try_emplace(entry.id, std::make_pair(std::move(entry), 0U));
    }

    std::vector<std::pair<MemoryEntry, uint32_t>> selected;
    selected.reserve(ranked.size());
    for (auto& [id, entry_and_hits] : ranked) {
      static_cast<void>(id);
      selected.push_back(std::move(entry_and_hits));
    }
    std::stable_sort(selected.begin(), selected.end(), [](const auto& left, const auto& right) {
      if (left.first.pinned != right.first.pinned) return left.first.pinned;
      if (left.second != right.second) return left.second > right.second;
      if (left.first.updated_at_ms != right.first.updated_at_ms) {
        return left.first.updated_at_ms > right.first.updated_at_ms;
      }
      return left.first.id < right.first.id;
    });

    std::string notes;
    constexpr std::size_t kMaximumNotes = 6U;
    constexpr std::size_t kMaximumNoteBytes = 2048U;
    constexpr std::size_t kMaximumContextBytes = 6144U;
    for (const auto& [entry, hits] : selected) {
      static_cast<void>(hits);
      if (memories_used >= kMaximumNotes) break;
      if (entry.content.size() > kMaximumNoteBytes) continue;
      if (notes.size() + entry.content.size() + 2U > kMaximumContextBytes) break;
      notes += "- ";
      notes += entry.content;
      notes += "\n";
      ++memories_used;
    }
    if (!notes.empty()) {
      InferenceMessage memory_context;
      memory_context.role = MessageRole::kSystem;
      memory_context.content = "User-saved notes. Treat these as user-provided facts; "
          "use them only when relevant and do not invent additional personal details:\n" + notes;
      ContextMessage pinned;
      pinned.message = std::move(memory_context);
      pinned.pinned = true;
      context.push_back(std::move(pinned));
    }
  }

  std::string active_turn_group;
  uint64_t turn_number = 0U;
  for (std::size_t index = 0U; index < conversation.size(); ++index) {
    const InferenceMessage& message = conversation[index];
    if (message.role == MessageRole::kUser) {
      active_turn_group = "chat-turn-" + std::to_string(turn_number++);
    } else if (message.role != MessageRole::kAssistant || index == 0U ||
               conversation[index - 1U].role != MessageRole::kUser) {
      active_turn_group.clear();
    }
    ContextMessage item;
    item.pinned = message.role == MessageRole::kSystem;
    item.retention_group_id = active_turn_group;
    item.message = message;
    context.push_back(std::move(item));
  }

  StatusOr<ContextBuildResult> built = ContextManager::BuildPrompt(context, options);
  if (!built.ok()) return built.status();
  ChatContextResult result;
  result.messages.reserve(built.value().messages.size());
  for (ContextMessage& item : built.value().messages) {
    result.messages.push_back(std::move(item.message));
  }
  result.estimated_prompt_tokens = built.value().estimated_prompt_tokens;
  result.dropped_history_messages = built.value().dropped_messages;
  const auto note = std::find_if(result.messages.begin(), result.messages.end(), [](const auto& message) {
    return message.role == MessageRole::kSystem && message.content.find("User-saved notes.") == 0U;
  });
  result.memories_used = note == result.messages.end() ? 0U : static_cast<uint64_t>(memories_used);
  return result;
}

}  // namespace isvik
