#ifndef ISVIK_CORE_MEMORY_SERVICE_H_
#define ISVIK_CORE_MEMORY_SERVICE_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "isvik/core/status.h"

namespace isvik {

inline constexpr std::size_t kMemoryMaxContentBytes = 64U * 1024U;
inline constexpr std::size_t kMemoryMaxTags = 32U;
inline constexpr std::size_t kMemoryMaxTagBytes = 64U;
inline constexpr std::size_t kMemoryMaxSessionIdBytes = 128U;

enum class MemoryScope {
  kPersistent = 0,
  kSession = 1,
};

struct MemoryDraft {
  std::string content;
  std::vector<std::string> tags;
  MemoryScope scope = MemoryScope::kPersistent;
  std::string session_id;
};

struct MemoryEntry {
  std::string id;
  std::string content;
  std::vector<std::string> tags;
  MemoryScope scope = MemoryScope::kPersistent;
  std::string session_id;
  std::int64_t created_at_ms = 0;
  std::int64_t updated_at_ms = 0;
  bool pinned = false;
};

// Stores user-approved local memories in SQLite. Session entries are visible
// only to the matching session, and disabling memory hides its session entries.
class MemoryService {
 public:
  [[nodiscard]] static StatusOr<std::unique_ptr<MemoryService>> Open(
      const std::filesystem::path& database_path);

  ~MemoryService();
  MemoryService(const MemoryService&) = delete;
  MemoryService& operator=(const MemoryService&) = delete;
  MemoryService(MemoryService&&) = delete;
  MemoryService& operator=(MemoryService&&) = delete;

  [[nodiscard]] StatusOr<MemoryEntry> Add(const MemoryDraft& draft);
  [[nodiscard]] StatusOr<MemoryEntry> Find(
      std::string_view id, std::string_view session_id = {}) const;
  [[nodiscard]] Status Update(std::string_view id,
                              std::string_view session_id,
                              std::string content,
                              std::vector<std::string> tags);
  [[nodiscard]] Status Remove(std::string_view id,
                              std::string_view session_id = {});
  [[nodiscard]] Status SetPinned(std::string_view id,
                                 std::string_view session_id, bool pinned);
  [[nodiscard]] StatusOr<std::vector<MemoryEntry>> List(
      std::string_view session_id = {}, std::size_t limit = 100U) const;
  [[nodiscard]] StatusOr<std::vector<MemoryEntry>> Search(
      std::string_view query, std::string_view session_id = {},
      std::size_t limit = 100U) const;
  [[nodiscard]] StatusOr<std::string> ExportJson(
      std::string_view session_id = {}, std::size_t limit = 10000U) const;

  [[nodiscard]] Status ClearPersistent();
  [[nodiscard]] Status ClearSession(std::string_view session_id);
  [[nodiscard]] Status ClearAll();
  [[nodiscard]] Status SetSessionMemoryEnabled(std::string_view session_id,
                                               bool enabled);
  [[nodiscard]] StatusOr<bool> IsSessionMemoryEnabled(
      std::string_view session_id) const;
  [[nodiscard]] Status CloseSession(std::string_view session_id);

 private:
  struct Impl;
  explicit MemoryService(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace isvik

#endif  // ISVIK_CORE_MEMORY_SERVICE_H_
