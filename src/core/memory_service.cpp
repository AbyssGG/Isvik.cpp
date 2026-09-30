#include "isvik/core/memory_service.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>
#include <sqlite3.h>

namespace isvik {

struct MemoryService::Impl {
  sqlite3* database = nullptr;
  mutable std::mutex mutex;
  std::unordered_set<std::string> disabled_sessions;

  ~Impl() {
    if (database != nullptr) sqlite3_close_v2(database);
  }
};

namespace {

constexpr int kSchemaVersion = 1;
constexpr std::size_t kMaxSearchBytes = 256U;
constexpr std::size_t kMaxListLimit = 1000U;
constexpr std::size_t kMaxExportLimit = 10000U;
constexpr std::size_t kMemoryIdLength = 36U;

class Statement {
 public:
  Statement() = default;
  ~Statement() {
    if (statement_ != nullptr) sqlite3_finalize(statement_);
  }

  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;

  sqlite3_stmt** out() { return &statement_; }
  sqlite3_stmt* get() const { return statement_; }

 private:
  sqlite3_stmt* statement_ = nullptr;
};

Status SqliteError(sqlite3* database, int result, std::string operation) {
  const int primary_result = result & 0xff;
  std::string detail = database == nullptr ? sqlite3_errstr(result)
                                           : sqlite3_errmsg(database);
  operation += ": ";
  operation += detail;
  if (primary_result == SQLITE_BUSY || primary_result == SQLITE_LOCKED ||
      primary_result == SQLITE_CANTOPEN || primary_result == SQLITE_FULL ||
      primary_result == SQLITE_IOERR || primary_result == SQLITE_READONLY) {
    return Status::Unavailable(std::move(operation));
  }
  return Status::Internal(std::move(operation));
}

Status Prepare(sqlite3* database, const char* sql, Statement* statement,
               const char* operation) {
  const int result = sqlite3_prepare_v2(database, sql, -1, statement->out(), nullptr);
  if (result != SQLITE_OK) {
    return SqliteError(database, result, operation);
  }
  return Status();
}

Status Execute(sqlite3* database, const char* sql, const char* operation) {
  char* error_message = nullptr;
  const int result = sqlite3_exec(database, sql, nullptr, nullptr, &error_message);
  if (result == SQLITE_OK) return Status();

  std::string message(operation);
  message += ": ";
  if (error_message != nullptr) {
    message += error_message;
    sqlite3_free(error_message);
  } else {
    message += sqlite3_errmsg(database);
  }
  const int primary_result = result & 0xff;
  if (primary_result == SQLITE_BUSY || primary_result == SQLITE_LOCKED ||
      primary_result == SQLITE_CANTOPEN || primary_result == SQLITE_FULL ||
      primary_result == SQLITE_IOERR || primary_result == SQLITE_READONLY) {
    return Status::Unavailable(std::move(message));
  }
  return Status::Internal(std::move(message));
}

Status BindText(sqlite3* database, sqlite3_stmt* statement, int index,
                std::string_view value, const char* operation) {
  const auto byte_count = static_cast<int>(value.size());
  const char* data = value.empty() ? "" : value.data();
  const int result = sqlite3_bind_text(statement, index, data, byte_count,
                                       SQLITE_TRANSIENT);
  if (result != SQLITE_OK) {
    return SqliteError(database, result, operation);
  }
  return Status();
}

bool IsValidUtf8(std::string_view value) {
  std::size_t index = 0U;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first == 0U) return false;
    if (first <= 0x7fU) {
      ++index;
      continue;
    }

    if (first >= 0xc2U && first <= 0xdfU) {
      if (index + 1U >= value.size()) return false;
      const auto second = static_cast<unsigned char>(value[index + 1U]);
      if (second < 0x80U || second > 0xbfU) return false;
      index += 2U;
      continue;
    }

    if (first >= 0xe0U && first <= 0xefU) {
      if (index + 2U >= value.size()) return false;
      const auto second = static_cast<unsigned char>(value[index + 1U]);
      const auto third = static_cast<unsigned char>(value[index + 2U]);
      const bool valid_second = first == 0xe0U
          ? (second >= 0xa0U && second <= 0xbfU)
          : first == 0xedU ? (second >= 0x80U && second <= 0x9fU)
                           : (second >= 0x80U && second <= 0xbfU);
      if (!valid_second || third < 0x80U || third > 0xbfU) return false;
      index += 3U;
      continue;
    }

    if (first >= 0xf0U && first <= 0xf4U) {
      if (index + 3U >= value.size()) return false;
      const auto second = static_cast<unsigned char>(value[index + 1U]);
      const auto third = static_cast<unsigned char>(value[index + 2U]);
      const auto fourth = static_cast<unsigned char>(value[index + 3U]);
      const bool valid_second = first == 0xf0U
          ? (second >= 0x90U && second <= 0xbfU)
          : first == 0xf4U ? (second >= 0x80U && second <= 0x8fU)
                           : (second >= 0x80U && second <= 0xbfU);
      if (!valid_second || third < 0x80U || third > 0xbfU ||
          fourth < 0x80U || fourth > 0xbfU) {
        return false;
      }
      index += 4U;
      continue;
    }
    return false;
  }
  return true;
}

Status ValidateSessionId(std::string_view session_id, bool allow_empty) {
  if (session_id.empty()) {
    return allow_empty ? Status() : Status::InvalidArgument("session ID is required");
  }
  if (session_id.size() > kMemoryMaxSessionIdBytes || !IsValidUtf8(session_id)) {
    return Status::InvalidArgument("session ID must be valid UTF-8 and at most 128 bytes");
  }
  return Status();
}

Status ValidateId(std::string_view id) {
  if (id.size() != kMemoryIdLength || id.substr(0U, 4U) != "mem-") {
    return Status::InvalidArgument("memory ID has an invalid format");
  }
  for (const char character : id.substr(4U)) {
    const bool hex = (character >= '0' && character <= '9') ||
        (character >= 'a' && character <= 'f') ||
        (character >= 'A' && character <= 'F');
    if (!hex) return Status::InvalidArgument("memory ID has an invalid format");
  }
  return Status();
}

Status ValidateContent(std::string_view content) {
  if (content.empty()) {
    return Status::InvalidArgument("memory content must not be empty");
  }
  if (content.size() > kMemoryMaxContentBytes) {
    return Status::InvalidArgument("memory content exceeds 64 KiB");
  }
  if (!IsValidUtf8(content)) {
    return Status::InvalidArgument("memory content must be valid UTF-8 without NUL bytes");
  }
  return Status();
}

Status ValidateTags(const std::vector<std::string>& tags) {
  if (tags.size() > kMemoryMaxTags) {
    return Status::InvalidArgument("memory has more than 32 tags");
  }
  for (const std::string& tag : tags) {
    if (tag.empty() || tag.size() > kMemoryMaxTagBytes || !IsValidUtf8(tag)) {
      return Status::InvalidArgument("tags must be non-empty valid UTF-8 strings up to 64 bytes");
    }
  }
  return Status();
}

Status ValidateLimit(std::size_t limit, std::size_t maximum) {
  if (limit == 0U || limit > maximum ||
      limit > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument("result limit is outside the supported range");
  }
  return Status();
}

std::string SerializeTags(const std::vector<std::string>& tags) {
  return nlohmann::json(tags).dump();
}

StatusOr<MemoryEntry> ReadEntry(sqlite3* database, sqlite3_stmt* statement) {
  MemoryEntry entry;
  const auto read_text = [statement](int column) {
    const auto* text = sqlite3_column_text(statement, column);
    const int byte_count = sqlite3_column_bytes(statement, column);
    if (text == nullptr || byte_count <= 0) return std::string();
    return std::string(reinterpret_cast<const char*>(text),
                       static_cast<std::size_t>(byte_count));
  };

  entry.id = read_text(0);
  entry.content = read_text(1);
  const std::string serialized_tags = read_text(2);
  const int scope = sqlite3_column_int(statement, 3);
  if (scope != static_cast<int>(MemoryScope::kPersistent) &&
      scope != static_cast<int>(MemoryScope::kSession)) {
    return Status::Internal("memory database contains an invalid scope");
  }
  entry.scope = static_cast<MemoryScope>(scope);
  if (sqlite3_column_type(statement, 4) != SQLITE_NULL) {
    entry.session_id = read_text(4);
  }
  entry.created_at_ms = sqlite3_column_int64(statement, 5);
  entry.updated_at_ms = sqlite3_column_int64(statement, 6);
  entry.pinned = sqlite3_column_int(statement, 7) != 0;

  try {
    const nlohmann::json parsed_tags = nlohmann::json::parse(serialized_tags);
    if (!parsed_tags.is_array()) {
      return Status::Internal("memory database contains malformed tags");
    }
    for (const nlohmann::json& tag : parsed_tags) {
      if (!tag.is_string()) {
        return Status::Internal("memory database contains a non-string tag");
      }
      entry.tags.push_back(tag.get<std::string>());
    }
  } catch (const nlohmann::json::exception& exception) {
    return Status::Internal(std::string("memory database contains malformed tags: ") +
                            exception.what());
  }
  static_cast<void>(database);
  return entry;
}

StatusOr<std::vector<MemoryEntry>> ReadVisibleEntries(
    sqlite3* database, std::string_view session_id, bool session_enabled,
    std::size_t limit, bool search, std::string_view query) {
  constexpr char kListSql[] =
      "SELECT id, content, tags_json, scope, session_id, created_at_ms, "
      "updated_at_ms, pinned FROM memories "
      "WHERE (scope = 0 OR (scope = 1 AND session_id = ?1 AND ?2 = 1)) "
      "ORDER BY pinned DESC, updated_at_ms DESC, id ASC LIMIT ?3";
  constexpr char kSearchSql[] =
      "SELECT id, content, tags_json, scope, session_id, created_at_ms, "
      "updated_at_ms, pinned FROM memories "
      "WHERE (scope = 0 OR (scope = 1 AND session_id = ?1 AND ?2 = 1)) "
      "AND (instr(lower(content), lower(?3)) > 0 OR "
      "instr(lower(tags_json), lower(?3)) > 0) "
      "ORDER BY pinned DESC, updated_at_ms DESC, id ASC LIMIT ?4";

  Statement statement;
  const Status prepare_status = Prepare(database, search ? kSearchSql : kListSql,
                                         &statement, "prepare memory query");
  if (!prepare_status.ok()) return prepare_status;
  Status status = BindText(database, statement.get(), 1, session_id,
                           "bind memory session ID");
  if (!status.ok()) return status;
  if (sqlite3_bind_int(statement.get(), 2, session_enabled ? 1 : 0) != SQLITE_OK) {
    return SqliteError(database, sqlite3_errcode(database), "bind session memory state");
  }
  if (search) {
    status = BindText(database, statement.get(), 3, query, "bind memory search query");
    if (!status.ok()) return status;
    if (sqlite3_bind_int(statement.get(), 4, static_cast<int>(limit)) != SQLITE_OK) {
      return SqliteError(database, sqlite3_errcode(database), "bind memory result limit");
    }
  } else if (sqlite3_bind_int(statement.get(), 3, static_cast<int>(limit)) != SQLITE_OK) {
    return SqliteError(database, sqlite3_errcode(database), "bind memory result limit");
  }

  std::vector<MemoryEntry> entries;
  int step_result = SQLITE_OK;
  while ((step_result = sqlite3_step(statement.get())) == SQLITE_ROW) {
    StatusOr<MemoryEntry> entry = ReadEntry(database, statement.get());
    if (!entry.ok()) return entry.status();
    entries.push_back(std::move(entry).value());
  }
  if (step_result != SQLITE_DONE) {
    return SqliteError(database, step_result, "read memory results");
  }
  return entries;
}

std::string NewMemoryId() {
  std::array<unsigned char, 16U> random_bytes{};
  sqlite3_randomness(static_cast<int>(random_bytes.size()), random_bytes.data());
  constexpr char kHex[] = "0123456789abcdef";
  std::string id = "mem-";
  id.reserve(kMemoryIdLength);
  for (const unsigned char value : random_bytes) {
    id.push_back(kHex[(value >> 4U) & 0x0fU]);
    id.push_back(kHex[value & 0x0fU]);
  }
  return id;
}

std::int64_t NowMilliseconds() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

template <typename ImplType>
bool IsSessionEnabled(const ImplType& impl, std::string_view session_id) {
  return impl.disabled_sessions.find(std::string(session_id)) ==
      impl.disabled_sessions.end();
}

Status InitializeSchema(sqlite3* database) {
  int version = 0;
  Status status;
  {
    Statement version_statement;
    status = Prepare(database, "PRAGMA user_version", &version_statement,
                     "read memory schema version");
    if (!status.ok()) return status;
    const int step_result = sqlite3_step(version_statement.get());
    if (step_result != SQLITE_ROW) {
      return SqliteError(database, step_result, "read memory schema version");
    }
    version = sqlite3_column_int(version_statement.get(), 0);
  }
  if (version > kSchemaVersion) {
    return Status::Unsupported("memory database schema is newer than this application");
  }
  if (version == kSchemaVersion) return Status();

  status = Execute(database, "BEGIN IMMEDIATE", "begin memory schema transaction");
  if (!status.ok()) return status;
  status = Execute(database,
      "CREATE TABLE IF NOT EXISTS memories ("
      "id TEXT PRIMARY KEY NOT NULL,"
      "content TEXT NOT NULL CHECK(length(content) BETWEEN 1 AND 65536),"
      "tags_json TEXT NOT NULL,"
      "scope INTEGER NOT NULL CHECK(scope IN (0, 1)),"
      "session_id TEXT,"
      "created_at_ms INTEGER NOT NULL,"
      "updated_at_ms INTEGER NOT NULL,"
      "pinned INTEGER NOT NULL CHECK(pinned IN (0, 1)),"
      "CHECK((scope = 0 AND session_id IS NULL) OR "
      "(scope = 1 AND session_id IS NOT NULL AND length(session_id) > 0))"
      ");"
      "CREATE INDEX IF NOT EXISTS memories_visibility_order "
      "ON memories(scope, session_id, pinned DESC, updated_at_ms DESC);"
      "PRAGMA user_version = 1;",
      "create memory schema");
  if (!status.ok()) {
    static_cast<void>(Execute(database, "ROLLBACK", "rollback memory schema transaction"));
    return status;
  }
  status = Execute(database, "COMMIT", "commit memory schema transaction");
  if (!status.ok()) {
    static_cast<void>(Execute(database, "ROLLBACK", "rollback memory schema transaction"));
    return status;
  }
  return Status();
}

Status ValidateSearch(std::string_view query) {
  if (query.empty() || query.size() > kMaxSearchBytes || !IsValidUtf8(query)) {
    return Status::InvalidArgument("search query must be valid UTF-8 and 1 to 256 bytes");
  }
  return Status();
}

}  // namespace

MemoryService::MemoryService(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

MemoryService::~MemoryService() = default;

StatusOr<std::unique_ptr<MemoryService>> MemoryService::Open(
    const std::filesystem::path& database_path) {
  if (database_path.empty()) {
    return Status::InvalidArgument("memory database path must not be empty");
  }

  const bool in_memory = database_path == std::filesystem::path(":memory:");
  if (!in_memory) {
    const std::filesystem::path parent = database_path.parent_path();
    if (!parent.empty()) {
      std::error_code error;
      std::filesystem::create_directories(parent, error);
      if (error) {
        return Status::Unavailable("create memory database directory failed: " +
                                   error.message());
      }
    }
  }

  std::string path_text;
  try {
    const auto utf8_path = database_path.u8string();
    path_text.assign(reinterpret_cast<const char*>(utf8_path.data()), utf8_path.size());
  } catch (const std::exception& exception) {
    return Status::InvalidArgument(std::string("memory database path is invalid: ") +
                                   exception.what());
  }
  if (path_text.find('\0') != std::string::npos) {
    return Status::InvalidArgument("memory database path contains a NUL byte");
  }

  sqlite3* database = nullptr;
  const int open_result = sqlite3_open_v2(
      path_text.c_str(), &database,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
      nullptr);
  if (open_result != SQLITE_OK) {
    const std::string detail = database == nullptr ? sqlite3_errstr(open_result)
                                                   : sqlite3_errmsg(database);
    if (database != nullptr) sqlite3_close_v2(database);
    return Status::Unavailable("open memory database failed: " + detail);
  }

  auto impl = std::make_unique<Impl>();
  impl->database = database;
  sqlite3_extended_result_codes(database, 1);
  sqlite3_busy_timeout(database, 5000);
  Status status = Execute(database, "PRAGMA journal_mode=WAL", "enable SQLite WAL");
  if (status.ok()) {
    status = Execute(database,
        "PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON; "
        "PRAGMA trusted_schema=OFF;",
        "configure memory database");
  }
  if (status.ok()) status = InitializeSchema(database);
  if (!status.ok()) return status;

  int ignored_value = 0;
  const int config_result = sqlite3_db_config(
      database, SQLITE_DBCONFIG_DEFENSIVE, 1, &ignored_value);
  if (config_result != SQLITE_OK) {
    return SqliteError(database, config_result, "enable defensive SQLite mode");
  }
  return std::unique_ptr<MemoryService>(
      new MemoryService(std::move(impl)));
}

StatusOr<MemoryEntry> MemoryService::Add(const MemoryDraft& draft) {
  Status status = ValidateContent(draft.content);
  if (!status.ok()) return status;
  status = ValidateTags(draft.tags);
  if (!status.ok()) return status;
  if (draft.scope == MemoryScope::kPersistent) {
    if (!draft.session_id.empty()) {
      return Status::InvalidArgument("persistent memory must not have a session ID");
    }
  } else if (draft.scope == MemoryScope::kSession) {
    status = ValidateSessionId(draft.session_id, false);
    if (!status.ok()) return status;
  } else {
    return Status::InvalidArgument("memory scope is invalid");
  }

  const std::string tags_json = SerializeTags(draft.tags);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (draft.scope == MemoryScope::kSession &&
      !IsSessionEnabled(*impl_, draft.session_id)) {
    return Status::Unavailable("session memory is disabled");
  }

  Statement statement;
  status = Prepare(impl_->database,
      "INSERT INTO memories(id, content, tags_json, scope, session_id, "
      "created_at_ms, updated_at_ms, pinned) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?6, 0)",
      &statement, "prepare memory insert");
  if (!status.ok()) return status;

  MemoryEntry entry;
  entry.id = NewMemoryId();
  entry.content = draft.content;
  entry.tags = draft.tags;
  entry.scope = draft.scope;
  entry.session_id = draft.session_id;
  entry.created_at_ms = NowMilliseconds();
  entry.updated_at_ms = entry.created_at_ms;

  status = BindText(impl_->database, statement.get(), 1, entry.id, "bind memory ID");
  if (status.ok()) status = BindText(impl_->database, statement.get(), 2,
                                     entry.content, "bind memory content");
  if (status.ok()) status = BindText(impl_->database, statement.get(), 3,
                                     tags_json, "bind memory tags");
  if (!status.ok()) return status;
  if (sqlite3_bind_int(statement.get(), 4, static_cast<int>(entry.scope)) != SQLITE_OK) {
    return SqliteError(impl_->database, sqlite3_errcode(impl_->database),
                       "bind memory scope");
  }
  if (entry.scope == MemoryScope::kSession) {
    status = BindText(impl_->database, statement.get(), 5, entry.session_id,
                      "bind memory session ID");
  } else if (sqlite3_bind_null(statement.get(), 5) != SQLITE_OK) {
    return SqliteError(impl_->database, sqlite3_errcode(impl_->database),
                       "bind empty memory session ID");
  }
  if (!status.ok()) return status;
  if (sqlite3_bind_int64(statement.get(), 6, entry.created_at_ms) != SQLITE_OK) {
    return SqliteError(impl_->database, sqlite3_errcode(impl_->database),
                       "bind memory timestamp");
  }
  const int result = sqlite3_step(statement.get());
  if (result != SQLITE_DONE) {
    if ((result & 0xff) == SQLITE_CONSTRAINT) {
      return Status::AlreadyExists("generated memory ID already exists");
    }
    return SqliteError(impl_->database, result, "insert memory");
  }
  return entry;
}

StatusOr<MemoryEntry> MemoryService::Find(std::string_view id,
                                          std::string_view session_id) const {
  Status status = ValidateId(id);
  if (!status.ok()) return status;
  status = ValidateSessionId(session_id, true);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  const bool session_enabled = IsSessionEnabled(*impl_, session_id);
  Statement statement;
  status = Prepare(impl_->database,
      "SELECT id, content, tags_json, scope, session_id, created_at_ms, "
      "updated_at_ms, pinned FROM memories WHERE id = ?1 AND "
      "(scope = 0 OR (scope = 1 AND session_id = ?2 AND ?3 = 1))",
      &statement, "prepare memory lookup");
  if (!status.ok()) return status;
  status = BindText(impl_->database, statement.get(), 1, id, "bind memory ID");
  if (status.ok()) status = BindText(impl_->database, statement.get(), 2,
                                     session_id, "bind memory session ID");
  if (!status.ok()) return status;
  if (sqlite3_bind_int(statement.get(), 3, session_enabled ? 1 : 0) != SQLITE_OK) {
    return SqliteError(impl_->database, sqlite3_errcode(impl_->database),
                       "bind session memory state");
  }
  const int result = sqlite3_step(statement.get());
  if (result == SQLITE_DONE) return Status::NotFound("memory was not found");
  if (result != SQLITE_ROW) return SqliteError(impl_->database, result, "read memory");
  return ReadEntry(impl_->database, statement.get());
}

Status MemoryService::Update(std::string_view id, std::string_view session_id,
                             std::string content,
                             std::vector<std::string> tags) {
  Status status = ValidateId(id);
  if (!status.ok()) return status;
  status = ValidateSessionId(session_id, true);
  if (!status.ok()) return status;
  status = ValidateContent(content);
  if (!status.ok()) return status;
  status = ValidateTags(tags);
  if (!status.ok()) return status;
  const std::string tags_json = SerializeTags(tags);

  std::lock_guard<std::mutex> lock(impl_->mutex);
  const bool session_enabled = IsSessionEnabled(*impl_, session_id);
  Statement statement;
  status = Prepare(impl_->database,
      "UPDATE memories SET content = ?1, tags_json = ?2, "
      "updated_at_ms = MAX(?3, updated_at_ms + 1) WHERE id = ?4 AND "
      "(scope = 0 OR (scope = 1 AND session_id = ?5 AND ?6 = 1))",
      &statement, "prepare memory update");
  if (!status.ok()) return status;
  status = BindText(impl_->database, statement.get(), 1, content,
                    "bind updated memory content");
  if (status.ok()) status = BindText(impl_->database, statement.get(), 2,
                                     tags_json, "bind updated memory tags");
  if (status.ok()) status = BindText(impl_->database, statement.get(), 4,
                                     id, "bind memory ID");
  if (status.ok()) status = BindText(impl_->database, statement.get(), 5,
                                     session_id, "bind memory session ID");
  if (!status.ok()) return status;
  if (sqlite3_bind_int64(statement.get(), 3, NowMilliseconds()) != SQLITE_OK ||
      sqlite3_bind_int(statement.get(), 6, session_enabled ? 1 : 0) != SQLITE_OK) {
    return SqliteError(impl_->database, sqlite3_errcode(impl_->database),
                       "bind memory update state");
  }
  const int result = sqlite3_step(statement.get());
  if (result != SQLITE_DONE) return SqliteError(impl_->database, result, "update memory");
  if (sqlite3_changes(impl_->database) == 0) {
    return Status::NotFound("memory was not found");
  }
  return Status();
}

Status MemoryService::Remove(std::string_view id, std::string_view session_id) {
  Status status = ValidateId(id);
  if (!status.ok()) return status;
  status = ValidateSessionId(session_id, true);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  const bool session_enabled = IsSessionEnabled(*impl_, session_id);
  Statement statement;
  status = Prepare(impl_->database,
      "DELETE FROM memories WHERE id = ?1 AND "
      "(scope = 0 OR (scope = 1 AND session_id = ?2 AND ?3 = 1))",
      &statement, "prepare memory deletion");
  if (!status.ok()) return status;
  status = BindText(impl_->database, statement.get(), 1, id, "bind memory ID");
  if (status.ok()) status = BindText(impl_->database, statement.get(), 2,
                                     session_id, "bind memory session ID");
  if (!status.ok()) return status;
  if (sqlite3_bind_int(statement.get(), 3, session_enabled ? 1 : 0) != SQLITE_OK) {
    return SqliteError(impl_->database, sqlite3_errcode(impl_->database),
                       "bind session memory state");
  }
  const int result = sqlite3_step(statement.get());
  if (result != SQLITE_DONE) return SqliteError(impl_->database, result, "delete memory");
  if (sqlite3_changes(impl_->database) == 0) {
    return Status::NotFound("memory was not found");
  }
  return Status();
}

Status MemoryService::SetPinned(std::string_view id, std::string_view session_id,
                                bool pinned) {
  Status status = ValidateId(id);
  if (!status.ok()) return status;
  status = ValidateSessionId(session_id, true);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  const bool session_enabled = IsSessionEnabled(*impl_, session_id);
  Statement statement;
  status = Prepare(impl_->database,
      "UPDATE memories SET pinned = ?1, "
      "updated_at_ms = MAX(?2, updated_at_ms + 1) WHERE id = ?3 AND "
      "(scope = 0 OR (scope = 1 AND session_id = ?4 AND ?5 = 1))",
      &statement, "prepare memory pin update");
  if (!status.ok()) return status;
  if (sqlite3_bind_int(statement.get(), 1, pinned ? 1 : 0) != SQLITE_OK ||
      sqlite3_bind_int64(statement.get(), 2, NowMilliseconds()) != SQLITE_OK) {
    return SqliteError(impl_->database, sqlite3_errcode(impl_->database),
                       "bind memory pin state");
  }
  status = BindText(impl_->database, statement.get(), 3, id, "bind memory ID");
  if (status.ok()) status = BindText(impl_->database, statement.get(), 4,
                                     session_id, "bind memory session ID");
  if (!status.ok()) return status;
  if (sqlite3_bind_int(statement.get(), 5, session_enabled ? 1 : 0) != SQLITE_OK) {
    return SqliteError(impl_->database, sqlite3_errcode(impl_->database),
                       "bind session memory state");
  }
  const int result = sqlite3_step(statement.get());
  if (result != SQLITE_DONE) return SqliteError(impl_->database, result, "pin memory");
  if (sqlite3_changes(impl_->database) == 0) {
    return Status::NotFound("memory was not found");
  }
  return Status();
}

StatusOr<std::vector<MemoryEntry>> MemoryService::List(
    std::string_view session_id, std::size_t limit) const {
  Status status = ValidateSessionId(session_id, true);
  if (!status.ok()) return status;
  status = ValidateLimit(limit, kMaxListLimit);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  return ReadVisibleEntries(impl_->database, session_id,
                            IsSessionEnabled(*impl_, session_id), limit, false, {});
}

StatusOr<std::vector<MemoryEntry>> MemoryService::Search(
    std::string_view query, std::string_view session_id,
    std::size_t limit) const {
  Status status = ValidateSearch(query);
  if (!status.ok()) return status;
  status = ValidateSessionId(session_id, true);
  if (!status.ok()) return status;
  status = ValidateLimit(limit, kMaxListLimit);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  return ReadVisibleEntries(impl_->database, session_id,
                            IsSessionEnabled(*impl_, session_id), limit, true, query);
}

StatusOr<std::string> MemoryService::ExportJson(
    std::string_view session_id, std::size_t limit) const {
  Status status = ValidateSessionId(session_id, true);
  if (!status.ok()) return status;
  status = ValidateLimit(limit, kMaxExportLimit);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  StatusOr<std::vector<MemoryEntry>> entries = ReadVisibleEntries(
      impl_->database, session_id, IsSessionEnabled(*impl_, session_id),
      limit, false, {});
  if (!entries.ok()) return entries.status();

  nlohmann::json memories = nlohmann::json::array();
  for (const MemoryEntry& entry : entries.value()) {
    nlohmann::json memory = {
        {"id", entry.id},
        {"content", entry.content},
        {"tags", entry.tags},
        {"scope", entry.scope == MemoryScope::kPersistent ? "persistent" : "session"},
        {"created_at_ms", entry.created_at_ms},
        {"updated_at_ms", entry.updated_at_ms},
        {"pinned", entry.pinned},
    };
    memory["session_id"] = entry.scope == MemoryScope::kSession
        ? nlohmann::json(entry.session_id) : nlohmann::json(nullptr);
    memories.push_back(std::move(memory));
  }
  return nlohmann::json{{"version", 1}, {"memories", std::move(memories)}}.dump(2);
}

Status MemoryService::ClearPersistent() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return Execute(impl_->database, "DELETE FROM memories WHERE scope = 0",
                 "clear persistent memories");
}

Status MemoryService::ClearSession(std::string_view session_id) {
  Status status = ValidateSessionId(session_id, false);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  Statement statement;
  status = Prepare(impl_->database,
      "DELETE FROM memories WHERE scope = 1 AND session_id = ?1",
      &statement, "prepare session memory clear");
  if (!status.ok()) return status;
  status = BindText(impl_->database, statement.get(), 1, session_id,
                    "bind session ID");
  if (!status.ok()) return status;
  const int result = sqlite3_step(statement.get());
  if (result != SQLITE_DONE) {
    return SqliteError(impl_->database, result, "clear session memories");
  }
  return Status();
}

Status MemoryService::ClearAll() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const Status status = Execute(impl_->database, "DELETE FROM memories",
                                "clear all memories");
  if (status.ok()) impl_->disabled_sessions.clear();
  return status;
}

Status MemoryService::SetSessionMemoryEnabled(std::string_view session_id,
                                             bool enabled) {
  Status status = ValidateSessionId(session_id, false);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (enabled) {
    impl_->disabled_sessions.erase(std::string(session_id));
  } else {
    impl_->disabled_sessions.insert(std::string(session_id));
  }
  return Status();
}

StatusOr<bool> MemoryService::IsSessionMemoryEnabled(
    std::string_view session_id) const {
  Status status = ValidateSessionId(session_id, false);
  if (!status.ok()) return status;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return IsSessionEnabled(*impl_, session_id);
}

Status MemoryService::CloseSession(std::string_view session_id) {
  Status status = ValidateSessionId(session_id, false);
  if (!status.ok()) return status;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  Statement statement;
  status = Prepare(impl_->database,
      "DELETE FROM memories WHERE scope = 1 AND session_id = ?1",
      &statement, "prepare session close cleanup");
  if (!status.ok()) return status;
  status = BindText(impl_->database, statement.get(), 1, session_id,
                    "bind session ID");
  if (!status.ok()) return status;
  const int result = sqlite3_step(statement.get());
  if (result != SQLITE_DONE) {
    return SqliteError(impl_->database, result, "clear closed session memories");
  }
  impl_->disabled_sessions.erase(std::string(session_id));
  return Status();
}

}  // namespace isvik
