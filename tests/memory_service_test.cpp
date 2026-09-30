#include <atomic>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include "isvik/core/memory_service.h"

namespace isvik {
namespace {

std::filesystem::path NewDatabasePath() {
  static std::atomic<std::uint64_t> sequence{0U};
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
      ("isvik_memory_test_" + std::to_string(ticks) + "_" +
       std::to_string(sequence.fetch_add(1U)) + ".sqlite3");
}

void RemoveDatabaseFiles(const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::remove(path, error);
  std::filesystem::remove(path.string() + "-wal", error);
  std::filesystem::remove(path.string() + "-shm", error);
}

class MemoryServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = NewDatabasePath();
    StatusOr<std::unique_ptr<MemoryService>> opened = MemoryService::Open(path_);
    ASSERT_TRUE(opened.ok()) << opened.status().message();
    service_ = std::move(opened).value();
  }

  void TearDown() override {
    service_.reset();
    RemoveDatabaseFiles(path_);
  }

  MemoryDraft Persistent(std::string content,
                         std::vector<std::string> tags = {}) const {
    MemoryDraft draft;
    draft.content = std::move(content);
    draft.tags = std::move(tags);
    return draft;
  }

  MemoryDraft Session(std::string content, std::string session_id,
                      std::vector<std::string> tags = {}) const {
    MemoryDraft draft = Persistent(std::move(content), std::move(tags));
    draft.scope = MemoryScope::kSession;
    draft.session_id = std::move(session_id);
    return draft;
  }

  std::filesystem::path path_;
  std::unique_ptr<MemoryService> service_;
};

TEST(MemoryServiceOpenTest, CreatesDatabaseAndParentDirectories) {
  const std::filesystem::path directory = NewDatabasePath().parent_path() /
      "isvik_memory_nested" / std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count());
  const std::filesystem::path database_path = directory / "memories.sqlite3";
  std::error_code error;
  std::filesystem::remove_all(directory, error);

  {
    StatusOr<std::unique_ptr<MemoryService>> opened = MemoryService::Open(database_path);
    ASSERT_TRUE(opened.ok()) << opened.status().message();
    EXPECT_TRUE(std::filesystem::is_regular_file(database_path));
  }

  RemoveDatabaseFiles(database_path);
  std::filesystem::remove_all(directory, error);
}

TEST(MemoryServiceOpenTest, SupportsInMemoryDatabase) {
  StatusOr<std::unique_ptr<MemoryService>> opened =
      MemoryService::Open(std::filesystem::path(":memory:"));
  ASSERT_TRUE(opened.ok()) << opened.status().message();
  EXPECT_TRUE(opened.value()->Add(MemoryDraft{"in memory", {},
                                               MemoryScope::kPersistent, {}}).ok());
}

TEST(MemoryServiceOpenTest, RejectsEmptyDatabasePath) {
  const StatusOr<std::unique_ptr<MemoryService>> opened =
      MemoryService::Open(std::filesystem::path());
  ASSERT_FALSE(opened.ok());
  EXPECT_EQ(opened.status().code(), StatusCode::kInvalidArgument);
}

TEST(MemoryServiceOpenTest, UsesPinnedSqliteRelease) {
  EXPECT_STREQ(sqlite3_libversion(), "3.53.4");
}

TEST(MemoryServiceOpenTest, OpensDatabasePathsWithUnicodeCharacters) {
  const std::filesystem::path database_path = NewDatabasePath().parent_path() /
      std::filesystem::path(u8"记忆存储.sqlite3");
  {
    StatusOr<std::unique_ptr<MemoryService>> opened = MemoryService::Open(database_path);
    ASSERT_TRUE(opened.ok()) << opened.status().message();
    EXPECT_TRUE(std::filesystem::is_regular_file(database_path));
  }
  RemoveDatabaseFiles(database_path);
}

TEST(MemoryServiceOpenTest, RejectsDatabaseWithNewerSchemaVersion) {
  const std::filesystem::path database_path = NewDatabasePath();
  sqlite3* database = nullptr;
  const int open_result = sqlite3_open(database_path.string().c_str(), &database);
  ASSERT_EQ(open_result, SQLITE_OK);
  const int version_result = sqlite3_exec(
      database, "PRAGMA user_version = 999", nullptr, nullptr, nullptr);
  ASSERT_EQ(version_result, SQLITE_OK);
  ASSERT_EQ(sqlite3_close(database), SQLITE_OK);

  const StatusOr<std::unique_ptr<MemoryService>> opened =
      MemoryService::Open(database_path);
  EXPECT_FALSE(opened.ok());
  if (!opened.ok()) EXPECT_EQ(opened.status().code(), StatusCode::kUnsupported);
  RemoveDatabaseFiles(database_path);
}

TEST_F(MemoryServiceTest, AddsPersistentMemoryWithMetadata) {
  StatusOr<MemoryEntry> added = service_->Add(Persistent("喜欢深色主题", {"ui", "中文"}));
  ASSERT_TRUE(added.ok()) << added.status().message();
  EXPECT_EQ(added.value().content, "喜欢深色主题");
  EXPECT_EQ(added.value().tags, (std::vector<std::string>{"ui", "中文"}));
  EXPECT_EQ(added.value().scope, MemoryScope::kPersistent);
  EXPECT_TRUE(added.value().session_id.empty());
  EXPECT_EQ(added.value().id.size(), 36U);
  EXPECT_EQ(added.value().id.substr(0U, 4U), "mem-");
  EXPECT_GT(added.value().created_at_ms, 0);
  EXPECT_EQ(added.value().created_at_ms, added.value().updated_at_ms);
  EXPECT_FALSE(added.value().pinned);
}

TEST_F(MemoryServiceTest, AddsSessionMemoryWithItsOwner) {
  StatusOr<MemoryEntry> added = service_->Add(Session("temporary note", "chat-a"));
  ASSERT_TRUE(added.ok()) << added.status().message();
  EXPECT_EQ(added.value().scope, MemoryScope::kSession);
  EXPECT_EQ(added.value().session_id, "chat-a");
}

TEST_F(MemoryServiceTest, RejectsEmptyContent) {
  const StatusOr<MemoryEntry> added = service_->Add(Persistent(""));
  ASSERT_FALSE(added.ok());
  EXPECT_EQ(added.status().code(), StatusCode::kInvalidArgument);
}

TEST_F(MemoryServiceTest, RejectsContentAboveSixtyFourKibibytes) {
  const StatusOr<MemoryEntry> added =
      service_->Add(Persistent(std::string(kMemoryMaxContentBytes + 1U, 'x')));
  ASSERT_FALSE(added.ok());
  EXPECT_EQ(added.status().code(), StatusCode::kInvalidArgument);
}

TEST_F(MemoryServiceTest, RejectsMalformedUtf8Content) {
  const StatusOr<MemoryEntry> added = service_->Add(Persistent(std::string("\xc0\xaf", 2U)));
  ASSERT_FALSE(added.ok());
  EXPECT_EQ(added.status().code(), StatusCode::kInvalidArgument);
}

TEST_F(MemoryServiceTest, RejectsNulBytesInContent) {
  const StatusOr<MemoryEntry> added =
      service_->Add(Persistent(std::string("before\0after", 12U)));
  ASSERT_FALSE(added.ok());
  EXPECT_EQ(added.status().code(), StatusCode::kInvalidArgument);
}

TEST_F(MemoryServiceTest, RejectsMoreThanThirtyTwoTags) {
  const StatusOr<MemoryEntry> added =
      service_->Add(Persistent("note", std::vector<std::string>(33U, "tag")));
  ASSERT_FALSE(added.ok());
  EXPECT_EQ(added.status().code(), StatusCode::kInvalidArgument);
}

TEST_F(MemoryServiceTest, RejectsEmptyAndOversizedTags) {
  EXPECT_FALSE(service_->Add(Persistent("note", {""})).ok());
  EXPECT_FALSE(service_->Add(Persistent(
      "note", {std::string(kMemoryMaxTagBytes + 1U, 'x')})).ok());
}

TEST_F(MemoryServiceTest, RequiresSessionIdOnlyForSessionScopedEntries) {
  EXPECT_FALSE(service_->Add(Session("temporary", "")).ok());
  EXPECT_FALSE(service_->Add(Session("temporary", std::string(129U, 's'))).ok());
  EXPECT_FALSE(service_->Add(MemoryDraft{"persistent", {},
                                          MemoryScope::kPersistent, "chat-a"}).ok());
  MemoryDraft invalid_scope = Persistent("note");
  invalid_scope.scope = static_cast<MemoryScope>(99);
  EXPECT_FALSE(service_->Add(invalid_scope).ok());
}

TEST_F(MemoryServiceTest, FindsOnlyPersistentOrMatchingSessionEntries) {
  StatusOr<MemoryEntry> persistent = service_->Add(Persistent("shared"));
  StatusOr<MemoryEntry> private_entry = service_->Add(Session("private", "chat-a"));
  ASSERT_TRUE(persistent.ok());
  ASSERT_TRUE(private_entry.ok());

  EXPECT_TRUE(service_->Find(persistent.value().id).ok());
  EXPECT_TRUE(service_->Find(private_entry.value().id, "chat-a").ok());
  const StatusOr<MemoryEntry> hidden = service_->Find(private_entry.value().id, "chat-b");
  ASSERT_FALSE(hidden.ok());
  EXPECT_EQ(hidden.status().code(), StatusCode::kNotFound);
}

TEST_F(MemoryServiceTest, UpdatesContentTagsAndMonotonicTimestamp) {
  StatusOr<MemoryEntry> added = service_->Add(Persistent("old", {"old-tag"}));
  ASSERT_TRUE(added.ok());
  ASSERT_TRUE(service_->Update(added.value().id, {}, "new", {"new-tag"}).ok());
  StatusOr<MemoryEntry> updated = service_->Find(added.value().id);
  ASSERT_TRUE(updated.ok()) << updated.status().message();
  EXPECT_EQ(updated.value().content, "new");
  EXPECT_EQ(updated.value().tags, (std::vector<std::string>{"new-tag"}));
  EXPECT_GT(updated.value().updated_at_ms, added.value().updated_at_ms);
  EXPECT_EQ(updated.value().created_at_ms, added.value().created_at_ms);
}

TEST_F(MemoryServiceTest, CannotUpdateAnotherSessionsEntry) {
  StatusOr<MemoryEntry> added = service_->Add(Session("private", "chat-a"));
  ASSERT_TRUE(added.ok());
  const Status status = service_->Update(added.value().id, "chat-b", "changed", {});
  EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST_F(MemoryServiceTest, RemovesMemoryAndReportsMissingId) {
  StatusOr<MemoryEntry> added = service_->Add(Persistent("remove me"));
  ASSERT_TRUE(added.ok());
  ASSERT_TRUE(service_->Remove(added.value().id).ok());
  EXPECT_EQ(service_->Find(added.value().id).status().code(), StatusCode::kNotFound);
  EXPECT_EQ(service_->Remove(added.value().id).code(), StatusCode::kNotFound);
}

TEST_F(MemoryServiceTest, PinsEntriesAheadOfMoreRecentUnpinnedEntries) {
  StatusOr<MemoryEntry> older = service_->Add(Persistent("older"));
  StatusOr<MemoryEntry> newer = service_->Add(Persistent("newer"));
  ASSERT_TRUE(older.ok());
  ASSERT_TRUE(newer.ok());
  ASSERT_TRUE(service_->SetPinned(older.value().id, {}, true).ok());
  StatusOr<std::vector<MemoryEntry>> listed = service_->List();
  ASSERT_TRUE(listed.ok()) << listed.status().message();
  ASSERT_EQ(listed.value().size(), 2U);
  EXPECT_EQ(listed.value().front().id, older.value().id);
  EXPECT_TRUE(listed.value().front().pinned);
  ASSERT_TRUE(service_->SetPinned(older.value().id, {}, false).ok());
}

TEST_F(MemoryServiceTest, ListsPersistentAndOnlyTheRequestedSession) {
  ASSERT_TRUE(service_->Add(Persistent("shared")).ok());
  ASSERT_TRUE(service_->Add(Session("a-only", "chat-a")).ok());
  ASSERT_TRUE(service_->Add(Session("b-only", "chat-b")).ok());
  StatusOr<std::vector<MemoryEntry>> entries = service_->List("chat-a");
  ASSERT_TRUE(entries.ok()) << entries.status().message();
  ASSERT_EQ(entries.value().size(), 2U);
  const bool has_session_entry = std::any_of(
      entries.value().begin(), entries.value().end(), [](const MemoryEntry& entry) {
        return entry.scope == MemoryScope::kSession && entry.session_id == "chat-a";
      });
  const bool has_persistent_entry = std::any_of(
      entries.value().begin(), entries.value().end(), [](const MemoryEntry& entry) {
        return entry.scope == MemoryScope::kPersistent;
      });
  EXPECT_TRUE(has_session_entry);
  EXPECT_TRUE(has_persistent_entry);
}

TEST_F(MemoryServiceTest, RejectsZeroAndOversizedListLimits) {
  EXPECT_FALSE(service_->List({}, 0U).ok());
  EXPECT_FALSE(service_->List({}, 1001U).ok());
}

TEST_F(MemoryServiceTest, SearchesContentAndTags) {
  ASSERT_TRUE(service_->Add(Persistent("喜欢安静配色", {"favorite", "color"})).ok());
  ASSERT_TRUE(service_->Add(Persistent("unrelated note")).ok());
  StatusOr<std::vector<MemoryEntry>> content = service_->Search("安静");
  StatusOr<std::vector<MemoryEntry>> tag = service_->Search("favorite");
  ASSERT_TRUE(content.ok());
  ASSERT_TRUE(tag.ok());
  ASSERT_EQ(content.value().size(), 1U);
  ASSERT_EQ(tag.value().size(), 1U);
}

TEST_F(MemoryServiceTest, TreatsSearchTextAsLiteralData) {
  ASSERT_TRUE(service_->Add(Persistent("ordinary memory")).ok());
  StatusOr<std::vector<MemoryEntry>> results = service_->Search("%' OR 1=1 --");
  ASSERT_TRUE(results.ok()) << results.status().message();
  EXPECT_TRUE(results.value().empty());
}

TEST_F(MemoryServiceTest, RejectsEmptyOversizedAndMalformedSearchText) {
  EXPECT_FALSE(service_->Search("").ok());
  EXPECT_FALSE(service_->Search(std::string(257U, 'x')).ok());
  EXPECT_FALSE(service_->Search(std::string("\xed\xa0\x80", 3U)).ok());
}

TEST_F(MemoryServiceTest, SearchResultsRespectSessionIsolation) {
  ASSERT_TRUE(service_->Add(Session("unique term", "chat-a")).ok());
  StatusOr<std::vector<MemoryEntry>> hidden = service_->Search("unique term", "chat-b");
  StatusOr<std::vector<MemoryEntry>> visible = service_->Search("unique term", "chat-a");
  ASSERT_TRUE(hidden.ok());
  ASSERT_TRUE(visible.ok());
  EXPECT_TRUE(hidden.value().empty());
  ASSERT_EQ(visible.value().size(), 1U);
}

TEST_F(MemoryServiceTest, DisablingHidesSessionMemoriesButKeepsPersistentMemories) {
  StatusOr<MemoryEntry> persistent = service_->Add(Persistent("persistent"));
  StatusOr<MemoryEntry> session = service_->Add(Session("private", "chat-a"));
  ASSERT_TRUE(persistent.ok());
  ASSERT_TRUE(session.ok());
  ASSERT_TRUE(service_->SetSessionMemoryEnabled("chat-a", false).ok());

  StatusOr<bool> enabled = service_->IsSessionMemoryEnabled("chat-a");
  StatusOr<std::vector<MemoryEntry>> listed = service_->List("chat-a");
  ASSERT_TRUE(enabled.ok());
  ASSERT_TRUE(listed.ok());
  EXPECT_FALSE(enabled.value());
  ASSERT_EQ(listed.value().size(), 1U);
  EXPECT_EQ(listed.value().front().id, persistent.value().id);
  EXPECT_EQ(service_->Find(session.value().id, "chat-a").status().code(),
            StatusCode::kNotFound);
}

TEST_F(MemoryServiceTest, DisabledSessionCannotAddOrChangeSessionMemory) {
  StatusOr<MemoryEntry> added = service_->Add(Session("private", "chat-a"));
  ASSERT_TRUE(added.ok());
  ASSERT_TRUE(service_->SetSessionMemoryEnabled("chat-a", false).ok());
  EXPECT_EQ(service_->Add(Session("blocked", "chat-a")).status().code(),
            StatusCode::kUnavailable);
  EXPECT_EQ(service_->Update(added.value().id, "chat-a", "changed", {}).code(),
            StatusCode::kNotFound);
  EXPECT_EQ(service_->SetPinned(added.value().id, "chat-a", true).code(),
            StatusCode::kNotFound);
}

TEST_F(MemoryServiceTest, ReenablingSessionMemoryRestoresExistingEntries) {
  StatusOr<MemoryEntry> added = service_->Add(Session("private", "chat-a"));
  ASSERT_TRUE(added.ok());
  ASSERT_TRUE(service_->SetSessionMemoryEnabled("chat-a", false).ok());
  ASSERT_TRUE(service_->SetSessionMemoryEnabled("chat-a", true).ok());
  StatusOr<MemoryEntry> restored = service_->Find(added.value().id, "chat-a");
  ASSERT_TRUE(restored.ok()) << restored.status().message();
  EXPECT_EQ(restored.value().content, "private");
}

TEST_F(MemoryServiceTest, ClearingSessionLeavesOtherAndPersistentMemories) {
  StatusOr<MemoryEntry> persistent = service_->Add(Persistent("shared"));
  StatusOr<MemoryEntry> first = service_->Add(Session("first", "chat-a"));
  StatusOr<MemoryEntry> second = service_->Add(Session("second", "chat-b"));
  ASSERT_TRUE(persistent.ok());
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  ASSERT_TRUE(service_->ClearSession("chat-a").ok());
  EXPECT_TRUE(service_->Find(first.value().id, "chat-a").status().code() ==
              StatusCode::kNotFound);
  EXPECT_TRUE(service_->Find(persistent.value().id).ok());
  EXPECT_TRUE(service_->Find(second.value().id, "chat-b").ok());
}

TEST_F(MemoryServiceTest, ClearingPersistentLeavesSessionMemories) {
  ASSERT_TRUE(service_->Add(Persistent("shared")).ok());
  StatusOr<MemoryEntry> session = service_->Add(Session("temporary", "chat-a"));
  ASSERT_TRUE(session.ok());
  ASSERT_TRUE(service_->ClearPersistent().ok());
  StatusOr<std::vector<MemoryEntry>> entries = service_->List("chat-a");
  ASSERT_TRUE(entries.ok());
  ASSERT_EQ(entries.value().size(), 1U);
  EXPECT_EQ(entries.value().front().id, session.value().id);
}

TEST_F(MemoryServiceTest, ClearAllRemovesDataAndReenablesSessions) {
  ASSERT_TRUE(service_->Add(Session("temporary", "chat-a")).ok());
  ASSERT_TRUE(service_->SetSessionMemoryEnabled("chat-a", false).ok());
  ASSERT_TRUE(service_->ClearAll().ok());
  StatusOr<bool> enabled = service_->IsSessionMemoryEnabled("chat-a");
  StatusOr<std::vector<MemoryEntry>> entries = service_->List("chat-a");
  ASSERT_TRUE(enabled.ok());
  ASSERT_TRUE(entries.ok());
  EXPECT_TRUE(enabled.value());
  EXPECT_TRUE(entries.value().empty());
}

TEST_F(MemoryServiceTest, ClosingSessionDeletesEntriesAndResetsItsSwitch) {
  StatusOr<MemoryEntry> added = service_->Add(Session("temporary", "chat-a"));
  ASSERT_TRUE(added.ok());
  ASSERT_TRUE(service_->SetSessionMemoryEnabled("chat-a", false).ok());
  ASSERT_TRUE(service_->CloseSession("chat-a").ok());
  StatusOr<bool> enabled = service_->IsSessionMemoryEnabled("chat-a");
  ASSERT_TRUE(enabled.ok());
  EXPECT_TRUE(enabled.value());
  EXPECT_EQ(service_->Find(added.value().id, "chat-a").status().code(),
            StatusCode::kNotFound);
}

TEST_F(MemoryServiceTest, ExportsStructuredJsonForOnlyTheRequestedSession) {
  ASSERT_TRUE(service_->Add(Persistent("shared", {"tag"})).ok());
  ASSERT_TRUE(service_->Add(Session("private-a", "chat-a")).ok());
  ASSERT_TRUE(service_->Add(Session("private-b", "chat-b")).ok());
  StatusOr<std::string> exported = service_->ExportJson("chat-a");
  ASSERT_TRUE(exported.ok()) << exported.status().message();
  const nlohmann::json parsed = nlohmann::json::parse(exported.value());
  ASSERT_EQ(parsed["version"], 1);
  ASSERT_EQ(parsed["memories"].size(), 2U);
  const std::string serialized = parsed.dump();
  EXPECT_NE(serialized.find("shared"), std::string::npos);
  EXPECT_NE(serialized.find("private-a"), std::string::npos);
  EXPECT_EQ(serialized.find("private-b"), std::string::npos);
}

TEST_F(MemoryServiceTest, PreservesPersistentMemoryAcrossServiceReopen) {
  StatusOr<MemoryEntry> added = service_->Add(Persistent("survives restart", {"durable"}));
  ASSERT_TRUE(added.ok());
  const std::string id = added.value().id;
  service_.reset();

  StatusOr<std::unique_ptr<MemoryService>> reopened = MemoryService::Open(path_);
  ASSERT_TRUE(reopened.ok()) << reopened.status().message();
  service_ = std::move(reopened).value();
  StatusOr<MemoryEntry> loaded = service_->Find(id);
  ASSERT_TRUE(loaded.ok()) << loaded.status().message();
  EXPECT_EQ(loaded.value().content, "survives restart");
  EXPECT_EQ(loaded.value().tags, (std::vector<std::string>{"durable"}));
}

TEST_F(MemoryServiceTest, SerializesConcurrentAddsSafely) {
  constexpr int kThreadCount = 8;
  constexpr int kEntriesPerThread = 16;
  std::atomic<int> successful_adds{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (int thread = 0; thread < kThreadCount; ++thread) {
    workers.emplace_back([&, thread] {
      for (int entry = 0; entry < kEntriesPerThread; ++entry) {
        const StatusOr<MemoryEntry> added = service_->Add(
            Persistent("thread-" + std::to_string(thread) + "-" +
                       std::to_string(entry)));
        if (added.ok()) successful_adds.fetch_add(1);
      }
    });
  }
  for (std::thread& worker : workers) worker.join();

  EXPECT_EQ(successful_adds.load(), kThreadCount * kEntriesPerThread);
  StatusOr<std::vector<MemoryEntry>> entries = service_->List({}, 1000U);
  ASSERT_TRUE(entries.ok()) << entries.status().message();
  EXPECT_EQ(entries.value().size(),
            static_cast<std::size_t>(kThreadCount * kEntriesPerThread));
}

TEST_F(MemoryServiceTest, RejectsInvalidMemoryIdBeforeQueryingDatabase) {
  const StatusOr<MemoryEntry> found = service_->Find("invalid");
  ASSERT_FALSE(found.ok());
  EXPECT_EQ(found.status().code(), StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace isvik
