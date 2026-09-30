#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <gtest/gtest.h>
#include "isvik/core/model_catalog.h"

namespace isvik {
namespace {
class CatalogTest : public testing::Test {
 protected:
  void SetUp() override {
    root = std::filesystem::temp_directory_path() /
        ("isvik-catalog-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
  }
  void TearDown() override { std::error_code error; std::filesystem::remove_all(root, error); }
  void Write(const std::filesystem::path& path, const std::string& text = "<net/>") {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
  }
  std::filesystem::path Model(const std::string& name, bool vlm = false, bool tokenizer = true) {
    auto directory = root / name;
    Write(directory / (vlm ? "openvino_language_model.xml" : "openvino_model.xml"));
    Write(directory / (vlm ? "openvino_language_model.bin" : "openvino_model.bin"), "weights");
    if (vlm) { Write(directory / "openvino_text_embeddings_model.xml"); Write(directory / "openvino_text_embeddings_model.bin", "embedding-weights"); }
    if (tokenizer) for (const char* file : {"openvino_tokenizer.xml", "openvino_tokenizer.bin", "openvino_detokenizer.xml", "openvino_detokenizer.bin"}) Write(directory / file);
    Write(directory / "config.json", "{\"model_type\":\"gemma4\",\"text_config\":{\"max_position_embeddings\":131072}}");
    return directory;
  }
  std::filesystem::path root;
};

TEST_F(CatalogTest, RecursiveScanGroupsAuxiliaryFilesIntoOneModel) {
  const auto directory = Model("nested/model");
  Write(directory / "openvino_vision_embeddings_model.xml");
  Write(directory / "openvino_vision_embeddings_model.bin");
  const auto scanned = ScanModelDirectory(root);
  ASSERT_TRUE(scanned.ok());
  ASSERT_EQ(scanned.value().models.size(), 1);
  const auto& entry = scanned.value().models.front();
  EXPECT_TRUE(entry.runnable);
  EXPECT_EQ(entry.model.path.filename(), "openvino_model.xml");
  EXPECT_EQ(entry.model.display_name, "model");
  EXPECT_EQ(entry.model.architecture, "gemma4");
  EXPECT_EQ(entry.model.context_length, 131072);
}
TEST_F(CatalogTest, VlmScanUsesLanguageEntryAndCountsEmbeddingWeights) {
  const auto directory = Model("vlm", true);
  Write(directory / "openvino_model.xml");
  Write(directory / "openvino_model.bin");
  const auto scanned = ScanModelDirectory(root);
  ASSERT_TRUE(scanned.ok());
  ASSERT_EQ(scanned.value().models.size(), 1);
  EXPECT_EQ(scanned.value().models.front().model.path.filename(), "openvino_language_model.xml");
  EXPECT_GT(scanned.value().models.front().model.file_size, 20);
}
TEST_F(CatalogTest, TokenizerSelectionResolvesToPrincipalModel) {
  const auto directory = Model("vlm", true);
  const auto resolved = ResolveModelEntry(directory / "openvino_tokenizer.xml");
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(resolved.value(), std::filesystem::canonical(directory / "openvino_language_model.xml"));
  EXPECT_EQ(ResolveModelEntry(directory).value(), resolved.value());
}
TEST_F(CatalogTest, AuxiliaryFilesAloneNeverBecomeModels) {
  Write(root / "openvino_tokenizer.xml"); Write(root / "openvino_tokenizer.bin");
  const auto scanned = ScanModelDirectory(root);
  ASSERT_TRUE(scanned.ok()); EXPECT_TRUE(scanned.value().models.empty());
  EXPECT_FALSE(ResolveModelEntry(root / "openvino_tokenizer.xml").ok());
}
TEST_F(CatalogTest, MissingTokenizerKeepsModelVisibleButDisablesLoading) {
  Model("incomplete", false, false);
  const auto scanned = ScanModelDirectory(root);
  ASSERT_TRUE(scanned.ok()); ASSERT_EQ(scanned.value().models.size(), 1);
  EXPECT_TRUE(scanned.value().models.front().available);
  EXPECT_FALSE(scanned.value().models.front().runnable);
  EXPECT_FALSE(scanned.value().models.front().issue.empty());
}
TEST_F(CatalogTest, MissingWeightsIsReportedInCatalog) {
  Write(root / "model/openvino_model.xml");
  const auto scanned = ScanModelDirectory(root);
  ASSERT_TRUE(scanned.ok()); ASSERT_EQ(scanned.value().models.size(), 1);
  EXPECT_FALSE(scanned.value().models.front().available);
}
TEST_F(CatalogTest, InvalidGgufDoesNotAbortOtherModelDiscovery) {
  Model("valid"); Write(root / "broken.gguf", "invalid");
  const auto scanned = ScanModelDirectory(root);
  ASSERT_TRUE(scanned.ok()); ASSERT_EQ(scanned.value().models.size(), 2);
  EXPECT_EQ(std::count_if(scanned.value().models.begin(), scanned.value().models.end(), [](const auto& entry) { return entry.runnable; }), 1);
}
TEST_F(CatalogTest, RepeatedScansKeepIdsAndUserChosenNames) {
  Model("model");
  ModelCatalog catalog(root / "catalog.json");
  const auto scanned = ScanModelDirectory(root);
  ASSERT_TRUE(scanned.ok()); catalog.Merge(scanned.value());
  const auto id = catalog.models().front().model.id;
  ASSERT_TRUE(catalog.Rename(id, "  自定义模型  ").ok());
  catalog.Merge(scanned.value());
  ASSERT_EQ(catalog.models().size(), 1);
  EXPECT_EQ(catalog.models().front().model.id, id);
  EXPECT_EQ(catalog.models().front().model.display_name, "自定义模型");
}
TEST_F(CatalogTest, PersistedCatalogRestoresFolderRenameAndDefault) {
  Model("model");
  const auto storage = root / "settings/models.json";
  ModelCatalog catalog(storage);
  catalog.Merge(ScanModelDirectory(root).value()); catalog.set_folder(root);
  const auto id = catalog.models().front().model.id;
  ASSERT_TRUE(catalog.Rename(id, "中文名称").ok()); ASSERT_TRUE(catalog.SetDefault(id).ok());
  ASSERT_TRUE(catalog.Save().ok());
  // Replace the existing file as well as creating it for the first time.
  ASSERT_TRUE(catalog.Save().ok());
  ModelCatalog restored(storage); ASSERT_TRUE(restored.Load().ok());
  EXPECT_EQ(restored.folder(), root); EXPECT_EQ(restored.default_id(), id);
  ASSERT_NE(restored.Find(id), nullptr);
  EXPECT_EQ(restored.Find(id)->model.display_name, "中文名称"); EXPECT_TRUE(restored.Find(id)->runnable);
}
TEST_F(CatalogTest, RemovingDefaultOnlyRemovesTheReference) {
  const auto directory = Model("model"); ModelCatalog catalog(root / "models.json");
  catalog.Merge(ScanModelDirectory(root).value()); const auto id = catalog.models().front().model.id;
  ASSERT_TRUE(catalog.SetDefault(id).ok()); ASSERT_TRUE(catalog.Remove(id).ok());
  EXPECT_TRUE(catalog.default_id().empty()); EXPECT_TRUE(catalog.models().empty());
  EXPECT_TRUE(std::filesystem::exists(directory / "openvino_model.bin"));
  EXPECT_EQ(catalog.Remove(id).code(), StatusCode::kNotFound);
}
TEST_F(CatalogTest, MovedModelRemainsVisibleAfterRestart) {
  const auto directory = Model("model"); const auto storage = root / "models.json";
  ModelCatalog catalog(storage); catalog.Merge(ScanModelDirectory(root).value()); ASSERT_TRUE(catalog.Save().ok());
  std::filesystem::rename(directory, root / "moved");
  ModelCatalog restored(storage); ASSERT_TRUE(restored.Load().ok());
  ASSERT_EQ(restored.models().size(), 1); EXPECT_FALSE(restored.models().front().available);
}
TEST_F(CatalogTest, RescanRefreshesMissingExistingEntries) {
  const auto directory = Model("model"); ModelCatalog catalog(root / "models.json");
  catalog.Merge(ScanModelDirectory(root).value()); std::filesystem::remove(directory / "openvino_model.bin");
  catalog.Merge(ScanModelDirectory(root).value()); ASSERT_EQ(catalog.models().size(), 1);
  EXPECT_FALSE(catalog.models().front().available); EXPECT_FALSE(catalog.models().front().runnable);
}
TEST_F(CatalogTest, CorruptCatalogDoesNotReplacePreviouslyLoadedState) {
  Model("model"); const auto storage = root / "models.json"; ModelCatalog catalog(storage);
  catalog.Merge(ScanModelDirectory(root).value()); ASSERT_TRUE(catalog.Save().ok());
  Write(storage, "{invalid"); EXPECT_FALSE(catalog.Load().ok());
  EXPECT_EQ(catalog.models().size(), 1);
  std::ifstream input(storage); std::string actual; input >> actual; EXPECT_EQ(actual, "{invalid");
}
TEST_F(CatalogTest, RejectsInvalidNamesAndUnknownDefaultIds) {
  Model("model"); ModelCatalog catalog(root / "models.json"); catalog.Merge(ScanModelDirectory(root).value());
  const auto id = catalog.models().front().model.id;
  EXPECT_FALSE(catalog.Rename(id, "   ").ok()); EXPECT_FALSE(catalog.Rename(id, std::string(257, 'x')).ok());
  EXPECT_EQ(catalog.SetDefault("missing").code(), StatusCode::kNotFound);
}
TEST_F(CatalogTest, Utf8DirectoryNamesRoundTripThroughStorage) {
  const auto directory = root / PathFromUtf8("中文模型");
  Write(directory / "openvino_model.xml"); Write(directory / "openvino_model.bin");
  const auto storage = root / "models.json"; ModelCatalog catalog(storage);
  catalog.Merge(ScanModelDirectory(root).value()); ASSERT_TRUE(catalog.Save().ok());
  ModelCatalog restored(storage); ASSERT_TRUE(restored.Load().ok());
  ASSERT_EQ(restored.models().size(), 1); EXPECT_EQ(restored.models().front().model.display_name, "中文模型");
  EXPECT_EQ(PathFromUtf8(PathUtf8(directory)), directory);
}
TEST_F(CatalogTest, SelectedBackendRoundTripsThroughStorage) {
  const auto storage = root / "models.json";
  ModelCatalog catalog(storage);
  catalog.set_backend(BackendType::kTensorRt);
  ASSERT_TRUE(catalog.Save().ok());
  ModelCatalog restored(storage);
  ASSERT_TRUE(restored.Load().ok());
  EXPECT_EQ(restored.backend(), BackendType::kTensorRt);
}
}  // namespace
}  // namespace isvik
