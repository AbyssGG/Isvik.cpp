#ifndef ISVIK_APP_NATIVE_DIALOG_H_
#define ISVIK_APP_NATIVE_DIALOG_H_
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include "isvik/core/status.h"
namespace isvik::app {
struct ApiKeyEntry {
  std::string name;
  std::string value;
};
[[nodiscard]] StatusOr<std::filesystem::path> ChooseModelFolder(
    const std::filesystem::path& initial_folder);
[[nodiscard]] std::filesystem::path CatalogStoragePath();
[[nodiscard]] std::filesystem::path MemoryStoragePath();
[[nodiscard]] std::filesystem::path ConversationStoragePath();
[[nodiscard]] StatusOr<std::vector<ApiKeyEntry>> LoadApiKeyVault();
[[nodiscard]] Status SaveApiKey(std::string_view name, std::string_view value);
[[nodiscard]] Status DeleteApiKey(std::string_view name);
}  // namespace isvik::app
#endif
