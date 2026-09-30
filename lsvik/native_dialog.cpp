#include "native_dialog.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <string_view>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <wincred.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <wrl/client.h>
#endif

namespace isvik::app {
namespace {
#ifdef _WIN32
constexpr std::wstring_view kApiKeyTargetPrefix = L"Isvik.ApiKey.";

StatusOr<std::wstring> ToWide(std::string_view value) {
  if (value.empty()) return std::wstring{};
  if (value.size() > static_cast<size_t>(INT_MAX))
    return Status::InvalidArgument("Text is too long for Windows Credential Manager");
  const int input_size = static_cast<int>(value.size());
  const int required = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
      value.data(), input_size, nullptr, 0);
  if (required <= 0) return Status::InvalidArgument("Text must be valid UTF-8");
  std::wstring result(static_cast<size_t>(required), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), input_size,
                          result.data(), required) != required)
    return Status::InvalidArgument("Text must be valid UTF-8");
  return result;
}

std::string FromWide(std::wstring_view value) {
  if (value.empty()) return {};
  const int input_size = static_cast<int>(value.size());
  const int required = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
      value.data(), input_size, nullptr, 0, nullptr, nullptr);
  if (required <= 0) return {};
  std::string result(static_cast<size_t>(required), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), input_size,
                          result.data(), required, nullptr, nullptr) != required)
    return {};
  return result;
}

std::wstring ApiKeyTarget(std::string_view name) {
  constexpr wchar_t kHex[] = L"0123456789abcdef";
  std::wstring target(kApiKeyTargetPrefix);
  target.reserve(target.size() + name.size() * 2U);
  for (const unsigned char value : name) {
    target.push_back(kHex[value >> 4U]);
    target.push_back(kHex[value & 0x0fU]);
  }
  return target;
}

Status CredentialError(std::string operation, DWORD error) {
  return Status::Unavailable(std::move(operation) +
      " (Windows error " + std::to_string(error) + ")");
}
#endif
}  // namespace

std::filesystem::path CatalogStoragePath() {
#ifdef _WIN32
  PWSTR folder = nullptr;
  if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &folder))) {
    const std::filesystem::path path(folder);
    CoTaskMemFree(folder);
    return path / L"Isvik" / L"models.json";
  }
#else
  if (const char* folder = std::getenv("XDG_DATA_HOME")) {
    return std::filesystem::path(folder) / "isvik" / "models.json";
  }
  if (const char* folder = std::getenv("HOME")) {
    return std::filesystem::path(folder) / ".local/share/isvik/models.json";
  }
#endif
  return std::filesystem::temp_directory_path() / "Isvik" / "models.json";
}

std::filesystem::path MemoryStoragePath() {
  return CatalogStoragePath().parent_path() / "memory.sqlite3";
}

std::filesystem::path ConversationStoragePath() {
  return CatalogStoragePath().parent_path() / "conversations.json";
}

StatusOr<std::vector<ApiKeyEntry>> LoadApiKeyVault() {
#ifdef _WIN32
  const std::wstring filter = std::wstring(kApiKeyTargetPrefix) + L"*";
  DWORD count = 0;
  PCREDENTIALW* credentials = nullptr;
  if (!CredEnumerateW(filter.c_str(), 0, &count, &credentials)) {
    const DWORD error = GetLastError();
    if (error == ERROR_NOT_FOUND) return std::vector<ApiKeyEntry>{};
    return CredentialError("Could not read API keys from Windows Credential Manager", error);
  }
  struct FreeCredentials {
    PCREDENTIALW* value;
    ~FreeCredentials() { if (value != nullptr) CredFree(value); }
  } free_credentials{credentials};

  std::vector<ApiKeyEntry> result;
  for (DWORD index = 0; index < count; ++index) {
    const CREDENTIALW& credential = *credentials[index];
    if (credential.Type != CRED_TYPE_GENERIC || credential.TargetName == nullptr ||
        credential.UserName == nullptr ||
        credential.CredentialBlob == nullptr ||
        !std::wstring_view(credential.TargetName).starts_with(kApiKeyTargetPrefix)) continue;
    const std::wstring_view name(credential.UserName);
    if (name.empty()) continue;
    const auto* bytes = reinterpret_cast<const char*>(credential.CredentialBlob);
    result.push_back({FromWide(name), std::string(bytes, credential.CredentialBlobSize)});
  }
  std::sort(result.begin(), result.end(), [](const ApiKeyEntry& left, const ApiKeyEntry& right) {
    return left.name < right.name;
  });
  return result;
#else
  return Status::Unsupported("The API key vault requires Windows Credential Manager");
#endif
}

Status SaveApiKey(std::string_view name, std::string_view value) {
#ifdef _WIN32
  if (name.empty() || name.size() > 100U)
    return Status::InvalidArgument("Key names must contain 1 to 100 UTF-8 bytes");
  if (value.empty()) return Status::InvalidArgument("API key cannot be empty");
  if (value.size() > CRED_MAX_CREDENTIAL_BLOB_SIZE)
    return Status::InvalidArgument("API key is too long for Windows Credential Manager");
  if (name.find('\0') != std::string_view::npos || value.find('\0') != std::string_view::npos)
    return Status::InvalidArgument("Key names and API keys cannot contain NUL characters");
  auto wide_name = ToWide(name);
  if (!wide_name.ok()) return wide_name.status();
  const std::wstring target = ApiKeyTarget(name);
  std::wstring comment = L"Isvik API key vault entry";
  CREDENTIALW credential{};
  credential.Type = CRED_TYPE_GENERIC;
  credential.TargetName = const_cast<LPWSTR>(target.c_str());
  credential.Comment = comment.data();
  credential.CredentialBlobSize = static_cast<DWORD>(value.size());
  credential.CredentialBlob = reinterpret_cast<LPBYTE>(const_cast<char*>(value.data()));
  credential.Persist = CRED_PERSIST_LOCAL_MACHINE;
  credential.UserName = wide_name.value().data();
  if (!CredWriteW(&credential, 0))
    return CredentialError("Could not save API key in Windows Credential Manager", GetLastError());
  return {};
#else
  (void)name;
  (void)value;
  return Status::Unsupported("The API key vault requires Windows Credential Manager");
#endif
}

Status DeleteApiKey(std::string_view name) {
#ifdef _WIN32
  if (name.empty() || name.size() > 100U)
    return Status::InvalidArgument("Key names must contain 1 to 100 UTF-8 bytes");
  if (name.find('\0') != std::string_view::npos)
    return Status::InvalidArgument("Key names cannot contain NUL characters");
  const std::wstring target = ApiKeyTarget(name);
  if (!CredDeleteW(target.c_str(), CRED_TYPE_GENERIC, 0)) {
    const DWORD error = GetLastError();
    if (error == ERROR_NOT_FOUND) return Status::NotFound("API key was not found");
    return CredentialError("Could not delete API key from Windows Credential Manager", error);
  }
  return {};
#else
  (void)name;
  return Status::Unsupported("The API key vault requires Windows Credential Manager");
#endif
}

StatusOr<std::filesystem::path> ChooseModelFolder(const std::filesystem::path& initial_folder) {
#ifdef _WIN32
  const HRESULT initialized = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  if (FAILED(initialized)) return Status::Unavailable("Cannot initialize folder dialog");
  struct ComSession { ~ComSession() { CoUninitialize(); } } session;
  Microsoft::WRL::ComPtr<IFileOpenDialog> dialog;
  HRESULT result = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                    IID_PPV_ARGS(dialog.GetAddressOf()));
  if (FAILED(result)) return Status::Unavailable("Cannot open Windows folder picker");
  DWORD options = 0;
  result = dialog->GetOptions(&options);
  if (SUCCEEDED(result)) result = dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR);
  if (FAILED(result)) return Status::Unavailable("Cannot configure folder picker");
  dialog->SetTitle(L"选择模型目录 / Choose model directory");
  if (!initial_folder.empty()) {
    Microsoft::WRL::ComPtr<IShellItem> initial;
    if (SUCCEEDED(SHCreateItemFromParsingName(initial_folder.c_str(), nullptr,
                                             IID_PPV_ARGS(initial.GetAddressOf())))) dialog->SetFolder(initial.Get());
  }
  result = dialog->Show(GetActiveWindow());
  if (result == HRESULT_FROM_WIN32(ERROR_CANCELLED)) return Status::Cancelled("Folder selection cancelled");
  if (FAILED(result)) return Status::Unavailable("Folder selection failed");
  Microsoft::WRL::ComPtr<IShellItem> selected;
  result = dialog->GetResult(selected.GetAddressOf());
  if (FAILED(result)) return Status::Unavailable("Cannot read selected folder");
  PWSTR path = nullptr;
  result = selected->GetDisplayName(SIGDN_FILESYSPATH, &path);
  if (FAILED(result)) return Status::Unavailable("Selected folder has no filesystem path");
  const std::filesystem::path directory(path);
  CoTaskMemFree(path);
  return directory;
#else
  (void)initial_folder;
  return Status::Unsupported("Enter a directory path in the model library on this platform");
#endif
}
}  // namespace isvik::app
