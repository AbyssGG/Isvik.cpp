#include "cli_app.h"
#include "native_dialog.h"
#include "isvik/app/api_server.h"
#include "isvik/i18n/localization.h"
#include "isvik/core/model_catalog.h"
#include "isvik/core/chat_context.h"
#include "isvik/core/memory_service.h"
#include "main_window.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#endif
#if defined(ISVIK_HAS_OPENVINO_RUNTIME)
#include "isvik/backends/openvino/device_provider.h"
#endif
#if defined(ISVIK_HAS_OPENVINO_GENAI)
#include "isvik/backends/openvino/genai_engine.h"
#endif
#if defined(ISVIK_HAS_TENSORRT)
#include "isvik/backends/tensorrt/engine.h"
#include "isvik/backends/tensorrt/gguf_genai_engine.h"
#endif
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
#include "isvik/backends/onnxruntime/genai_engine.h"
#endif

namespace {
using Window = slint::ComponentHandle<MainWindow>;
using WeakWindow = slint::ComponentWeakHandle<MainWindow>;
std::string Text(const slint::SharedString& value) { return {value.data(), value.size()}; }
slint::SharedString Shared(const std::string& value) { return slint::SharedString(value); }
bool Chinese(const isvik::LocalizationService& language) { return language.language() == isvik::Language::kChineseSimplified; }
std::string Choose(bool chinese, const char* zh, const char* en) { return chinese ? zh : en; }
using Json = nlohmann::json;
struct ConversationRecord {
  std::string id;
  std::string title;
  int64_t updated_at_ms = 0;
  std::vector<isvik::InferenceMessage> context;
  std::vector<ChatMessage> messages;
};
bool IsSupportedDecoding(int index) {
  return index == 0 || index == 1 || index == 2 || index == 3 || index == 6 || index == 8;
}
std::string DecodingInfo(int index, const std::string& value) {
  switch (index) {
    case 0: return "温度采样 / Temperature Sampling · temperature=" + value;
    case 1: return "束搜索 / Beam Search · beams=" + value;
    case 2: return "Top-k 采样 / Top-k Sampling · k=" + value;
    case 3: return "Top-p 核采样 / Top-p / Nucleus Sampling · p=" + value;
    case 6: return "多样化束搜索 / Diverse Beam Search · beams=" + value + " · groups=2 · diversity=0.5";
    case 8: return "贪心解码 / Greedy Decoding";
    default: return "暂不可用 / Unavailable";
  }
}
bool ParseFloat(std::string_view text, float& value) {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value,
                                      std::chars_format::general);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
         std::isfinite(value);
}
int EstimatePromptLines(std::string_view text) {
  constexpr double kWrapWidth = 680.0;
  int lines = 1;
  double line_width = 0.0;
  for (std::size_t index = 0; index < text.size(); ++index) {
    const auto byte = static_cast<unsigned char>(text[index]);
    if (byte == '\n') {
      ++lines;
      line_width = 0.0;
      continue;
    }
    if (byte == '\r' || (byte & 0xC0U) == 0x80U) continue;
    const double glyph_width = byte == '\t' ? 32.0 : byte == ' ' ? 4.0 : byte < 0x80U ? 8.0 : 16.0;
    if (line_width > 0.0 && line_width + glyph_width > kWrapWidth) {
      ++lines;
      line_width = 0.0;
    }
    line_width += glyph_width;
  }
  return std::clamp(lines, 1, 8);
}
std::string ArchitectureLabel(std::string architecture) {
  std::transform(architecture.begin(), architecture.end(), architecture.begin(), [](unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  if (architecture == "gemma4") return "gemma-4";
  if (architecture == "gpt_oss") return "gpt-oss";
  return architecture;
}
void Log(const char* level, std::string_view message) {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
#ifdef _WIN32
  localtime_s(&local, &time);
#else
  localtime_r(&time, &local);
#endif
  std::cout << '[' << std::put_time(&local, "%Y-%m-%d %H:%M:%S") << "] ["
            << level << "] " << message << std::endl;
}
std::string Lower(std::string text) {
  for (char& value : text) value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
  return text;
}
std::string GenerateApiKey() {
  std::array<unsigned char, 32> random_bytes{};
#ifdef _WIN32
  if (BCryptGenRandom(nullptr, random_bytes.data(),
                      static_cast<ULONG>(random_bytes.size()),
                      BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0) {
    return {};
  }
#else
  try {
    std::random_device random;
    for (auto& value : random_bytes) value = static_cast<unsigned char>(random());
  } catch (...) {
    return {};
  }
#endif
  constexpr char kHex[] = "0123456789abcdef";
  std::string key = "isvik_";
  key.reserve(6U + random_bytes.size() * 2U);
  for (const unsigned char value : random_bytes) {
    key.push_back(kHex[value >> 4U]);
    key.push_back(kHex[value & 0x0fU]);
  }
  return key;
}
struct GuiState {
  isvik::ModelCatalog catalog{isvik::app::CatalogStoragePath()};
  bool catalog_writable = true;
  std::unique_ptr<isvik::MemoryService> memories;
  std::string memory_session;
  bool use_memories = true;
  bool session_memory_enabled = true;
  uint64_t context_limit = 8192U;
  std::filesystem::path conversation_path;
  std::vector<ConversationRecord> conversation_history;
  std::string active_conversation_id;
  std::vector<isvik::InferenceMessage> conversation;
  std::shared_ptr<slint::VectorModel<ChatMessage>> chat_messages;
  std::optional<std::size_t> latest_assistant_row;
  bool can_remember_exchange = false;
  bool exchange_saved = false;
#if defined(ISVIK_HAS_OPENVINO_GENAI)
  isvik::openvino_backend::GenAiEngine engine;
#endif
#if defined(ISVIK_HAS_TENSORRT)
  isvik::tensorrt_backend::GgufGenAiEngine tensorrt_engine{
      [](std::string_view message) { Log("TensorRT", message); }};
#endif
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
  isvik::onnxruntime_backend::GenAiEngine onnxruntime_engine;
#endif
  std::optional<isvik::ModelDescriptor> loaded_model;
  std::optional<isvik::BackendType> loaded_backend;
  isvik::CancellationSource cancellation;
  bool busy = false;
  std::string selected_id;
  std::vector<std::string> chat_ids;
  std::jthread worker;
  isvik::app::ApiServer api_server;
  std::jthread api_worker;
  bool api_server_transition = false;
};
void Status(const Window& window, std::string message, std::string detail = {}) {
  if (!message.empty()) Log(detail.empty() ? "INFO" : "ERROR", detail.empty() ? message : message + " · " + detail);
  window->set_status_message(Shared(message));
  window->set_status_error(!detail.empty());
  window->set_error_detail(Shared(detail));
  window->set_show_details(false);
}
std::string Metadata(const isvik::CatalogModel& entry) {
  std::ostringstream text;
  text << (entry.model.architecture.empty()
      ? (entry.model.format == isvik::ModelFormat::kGguf ? "GGUF"
          : entry.model.format == isvik::ModelFormat::kOnnx ? "ONNX" : "OpenVINO")
      : ArchitectureLabel(entry.model.architecture));
  if (!entry.model.parameter_label.empty()) text << " · " << entry.model.parameter_label;
  if (!entry.model.quantization.empty()) text << " · " << entry.model.quantization;
  if (!entry.model.variant.empty()) text << " · " << entry.model.variant;
  if (entry.model.context_length > 0U) text << " · " << entry.model.context_length << " ctx";
  text << " · " << std::fixed << std::setprecision(2)
       << static_cast<double>(entry.model.file_size) / (1024.0 * 1024.0 * 1024.0) << " GiB";
  return text.str();
}
std::string Metadata(const isvik::ModelDescriptor& model, std::string_view device = {}) {
  std::ostringstream text;
  text << (model.architecture.empty()
      ? (model.format == isvik::ModelFormat::kGguf ? "GGUF"
          : model.format == isvik::ModelFormat::kOnnx ? "ONNX" : "OpenVINO")
      : ArchitectureLabel(model.architecture));
  if (!model.parameter_label.empty()) text << " · " << model.parameter_label;
  if (!model.quantization.empty()) text << " · " << model.quantization;
  if (!model.variant.empty()) text << " · " << model.variant;
  if (model.context_length > 0U) text << " · " << model.context_length << " ctx";
  if (!device.empty()) text << " · " << device;
  return text.str();
}
std::string Statistics(const isvik::InferenceCompleted& metrics) {
  std::ostringstream text;
  text << metrics.output_tokens << " tokens · " << std::fixed << std::setprecision(2)
       << metrics.duration_seconds << " s · " << std::setprecision(2)
       << metrics.tokens_per_second << " tokens/s";
  return text.str();
}
ChatMessage Message(std::string_view role, std::string content = {},
                    std::string model_info = {}, std::string request_info = {},
                    std::string statistics = {}, bool latest = false,
                    bool saved = false) {
  ChatMessage item;
  item.role = Shared(std::string(role));
  item.content = Shared(content);
  item.model_info = Shared(model_info);
  item.request_info = Shared(request_info);
  item.statistics = Shared(statistics);
  item.is_latest = latest;
  item.saved = saved;
  return item;
}
const char* RoleName(isvik::MessageRole role) {
  switch (role) {
    case isvik::MessageRole::kSystem: return "system";
    case isvik::MessageRole::kUser: return "user";
    case isvik::MessageRole::kAssistant: return "assistant";
    case isvik::MessageRole::kTool: return "tool";
  }
  return "user";
}
std::optional<isvik::MessageRole> ParseRole(std::string_view role) {
  if (role == "system") return isvik::MessageRole::kSystem;
  if (role == "user") return isvik::MessageRole::kUser;
  if (role == "assistant") return isvik::MessageRole::kAssistant;
  if (role == "tool") return isvik::MessageRole::kTool;
  return std::nullopt;
}
std::string ConversationTitle(std::string_view prompt) {
  const std::size_t first = prompt.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return "新聊天 / New chat";
  prompt.remove_prefix(first);
  const std::size_t line_end = prompt.find_first_of("\r\n");
  if (line_end != std::string_view::npos) prompt = prompt.substr(0, line_end);
  constexpr std::size_t kTitleCharacters = 36U;
  std::size_t byte_end = 0U;
  std::size_t characters = 0U;
  while (byte_end < prompt.size()) {
    const auto byte = static_cast<unsigned char>(prompt[byte_end]);
    if ((byte & 0xC0U) != 0x80U) {
      if (characters == kTitleCharacters) return std::string(prompt.substr(0U, byte_end)) + "…";
      ++characters;
    }
    ++byte_end;
  }
  return std::string(prompt);
}
int64_t NowMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
isvik::Status LoadConversationHistory(GuiState& state) {
  std::error_code error;
  if (!std::filesystem::exists(state.conversation_path, error)) {
    if (error) return isvik::Status::Unavailable("cannot inspect conversation history: " + error.message());
    return {};
  }
  const auto file_size = std::filesystem::file_size(state.conversation_path, error);
  if (error) return isvik::Status::Unavailable("cannot read conversation history size: " + error.message());
  if (file_size > 64U * 1024U * 1024U) {
    return isvik::Status::InvalidArgument("conversation history is larger than 64 MiB");
  }
  std::ifstream input(state.conversation_path, std::ios::binary);
  if (!input) return isvik::Status::Unavailable("cannot open conversation history");
  Json document = Json::parse(input, nullptr, false);
  if (document.is_discarded() || !document.is_object() ||
      document.value("version", 0) != 1 || !document.contains("conversations") ||
      !document["conversations"].is_array()) {
    return isvik::Status::InvalidArgument("conversation history has an invalid format");
  }
  std::vector<ConversationRecord> records;
  for (const Json& saved : document["conversations"]) {
    if (!saved.is_object() || records.size() >= 100U) continue;
    try {
      ConversationRecord record;
      record.id = saved.value("id", std::string{});
      record.title = saved.value("title", std::string{});
      record.updated_at_ms = saved.value("updated_at_ms", int64_t{0});
      if (record.id.empty() || !saved.contains("context") || !saved["context"].is_array() ||
          !saved.contains("messages") || !saved["messages"].is_array()) continue;
      for (const Json& row : saved["context"]) {
        if (!row.is_object()) continue;
        const auto role = ParseRole(row.value("role", std::string{}));
        if (!role) continue;
        record.context.push_back({*role, row.value("content", std::string{}),
            row.value("name", std::string{}), row.value("tool_call_id", std::string{})});
      }
      for (const Json& row : saved["messages"]) {
        if (!row.is_object()) continue;
        const std::string role = row.value("role", std::string{});
        if (role != "user" && role != "assistant") continue;
        record.messages.push_back(Message(role, row.value("content", std::string{}),
            row.value("model_info", std::string{}), row.value("request_info", std::string{}),
            row.value("statistics", std::string{}), false, row.value("saved", false)));
      }
      if (!record.context.empty() && !record.messages.empty()) records.push_back(std::move(record));
    } catch (const Json::exception&) {
      continue;
    }
  }
  std::sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
    return left.updated_at_ms > right.updated_at_ms;
  });
  state.conversation_history = std::move(records);
  return {};
}
isvik::Status WriteConversationHistory(const GuiState& state) {
  std::error_code error;
  std::filesystem::create_directories(state.conversation_path.parent_path(), error);
  if (error) return isvik::Status::Unavailable("cannot create conversation history directory: " + error.message());
  Json conversations = Json::array();
  for (const ConversationRecord& record : state.conversation_history) {
    Json context = Json::array();
    for (const auto& message : record.context) {
      context.push_back({{"role", RoleName(message.role)}, {"content", message.content},
          {"name", message.name}, {"tool_call_id", message.tool_call_id}});
    }
    Json messages = Json::array();
    for (const ChatMessage& message : record.messages) {
      messages.push_back({{"role", Text(message.role)}, {"content", Text(message.content)},
          {"model_info", Text(message.model_info)}, {"request_info", Text(message.request_info)},
          {"statistics", Text(message.statistics)}, {"saved", message.saved}});
    }
    conversations.push_back({{"id", record.id}, {"title", record.title},
        {"updated_at_ms", record.updated_at_ms}, {"context", std::move(context)},
        {"messages", std::move(messages)}});
  }
  auto temporary_path = state.conversation_path;
  temporary_path += ".tmp";
  {
    std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
    if (!output) return isvik::Status::Unavailable("cannot write conversation history");
    output << Json{{"version", 1}, {"conversations", std::move(conversations)}}.dump(2);
    output.flush();
    if (!output) return isvik::Status::Unavailable("cannot flush conversation history");
  }
#ifdef _WIN32
  if (!MoveFileExW(temporary_path.c_str(),
                   state.conversation_path.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    std::filesystem::remove(temporary_path, error);
    return isvik::Status::Unavailable("cannot replace conversation history file");
  }
#else
  std::filesystem::rename(temporary_path, state.conversation_path, error);
  if (error) {
    std::filesystem::remove(temporary_path);
    return isvik::Status::Unavailable("cannot replace conversation history file: " + error.message());
  }
#endif
  return {};
}
void RefreshConversationItems(const Window& window, const GuiState& state) {
  auto items = std::make_shared<slint::VectorModel<ConversationItem>>();
  for (const ConversationRecord& record : state.conversation_history) {
    ConversationItem item;
    item.id = Shared(record.id);
    item.title = Shared(record.title);
    items->push_back(item);
  }
  window->set_conversation_items(items);
  window->set_active_conversation_id(Shared(state.active_conversation_id));
  const auto active = std::find_if(state.conversation_history.begin(), state.conversation_history.end(),
      [&state](const auto& record) { return record.id == state.active_conversation_id; });
  window->set_current_conversation_title(Shared(active == state.conversation_history.end() ? "" : active->title));
}
isvik::Status SaveActiveConversation(const Window& window, GuiState& state) {
  if (state.conversation.empty()) return {};
  if (state.active_conversation_id.empty()) {
    state.active_conversation_id = "chat-" + std::to_string(NowMilliseconds());
    while (std::any_of(state.conversation_history.begin(), state.conversation_history.end(),
        [&state](const auto& item) { return item.id == state.active_conversation_id; })) {
      state.active_conversation_id += "-1";
    }
  }
  const auto existing = std::find_if(state.conversation_history.begin(), state.conversation_history.end(),
      [&state](const auto& record) { return record.id == state.active_conversation_id; });
  ConversationRecord record;
  if (existing != state.conversation_history.end()) record = *existing;
  record.id = state.active_conversation_id;
  record.context = state.conversation;
  record.updated_at_ms = NowMilliseconds();
  for (const auto& message : state.conversation) {
    if (message.role == isvik::MessageRole::kUser) {
      record.title = ConversationTitle(message.content);
      break;
    }
  }
  record.messages.clear();
  for (std::size_t index = 0; index < state.chat_messages->row_count(); ++index) {
    const auto row = state.chat_messages->row_data(index);
    if (row && (Text(row->role) == "user" || Text(row->role) == "assistant")) {
      ChatMessage saved = *row;
      saved.is_latest = false;
      record.messages.push_back(std::move(saved));
    }
  }
  if (record.messages.empty()) return {};
  if (existing == state.conversation_history.end()) state.conversation_history.push_back(std::move(record));
  else *existing = std::move(record);
  std::sort(state.conversation_history.begin(), state.conversation_history.end(), [](const auto& left, const auto& right) {
    return left.updated_at_ms > right.updated_at_ms;
  });
  if (state.conversation_history.size() > 100U) state.conversation_history.resize(100U);
  const isvik::Status saved = WriteConversationHistory(state);
  RefreshConversationItems(window, state);
  if (!saved.ok()) Log("ERROR", "Could not save conversation history: " + saved.message());
  return saved;
}
bool RestoreConversation(const Window& window, GuiState& state,
                        const std::string& id, bool chinese) {
  const auto found = std::find_if(state.conversation_history.begin(), state.conversation_history.end(),
      [&id](const auto& record) { return record.id == id; });
  if (found == state.conversation_history.end()) return false;
  state.active_conversation_id = found->id;
  state.conversation = found->context;
  state.chat_messages->clear();
  state.latest_assistant_row.reset();
  std::optional<std::size_t> latest_assistant;
  std::string latest_answer;
  bool latest_saved = false;
  std::string last_prompt;
  for (std::size_t index = 0; index < found->messages.size(); ++index) {
    ChatMessage row = found->messages[index];
    row.is_latest = false;
    const std::string role = Text(row.role);
    if (role == "user") last_prompt = Text(row.content);
    if (role == "assistant") {
      latest_assistant = index;
      latest_answer = Text(row.content);
      latest_saved = row.saved;
    }
    state.chat_messages->push_back(row);
  }
  if (latest_assistant && *latest_assistant < state.chat_messages->row_count()) {
    auto row = state.chat_messages->row_data(*latest_assistant);
    if (row) { row->is_latest = true; state.chat_messages->set_row_data(*latest_assistant, *row); }
    state.latest_assistant_row = latest_assistant;
  }
  state.can_remember_exchange = state.conversation.size() >= 2U && latest_assistant.has_value() && !latest_saved;
  state.exchange_saved = latest_saved;
  window->set_active_conversation_id(Shared(state.active_conversation_id));
  window->set_current_conversation_title(Shared(found->title));
  window->set_last_prompt(Shared(last_prompt));
  window->set_prompt_text(Shared(""));
  window->set_prompt_line_count(2);
  window->set_response_text(Shared(latest_answer));
  window->set_can_remember_exchange(state.can_remember_exchange);
  window->set_exchange_saved(state.exchange_saved);
  window->set_chat_messages(state.chat_messages);
  window->set_active_page(Shared("playground"));
  RefreshConversationItems(window, state);
  Status(window, Choose(chinese, "已打开历史对话", "Conversation loaded"));
  return true;
}
bool HasOnnxRuntimeAssets(const std::filesystem::path& graph_path) {
  const auto resolved = isvik::ResolveModelEntry(graph_path);
  if (!resolved.ok()) return false;
  std::error_code error;
  auto directory = resolved.value().parent_path();
  while (!directory.empty()) {
    if (std::filesystem::is_regular_file(directory / "genai_config.json", error) &&
        std::filesystem::is_regular_file(directory / "tokenizer_config.json", error) &&
        std::filesystem::is_regular_file(directory / "tokenizer.json", error)) {
      return true;
    }
    const auto parent = directory.parent_path();
    if (parent == directory) break;
    directory = parent;
  }
  return false;
}
bool CanRun(const isvik::CatalogModel& entry, isvik::BackendType backend) {
  if (backend == isvik::BackendType::kOnnxRuntime) {
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
    return entry.available && entry.model.format == isvik::ModelFormat::kOnnx &&
        HasOnnxRuntimeAssets(entry.model.path);
#else
    (void)entry;
    return false;
#endif
  }
  if (backend == isvik::BackendType::kTensorRt) {
#if defined(ISVIK_HAS_TENSORRT)
    if (entry.model.format == isvik::ModelFormat::kOnnx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP)
      return entry.available && HasOnnxRuntimeAssets(entry.model.path);
#else
      return false;
#endif
    }
    return entry.available && entry.model.format == isvik::ModelFormat::kGguf &&
        std::find(entry.model.compatible_backends.begin(),
                  entry.model.compatible_backends.end(),
                  isvik::BackendType::kTensorRt) != entry.model.compatible_backends.end();
#else
    (void)entry;
    return false;
#endif
  }
#if !defined(ISVIK_HAS_OPENVINO_GENAI)
  (void)entry;
  return false;
#else
#if defined(_WIN32) && defined(_DEBUG)
  if (entry.model.path.filename() == "openvino_language_model.xml") return false;
#endif
  return entry.runnable;
#endif
}
std::string BackendLabel(isvik::BackendType backend) {
  if (backend == isvik::BackendType::kTensorRt) return "TensorRT";
  if (backend == isvik::BackendType::kOnnxRuntime) return "ONNX Runtime";
  return "OpenVINO";
}
bool BackendBuilt(isvik::BackendType backend) {
  if (backend == isvik::BackendType::kOnnxRuntime) {
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
    return true;
#else
    return false;
#endif
  }
  if (backend == isvik::BackendType::kTensorRt) {
#if defined(ISVIK_HAS_TENSORRT)
    return true;
#else
    return false;
#endif
  }
#if defined(ISVIK_HAS_OPENVINO_GENAI)
  return true;
#else
  return false;
#endif
}
isvik::BackendType FallbackBackend() {
  if (BackendBuilt(isvik::BackendType::kOpenVino)) return isvik::BackendType::kOpenVino;
  if (BackendBuilt(isvik::BackendType::kTensorRt)) return isvik::BackendType::kTensorRt;
  if (BackendBuilt(isvik::BackendType::kOnnxRuntime)) return isvik::BackendType::kOnnxRuntime;
  return isvik::BackendType::kOpenVino;
}
bool BackendFromLabel(std::string_view label, isvik::BackendType* backend) {
  if (label == "OpenVINO") {
    *backend = isvik::BackendType::kOpenVino;
    return true;
  }
  if (label == "TensorRT") {
    *backend = isvik::BackendType::kTensorRt;
    return true;
  }
  if (label == "ONNX Runtime") {
    *backend = isvik::BackendType::kOnnxRuntime;
    return true;
  }
  return false;
}
std::string BackendIssue(isvik::BackendType backend, bool chinese) {
  if (backend == isvik::BackendType::kOnnxRuntime) {
    return Choose(chinese,
        "ONNX Runtime 需要带有 GenAI 配置和分词器文件的 ONNX 模型目录。",
        "ONNX Runtime needs an ONNX model directory with GenAI and tokenizer configuration files.");
  }
  if (backend == isvik::BackendType::kTensorRt) {
    return Choose(chinese,
        "TensorRT 原生聊天当前支持内嵌分词器且量化编码受支持的 Gemma 4 GGUF。",
        "Native TensorRT chat currently supports Gemma 4 GGUF files with an embedded tokenizer and supported quantization.");
  }
#if !defined(ISVIK_HAS_OPENVINO_GENAI)
  return Choose(chinese, "此版本未启用 OpenVINO GenAI。", "OpenVINO GenAI is not enabled in this build.");
#else
  return Choose(chinese, "此模型不受当前 OpenVINO 运行时支持。", "This model is not supported by the current OpenVINO runtime.");
#endif
}
void SelectModel(const Window& window, GuiState& state, const std::string& id,
                 bool chinese, isvik::BackendType backend) {
  const auto* entry = state.catalog.Find(id);
  if (state.selected_id != id) window->set_dismissed_issue(Shared(""));
  state.selected_id = entry ? id : "";
  window->set_selected_id(Shared(state.selected_id));
  window->set_selected_name(Shared(entry ? entry->model.display_name : ""));
  window->set_selected_path(Shared(entry ? isvik::PathUtf8(entry->model.path) : ""));
  window->set_selected_meta(Shared(entry ? Metadata(*entry) : ""));
  window->set_rename_text(Shared(entry ? entry->model.display_name : ""));
  window->set_selected_runnable(entry && CanRun(*entry, backend));
  std::string issue;
  if (entry && !CanRun(*entry, backend)) {
    if (!entry->available) {
      issue = Choose(chinese, "模型文件不完整或已移动，请重新扫描目录。", "Model files are incomplete or moved. Rescan the directory.");
    } else if (backend == isvik::BackendType::kTensorRt &&
               entry->model.format == isvik::ModelFormat::kOnnx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP)
      issue = BackendIssue(isvik::BackendType::kOnnxRuntime, chinese);
#else
      issue = Choose(chinese,
          "此版本没有包含 TensorRT-RTX 的 ONNX 执行组件。",
          "This build does not include the TensorRT-RTX ONNX execution component.");
#endif
    } else if (backend == isvik::BackendType::kTensorRt || !entry->runnable) {
      issue = BackendIssue(backend, chinese);
    }
  }
  if (entry && backend == isvik::BackendType::kOpenVino && !entry->runnable) {
    if (entry->model.format == isvik::ModelFormat::kGguf) {
      issue = chinese ? "已识别 GGUF；当前 OpenVINO 版本不能运行此架构或量化编码。"
                      : entry->issue.empty() ? "This GGUF architecture or encoding is unsupported."
                                             : entry->issue;
    } else if (entry->model.format == isvik::ModelFormat::kOnnx) {
      issue = Choose(chinese, "已识别 ONNX 模型；当前版本还不能用它进行对话。",
                     "ONNX model recognized; this build cannot use it for chat yet.");
    } else {
      issue = Choose(chinese, "模型文件不完整或已移动，请重新扫描目录。", "Model files are incomplete or moved. Rescan the directory.");
    }
  }
#if defined(_WIN32) && defined(_DEBUG)
  if (entry && backend == isvik::BackendType::kOpenVino && entry->runnable && !CanRun(*entry, backend)) {
    issue = Choose(chinese, "此多模态模型请使用 Release 版加载；本机 Debug SDK 存在断点退出问题。", "Use Release for this VLM. The local Debug SDK aborts while loading this layout.");
  }
#endif
  window->set_selected_issue(Shared(issue));
}
void RefreshModels(const Window& window, GuiState& state, bool chinese) {
  const auto filter = Lower(Text(window->get_search_text()));
  auto rows = std::make_shared<slint::VectorModel<ModelItem>>();
  auto chat_names = std::make_shared<slint::VectorModel<slint::SharedString>>();
  state.chat_ids.clear();
  int selected_index = -1;
  for (const auto& entry : state.catalog.models()) {
    const bool runnable = CanRun(entry, state.catalog.backend());
    if (runnable) {
      if ((state.loaded_model && entry.model.id == state.loaded_model->id) ||
          (!state.loaded_model && entry.model.id == state.selected_id)) selected_index = static_cast<int>(state.chat_ids.size());
      state.chat_ids.push_back(entry.model.id);
      chat_names->push_back(Shared(entry.model.display_name));
    }
    if (!filter.empty() && Lower(entry.model.display_name + " " + entry.model.architecture + " " + isvik::PathUtf8(entry.model.path)).find(filter) == std::string::npos) continue;
    ModelItem item;
    item.id = Shared(entry.model.id);
    item.name = Shared(entry.model.display_name);
    item.path = Shared(isvik::PathUtf8(entry.model.path));
    item.meta = Shared(Metadata(entry));
    item.state = Shared(runnable ? Choose(chinese, "可加载", "Ready to load") : !entry.available ? Choose(chinese, "文件缺失或无效", "Missing or invalid files") : state.catalog.backend() == isvik::BackendType::kTensorRt ? Choose(chinese, "聊天暂不可用", "Chat unavailable") : Choose(chinese, "暂不可运行", "Not runnable"));
    if (entry.runnable && state.catalog.backend() == isvik::BackendType::kOpenVino && !runnable) item.state = Shared(Choose(chinese, "使用 Release 版加载", "Load with Release"));
    if (state.loaded_model && state.loaded_model->id == entry.model.id) item.state = Shared(Choose(chinese, "已加载", "Loaded"));
    item.runnable = runnable;
    item.is_default = entry.model.id == state.catalog.default_id();
    rows->push_back(item);
  }
  window->set_model_items(rows);
  window->set_model_count(static_cast<int>(state.catalog.models().size()));
  window->set_visible_model_count(static_cast<int>(rows->row_count()));
  window->set_chat_model_items(chat_names);
  window->set_chat_model_index(selected_index);
  SelectModel(window, state, state.selected_id, chinese, state.catalog.backend());
  if (state.loaded_model) window->set_loaded_name(Shared(state.loaded_model->display_name));
}
void RefreshDevices(const Window& window, bool chinese, isvik::BackendType backend) {
  auto choices = std::make_shared<slint::VectorModel<slint::SharedString>>();
  auto rows = std::make_shared<slint::VectorModel<DeviceItem>>();
  if (backend == isvik::BackendType::kOnnxRuntime) {
    choices->push_back(Shared("CPU"));
    DeviceItem row;
    row.id = Shared("CPU");
    row.name = Shared(Choose(chinese, "ONNX Runtime CPU", "ONNX Runtime CPU"));
    rows->push_back(row);
    window->set_device_count(1);
    window->set_device_list_text(Shared(Choose(chinese, "ONNX Runtime CPU", "ONNX Runtime CPU")));
    window->set_device_items(choices);
    window->set_device_rows(rows);
    return;
  }
  if (backend == isvik::BackendType::kTensorRt) {
#if defined(ISVIK_HAS_TENSORRT)
    const auto devices = isvik::tensorrt_backend::EnumerateDevices();
    if (!devices.ok()) {
      window->set_device_count(0);
      window->set_device_list_text(Shared(devices.status().message()));
    } else {
      std::string listing;
      for (const auto& device : devices.value()) {
        const std::string id = std::to_string(device.index);
        choices->push_back(Shared("GPU." + id));
        DeviceItem row;
        row.id = Shared("GPU." + id);
        std::ostringstream name;
        name << device.name << " · " << std::fixed << std::setprecision(1)
             << static_cast<double>(device.total_memory_bytes) / (1024.0 * 1024.0 * 1024.0)
             << " GiB · SM " << device.compute_capability_major << '.'
             << device.compute_capability_minor;
        row.name = Shared(name.str());
        rows->push_back(row);
        listing += "GPU." + id + "   ·   " + name.str() + "\n\n";
      }
      window->set_device_count(static_cast<int>(devices.value().size()));
      window->set_device_list_text(Shared(listing.empty()
          ? Choose(chinese, "没有可用的 NVIDIA GPU", "No NVIDIA GPU is available")
          : listing));
    }
#else
    window->set_device_count(0);
    window->set_device_list_text(Shared(Choose(chinese, "此版本未启用 TensorRT", "TensorRT is not enabled in this build")));
#endif
    window->set_device_items(choices);
    window->set_device_rows(rows);
    return;
  }
  choices->push_back(Shared("AUTO"));
#if defined(ISVIK_HAS_OPENVINO_RUNTIME)
  const auto devices = isvik::openvino_backend::EnumerateDevices();
  if (!devices.ok()) {
    Status(window, Choose(chinese, "无法枚举设备", "Cannot enumerate devices"), devices.status().message());
    return;
  }
  std::string listing;
  for (const auto& device : devices.value()) {
    choices->push_back(Shared(device.id));
    DeviceItem row;
    row.id = Shared(device.id);
    row.name = Shared(device.full_name);
    rows->push_back(row);
    listing += device.id + "   ·   " + device.full_name + "\n\n";
  }
  window->set_device_count(static_cast<int>(devices.value().size()));
  window->set_device_list_text(Shared(listing));
#else
  choices->push_back(Shared("CPU"));
  window->set_device_list_text(Shared(Choose(chinese, "未启用 OpenVINO", "OpenVINO is not enabled")));
#endif
  window->set_device_items(choices);
  window->set_device_rows(rows);
}
void RefreshText(const Window& window, const isvik::LocalizationService& language) {
  window->set_chinese(Chinese(language));
#define ISVIK_LABEL(property, message) window->set_##property(slint::SharedString(language.GetText(isvik::MessageId::message)))
  ISVIK_LABEL(app_title, kWindowTitle);
  ISVIK_LABEL(tagline, kTagline);
  ISVIK_LABEL(overview, kOverview);
  ISVIK_LABEL(playground, kPlayground);
  ISVIK_LABEL(devices, kDevices);
  ISVIK_LABEL(device_label, kDeviceLabel);
  ISVIK_LABEL(max_tokens_label, kMaxTokensLabel);
  ISVIK_LABEL(load_model_label, kLoadModel);
  ISVIK_LABEL(prompt_label, kPromptLabel);
  ISVIK_LABEL(prompt_placeholder, kPromptPlaceholder);
  ISVIK_LABEL(generate_label, kGenerate);
  ISVIK_LABEL(cancel_label, kCancel);
  ISVIK_LABEL(response_label, kResponse);
  ISVIK_LABEL(clear_label, kClear);
  ISVIK_LABEL(refresh_devices_label, kRefreshDevices);
  ISVIK_LABEL(device_list_label, kDeviceListLabel);
  ISVIK_LABEL(language_button, kLanguageButton);
#undef ISVIK_LABEL
  window->set_models(Shared(Choose(Chinese(language), "模型库", "Model library")));
}
void RefreshApiKeyNames(const Window& window) {
  auto names = std::make_shared<slint::VectorModel<slint::SharedString>>();
  auto entries = isvik::app::LoadApiKeyVault();
  if (!entries.ok()) {
    Log("ERROR", "API key vault could not be loaded: " + entries.status().message());
    window->set_api_key_names(names);
    return;
  }
  for (const auto& entry : entries.value()) names->push_back(Shared(entry.name));
  window->set_api_key_names(names);
}
void Busy(const Window& window, GuiState& state, bool busy) {
  state.busy = busy;
  window->set_operation_busy(busy);
}
// Persist a proposed catalog before making it visible. Failures leave the old state intact.
isvik::Status Commit(GuiState& state, isvik::ModelCatalog proposed) {
  if (!state.catalog_writable) return isvik::Status::Unavailable("Existing catalog could not be read. Repair or back up models.json before changing the library.");
  const auto saved = proposed.Save();
  if (saved.ok()) state.catalog = std::move(proposed);
  return saved;
}

std::vector<std::string> ParseTags(std::string_view text) {
  std::vector<std::string> tags;
  std::string current;
  const auto flush = [&] {
  const auto first = current.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) { current.clear(); return; }
    const auto last = current.find_last_not_of(" \t\r\n");
    std::string tag = current.substr(first, last - first + 1U);
    if (std::find(tags.begin(), tags.end(), tag) == tags.end()) tags.push_back(std::move(tag));
    current.clear();
  };
  for (std::size_t index = 0U; index < text.size();) {
    if (text[index] == ',') { flush(); ++index; }
    else if (text.substr(index, 3U) == "，") { flush(); index += 3U; }
    else current.push_back(text[index++]);
  }
  flush();
  return tags;
}

void RefreshMemories(const Window& window, GuiState& state, bool chinese,
                     std::string_view query = {}) {
  auto rows = std::make_shared<slint::VectorModel<MemoryItem>>();
  if (!state.memories) {
    window->set_memory_ready(false);
    window->set_memory_items(rows);
    return;
  }
  isvik::StatusOr<std::vector<isvik::MemoryEntry>> entries = query.empty()
      ? state.memories->List(state.memory_session, 200U)
      : state.memories->Search(query, state.memory_session, 200U);
  if (!entries.ok()) {
    window->set_memory_status(Shared(entries.status().message()));
    return;
  }
  for (const auto& entry : entries.value()) {
    MemoryItem item;
    item.id = Shared(entry.id);
    item.content = Shared(entry.content);
    item.scope = Shared(entry.scope == isvik::MemoryScope::kSession ? "session" : "persistent");
    item.pinned = entry.pinned;
    std::string tags;
    for (const auto& tag : entry.tags) {
      if (!tags.empty()) tags += ", ";
      tags += tag;
    }
    item.tags = Shared(tags);
    rows->push_back(item);
  }
  window->set_memory_ready(true);
  window->set_memory_items(rows);
  const std::string count = std::to_string(entries.value().size());
  window->set_memory_status(Shared(chinese ? "共 " + count + " 条记忆" : count + " memories"));
}

void SetContextStatus(const Window& window, const GuiState& state,
                      uint64_t estimated_tokens = 0U, uint64_t dropped = 0U,
                      uint64_t memories_used = 0U, bool chinese = true,
                      uint64_t effective_limit = 0U) {
  std::ostringstream text;
  const uint64_t shown_limit = effective_limit == 0U ? state.context_limit : effective_limit;
  if (estimated_tokens == 0U) {
    text << (chinese ? "预算 " : "Budget ") << shown_limit << (chinese ? " 词元" : " tokens");
  } else {
    text << estimated_tokens << "/" << shown_limit << (chinese ? " 词元" : " tokens");
    if (dropped != 0U) text << (chinese ? " · 已收起 " : " · trimmed ") << dropped;
    text << (chinese ? " · 使用记忆 " : " · memories ") << memories_used;
  }
  window->set_context_status(Shared(text.str()));
}
}  // namespace

int RunApplication(int argc, char** argv) {
  isvik::app::ConfigureOpenVinoRuntime();
  if (argc > 1) return isvik::app::RunCli(argc, argv);
  auto window = MainWindow::create();
  const WeakWindow weak(window);
  auto state = std::make_shared<GuiState>();
  state->chat_messages = std::make_shared<slint::VectorModel<ChatMessage>>();
  window->set_chat_messages(state->chat_messages);
  state->memory_session = "gui-" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  state->conversation_path = isvik::app::ConversationStoragePath();
  const isvik::Status history_status = LoadConversationHistory(*state);
  if (!history_status.ok()) Log("ERROR", "Conversation history could not be loaded: " + history_status.message());
  RefreshConversationItems(window, *state);
  auto memory_service = isvik::MemoryService::Open(isvik::app::MemoryStoragePath());
  if (memory_service.ok()) {
    state->memories = std::move(memory_service).value();
    (void)state->memories->SetSessionMemoryEnabled(state->memory_session, true);
  }
  isvik::LocalizationService language;
  RefreshText(window, language);
  window->set_catalog_location(Shared(isvik::PathUtf8(isvik::app::CatalogStoragePath())));
  window->set_memory_location(Shared(isvik::PathUtf8(isvik::app::MemoryStoragePath())));
  window->set_memory_ready(state->memories != nullptr);
  window->set_context_memories_enabled(true);
  window->set_session_memory_enabled(true);
  window->set_context_tokens_text(Shared("8192"));
  window->set_api_host(Shared("127.0.0.1"));
  window->set_api_port(Shared("1234"));
  window->set_api_key(Shared(""));
  window->set_api_key_names(std::make_shared<slint::VectorModel<slint::SharedString>>());
  window->set_api_key_name(Shared(""));
  window->set_api_key_label(Shared(""));
  RefreshApiKeyNames(window);
  window->set_api_server_running(false);
  window->set_api_server_transition(false);
  window->set_api_model_info(Shared(""));
  RefreshMemories(window, *state, Chinese(language));
  SetContextStatus(window, *state, 0U, 0U, 0U, Chinese(language));
  const auto restored = state->catalog.Load();
  state->catalog_writable = restored.ok();
  if (!BackendBuilt(state->catalog.backend())) {
    state->catalog.set_backend(FallbackBackend());
    if (state->catalog_writable) {
      const auto saved = state->catalog.Save();
      if (!saved.ok()) Log("ERROR", "Could not reset unavailable backend preference: " + saved.message());
    }
  }
  state->selected_id = state->catalog.default_id();
  if (state->selected_id.empty() && !state->catalog.models().empty()) {
    state->selected_id = state->catalog.models().front().model.id;
  }
  bool startup_switched_to_onnx_runtime = false;
  if (const auto* selected = state->catalog.Find(state->selected_id);
      selected && BackendBuilt(isvik::BackendType::kOnnxRuntime) &&
      !CanRun(*selected, state->catalog.backend()) &&
      CanRun(*selected, isvik::BackendType::kOnnxRuntime) && state->catalog_writable) {
    auto proposed = state->catalog;
    proposed.set_backend(isvik::BackendType::kOnnxRuntime);
    const auto saved = Commit(*state, std::move(proposed));
    if (saved.ok()) {
      startup_switched_to_onnx_runtime = true;
      Log("INFO", "Selected ONNX model requires the ONNX Runtime backend; switched automatically");
    } else {
      Log("ERROR", "Could not select the ONNX Runtime backend for the selected ONNX model: " + saved.message());
    }
  }
  auto backend_items = std::make_shared<slint::VectorModel<slint::SharedString>>();
#if defined(ISVIK_HAS_OPENVINO_GENAI)
  backend_items->push_back(Shared("OpenVINO"));
#endif
#if defined(ISVIK_HAS_TENSORRT)
  backend_items->push_back(Shared("TensorRT"));
#endif
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
  backend_items->push_back(Shared("ONNX Runtime"));
#endif
  if (backend_items->row_count() == 0U) backend_items->push_back(Shared("OpenVINO"));
  window->set_backend_items(backend_items);
  window->set_backend_name(Shared(BackendLabel(state->catalog.backend())));
  window->set_device_name(Shared(state->catalog.backend() == isvik::BackendType::kTensorRt ? "GPU.0" : "CPU"));
  RefreshDevices(window, Chinese(language), state->catalog.backend());
  window->set_folder_path(Shared(isvik::PathUtf8(state->catalog.folder())));
  RefreshModels(window, *state, Chinese(language));
  if (!restored.ok()) Status(window, "模型库读取失败 / Cannot restore model library", restored.message());
  else if (!memory_service.ok()) Status(window, Choose(Chinese(language), "记忆服务打开失败", "Cannot open memory service"), memory_service.status().message());
  else Status(window, "");
  Log("INFO", "GUI ready; model catalog=" + isvik::PathUtf8(isvik::app::CatalogStoragePath()) +
      "; memory database=" + isvik::PathUtf8(isvik::app::MemoryStoragePath()) +
      "; conversation history=" + isvik::PathUtf8(isvik::app::ConversationStoragePath()) +
      "; models=" + std::to_string(state->catalog.models().size()));

  auto backend_notice_generation = std::make_shared<uint64_t>(0U);
  if (startup_switched_to_onnx_runtime) {
    const std::string notice = Choose(Chinese(language),
        "已为所选 ONNX 模型自动切换到 ONNX Runtime（CPU）",
        "Automatically switched to ONNX Runtime (CPU) for the selected ONNX model");
    Status(window, notice);
    const uint64_t generation = ++*backend_notice_generation;
    slint::Timer::single_shot(std::chrono::milliseconds(1000), [weak, backend_notice_generation, generation, notice] {
      if (*backend_notice_generation != generation) return;
      auto latest = weak.lock();
      if (!latest || (*latest)->get_status_error() || Text((*latest)->get_status_message()) != notice) return;
      Status(*latest, "");
    });
  }
  window->on_toggle_language([&] { language.ToggleLanguage(); RefreshText(window, language); RefreshModels(window, *state, Chinese(language)); });
  window->on_toggle_api_server([weak, state, &language](slint::SharedString host_text,
      slint::SharedString port_text, slint::SharedString key_text) {
    auto current = weak.lock();
    if (!current || state->api_server_transition) return;
    const bool chinese = Chinese(language);
    if (state->api_server.IsRunning()) {
      state->api_server_transition = true;
      (*current)->set_api_server_transition(true);
      Status(*current, Choose(chinese, "正在关闭 API 服务…", "Stopping API server…"));
      state->api_worker = std::jthread([weak, state, chinese] {
        state->api_server.Stop();
        (void)slint::invoke_from_event_loop([weak, state, chinese] {
          state->api_server_transition = false;
          if (auto latest = weak.lock()) {
            (*latest)->set_api_server_running(false);
            (*latest)->set_api_server_transition(false);
            Status(*latest, Choose(chinese, "API 服务已停止", "API server stopped"));
          }
        });
      });
      return;
    }
    if (state->busy || !state->loaded_model || !state->loaded_backend) {
      Status(*current, Choose(chinese, "请先加载一个模型，再启动 API 服务", "Load a model before starting the API server"));
      return;
    }

    const std::string host = Text(host_text);
    const std::string port_text_value = Text(port_text);
    int port = 0;
    const auto parsed_port = std::from_chars(port_text_value.data(),
        port_text_value.data() + port_text_value.size(), port);
    if (host.empty() || parsed_port.ec != std::errc{} ||
        parsed_port.ptr != port_text_value.data() + port_text_value.size() ||
        port < 1 || port > 65535) {
      Status(*current, Choose(chinese, "请检查监听地址和端口（1–65535）", "Check the host and port (1–65535)"));
      return;
    }

    const isvik::ModelDescriptor model = *state->loaded_model;
    const isvik::BackendType backend = *state->loaded_backend;
    isvik::app::ApiServerConfig config;
    config.host = host;
    config.port = port;
    config.api_key = Text(key_text);
    config.backend = BackendLabel(backend);
    const auto generate = [state, backend, model](
        const isvik::UnifiedInferenceRequest& request,
        const isvik::CancellationToken& cancellation,
        const std::function<void(const isvik::InferenceEvent&)>& handler) -> isvik::Status {
      if (backend == isvik::BackendType::kTensorRt) {
        if (model.format == isvik::ModelFormat::kOnnx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP) && defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
          return state->onnxruntime_engine.Generate(request, cancellation, handler);
#else
          return isvik::Status::Unavailable("TensorRT-RTX ONNX inference is not available in this build");
#endif
        }
#if defined(ISVIK_HAS_TENSORRT)
        return state->tensorrt_engine.Generate(request, cancellation, handler);
#else
        return isvik::Status::Unavailable("TensorRT inference is not available in this build");
#endif
      }
      if (backend == isvik::BackendType::kOnnxRuntime) {
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
        return state->onnxruntime_engine.Generate(request, cancellation, handler);
#else
        return isvik::Status::Unavailable("ONNX Runtime inference is not available in this build");
#endif
      }
#if defined(ISVIK_HAS_OPENVINO_GENAI)
      return state->engine.Generate(request, cancellation, handler);
#else
      return isvik::Status::Unavailable("OpenVINO inference is not available in this build");
#endif
    };
    state->api_server_transition = true;
    (*current)->set_api_server_transition(true);
    Status(*current, Choose(chinese, "正在启动 API 服务…", "Starting API server…"));
    const isvik::Status started = state->api_server.Start(config, model, generate);
    state->api_server_transition = false;
    (*current)->set_api_server_transition(false);
    if (!started.ok()) {
      (*current)->set_api_server_running(false);
      Status(*current, Choose(chinese, "API 服务启动失败", "Could not start API server"), started.message());
      return;
    }
    (*current)->set_api_server_running(true);
    (*current)->set_api_model_info(Shared(model.display_name + " · " + BackendLabel(backend) +
        " · " + Text((*current)->get_loaded_device())));
    Status(*current, Choose(chinese, "API 服务已启动", "API server started"));
  });
  window->on_generate_api_key([weak, state, &language] {
    auto current = weak.lock();
    if (!current || state->api_server_transition || state->api_server.IsRunning()) return;
    const std::string key = GenerateApiKey();
    if (key.empty()) {
      Status(*current, Choose(Chinese(language), "生成随机密钥失败", "Could not generate a random API key"));
      return;
    }
    (*current)->set_api_key(Shared(key));
    (*current)->set_api_key_hidden(false);
    Status(*current, Choose(Chinese(language), "已生成随机 API 密钥", "Random API key generated"));
  });
  window->on_save_api_key([weak, &language](slint::SharedString name_text,
                                            slint::SharedString key_text) {
    auto current = weak.lock();
    if (!current) return;
    const std::string name = Text(name_text);
    const std::string key = Text(key_text);
    const isvik::Status saved = isvik::app::SaveApiKey(name, key);
    if (!saved.ok()) {
      Status(*current, Choose(Chinese(language), "保存密钥失败", "Could not save API key"), saved.message());
      return;
    }
    RefreshApiKeyNames(*current);
    (*current)->set_api_key_name(Shared(name));
    Status(*current, Choose(Chinese(language), "密钥已安全保存", "API key saved securely"));
  });
  window->on_select_api_key([weak, &language](slint::SharedString name_text) {
    auto current = weak.lock();
    if (!current) return;
    const std::string name = Text(name_text);
    if (name.empty()) return;
    auto entries = isvik::app::LoadApiKeyVault();
    if (!entries.ok()) {
      Status(*current, Choose(Chinese(language), "读取密钥失败", "Could not read API key"), entries.status().message());
      return;
    }
    const auto found = std::find_if(entries.value().begin(), entries.value().end(),
        [&name](const isvik::app::ApiKeyEntry& entry) { return entry.name == name; });
    if (found == entries.value().end()) {
      Status(*current, Choose(Chinese(language), "找不到已保存的密钥", "Saved API key was not found"));
      RefreshApiKeyNames(*current);
      return;
    }
    (*current)->set_api_key(Shared(found->value));
    (*current)->set_api_key_hidden(true);
    Status(*current, Choose(Chinese(language), "密钥已填入", "API key loaded"));
  });
  window->on_delete_api_key([weak, &language](slint::SharedString name_text) {
    auto current = weak.lock();
    if (!current) return;
    const std::string name = Text(name_text);
    if (name.empty()) return;
    const isvik::Status removed = isvik::app::DeleteApiKey(name);
    if (!removed.ok()) {
      Status(*current, Choose(Chinese(language), "删除密钥失败", "Could not delete API key"), removed.message());
      return;
    }
    if (Text((*current)->get_api_key_name()) == name) {
      (*current)->set_api_key(Shared(""));
      (*current)->set_api_key_hidden(true);
      (*current)->set_api_key_name(Shared(""));
    }
    RefreshApiKeyNames(*current);
    Status(*current, Choose(Chinese(language), "已从密钥库删除", "API key deleted from vault"));
  });
  window->on_select_model([weak, state, &language, backend_notice_generation](slint::SharedString id_text) {
    auto current = weak.lock();
    if (!current) return;
    const std::string id = Text(id_text);
    const auto* selected = state->catalog.Find(id);
    const auto active_backend = state->catalog.backend();
    const bool should_use_onnx_runtime = selected &&
        selected->model.format == isvik::ModelFormat::kOnnx &&
        BackendBuilt(isvik::BackendType::kOnnxRuntime) &&
        !CanRun(*selected, active_backend) &&
        CanRun(*selected, isvik::BackendType::kOnnxRuntime) &&
        !state->loaded_model && state->catalog_writable;
    if (!should_use_onnx_runtime) {
      SelectModel(*current, *state, id, Chinese(language), active_backend);
      return;
    }
    auto proposed = state->catalog;
    proposed.set_backend(isvik::BackendType::kOnnxRuntime);
    const auto saved = Commit(*state, std::move(proposed));
    if (!saved.ok()) {
      SelectModel(*current, *state, id, Chinese(language), active_backend);
      Status(*current, Choose(Chinese(language), "无法自动切换 ONNX Runtime", "Could not switch to ONNX Runtime"), saved.message());
      return;
    }
    (*current)->set_backend_name(Shared(BackendLabel(isvik::BackendType::kOnnxRuntime)));
    (*current)->set_device_name(Shared("CPU"));
    (*current)->set_dismissed_issue(Shared(""));
    RefreshDevices(*current, Chinese(language), isvik::BackendType::kOnnxRuntime);
    state->selected_id = id;
    RefreshModels(*current, *state, Chinese(language));
    SelectModel(*current, *state, id, Chinese(language), isvik::BackendType::kOnnxRuntime);
    const std::string notice = Choose(Chinese(language),
        "已为此 ONNX 模型切换到 ONNX Runtime（CPU）",
        "Switched to ONNX Runtime (CPU) for this ONNX model");
    Status(*current, notice);
    const uint64_t generation = ++*backend_notice_generation;
    slint::Timer::single_shot(std::chrono::milliseconds(1000), [weak, backend_notice_generation, generation, notice] {
      if (*backend_notice_generation != generation) return;
      auto latest = weak.lock();
      if (!latest || (*latest)->get_status_error() || Text((*latest)->get_status_message()) != notice) return;
      Status(*latest, "");
    });
  });
  window->on_filter_models([&](slint::SharedString) { RefreshModels(window, *state, Chinese(language)); });
  window->on_refresh_devices([&] { RefreshDevices(window, Chinese(language), state->catalog.backend()); });
  window->on_switch_backend([weak, state, &language, backend_notice_generation](slint::SharedString name) {
    auto current = weak.lock();
    if (!current || state->busy) return;
    if (state->api_server_transition || state->api_server.IsRunning()) {
      (*current)->set_backend_name(Shared(BackendLabel(state->catalog.backend())));
      Status(*current, Choose(Chinese(language), "请先停止 API 服务再切换后端。", "Stop the API server before switching backends."));
      return;
    }
    isvik::BackendType selected_backend = isvik::BackendType::kOpenVino;
    if (!BackendFromLabel(Text(name), &selected_backend) || !BackendBuilt(selected_backend)) {
      (*current)->set_backend_name(Shared(BackendLabel(state->catalog.backend())));
      Status(*current, Choose(Chinese(language), "此版本没有启用所选后端", "The selected backend is not enabled in this build"));
      return;
    }
    if (selected_backend == state->catalog.backend()) return;
    if (state->loaded_model) {
      (*current)->set_backend_name(Shared(BackendLabel(state->catalog.backend())));
      Status(*current, Choose(Chinese(language), "请先卸载当前模型，再切换推理后端。", "Unload the current model before switching backends."));
      return;
    }
    if (!state->catalog_writable) {
      (*current)->set_backend_name(Shared(BackendLabel(state->catalog.backend())));
      Status(*current, Choose(Chinese(language), "后端设置无法保存", "Backend selection cannot be saved"),
          "Repair or back up models.json before changing the model library.");
      return;
    }
    auto proposed = state->catalog;
    proposed.set_backend(selected_backend);
    const auto saved = Commit(*state, std::move(proposed));
    if (!saved.ok()) {
      (*current)->set_backend_name(Shared(BackendLabel(state->catalog.backend())));
      Status(*current, Choose(Chinese(language), "后端设置保存失败", "Could not save backend selection"), saved.message());
      return;
    }
    (*current)->set_backend_name(Shared(BackendLabel(selected_backend)));
    (*current)->set_device_name(Shared(selected_backend == isvik::BackendType::kTensorRt ? "GPU.0" : "CPU"));
    (*current)->set_dismissed_issue(Shared(""));
    RefreshDevices(*current, Chinese(language), selected_backend);
    RefreshModels(*current, *state, Chinese(language));
    const std::string notice = selected_backend == isvik::BackendType::kTensorRt
        ? Choose(Chinese(language), "已切换到 TensorRT", "Switched to TensorRT")
        : selected_backend == isvik::BackendType::kOnnxRuntime
            ? Choose(Chinese(language), "已切换到 ONNX Runtime", "Switched to ONNX Runtime")
            : Choose(Chinese(language), "已切换到 OpenVINO", "Switched to OpenVINO");
    Status(*current, notice);
    const uint64_t generation = ++*backend_notice_generation;
    slint::Timer::single_shot(std::chrono::milliseconds(1000), [weak, backend_notice_generation, generation, notice] {
      if (*backend_notice_generation != generation) return;
      auto latest = weak.lock();
      if (!latest || (*latest)->get_status_error() || Text((*latest)->get_status_message()) != notice) return;
      Status(*latest, "");
    });
  });
  window->on_search_memories([weak, state, &language](slint::SharedString query) {
    if (auto current = weak.lock()) RefreshMemories(*current, *state, Chinese(language), Text(query));
  });
  window->on_select_memory([weak, state, &language](slint::SharedString id_text) {
    auto current = weak.lock();
    if (!current || !state->memories) return;
    const std::string id = Text(id_text);
    const auto entry = state->memories->Find(id, state->memory_session);
    const bool chinese = Chinese(language);
    if (!entry.ok()) { (*current)->set_memory_status(Shared(entry.status().message())); return; }
    std::string tags;
    for (const auto& tag : entry.value().tags) { if (!tags.empty()) tags += ", "; tags += tag; }
    (*current)->set_selected_memory_id(Shared(entry.value().id));
    (*current)->set_selected_memory_pinned(entry.value().pinned);
    (*current)->set_memory_content(Shared(entry.value().content));
    (*current)->set_memory_tags(Shared(tags));
    (*current)->set_session_memory_scope(entry.value().scope == isvik::MemoryScope::kSession);
    (*current)->set_memory_remove_pending(false);
    (*current)->set_memory_status(Shared(Choose(chinese, "已选择记忆", "Memory selected")));
  });
  window->on_save_memory([weak, state, &language](slint::SharedString content_text,
                                                  slint::SharedString tags_text,
                                                  bool session_scope) {
    auto current = weak.lock();
    if (!current || !state->memories) return;
    const std::string content = Text(content_text);
    if (content.find_first_not_of(" \t\r\n") == std::string::npos) {
      (*current)->set_memory_status(Shared(Choose(Chinese(language), "记忆内容不能为空", "Memory content is required")));
      return;
    }
    const std::string id = Text((*current)->get_selected_memory_id());
    const auto tags = ParseTags(Text(tags_text));
    isvik::Status status;
    std::string saved_id = id;
    if (id.empty()) {
      isvik::MemoryDraft draft;
      draft.content = content;
      draft.tags = tags;
      draft.scope = session_scope ? isvik::MemoryScope::kSession : isvik::MemoryScope::kPersistent;
      if (session_scope) draft.session_id = state->memory_session;
      const auto added = state->memories->Add(draft);
      if (!added.ok()) status = added.status();
      else saved_id = added.value().id;
    } else {
      status = state->memories->Update(id, state->memory_session, content, tags);
    }
    const bool chinese = Chinese(language);
    if (!status.ok()) { (*current)->set_memory_status(Shared(status.message())); return; }
    (*current)->set_memory_query(Shared(""));
    (*current)->set_selected_memory_id(Shared(saved_id));
    (*current)->set_memory_remove_pending(false);
    RefreshMemories(*current, *state, chinese);
    const auto saved = state->memories->Find(saved_id, state->memory_session);
    if (saved.ok()) (*current)->set_selected_memory_pinned(saved.value().pinned);
    (*current)->set_memory_status(Shared(Choose(chinese, "记忆已保存", "Memory saved")));
  });
  window->on_remember_exchange([weak, state, &language] {
    auto current = weak.lock();
    if (!current || !state->memories || !state->can_remember_exchange ||
        state->exchange_saved || state->conversation.size() < 2U) return;
    const auto& user_message = state->conversation[state->conversation.size() - 2U];
    const auto& assistant_message = state->conversation.back();
    if (user_message.role != isvik::MessageRole::kUser ||
        assistant_message.role != isvik::MessageRole::kAssistant) return;

    isvik::MemoryDraft draft;
    draft.content = "用户：" + user_message.content + "\n助手：" + assistant_message.content;
    draft.tags = {"对话"};
    draft.scope = isvik::MemoryScope::kPersistent;
    const auto added = state->memories->Add(draft);
    const bool chinese = Chinese(language);
    if (!added.ok()) {
      Status(*current, Choose(chinese, "保存问答到记忆失败", "Could not save exchange to memory"), added.status().message());
      return;
    }

    state->exchange_saved = true;
    (*current)->set_exchange_saved(true);
    if (state->latest_assistant_row && *state->latest_assistant_row < state->chat_messages->row_count()) {
      auto latest = state->chat_messages->row_data(*state->latest_assistant_row);
      if (latest) {
        latest->saved = true;
        state->chat_messages->set_row_data(*state->latest_assistant_row, *latest);
      }
    }
    (void)SaveActiveConversation(*current, *state);
    Status(*current, Choose(chinese, "这次问答已保存到记忆", "Exchange saved to memory"));
  });
  window->on_update_prompt_lines([weak](slint::SharedString text) {
    if (auto current = weak.lock()) (*current)->set_prompt_line_count(EstimatePromptLines(Text(text)));
  });
  window->on_open_conversation([weak, state, &language](slint::SharedString id_text) {
    if (state->busy) return;
    auto current = weak.lock();
    if (!current) return;
    (*current)->set_active_page(Shared("playground"));
    const std::string id = Text(id_text);
    if (id == state->active_conversation_id) return;
    const bool chinese = Chinese(language);
    const auto saved = SaveActiveConversation(*current, *state);
    if (!saved.ok()) Status(*current, Choose(chinese, "保存当前对话失败", "Could not save the current conversation"), saved.message());
    if (!RestoreConversation(*current, *state, id, chinese)) {
      Status(*current, Choose(chinese, "找不到这条历史对话", "Conversation was not found"));
    }
  });
  window->on_delete_conversation([weak, state, &language](slint::SharedString id_text) {
    if (state->busy) return;
    auto current = weak.lock();
    if (!current) return;
    const std::string id = Text(id_text);
    const auto found = std::find_if(state->conversation_history.begin(), state->conversation_history.end(),
        [&id](const auto& record) { return record.id == id; });
    if (found == state->conversation_history.end()) return;

    const auto removed_index = static_cast<std::size_t>(
        std::distance(state->conversation_history.begin(), found));
    const bool deleting_active = state->active_conversation_id == id;
    ConversationRecord removed = std::move(*found);
    state->conversation_history.erase(found);
    const isvik::Status saved = WriteConversationHistory(*state);
    const bool chinese = Chinese(language);
    if (!saved.ok()) {
      state->conversation_history.insert(
          state->conversation_history.begin() +
              static_cast<std::vector<ConversationRecord>::difference_type>(removed_index),
          std::move(removed));
      RefreshConversationItems(*current, *state);
      Status(*current, Choose(chinese, "删除对话失败", "Could not delete conversation"), saved.message());
      return;
    }

    if (deleting_active) {
      state->active_conversation_id.clear();
      state->conversation.clear();
      state->chat_messages->clear();
      state->latest_assistant_row.reset();
      state->can_remember_exchange = false;
      state->exchange_saved = false;
      (*current)->set_can_remember_exchange(false);
      (*current)->set_exchange_saved(false);
      (*current)->set_response_text(Shared(""));
      (*current)->set_last_prompt(Shared(""));
      (*current)->set_prompt_text(Shared(""));
      (*current)->set_prompt_line_count(2);
      SetContextStatus(*current, *state, 0U, 0U, 0U, chinese);
    }
    RefreshConversationItems(*current, *state);
    Status(*current, Choose(chinese, "对话已删除", "Conversation deleted"));
  });
  window->on_pin_memory([weak, state, &language](slint::SharedString id_text, bool pinned) {
    auto current = weak.lock();
    if (!current || !state->memories) return;
    const auto status = state->memories->SetPinned(Text(id_text), state->memory_session, pinned);
    const bool chinese = Chinese(language);
    if (!status.ok()) { (*current)->set_memory_status(Shared(status.message())); return; }
    (*current)->set_selected_memory_pinned(pinned);
    RefreshMemories(*current, *state, chinese, Text((*current)->get_memory_query()));
    (*current)->set_memory_status(Shared(Choose(chinese, pinned ? "记忆已固定" : "已取消固定", pinned ? "Memory pinned" : "Pin removed")));
  });
  window->on_remove_memory([weak, state, &language](slint::SharedString id_text) {
    auto current = weak.lock();
    if (!current || !state->memories) return;
    const auto status = state->memories->Remove(Text(id_text), state->memory_session);
    const bool chinese = Chinese(language);
    if (!status.ok()) { (*current)->set_memory_status(Shared(status.message())); return; }
    (*current)->set_selected_memory_id(Shared(""));
    (*current)->set_selected_memory_pinned(false);
    (*current)->set_memory_content(Shared(""));
    (*current)->set_memory_tags(Shared(""));
    RefreshMemories(*current, *state, chinese, Text((*current)->get_memory_query()));
    (*current)->set_memory_status(Shared(Choose(chinese, "记忆已删除", "Memory deleted")));
  });
  window->on_update_context([weak, state, &language](slint::SharedString limit_text,
                                                     bool use_memories,
                                                     bool session_enabled) {
    auto current = weak.lock();
    if (!current) return;
    const std::string text = Text(limit_text);
    uint64_t limit = 0U;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), limit);
    const bool chinese = Chinese(language);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || limit == 0U || limit > 1000000U) {
      (*current)->set_memory_status(Shared(Choose(chinese, "上下文上限必须是 1 至 1000000 的整数", "Context limit must be an integer from 1 to 1000000")));
      return;
    }
    if (state->memories) {
      const auto status = state->memories->SetSessionMemoryEnabled(state->memory_session, session_enabled);
      if (!status.ok()) { (*current)->set_memory_status(Shared(status.message())); return; }
    }
    state->context_limit = limit;
    state->use_memories = use_memories;
    state->session_memory_enabled = session_enabled;
    (*current)->set_context_memories_enabled(use_memories);
    (*current)->set_session_memory_enabled(session_enabled);
    SetContextStatus(*current, *state, 0U, 0U, 0U, chinese);
    RefreshMemories(*current, *state, chinese, Text((*current)->get_memory_query()));
    (*current)->set_memory_status(Shared(Choose(chinese, "上下文设置已应用", "Context settings applied")));
  });

  auto scan = [weak, state, &language](std::filesystem::path directory) {
    if (state->busy) return;
    const bool chinese = Chinese(language);
    if (auto current = weak.lock()) {
      Busy(*current, *state, true);
      Status(*current, Choose(chinese, "正在扫描模型目录…", "Scanning model directory…"));
    }
    state->worker = std::jthread([weak, state, directory = std::move(directory), chinese] {
      auto result = [&]() -> isvik::StatusOr<isvik::ModelScan> {
        try { return isvik::ScanModelDirectory(directory); }
        catch (const std::exception& error) { return isvik::Status::Unavailable(error.what()); }
      }();
      (void)slint::invoke_from_event_loop([weak, state, directory, result = std::move(result), chinese] {
        if (auto current = weak.lock()) {
          Busy(*current, *state, false);
          if (!result.ok()) { Status(*current, Choose(chinese, "扫描失败，请检查目录", "Scan failed. Check the directory"), result.status().message()); return; }
          auto proposed = state->catalog;
          proposed.Merge(result.value());
          proposed.set_folder(directory);
          const auto status = Commit(*state, std::move(proposed));
          if (!status.ok()) { Status(*current, Choose(chinese, "无法保存模型库", "Cannot save model library"), status.message()); return; }
          if (state->selected_id.empty() && !state->catalog.models().empty()) state->selected_id = state->catalog.models().front().model.id;
          (*current)->set_folder_path(Shared(isvik::PathUtf8(directory)));
          (*current)->set_search_text(Shared(""));
          RefreshModels(*current, *state, chinese);
          std::string message = Choose(chinese, "扫描完成，发现 ", "Scan complete: ") + std::to_string(result.value().models.size()) + Choose(chinese, " 个模型", " models found");
          std::string warnings;
          for (const auto& warning : result.value().warnings) warnings += warning + "\n";
          Status(*current, message, warnings);
        }
      });
    });
  };
  window->on_scan_folder([scan](slint::SharedString path) { scan(isvik::PathFromUtf8(Text(path))); });
  window->on_browse_folder([weak, state, scan, &language] {
    if (state->busy) return;
    auto current = weak.lock();
    if (!current) return;
    const auto selected = isvik::app::ChooseModelFolder(isvik::PathFromUtf8(Text((*current)->get_folder_path())));
    if (selected.ok()) scan(selected.value());
    else if (selected.status().code() != isvik::StatusCode::kCancelled) Status(*current, Choose(Chinese(language), "无法选择目录", "Cannot choose directory"), selected.status().message());
  });

  auto load = [weak, state, &language](slint::SharedString path_text, slint::SharedString device_text) {
    if (state->busy || state->api_server_transition || state->api_server.IsRunning()) return;
    auto current = weak.lock();
    if (!current) return;
    const bool chinese = Chinese(language);
    const auto resolved = isvik::ResolveModelEntry(isvik::PathFromUtf8(Text(path_text)));
    if (!resolved.ok()) { Status(*current, Choose(chinese, "找不到模型入口，请重新扫描", "Model entry not found. Rescan the folder"), resolved.status().message()); return; }
    const isvik::CatalogModel* entry = nullptr;
    for (const auto& candidate : state->catalog.models()) if (candidate.model.path == resolved.value()) { entry = &candidate; break; }
    if (!entry || !CanRun(*entry, state->catalog.backend())) {
      std::string reason = "Model is not in the catalog";
      if (entry) {
        if (!entry->available) reason = Choose(chinese, "模型文件缺失或无效", "Model files are missing or invalid");
        else if (state->catalog.backend() == isvik::BackendType::kTensorRt && entry->model.format == isvik::ModelFormat::kOnnx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP)
          reason = BackendIssue(isvik::BackendType::kOnnxRuntime, chinese);
#else
          reason = Choose(chinese,
              "此版本没有包含 TensorRT-RTX 的 ONNX 执行组件。",
              "This build does not include the TensorRT-RTX ONNX execution component.");
#endif
        }
        else if (state->catalog.backend() == isvik::BackendType::kTensorRt) reason = BackendIssue(state->catalog.backend(), chinese);
        else if (entry->model.format == isvik::ModelFormat::kGguf && !entry->issue.empty()) reason = entry->issue;
        else if (entry->model.format == isvik::ModelFormat::kOnnx) reason = Choose(chinese, "已识别 ONNX 模型；当前版本还不能用它进行对话。", "ONNX model recognized; this build cannot use it for chat yet.");
#if defined(_WIN32) && defined(_DEBUG)
        else if (entry->model.path.filename() == "openvino_language_model.xml") reason = Choose(chinese, "此多模态模型请使用 Release 版加载；本机 Debug SDK 存在断点退出问题。", "Use Release for this VLM. The local Debug SDK aborts while loading this layout.");
#endif
        else reason = BackendIssue(state->catalog.backend(), chinese);
      }
      Status(*current, Choose(chinese, "模型暂不可运行", "Model cannot be run"), reason);
      return;
    }
    const auto model = entry->model;
    const auto device = Text(device_text);
    Busy(*current, *state, true);
    Status(*current, Choose(chinese, "正在加载 ", "Loading ") + model.display_name + " · " + device);
    const auto backend = state->catalog.backend();
    const std::string execution_route =
        backend == isvik::BackendType::kTensorRt && model.format == isvik::ModelFormat::kOnnx
            ? " · runtime=TensorRT-RTX EP (ONNX Runtime GenAI + CPU fallback)"
            : "";
    Log("INFO", "Loading model: " + model.display_name + " · " + Metadata(model, device) +
        execution_route + " · path=" + isvik::PathUtf8(model.path));
#if defined(ISVIK_HAS_OPENVINO_GENAI) || defined(ISVIK_HAS_TENSORRT) || defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
    const auto load_started = std::chrono::steady_clock::now();
    state->worker = std::jthread([weak, state, model, device, backend, chinese, load_started] {
      isvik::Status status = isvik::Status::Unavailable("selected inference backend is not built");
      if (backend == isvik::BackendType::kTensorRt) {
#if defined(ISVIK_HAS_TENSORRT)
        int device_index = 0;
        const std::string prefix = "GPU.";
        if (device.starts_with(prefix)) {
          const auto suffix = std::string_view(device).substr(prefix.size());
          const auto parsed = std::from_chars(suffix.data(), suffix.data() + suffix.size(), device_index);
          if (parsed.ec != std::errc{} || parsed.ptr != suffix.data() + suffix.size()) device_index = -1;
        }
        if (model.format == isvik::ModelFormat::kOnnx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP) && defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
          status = state->onnxruntime_engine.LoadModel(model, device);
#else
          status = isvik::Status::Unsupported("TensorRT-RTX ONNX support is not included in this build");
#endif
        } else {
          status = state->tensorrt_engine.LoadModel(model, device_index);
        }
#endif
      } else if (backend == isvik::BackendType::kOnnxRuntime) {
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
        status = state->onnxruntime_engine.LoadModel(model, device);
#endif
      } else {
#if defined(ISVIK_HAS_OPENVINO_GENAI)
        status = state->engine.LoadModel(model, device);
#endif
      }
      (void)slint::invoke_from_event_loop([weak, state, model, device, backend, status, chinese, load_started] {
        if (auto current = weak.lock()) {
          Busy(*current, *state, false);
          if (status.ok()) {
            state->loaded_model = model;
            state->loaded_backend = backend;
            std::string actual_device = device;
            if (backend == isvik::BackendType::kTensorRt && model.format == isvik::ModelFormat::kOnnx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP) && defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
              actual_device = state->onnxruntime_engine.loaded_device();
#endif
            }
            (*current)->set_model_loaded(true);
            (*current)->set_loaded_name(Shared(model.display_name));
            (*current)->set_loaded_device(Shared(actual_device));
            (*current)->set_loaded_model_info(Shared(Metadata(model, actual_device)));
            (*current)->set_api_model_info(Shared(model.display_name + " · " + BackendLabel(backend) + " · " + actual_device));
            (*current)->set_active_page(Shared("playground"));
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - load_started).count();
            std::ostringstream log;
            log << "Model ready: " << model.display_name << " · " << Metadata(model, actual_device)
                << " · load " << std::fixed << std::setprecision(2) << seconds << " s";
            Log("INFO", log.str());
            Status(*current, Choose(chinese, "模型已就绪", "Model is ready"));
          } else {
            Log("ERROR", "Model load failed: " + status.message());
            Status(*current, Choose(chinese, "模型加载失败，查看详细信息", "Model load failed. View details"), status.message());
          }
          RefreshModels(*current, *state, chinese);
        }
      });
    });
#else
    Busy(*current, *state, false);
    Status(*current, "Inference backend unavailable", "Enable OpenVINO GenAI, TensorRT, or ONNX Runtime GenAI");
#endif
  };
  window->on_load_model(load);
  window->on_choose_chat_model([weak, state, load, &language](int index) {
    if (state->busy || state->api_server_transition || state->api_server.IsRunning() || index < 0 || static_cast<std::size_t>(index) >= state->chat_ids.size()) return;
    if (auto current = weak.lock()) {
      SelectModel(*current, *state, state->chat_ids[static_cast<std::size_t>(index)], Chinese(language), state->catalog.backend());
      load((*current)->get_selected_path(), (*current)->get_device_name());
    }
  });

  auto update_catalog = [weak, state, &language](const std::string& id, auto change) {
    if (state->busy || state->api_server_transition || state->api_server.IsRunning()) return;
    auto proposed = state->catalog;
    auto status = change(proposed);
    if (status.ok()) status = Commit(*state, std::move(proposed));
    if (auto current = weak.lock()) {
      const bool chinese = Chinese(language);
      if (!status.ok()) { Status(*current, Choose(chinese, "模型库更新失败", "Library update failed"), status.message()); return; }
      if (state->loaded_model && state->loaded_model->id == id && state->catalog.Find(id)) state->loaded_model->display_name = state->catalog.Find(id)->model.display_name;
      RefreshModels(*current, *state, chinese);
      Status(*current, Choose(chinese, "模型库已保存", "Model library saved"));
    }
  };
  window->on_rename_model([update_catalog](slint::SharedString id, slint::SharedString name) { update_catalog(Text(id), [&](auto& catalog) { return catalog.Rename(Text(id), Text(name)); }); });
  window->on_default_model([update_catalog](slint::SharedString id) { update_catalog(Text(id), [&](auto& catalog) { return catalog.SetDefault(Text(id)); }); });
  window->on_remove_model([update_catalog](slint::SharedString id) { update_catalog(Text(id), [&](auto& catalog) { return catalog.Remove(Text(id)); }); });
  window->on_unload_model([weak, state, &language] {
    if (state->busy || state->api_server_transition || state->api_server.IsRunning()) return;
    const bool chinese = Chinese(language);
    if (auto current = weak.lock()) Busy(*current, *state, true);
    state->worker = std::jthread([weak, state, chinese] {
#if defined(ISVIK_HAS_OPENVINO_GENAI)
      state->engine.UnloadModel();
#endif
#if defined(ISVIK_HAS_TENSORRT)
      state->tensorrt_engine.UnloadModel();
#endif
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
      state->onnxruntime_engine.UnloadModel();
#endif
      (void)slint::invoke_from_event_loop([weak, state, chinese] {
        if (auto current = weak.lock()) {
          state->loaded_model.reset();
          state->loaded_backend.reset();
          Busy(*current, *state, false);
          (*current)->set_model_loaded(false);
          (*current)->set_loaded_name(Shared(""));
          (*current)->set_loaded_device(Shared(""));
          (*current)->set_loaded_model_info(Shared(""));
          (*current)->set_api_model_info(Shared(""));
          RefreshModels(*current, *state, chinese);
          Status(*current, Choose(chinese, "模型已卸载", "Model unloaded"));
        }
      });
    });
  });

  window->on_generate_text([weak, state, &language](slint::SharedString prompt_text,
      slint::SharedString token_text, int decoding_index, slint::SharedString decoding_value_text) {
#if defined(ISVIK_HAS_OPENVINO_GENAI) || defined(ISVIK_HAS_TENSORRT) || defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
    if (state->busy || !state->loaded_model) return;
    auto current = weak.lock();
    if (!current) return;
    int max_tokens = 0;
    const auto tokens = Text(token_text);
    const auto parsed = std::from_chars(tokens.data(), tokens.data() + tokens.size(), max_tokens);
    const bool chinese = Chinese(language);
    if (parsed.ec != std::errc{} || parsed.ptr != tokens.data() + tokens.size() || max_tokens <= 0 || max_tokens > 32768) {
      Status(*current, Choose(chinese, "最大词元数应为 1–32768", "Max tokens must be between 1 and 32768")); return;
    }
    if (!IsSupportedDecoding(decoding_index)) {
      Status(*current, Choose(chinese, "当前运行时不支持此解码算法", "The current runtime does not support this decoding algorithm"));
      return;
    }
    const std::string decoding_value = Text(decoding_value_text);
    isvik::GenerationConfig generation;
    generation.max_tokens = max_tokens;
    generation.stream = true;
    if (decoding_index == 0) {
      float temperature = 0.0F;
      if (!ParseFloat(decoding_value, temperature) || temperature <= 0.0F || temperature > 5.0F) {
        Status(*current, Choose(chinese, "温度应大于 0 且不超过 5", "Temperature must be greater than 0 and at most 5")); return;
      }
      generation.decoding_mode = isvik::DecodingMode::kSampling;
      generation.sampling.temperature = temperature;
    } else if (decoding_index == 2) {
      int top_k = 0;
      const auto parsed = std::from_chars(decoding_value.data(),
          decoding_value.data() + decoding_value.size(), top_k);
      if (parsed.ec != std::errc{} || parsed.ptr != decoding_value.data() + decoding_value.size() ||
          top_k < 1 || top_k > 1000) {
        Status(*current, Choose(chinese, "Top-k 应为 1–1000 的整数", "Top-k must be an integer from 1 to 1000")); return;
      }
      generation.decoding_mode = isvik::DecodingMode::kSampling;
      generation.sampling.top_k = top_k;
    } else if (decoding_index == 3) {
      float top_p = 0.0F;
      if (!ParseFloat(decoding_value, top_p) || top_p <= 0.0F || top_p > 1.0F) {
        Status(*current, Choose(chinese, "Top-p 应大于 0 且不超过 1", "Top-p must be greater than 0 and at most 1")); return;
      }
      generation.decoding_mode = isvik::DecodingMode::kSampling;
      generation.sampling.top_p = top_p;
    } else if (decoding_index == 1 || decoding_index == 6) {
      int num_beams = 0;
      const auto parsed = std::from_chars(decoding_value.data(),
          decoding_value.data() + decoding_value.size(), num_beams);
      if (parsed.ec != std::errc{} || parsed.ptr != decoding_value.data() + decoding_value.size() ||
          num_beams < 2 || num_beams > 16 || (decoding_index == 6 && num_beams % 2 != 0)) {
        Status(*current, decoding_index == 6
            ? Choose(chinese, "多样化束搜索需要 2–16 的偶数光束数", "Diverse beam search requires an even beam count from 2 to 16")
            : Choose(chinese, "光束数应为 2–16 的整数", "Beam count must be an integer from 2 to 16")); return;
      }
      generation.num_beams = num_beams;
      generation.decoding_mode = decoding_index == 6
          ? isvik::DecodingMode::kDiverseBeamSearch : isvik::DecodingMode::kBeamSearch;
    } else {
      generation.decoding_mode = isvik::DecodingMode::kGreedy;
    }
    const std::string decoding_info = DecodingInfo(decoding_index, decoding_value);
    const auto prompt = Text(prompt_text);
    if (prompt.find_first_not_of(" \t\r\n") == std::string::npos) return;
    isvik::ContextOptions context_options;
    context_options.max_context_tokens = state->context_limit;
    if (state->loaded_model->context_length > 0U) {
      context_options.max_context_tokens = std::min<uint64_t>(
          context_options.max_context_tokens, state->loaded_model->context_length);
    }
    context_options.reserved_output_tokens = static_cast<uint64_t>(max_tokens);
    context_options.compression = isvik::ContextCompressionMode::kHybrid;
    if (context_options.reserved_output_tokens >= context_options.max_context_tokens) {
      Status(*current, Choose(chinese, "上下文上限必须大于最大输出词元数", "Context limit must exceed the output token limit"));
      return;
    }
    std::vector<isvik::InferenceMessage> conversation = state->conversation;
    conversation.push_back({isvik::MessageRole::kUser, prompt, {}, {}});
    const auto built_context = isvik::BuildChatContext(
        conversation, state->memories.get(), state->memory_session,
        state->use_memories, context_options);
    if (!built_context.ok()) {
      Status(*current, Choose(chinese, "上下文无法放入预算", "Context does not fit the budget"),
             built_context.status().message());
      return;
    }
    isvik::UnifiedInferenceRequest request;
    request.request_id = "gui-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    request.model_id = state->loaded_model->id;
    request.messages = built_context.value().messages;
    request.generation = generation;
    state->cancellation = isvik::CancellationSource();
    const auto token = state->cancellation.token();
    Busy(*current, *state, true);
    (*current)->set_generating(true);
    state->can_remember_exchange = false;
    state->exchange_saved = false;
    (*current)->set_can_remember_exchange(false);
    (*current)->set_exchange_saved(false);
    (*current)->set_response_text(Shared(""));
    (*current)->set_last_prompt(Shared(prompt));
    Status(*current, Choose(chinese, "正在生成…", "Generating…"));
    const auto estimated_tokens = built_context.value().estimated_prompt_tokens;
    const auto dropped_messages = built_context.value().dropped_history_messages;
    const auto memories_used = built_context.value().memories_used;
    const auto effective_context_limit = context_options.max_context_tokens;
    const std::string model_info = Metadata(*state->loaded_model, Text((*current)->get_loaded_device()));
    const std::string request_info =
        "context " + std::to_string(estimated_tokens) + "/" +
        std::to_string(effective_context_limit) + " estimated · max output " +
        std::to_string(max_tokens) + " · memories " + std::to_string(memories_used) +
        " · trimmed " + std::to_string(dropped_messages) + " · " +
        Text((*current)->get_loaded_device()) + " · " + decoding_info;
    if (state->latest_assistant_row && *state->latest_assistant_row < state->chat_messages->row_count()) {
      auto previous = state->chat_messages->row_data(*state->latest_assistant_row);
      if (previous) {
        previous->is_latest = false;
        state->chat_messages->set_row_data(*state->latest_assistant_row, *previous);
      }
    }
    state->chat_messages->push_back(Message("user", prompt));
    const std::size_t assistant_row = state->chat_messages->row_count();
    state->chat_messages->push_back(Message("assistant", {}, model_info, request_info, {}, true));
    state->latest_assistant_row = assistant_row;
    (*current)->set_chat_messages(state->chat_messages);
    auto generated_text = std::make_shared<std::string>();
    auto completion = std::make_shared<std::optional<isvik::InferenceCompleted>>();
    Log("INFO", "Generation started: model=" + state->loaded_model->display_name +
        "; device=" + Text((*current)->get_loaded_device()) +
        "; max_output_tokens=" + std::to_string(max_tokens) +
        "; context_estimate=" + std::to_string(estimated_tokens) + "/" +
        std::to_string(effective_context_limit) +
        "; decoding=" + decoding_info +
        "; memories=" + std::to_string(memories_used) +
        "; trimmed_messages=" + std::to_string(dropped_messages) + "; prompt text omitted");
    const auto generation_backend = state->loaded_backend.value_or(state->catalog.backend());
    const bool generation_uses_tensorrt_rtx_onnx =
        generation_backend == isvik::BackendType::kTensorRt && state->loaded_model &&
        state->loaded_model->format == isvik::ModelFormat::kOnnx;
    state->worker = std::jthread([weak, state, request = std::move(request), token, chinese,
                                  prompt, generated_text, estimated_tokens,
                                  dropped_messages, memories_used, effective_context_limit,
                                  model_info, request_info, assistant_row, completion,
                                  generation_backend, generation_uses_tensorrt_rtx_onnx] {
      const auto event_handler = [weak, state, generated_text, assistant_row, completion](const isvik::InferenceEvent& event) {
        const auto* delta = std::get_if<isvik::ContentDelta>(&event.payload);
        if (delta) {
          *generated_text += delta->text;
          (void)slint::invoke_from_event_loop([weak, state, piece = delta->text, assistant_row] {
            if (auto current = weak.lock()) {
              (*current)->set_response_text(Shared(Text((*current)->get_response_text()) + piece));
              if (assistant_row < state->chat_messages->row_count()) {
                auto item = state->chat_messages->row_data(assistant_row);
                if (item) {
                  item->content = Shared(Text(item->content) + piece);
                  state->chat_messages->set_row_data(assistant_row, *item);
                }
              }
            }
          });
          return;
        }
        if (const auto* metrics = std::get_if<isvik::InferenceCompleted>(&event.payload)) {
          *completion = *metrics;
        }
      };
      isvik::Status status = isvik::Status::Unavailable("selected inference backend is not built");
      if (generation_backend == isvik::BackendType::kTensorRt) {
        if (generation_uses_tensorrt_rtx_onnx) {
#if defined(ISVIK_HAS_TENSORRT_RTX_EP) && defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
          status = state->onnxruntime_engine.Generate(request, token, event_handler);
#endif
        } else {
#if defined(ISVIK_HAS_TENSORRT)
          status = state->tensorrt_engine.Generate(request, token, event_handler);
#endif
        }
      } else if (generation_backend == isvik::BackendType::kOnnxRuntime) {
#if defined(ISVIK_HAS_ONNXRUNTIME_GENAI)
        status = state->onnxruntime_engine.Generate(request, token, event_handler);
#endif
      } else {
#if defined(ISVIK_HAS_OPENVINO_GENAI)
        status = state->engine.Generate(request, token, event_handler);
#endif
      }
      (void)slint::invoke_from_event_loop([weak, state, status, chinese, prompt, generated_text,
                                           estimated_tokens, dropped_messages, memories_used,
                                           effective_context_limit, assistant_row, completion,
                                           model_info, request_info] {
        if (auto current = weak.lock()) {
          Busy(*current, *state, false);
          (*current)->set_generating(false);
          if (assistant_row < state->chat_messages->row_count()) {
            auto item = state->chat_messages->row_data(assistant_row);
            if (item) {
              item->content = Shared(*generated_text);
              item->model_info = Shared(model_info);
              item->request_info = Shared(request_info);
              if (completion->has_value()) {
                const auto& metrics = completion->value();
                item->statistics = Shared(Statistics(metrics));
                item->request_info = Shared(
                    "prompt " + std::to_string(metrics.input_tokens) + " tokens · " + request_info);
                std::ostringstream log;
                log << "Generation completed: input=" << metrics.input_tokens
                    << " tokens; output=" << metrics.output_tokens << " tokens; duration="
                    << std::fixed << std::setprecision(2) << metrics.duration_seconds
                    << " s; throughput=" << metrics.tokens_per_second << " tokens/s";
                Log("INFO", log.str());
              } else if (!status.ok()) {
                item->request_info = Shared(request_info + (status.code() == isvik::StatusCode::kCancelled
                    ? " · stopped" : " · failed"));
              }
              state->chat_messages->set_row_data(assistant_row, *item);
            }
          }
          if (status.ok()) {
            state->conversation.push_back({isvik::MessageRole::kUser, prompt, {}, {}});
            state->conversation.push_back({isvik::MessageRole::kAssistant, *generated_text, {}, {}});
            (void)SaveActiveConversation(*current, *state);
            state->can_remember_exchange = true;
            (*current)->set_can_remember_exchange(true);
            Status(*current, Choose(chinese, "生成完成", "Generation complete"));
            (*current)->set_prompt_text(Shared(""));
            (*current)->set_prompt_line_count(2);
            SetContextStatus(*current, *state, estimated_tokens, dropped_messages,
                             memories_used, chinese, effective_context_limit);
          }
          else if (status.code() == isvik::StatusCode::kCancelled) Status(*current, Choose(chinese, "生成已停止", "Generation stopped"));
          else Status(*current, Choose(chinese, "生成失败，查看详细信息", "Generation failed. View details"), status.message());
        }
      });
    });
#else
    (void)prompt_text; (void)token_text;
#endif
  });
  window->on_cancel_generation([state] { state->cancellation.Cancel(); });
  window->on_clear_response([&] {
    if (state->busy) return;
    const auto saved = SaveActiveConversation(window, *state);
    if (!saved.ok()) Log("ERROR", "Could not save conversation before starting a new one: " + saved.message());
    state->conversation.clear();
    state->chat_messages->clear();
    state->latest_assistant_row.reset();
    state->active_conversation_id.clear();
    state->can_remember_exchange = false;
    state->exchange_saved = false;
    window->set_can_remember_exchange(false);
    window->set_exchange_saved(false);
    window->set_response_text(Shared(""));
    window->set_last_prompt(Shared(""));
    window->set_current_conversation_title(Shared(""));
    window->set_active_conversation_id(Shared(""));
    window->set_prompt_text(Shared(""));
    window->set_prompt_line_count(2);
    SetContextStatus(window, *state, 0U, 0U, 0U, Chinese(language));
    Status(window, "");
  });
  window->run();
  state->cancellation.Cancel();
  if (state->worker.joinable()) state->worker.join();
  if (state->memories) (void)state->memories->CloseSession(state->memory_session);
  Log("INFO", "GUI closed");
  return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  SetConsoleCP(CP_UTF8);
  SetConsoleOutputCP(CP_UTF8);
  std::vector<std::string> arguments;
  std::vector<char*> pointers;
  arguments.reserve(static_cast<std::size_t>(argc));
  pointers.reserve(static_cast<std::size_t>(argc));
  for (int index = 0; index < argc; ++index) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, argv[index], -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) return 2;
    std::string argument(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, argv[index], -1, argument.data(), size, nullptr, nullptr);
    argument.pop_back();
    arguments.push_back(std::move(argument));
  }
  for (auto& argument : arguments) pointers.push_back(argument.data());
  return RunApplication(argc, pointers.data());
}
#else
int main(int argc, char** argv) { return RunApplication(argc, argv); }
#endif
