#include "isvik/i18n/localization.h"

#include <array>
#include <cstddef>

namespace isvik {
namespace {

constexpr std::array<std::string_view, static_cast<std::size_t>(MessageId::kCount)> kChinese = {
    "Isvik",
    "本地 AI 运行时",
    "概览",
    "试验场",
    "模型",
    "设备",
    "运行状态",
    "运行时已就绪，可加载本地模型",
    "OpenVINO 模型路径（.xml）",
    "选择模型 XML 文件",
    "设备",
    "最大生成词元数",
    "加载模型",
    "提示词",
    "输入提示词",
    "生成",
    "取消",
    "回答",
    "清空",
    "刷新设备",
    "已导入模型",
    "可用设备",
    "尚未加载模型",
    "正在加载模型…",
    "模型已加载",
    "正在生成…",
    "生成已取消",
    "设备列表暂不可用",
    "English",
};

constexpr std::array<std::string_view, static_cast<std::size_t>(MessageId::kCount)> kEnglish = {
    "Isvik",
    "Local AI Runtime",
    "Overview",
    "Playground",
    "Models",
    "Devices",
    "Runtime status",
    "Runtime ready. Load a local model.",
    "OpenVINO model path (.xml)",
    "Select the model XML file",
    "Device",
    "Max tokens",
    "Load model",
    "Prompt",
    "Enter a prompt",
    "Generate",
    "Cancel",
    "Response",
    "Clear",
    "Refresh devices",
    "Imported model",
    "Available devices",
    "No model loaded",
    "Loading model…",
    "Model loaded",
    "Generating…",
    "Generation cancelled",
    "Device list is not available",
    "中文",
};

}  // namespace

void LocalizationService::ToggleLanguage() {
  language_ = language_ == Language::kChineseSimplified
      ? Language::kEnglish
      : Language::kChineseSimplified;
}

std::string_view LocalizationService::GetText(MessageId message_id) const {
  const std::size_t index = static_cast<std::size_t>(message_id);
  if (index >= static_cast<std::size_t>(MessageId::kCount)) {
    return {};
  }
  return language_ == Language::kEnglish ? kEnglish[index] : kChinese[index];
}

}  // namespace isvik
