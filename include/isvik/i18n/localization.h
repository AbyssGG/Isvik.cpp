#ifndef ISVIK_I18N_LOCALIZATION_H_
#define ISVIK_I18N_LOCALIZATION_H_

#include <string_view>

namespace isvik {

enum class Language {
  kChineseSimplified,
  kEnglish,
};

enum class MessageId {
  kWindowTitle,
  kTagline,
  kOverview,
  kPlayground,
  kModels,
  kDevices,
  kStatusLabel,
  kStatusReady,
  kModelPathLabel,
  kModelPathPlaceholder,
  kDeviceLabel,
  kMaxTokensLabel,
  kLoadModel,
  kPromptLabel,
  kPromptPlaceholder,
  kGenerate,
  kCancel,
  kResponse,
  kClear,
  kRefreshDevices,
  kModelListLabel,
  kDeviceListLabel,
  kNoModelLoaded,
  kLoadingModel,
  kModelLoaded,
  kGenerating,
  kGenerationCancelled,
  kDevicesUnavailable,
  kLanguageButton,
  kCount,
};

class LocalizationService {
 public:
  explicit LocalizationService(Language language = Language::kChineseSimplified)
      : language_(language) {}

  [[nodiscard]] Language language() const { return language_; }
  void SetLanguage(Language language) { language_ = language; }
  void ToggleLanguage();
  [[nodiscard]] std::string_view GetText(MessageId message_id) const;

 private:
  Language language_;
};

}  // namespace isvik

#endif  // ISVIK_I18N_LOCALIZATION_H_
