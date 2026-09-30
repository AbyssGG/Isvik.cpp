#include <cmath>
#include <limits>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include "isvik/core/backend_capabilities.h"
#include "isvik/core/cancellation.h"
#include "isvik/core/generation_config.h"
#include "isvik/core/inference_event.h"
#include "isvik/core/inference_request.h"
#include "isvik/core/model_registry.h"
#include "isvik/core/status.h"
#include "isvik/i18n/localization.h"

#if defined(ISVIK_TEST_OPENVINO_BACKEND)
#include "isvik/backends/openvino/device_provider.h"
#endif

namespace isvik {
namespace {

ModelDescriptor MakeModel(std::string id = "tinyllama") {
  ModelDescriptor descriptor;
  descriptor.id = std::move(id);
  descriptor.display_name = "TinyLlama";
  descriptor.path = "models/tinyllama.xml";
  descriptor.format = ModelFormat::kOpenVinoIr;
  descriptor.compatible_backends = {BackendType::kOpenVino};
  return descriptor;
}

TEST(StatusTest, DefaultStatusIsOk) {
  const Status status;
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kOk);
  EXPECT_TRUE(status.message().empty());
}

TEST(StatusTest, ErrorStatusKeepsCodeAndMessage) {
  const Status status = Status::NotFound("missing model");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kNotFound);
  EXPECT_EQ(status.message(), "missing model");
}

TEST(StatusOrTest, ReturnsStoredValue) {
  const StatusOr<int> result(42);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), 42);
}

TEST(StatusOrTest, PreservesErrorStatus) {
  const StatusOr<int> result(Status::InvalidArgument("bad value"));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(StatusOrTest, ReadingAnErrorThrows) {
  const StatusOr<int> result(Status::NotFound("missing"));
  EXPECT_THROW(static_cast<void>(result.value()), std::logic_error);
}

TEST(GenerationConfigTest, DefaultsToSamplingAndStreaming) {
  const GenerationConfig config;
  EXPECT_EQ(config.decoding_mode, DecodingMode::kSampling);
  EXPECT_TRUE(config.stream);
  EXPECT_TRUE(ValidateGenerationConfig(config).ok());
}

TEST(GenerationConfigTest, RejectsZeroMaximumTokens) {
  GenerationConfig config;
  config.max_tokens = 0;
  EXPECT_EQ(ValidateGenerationConfig(config).code(), StatusCode::kInvalidArgument);
}

TEST(GenerationConfigTest, AcceptsTopPInRange) {
  GenerationConfig config;
  config.sampling.top_p = 0.9F;
  EXPECT_TRUE(ValidateGenerationConfig(config).ok());
}

TEST(GenerationConfigTest, RejectsTopPOutsideRange) {
  GenerationConfig config;
  config.sampling.top_p = 1.1F;
  EXPECT_EQ(ValidateGenerationConfig(config).code(), StatusCode::kInvalidArgument);
}

TEST(GenerationConfigTest, RejectsZeroTopK) {
  GenerationConfig config;
  config.sampling.top_k = 0;
  EXPECT_EQ(ValidateGenerationConfig(config).code(), StatusCode::kInvalidArgument);
}

TEST(GenerationConfigTest, RejectsNonFiniteTemperature) {
  GenerationConfig config;
  config.sampling.temperature = std::numeric_limits<float>::infinity();
  EXPECT_EQ(ValidateGenerationConfig(config).code(), StatusCode::kInvalidArgument);
}

TEST(GenerationConfigTest, RejectsEmptyStopSequence) {
  GenerationConfig config;
  config.stop_sequences.emplace_back();
  EXPECT_EQ(ValidateGenerationConfig(config).code(), StatusCode::kInvalidArgument);
}

TEST(InferenceRequestTest, AcceptsAValidUserRequest) {
  UnifiedInferenceRequest request;
  request.model_id = "tinyllama";
  request.messages.push_back({MessageRole::kUser, "Hello", {}, {}});
  EXPECT_TRUE(ValidateInferenceRequest(request).ok());
}

TEST(InferenceRequestTest, RequiresAModelId) {
  UnifiedInferenceRequest request;
  request.messages.push_back({MessageRole::kUser, "Hello", {}, {}});
  EXPECT_EQ(ValidateInferenceRequest(request).code(), StatusCode::kInvalidArgument);
}

TEST(InferenceRequestTest, RequiresAtLeastOneMessage) {
  UnifiedInferenceRequest request;
  request.model_id = "tinyllama";
  EXPECT_EQ(ValidateInferenceRequest(request).code(), StatusCode::kInvalidArgument);
}

TEST(InferenceRequestTest, RequiresToolCallIdForToolMessages) {
  UnifiedInferenceRequest request;
  request.model_id = "tinyllama";
  request.messages.push_back({MessageRole::kTool, "result", {}, {}});
  EXPECT_EQ(ValidateInferenceRequest(request).code(), StatusCode::kInvalidArgument);
}

TEST(InferenceEventTest, CarriesContentDeltaPayload) {
  const InferenceEvent event{"request-1", ContentDelta{"hello"}};
  ASSERT_TRUE(std::holds_alternative<ContentDelta>(event.payload));
  EXPECT_EQ(std::get<ContentDelta>(event.payload).text, "hello");
}

TEST(CancellationTest, DefaultTokenIsNotCancelled) {
  const CancellationToken token;
  EXPECT_FALSE(token.IsCancellationRequested());
}

TEST(CancellationTest, SourceCancelsSharedTokens) {
  const CancellationSource source;
  const CancellationToken token = source.token();
  EXPECT_FALSE(token.IsCancellationRequested());
  source.Cancel();
  EXPECT_TRUE(token.IsCancellationRequested());
}

TEST(CapabilityTest, CapabilitiesDefaultToUnsupported) {
  const BackendCapabilities capabilities;
  EXPECT_FALSE(capabilities.supports_cpu);
  EXPECT_FALSE(capabilities.supports_sampling);
  EXPECT_FALSE(capabilities.supports_int4);
}

TEST(ModelRegistryTest, RegistersAndFindsModel) {
  ModelRegistry registry;
  ASSERT_TRUE(registry.Register(MakeModel()).ok());
  const StatusOr<ModelDescriptor> found = registry.Find("tinyllama");
  ASSERT_TRUE(found.ok());
  EXPECT_EQ(found.value().display_name, "TinyLlama");
  EXPECT_EQ(registry.size(), 1U);
}

TEST(ModelRegistryTest, RejectsDuplicateIds) {
  ModelRegistry registry;
  ASSERT_TRUE(registry.Register(MakeModel()).ok());
  EXPECT_EQ(registry.Register(MakeModel()).code(), StatusCode::kAlreadyExists);
}

TEST(ModelRegistryTest, RejectsEmptyId) {
  ModelRegistry registry;
  EXPECT_EQ(registry.Register(MakeModel("")).code(), StatusCode::kInvalidArgument);
}

TEST(ModelRegistryTest, RejectsEmptyDisplayName) {
  ModelRegistry registry;
  ModelDescriptor descriptor = MakeModel();
  descriptor.display_name.clear();
  EXPECT_EQ(registry.Register(std::move(descriptor)).code(), StatusCode::kInvalidArgument);
}

TEST(ModelRegistryTest, ListsModelsInStableIdOrder) {
  ModelRegistry registry;
  ASSERT_TRUE(registry.Register(MakeModel("z-model")).ok());
  ASSERT_TRUE(registry.Register(MakeModel("a-model")).ok());
  const std::vector<ModelDescriptor> models = registry.List();
  ASSERT_EQ(models.size(), 2U);
  EXPECT_EQ(models.front().id, "a-model");
}

TEST(ModelRegistryTest, RemovesRegisteredModel) {
  ModelRegistry registry;
  ASSERT_TRUE(registry.Register(MakeModel()).ok());
  EXPECT_TRUE(registry.Remove("tinyllama").ok());
  EXPECT_EQ(registry.size(), 0U);
}

TEST(ModelRegistryTest, ReportsMissingModel) {
  ModelRegistry registry;
  EXPECT_EQ(registry.Find("missing").status().code(), StatusCode::kNotFound);
  EXPECT_EQ(registry.Remove("missing").code(), StatusCode::kNotFound);
}

TEST(ModelRegistryTest, PreservesDescriptorWhenDuplicateIdIsRejected) {
  ModelRegistry registry;
  ModelDescriptor first = MakeModel();
  first.display_name = "Original";
  ASSERT_TRUE(registry.Register(std::move(first)).ok());

  ModelDescriptor duplicate = MakeModel();
  duplicate.display_name = "Duplicate";
  EXPECT_EQ(registry.Register(std::move(duplicate)).code(), StatusCode::kAlreadyExists);
  EXPECT_EQ(registry.Find("tinyllama").value().display_name, "Original");
}

TEST(LocalizationTest, DefaultsToSimplifiedChinese) {
  const LocalizationService localization;
  EXPECT_EQ(localization.GetText(MessageId::kOverview), "概览");
}

TEST(LocalizationTest, SwitchesLanguagesAtRuntime) {
  LocalizationService localization;
  localization.ToggleLanguage();
  EXPECT_EQ(localization.language(), Language::kEnglish);
  EXPECT_EQ(localization.GetText(MessageId::kOverview), "Overview");
  localization.ToggleLanguage();
  EXPECT_EQ(localization.language(), Language::kChineseSimplified);
}

TEST(LocalizationTest, InvalidMessageIdReturnsEmptyText) {
  const LocalizationService localization;
  EXPECT_TRUE(localization.GetText(static_cast<MessageId>(-1)).empty());
}

#if defined(ISVIK_TEST_OPENVINO_BACKEND)
TEST(OpenVinoDeviceTest, ClassifiesBuiltInPluginIds) {
  using openvino_backend::ClassifyDevice;
  using openvino_backend::DeviceType;
  EXPECT_EQ(ClassifyDevice("CPU"), DeviceType::kCpu);
  EXPECT_EQ(ClassifyDevice("GPU.0"), DeviceType::kGpu);
  EXPECT_EQ(ClassifyDevice("NPU"), DeviceType::kNpu);
  EXPECT_EQ(ClassifyDevice("AUTO"), DeviceType::kOther);
}
#endif

}  // namespace
}  // namespace isvik
