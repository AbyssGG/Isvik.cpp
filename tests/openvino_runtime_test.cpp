#include <algorithm>

#include <gtest/gtest.h>

#include "isvik/backends/openvino/device_provider.h"

namespace isvik::openvino_backend {
namespace {

TEST(OpenVinoRuntimeTest, EnumeratesTheInstalledCpuPlugin) {
  const StatusOr<std::vector<DeviceInfo>> result = EnumerateDevices();
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_TRUE(std::any_of(result.value().begin(), result.value().end(), [](const DeviceInfo& device) {
    return device.type == DeviceType::kCpu;
  }));
}

}  // namespace
}  // namespace isvik::openvino_backend
