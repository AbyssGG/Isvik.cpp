#ifndef ISVIK_BACKENDS_OPENVINO_DEVICE_PROVIDER_H_
#define ISVIK_BACKENDS_OPENVINO_DEVICE_PROVIDER_H_

#include <string>
#include <string_view>
#include <vector>

#include "isvik/core/status.h"

namespace isvik::openvino_backend {

enum class DeviceType {
  kCpu,
  kGpu,
  kNpu,
  kOther,
};

struct DeviceInfo {
  std::string id;
  std::string full_name;
  DeviceType type = DeviceType::kOther;
};

[[nodiscard]] inline DeviceType ClassifyDevice(std::string_view device_id) {
  if (device_id.starts_with("CPU")) {
    return DeviceType::kCpu;
  }
  if (device_id.starts_with("GPU")) {
    return DeviceType::kGpu;
  }
  if (device_id.starts_with("NPU")) {
    return DeviceType::kNpu;
  }
  return DeviceType::kOther;
}

[[nodiscard]] StatusOr<std::vector<DeviceInfo>> EnumerateDevices();

}  // namespace isvik::openvino_backend

#endif  // ISVIK_BACKENDS_OPENVINO_DEVICE_PROVIDER_H_
