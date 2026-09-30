#include "isvik/backends/openvino/device_provider.h"

#include <openvino/openvino.hpp>

#include <exception>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace isvik::openvino_backend {
namespace {

std::filesystem::path PluginDirectory(const std::filesystem::path& install_root) {
#if defined(_WIN32)
#if defined(_DEBUG)
  constexpr char kConfiguration[] = "Debug";
#else
  constexpr char kConfiguration[] = "Release";
#endif
  return install_root / "runtime" / "bin" / "intel64" / kConfiguration;
#else
  return install_root / "runtime" / "lib" / "intel64";
#endif
}

std::vector<std::string> RegisterInstalledPlugins(
    ov::Core* core,
    const std::filesystem::path& plugin_directory) {
#if defined(_WIN32)
#if defined(_DEBUG)
  constexpr char kSuffix[] = "d.dll";
#else
  constexpr char kSuffix[] = ".dll";
#endif
#elif defined(__APPLE__)
  constexpr char kPrefix[] = "lib";
  constexpr char kSuffix[] = ".dylib";
#else
  constexpr char kPrefix[] = "lib";
  constexpr char kSuffix[] = ".so";
#endif

  const std::vector<std::string> plugin_ids = {"CPU", "GPU", "NPU"};
  std::vector<std::string> registration_errors;
  for (const std::string& plugin_id : plugin_ids) {
#if defined(_WIN32)
    const std::filesystem::path plugin_path = plugin_directory
        / ("openvino_intel_" + std::string(plugin_id == "CPU" ? "cpu_plugin" :
                                           plugin_id == "GPU" ? "gpu_plugin" : "npu_plugin")
           + kSuffix);
#else
    const std::string plugin_name = plugin_id == "CPU" ? "cpu_plugin" :
        plugin_id == "GPU" ? "gpu_plugin" : "npu_plugin";
    const std::filesystem::path plugin_path = plugin_directory
        / (std::string(kPrefix) + "openvino_intel_" + plugin_name + kSuffix);
#endif
    if (!std::filesystem::exists(plugin_path)) {
      if (plugin_id == "CPU") {
        registration_errors.push_back("CPU plugin not found at " + plugin_path.string());
      }
      continue;
    }
    try {
      core->register_plugin(plugin_path, plugin_id);
    } catch (const ov::Exception& error) {
      // One device plugin may be unusable while the other installed plugins work.
      registration_errors.push_back(plugin_id + ": " + error.what());
    }
  }
  return registration_errors;
}

}  // namespace

StatusOr<std::vector<DeviceInfo>> EnumerateDevices() {
  try {
    ov::Core core;
    std::vector<std::string> available_devices = core.get_available_devices();
    if (available_devices.empty()) {
#if defined(_WIN32)
      char* install_root_buffer = nullptr;
      std::size_t install_root_size = 0;
      if (_dupenv_s(&install_root_buffer, &install_root_size, "INTEL_OPENVINO_DIR") == 0
          && install_root_buffer != nullptr) {
        const std::filesystem::path plugin_directory = PluginDirectory(install_root_buffer);
        std::free(install_root_buffer);
        const std::vector<std::string> registration_errors =
            RegisterInstalledPlugins(&core, plugin_directory);
        available_devices = core.get_available_devices();
        if (available_devices.empty()) {
          std::ostringstream message;
          message << "OpenVINO registered no devices from " << plugin_directory.string();
          for (const std::string& registration_error : registration_errors) {
            message << "; " << registration_error;
          }
          return Status::Unavailable(message.str());
        }
      }
#else
      const char* install_root = std::getenv("INTEL_OPENVINO_DIR");
      if (install_root != nullptr && install_root[0] != '\0') {
        const std::filesystem::path plugin_directory = PluginDirectory(install_root);
        const std::vector<std::string> registration_errors =
            RegisterInstalledPlugins(&core, plugin_directory);
        available_devices = core.get_available_devices();
        if (available_devices.empty()) {
          std::ostringstream message;
          message << "OpenVINO registered no devices from " << plugin_directory.string();
          for (const std::string& registration_error : registration_errors) {
            message << "; " << registration_error;
          }
          return Status::Unavailable(message.str());
        }
      }
#endif
    }
    std::vector<DeviceInfo> devices;
    devices.reserve(available_devices.size());

    for (const std::string& device_id : available_devices) {
      std::string full_name = device_id;
      try {
        full_name = core.get_property(device_id, ov::device::full_name);
      } catch (const ov::Exception&) {
        // Keep the stable plugin/device id when a plugin lacks FULL_DEVICE_NAME.
      }
      devices.push_back({device_id, std::move(full_name), ClassifyDevice(device_id)});
    }
    return devices;
  } catch (const ov::Exception& error) {
    return Status::Unavailable(std::string("OpenVINO device enumeration failed: ") + error.what());
  } catch (const std::exception& error) {
    return Status::Internal(std::string("Unexpected device enumeration error: ") + error.what());
  }
}

}  // namespace isvik::openvino_backend
