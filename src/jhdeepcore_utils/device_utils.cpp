#include "jhdeepcore_utils/device_utils.h"
#include <algorithm>
#include <cstring>

#ifdef ONNXRUNTIME_FOUND
#include <onnxruntime_cxx_api.h>
#endif

namespace JHDeepCore {
namespace utils {

bool DeviceUtils::IsCudaAvailable() {
#ifdef ONNXRUNTIME_FOUND
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DeviceCheck");
        Ort::SessionOptions session_options;

        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        for (const std::string &provider : available_providers) {
            if (provider == "CUDAExecutionProvider") {
                return true;
            }
        }
    } catch (...) {
        return false;
    }
#endif
    return false;
}

std::string DeviceUtils::GetOptimalDevice() {
    if (IsCudaAvailable()) {
        return "cuda";
    }
    return "cpu";
}

bool DeviceUtils::IsValidDevice(const std::string &device) {
    std::string normalized = NormalizeDevice(device);
    return normalized == "cpu" || normalized == "cuda";
}

std::string DeviceUtils::NormalizeDevice(const std::string &device) {
    std::string normalized = device;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

    if (normalized == "gpu" || normalized.find("cuda") != std::string::npos) {
        return "cuda";
    }
    if (normalized == "cpu") {
        return "cpu";
    }

    return normalized;
}

} // namespace utils
} // namespace JHDeepCore
