#pragma once

#include <string>

namespace JHDeepCore {
namespace utils {

class DeviceUtils {
  public:
    static std::string GetOptimalDevice();
    static bool IsCudaAvailable();
    static bool IsValidDevice(const std::string &device);
    static std::string NormalizeDevice(const std::string &device);
};

} // namespace utils
} // namespace JHDeepCore
