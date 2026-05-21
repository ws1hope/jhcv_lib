#include "JHDeepCore.h"
#include "jhdeepcore_utils/device_utils.h"

namespace JHDeepCore {

std::string GetOptimalDevice() {
    return utils::DeviceUtils::GetOptimalDevice();
}

} // namespace JHDeepCore
