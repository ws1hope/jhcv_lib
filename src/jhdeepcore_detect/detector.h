#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include "JHDeepCore.h"

namespace JHDeepCore {
namespace detect {

class DetectorImpl {
  public:
    DetectorImpl(const std::string &model_path, const std::string &device = "cpu",
                 const std::vector<std::string> &class_names = {});

    ~DetectorImpl();

    JHDeepCore::DetectionResult DetectSingle(const cv::Mat &image);
    std::vector<JHDeepCore::DetectionResult> DetectBatch(const std::vector<cv::Mat> &images);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace detect
} // namespace JHDeepCore
