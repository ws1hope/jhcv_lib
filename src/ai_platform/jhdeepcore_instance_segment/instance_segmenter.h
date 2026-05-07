#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include "JHDeepCore.h"

namespace JHDeepCore {
namespace instance_segment {

class InstanceSegmenterImpl {
  public:
    InstanceSegmenterImpl(const std::string &model_path, const std::string &device = "cpu",
                         const std::vector<std::string> &class_names = {});

    ~InstanceSegmenterImpl();

    JHDeepCore::InstanceSegmentationResult SegmentSingle(const cv::Mat &image);
    std::vector<JHDeepCore::InstanceSegmentationResult> SegmentBatch(const std::vector<cv::Mat> &images);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace instance_segment
} // namespace JHDeepCore
