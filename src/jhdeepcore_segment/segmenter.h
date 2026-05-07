#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include "JHDeepCore.h"

namespace JHDeepCore {
namespace segment {

class SegmenterImpl {
  public:
    SegmenterImpl(const std::string &model_path, const std::string &device = "cpu",
                  const std::vector<std::string> &class_names = {});

    ~SegmenterImpl();

    JHDeepCore::SegmentationResult SegmentSingle(const cv::Mat &image);
    std::vector<JHDeepCore::SegmentationResult> SegmentBatch(const std::vector<cv::Mat> &images);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace segment
} // namespace JHDeepCore
