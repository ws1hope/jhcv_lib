#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include "JHDeepCore.h"

namespace JHDeepCore {
namespace classify {

class ClassifierImpl {
  public:
    ClassifierImpl(const std::string &model_path, const std::string &device = "cpu",
                   const std::vector<std::string> &class_names = {});

    ~ClassifierImpl();

    JHDeepCore::ClassificationResult ClassifySingle(const cv::Mat &image);
    std::vector<JHDeepCore::ClassificationResult> ClassifyBatch(const std::vector<cv::Mat> &images);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace classify
} // namespace JHDeepCore
