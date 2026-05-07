#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include "JHDeepCore.h"

namespace JHDeepCore {
namespace ocr {

class OCRRecognizerImpl {
  public:
    struct Params {
        std::string rec_model_path;
        std::string rec_label_path;
        std::string device = "cpu";
        float rec_score_thresh = 0.5f;
        bool useGPU = true;
        int gpuId = 0;
    };

    explicit OCRRecognizerImpl(const Params &params);
    ~OCRRecognizerImpl();

    JHDeepCore::OCRResult Recognize(const cv::Mat &text_image);
    JHDeepCore::OCRResult Recognize(const std::string &image_path);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace ocr
} // namespace JHDeepCore
