#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include "file_utils.h"
#include "infer_utils.h"

namespace JHDeepCore {
namespace Pipeline {

struct BilletCharInfo {
    cv::Rect bbox_on_billet;
    cv::Rect bbox_on_src;
    cv::Mat mask;
    std::string ocr_text;
};

struct BilletResult {
    cv::Rect bbox_on_src;
    std::string class_name;
    float confidence;
    std::vector<BilletCharInfo> chars;
    std::string ocr_text;
};

struct ZbhcPipelineResult {
    std::vector<Detection> det1_detections;
    std::vector<BilletResult> billets;
    cv::Mat annotated_image;
};

class ZbhcPipeline {
public:
    explicit ZbhcPipeline(const ZbhcServerConfig& config);

    ZbhcPipelineResult process(const cv::Mat& image, bool verbose = false);

private:
    void warmup();

    cv::Mat createAnnotatedImage(
        const cv::Mat& src_img,
        const std::vector<Detection>& det1_dets,
        const std::vector<BilletResult>& billets);

    std::unique_ptr<Detector> det1_;
    std::unique_ptr<Detector> det2_;
    std::unique_ptr<InstanceSegmenter> seg_;
    std::unique_ptr<OCRRecognizer> ocr_;
    ZbhcServerConfig config_;
};

} // namespace Pipeline
} // namespace JHDeepCore
