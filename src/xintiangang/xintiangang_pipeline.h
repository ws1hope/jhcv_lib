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

// 单个检测目标 + 其 OCR 识别结果
struct XintiangangTargetResult {
    cv::Rect bbox_on_src;     // 检测框在原图上的 ROI
    std::string class_name;
    float confidence;
    std::string ocr_text;     // 该框内图像送 OCR 的识别结果
};

// 单图 pipeline 结果
struct XintiangangPipelineResult {
    std::vector<Detection> detections;
    std::vector<XintiangangTargetResult> targets;
    std::string all_results;       // 所有目标 OCR 结果按从左到右拼接
    cv::Mat annotated_image;       // 画框 + OCR 文字的结果图
};

// 目标检测 -> 框内图像送 OCR
class XintiangangPipeline {
public:
    explicit XintiangangPipeline(const XintiangangServerConfig& config);

    XintiangangPipelineResult process(const cv::Mat& image, bool verbose = false);

private:
    void warmup();
    cv::Mat createAnnotatedImage(
        const cv::Mat& src_img,
        const std::vector<XintiangangTargetResult>& targets);

    std::unique_ptr<Detector> det_;
    std::unique_ptr<OCRRecognizer> ocr_;
    XintiangangServerConfig config_;
};

} // namespace Pipeline
} // namespace JHDeepCore
