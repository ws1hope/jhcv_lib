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

// 单个 det2 检测框 + 其 OCR 识别结果
struct GuokuacheTargetResult {
    cv::Rect bbox_on_src;     // det2 检测框在原图上的 ROI（已映射回原图坐标）
    std::string class_name;
    float confidence;
    std::string ocr_text;     // 该框内图像送 OCR 的识别结果
};

// 单图 pipeline 结果
struct GuokuachePipelineResult {
    std::vector<Detection> det1_detections;   // 第一个检测模型全部输出
    int selected_index = -1;                  // 送入第二个模型的 det1 框下标（最左框）
    cv::Rect selected_roi;                    // 该框在原图上的 ROI
    std::vector<GuokuacheTargetResult> targets;  // det2 框按 Z 型排列 + OCR 结果
    cv::Mat annotated_image;                  // 画框 + OCR 文字的结果图
};

// 目标检测(最左框) -> 框内送第二个检测模型 -> 框按 Z 型排序 -> 逐框送 OCR
class GuokuachePipeline {
public:
    explicit GuokuachePipeline(const GuokuacheServerConfig& config);

    GuokuachePipelineResult process(const cv::Mat& image, bool verbose = false);

private:
    void warmup();
    cv::Mat createAnnotatedImage(
        const cv::Mat& src_img,
        const GuokuachePipelineResult& result);

    std::unique_ptr<Detector> det1_;
    std::unique_ptr<Detector> det2_;
    std::unique_ptr<OCRRecognizer> ocr_;
    GuokuacheServerConfig config_;
};

} // namespace Pipeline
} // namespace JHDeepCore
