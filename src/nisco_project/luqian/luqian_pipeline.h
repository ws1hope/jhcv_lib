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

struct LuqianCharInfo {
    cv::Rect bbox_on_target;
    cv::Rect bbox_on_src;
    cv::Mat mask;
    cv::Mat image_before_flip;   // 方向矫正前（原始字符 crop）
    cv::Mat image_after_flip;    // 方向矫正后（实际送 OCR 的字符 crop）
    std::string ocr_text;
    float ocr_confidence = 0.f;  // 该字符 OCR 置信度（无 box 时为 0）
    int direction_flag = 0;     // 该字符角度分类结果（0 或 180）
};

struct LuqianTargetResult {
    cv::Rect bbox_on_src;
    std::string class_name;
    float confidence;
    int direction_flag = 0;   // 0 或 180
    cv::Mat target_image;     // 目标裁剪图（用于结果图可视化缩略图）
    std::vector<LuqianCharInfo> chars;
    std::string ocr_text;
    std::string ocr1_text;   // 第一次识别(模型1)完整拼接结果
    std::string ocr2_text;   // 第二次识别(模型2)完整拼接结果（触发二次识别时填充）
    float ocr_confidence = 0.f;  // 该目标各已识别字符 OCR 置信度的最小值
};

struct LuqianPipelineResult {
    std::vector<Detection> det_detections;
    std::vector<LuqianTargetResult> targets;
    cv::Mat annotated_image;
};

class LuqianPipeline {
public:
    explicit LuqianPipeline(const LuqianServerConfig& config);

    LuqianPipelineResult process(const cv::Mat& image, bool verbose = false,
                                 const std::string& heat_number = "");

private:
    void warmup();

    cv::Mat createAnnotatedImage(
        const cv::Mat& src_img,
        const std::vector<Detection>& det_dets,
        const std::vector<LuqianTargetResult>& targets);

    std::unique_ptr<Detector> det_;
    std::unique_ptr<InstanceSegmenter> seg_;
    std::unique_ptr<OCRRecognizer> ocr_;
    std::unique_ptr<OCRRecognizer> ocr2_;   // 二次识别 OCR（可选，为空不启用）
    std::unique_ptr<Classifier> direction_cls_;
    LuqianServerConfig config_;
};

} // namespace Pipeline
} // namespace JHDeepCore
