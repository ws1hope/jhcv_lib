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
    cv::Mat image_before_flip;   // 方向矫正前（原始字符 crop）
    cv::Mat image_after_flip;    // 方向矫正后（实际送 OCR 的字符 crop）
    std::string ocr_text;
};

struct BilletResult {
    cv::Rect bbox_on_src;
    std::string class_name;
    float confidence;
    int direction_flag = 0;   // 0 或 180
    cv::Mat billet_image;      // 坯料裁剪图（用于结果图可视化缩略图）
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

    int classifyDirection(const std::vector<cv::Mat>& char_images);

    cv::Mat createAnnotatedImage(
        const cv::Mat& src_img,
        const std::vector<Detection>& det1_dets,
        const std::vector<BilletResult>& billets);

    std::unique_ptr<Detector> det1_;
    std::unique_ptr<Detector> det2_;
    std::unique_ptr<InstanceSegmenter> seg_;
    std::unique_ptr<OCRRecognizer> ocr_;
    std::unique_ptr<Classifier> direction_cls_;
    ZbhcServerConfig config_;
};

} // namespace Pipeline
} // namespace JHDeepCore
