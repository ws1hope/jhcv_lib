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
    float ocr_confidence = 0.f;  // 该字符 OCR 置信度（无 box 时为 0）
    int direction_flag = 0;     // 该字符角度分类结果（0 或 180）
};

struct BilletResult {
    cv::Rect bbox_on_src;
    std::string class_name;
    float confidence;
    int direction_flag = 0;   // 0 或 180
    cv::Mat billet_image;      // 坯料裁剪图（用于结果图可视化缩略图）
    std::vector<BilletCharInfo> chars;
    std::string ocr_text;
    float ocr_confidence = 0.f;  // 该坯料各已识别字符 OCR 置信度的最小值
};

struct ZbhcPipelineResult {
    std::vector<Detection> det1_detections;
    std::vector<BilletResult> billets;
    cv::Mat annotated_image;
};

class ZbhcPipeline {
public:
    explicit ZbhcPipeline(const ZbhcServerConfig& config);

    ZbhcPipelineResult process(const cv::Mat& image, bool verbose = false,
                               const std::string& heat_number = "");

private:
    void warmup();

    // pred_out/conf_out: 输出模型原始预测类别(0/180)及置信度，供调试打印
    int classifyDirection(const cv::Mat& char_image,
                          int* pred_out = nullptr,
                          float* conf_out = nullptr);

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
