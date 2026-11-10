#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include "image_utils.h"
#include "file_utils.h"
#include "infer_utils.h"

namespace JHDeepCore {
namespace Pipeline {

// 断面内单个矫正后字符片段 + OCR 结果
struct TiangangfujianCharResult {
    cv::Rect bbox_on_rotated;      // 字符框在矫正图上的 ROI
    cv::Mat image_before_flip;     // 方向矫正前（旋转矫正后的原始字符 crop）
    cv::Mat image_after_flip;      // 方向矫正后（实际送 OCR 的字符 crop）
    int direction_flag = 0;        // 所属断面方向投票结果（0 或 180）
    std::string ocr_text;          // 该字符识别结果
};

// 单个断面结果
struct TiangangfujianSectionResult {
    cv::Rect bbox_on_src;          // 断面框在原图上的 ROI（已含 ROI 区域偏移）
    int roi_index = 0;             // 所属全局 ROI 编号（1-N，未配置 ROI 时为 0）
    std::string class_name;
    float confidence = 0.f;
    int class_id = 0;
    cv::Mat section_image;         // 断面裁剪图（送字符分割的输入）
    cv::Mat rotated_image;         // 角度矫正后的断面图
    int direction_flag = 0;        // 断面方向投票结果（0 或 180）
    std::vector<TiangangfujianCharResult> chars;  // 按矫正后 y 从小到大排序
    std::string ocr_text;          // 各字符 OCR 结果按顺序拼接
};

// 单图 pipeline 结果
struct TiangangfujianPipelineResult {
    std::vector<TiangangfujianSectionResult> sections;  // 断面按原图 x 从左到右排序
    std::string all_results;       // 各断面 OCR 文本按顺序拼接
    cv::Mat annotated_image;       // 断面框 + OCR 文本的结果图
};

// 目标检测定位断面 -> 逐断面字符实例分割+角度矫正 -> 方向分类矫正为 0 度 -> 逐字符 OCR
class TiangangfujianPipeline {
public:
    explicit TiangangfujianPipeline(const TiangangfujianServerConfig& config);

    TiangangfujianPipelineResult process(const cv::Mat& image, bool verbose = false);

private:
    // 单个区域内检测断面并逐断面处理
    // region_offset：该区域在原图上的偏移，用于把断面框映射回原图坐标；
    // roi_index：全局 ROI 编号（1-N），未配置 ROI 时传 0
    void processRegion(const cv::Mat& region_img,
                       const cv::Point& region_offset,
                       int roi_index,
                       bool verbose,
                       std::vector<TiangangfujianSectionResult>& sections);

    // 字符实例分割 + 断面角度矫正（同 tiebiao detectAndRotateChars）
    void detectAndRotateChars(const cv::Mat& section_roi,
                              cv::Mat& rotated_image,
                              std::vector<CharCropInfo>& char_crops);

    // 断面内全部字符投票判向（同 tiebiao classifyDirection）
    int classifyDirection(const std::vector<cv::Mat>& char_images);

    void warmup();
    cv::Mat createAnnotatedImage(
        const cv::Mat& src_img,
        const std::vector<TiangangfujianSectionResult>& sections);

    std::unique_ptr<Detector> det_;                  // 断面定位（目标检测）
    std::unique_ptr<InstanceSegmenter> char_seg_;    // 字符实例分割
    std::unique_ptr<Classifier> direction_cls_;      // 字符方向分类
    std::unique_ptr<OCRRecognizer> ocr_;
    TiangangfujianServerConfig config_;
};

} // namespace Pipeline
} // namespace JHDeepCore
