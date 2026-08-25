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

// 单个字符实例的分割框 + OCR 结果
struct FujianCharResult {
    cv::Rect bbox_on_src;     // 字符实例框在原图上的 ROI（已映射回原图坐标）
    std::string class_name;
    float confidence;
    std::string ocr_text;     // 该字符实例送 OCR 的识别结果
    cv::Mat image;            // 送 OCR 的字符裁剪图（空则不保存）
};

// 单图 pipeline 结果
struct FujianPipelineResult {
    int roi_index = 0;                        // 使用的 ROI 编号（1-5）
    cv::Rect source_roi;                      // 配置读取的原始 ROI（未裁剪）
    cv::Rect used_roi;                        // 本次使用的 ROI（原图坐标）
    cv::Mat roi_image;                        // ROI 裁剪图（送分割的输入）
    std::vector<FujianCharResult> chars;      // 字符实例按 Z 型排列 + OCR 结果
    std::string full_text;                    // 逐字符 OCR 拼接
    cv::Mat annotated_image;                  // ROI 框 + 字符框 + OCR 文字的结果图
};

// ROI 固定区域裁剪 -> 实例分割检出字符实例 -> Z 型排序 -> 逐字符裁剪送 OCR -> 拼接
class FujianPipeline {
public:
    FujianPipeline(const FujianStationConfig& station_config, const std::string& device);

    // heat_number：用户输入炉号。分割恰好检出 2 个框且宽度差 < 30px 时，
    // 跳过 OCR，直接以 heat_number 作为识别结果
    FujianPipelineResult process(const cv::Mat& image, const cv::Rect& roi,
                                 int roi_index, const std::string& heat_number,
                                 bool verbose = false);

private:
    void warmup();
    cv::Mat createAnnotatedImage(
        const cv::Mat& src_img,
        const FujianPipelineResult& result);

    std::unique_ptr<InstanceSegmenter> seg_;
    std::unique_ptr<OCRRecognizer> ocr_;
    FujianStationConfig station_config_;
    std::string device_;
};

} // namespace Pipeline
} // namespace JHDeepCore
