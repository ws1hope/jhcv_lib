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

// 第二个检测模型的目标（bbox 已映射回原图坐标）
struct HuaxinDet2Target {
    cv::Rect bbox_on_src;     // 在原图上的 ROI
    std::string class_name;
    float confidence;
};

// 单图 pipeline 结果
struct HuaxinPipelineResult {
    std::vector<Detection> det1_detections;   // 第一个检测模型的全部输出框
    int leftmost_index = -1;                  // 送入第二个模型的框在 det1_detections 中的下标
    cv::Rect leftmost_roi;                    // 送入第二个模型的框在原图上的 ROI
    std::vector<HuaxinDet2Target> det2_targets; // 第二个检测模型的输出（已映射回原图）
    std::string all_results;                  // 铸坯端面识别结果（det2 类别名从左到右拼接）
    cv::Mat annotated_image;                  // 画框 + 文字的结果图
};

// 目标检测 -> 最左框裁剪 -> 第二个目标检测
class HuaxinPipeline {
public:
    explicit HuaxinPipeline(const HuaxinServerConfig& config);

    HuaxinPipelineResult process(const cv::Mat& image, bool verbose = false);

private:
    void warmup();
    cv::Mat createAnnotatedImage(
        const cv::Mat& src_img,
        const HuaxinPipelineResult& result);

    std::unique_ptr<Detector> det1_;  // 第一个检测模型（定位）
    std::unique_ptr<Detector> det2_;  // 第二个检测模型（端面识别）
    HuaxinServerConfig config_;
};

} // namespace Pipeline
} // namespace JHDeepCore
