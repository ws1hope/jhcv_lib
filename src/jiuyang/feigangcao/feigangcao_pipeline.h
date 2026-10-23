#pragma once

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "JHDeepCore.h"
#include "file_utils.h"
#include "infer_utils.h"

namespace JHDeepCore {
namespace Pipeline {

// 单图 pipeline 结果
struct FeigangcaoPipelineResult {
    bool detected = false;              // 定位模型是否检出目标
    cv::Rect roi;                       // 送入分类模型的定位框（原图坐标）
    float det_confidence = 0.0f;        // 定位框置信度
    std::string result;                 // 分类结果（class_name），无结果时为空
    float cls_confidence = 0.0f;        // 分类置信度
    cv::Mat det_crop;                   // 定位框裁剪图（送入分类模型的图像）
    bool out_of_roi = false;            // 检测框均不在当前工位有效检测范围内
    cv::Mat annotated_image;            // 画框 + 文字的结果图
    double inference_time_ms = 0.0;     // 纯模型推理耗时（定位 + 分类）
};

// 定位模型 -> 按有效检测范围筛选框后取置信度最高 -> 裁剪 -> 分类模型
class FeigangcaoPipeline {
public:
    explicit FeigangcaoPipeline(const FeigangcaoServerConfig& config);

    FeigangcaoPipelineResult process(const cv::Mat& image,
                                     const std::string& station_id,
                                     bool verbose = false);

private:
    void warmup();
    cv::Mat createAnnotatedImage(const cv::Mat& src_img,
                                 const FeigangcaoPipelineResult& result);

    std::unique_ptr<Detector> detector_;      // 定位模型
    std::unique_ptr<Classifier> classifier_;  // 分类模型
    FeigangcaoServerConfig config_;
    std::vector<cv::Point> active_roi_poly_;  // 当前工位生效的有效检测范围多边形（空则不过滤）
};

} // namespace Pipeline
} // namespace JHDeepCore
