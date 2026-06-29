#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "JHDeepCore.h"
#include "file_utils.h"
#include "image_utils.h"
#include "infer_utils.h"
#include "tiebiao_pipeline.h"

namespace JHDeepCore {
namespace Pipeline {

// 单个字符片段（zifu 分支专用，模型存在的展示信息）
struct GxJingzhengCharInfo {
    cv::Rect bbox;            // 字符在旋转后图像上的 bbox
    cv::Mat image_before_flip; // 方向矫正前（旋转后裁出的原始片段）
    cv::Mat image_after_flip;  // 方向矫正后（送 OCR 的片段）
    std::string ocr_text;
};

struct GxJingzhengPipelineResult {
    std::string state_flag;       // "OK" / "NG"
    std::string branch;           // "zifu" / "gangbiao" / ""
    std::string ocr_text;
    cv::Mat annotated_image;

    // det 第一阶段结果（用于可视化）
    std::vector<Detection> det_detections;
    cv::Rect chosen_bbox;         // 选中的最左 det 框

    // zifu 分支展示信息
    cv::Mat rotated_crop;
    int direction_flag = 0;       // 0 / 180
    std::vector<GxJingzhengCharInfo> chars;

    // gangbiao 分支：直接复用 tiebiao 的标注图
    cv::Mat tiebiao_annotated;
};

class GxJingzhengPipeline {
public:
    explicit GxJingzhengPipeline(const GxJingzhengServerConfig& config);

    GxJingzhengPipelineResult process(const cv::Mat& image,
                                      int station_id,
                                      const std::string& heat_number,
                                      bool verbose = false);

private:
    void warmup();

    // 在 crop 上跑语义分割，统计前景像素决定分支
    std::string decideBranch(const cv::Mat& crop, cv::Mat& seg_mask_full_size);

    // zifu 分支：从 zifu 二值掩码切连通域 → 计算字符角度 → 整体矫正 → 排版
    bool handleZifuBranch(const cv::Mat& crop,
                          const cv::Mat& zifu_binary,
                          GxJingzhengPipelineResult& result,
                          bool verbose);

    // 方向分类（按字符片段多数投票）
    int classifyDirection(const std::vector<cv::Mat>& char_images);

    cv::Mat createAnnotatedImage(const cv::Mat& src_img,
                                  const GxJingzhengPipelineResult& result);

    GxJingzhengServerConfig config_;
    std::vector<std::string> seg_class_names_;   // 来自 seg_label
    int zifu_class_id_ = -1;
    int gangbiao_class_id_ = -1;

    std::unique_ptr<Detector> det_;
    std::unique_ptr<Segmenter> seg_;
    std::unique_ptr<Classifier> direction_cls_;
    std::unique_ptr<OCRRecognizer> ocr_;

    // gangbiao 分支复用
    std::unique_ptr<TiebiaoPipeline> tiebiao_pipeline_;
};

} // namespace Pipeline
} // namespace JHDeepCore
