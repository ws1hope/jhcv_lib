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

// 实例分割的一条结果（crop 坐标系下）
struct GxJingzhengSegInstance {
    std::string class_name;
    cv::Rect bbox;
    cv::Mat mask;            // 与 crop 同尺寸的二值 mask (CV_8UC1)
    float confidence = 0.f;
};

struct GxJingzhengPipelineResult {
    std::string state_flag;       // "OK" / "NG"
    std::string branch;           // "zifu" / "gangbiao" / ""
    std::string zifu_type;        // zifu 分支->"Penma", gangbiao 分支->"Tiebiao"
    std::string penma_version;    // 固定 "new"
    std::string duanmian;         // 第一阶段 det 有输出 -> "yes" / "no"
    std::string ocr_text;
    cv::Mat annotated_image;

    // det 第一阶段结果（用于可视化）
    std::vector<Detection> det_detections;
    cv::Rect chosen_bbox;         // 选中的最左 det 框

    // 实例分割结果（crop 坐标系），绘图时每个实例独立上色
    std::vector<GxJingzhengSegInstance> seg_instances;

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

    // 在 crop 上跑实例分割，把结果转为 crop 坐标系的实例列表，并按类别像素数选择分支
    std::string decideBranch(const cv::Mat& crop,
                              std::vector<GxJingzhengSegInstance>& instances_out);

    // zifu 分支：对每个 zifu 实例 mask 各自算最小外接矩 → 透视裁剪 → 方向分类 + OCR，
    // 再依据 heat_number 排列片段顺序拼接结果
    bool handleZifuBranch(const cv::Mat& crop,
                          const std::vector<GxJingzhengSegInstance>& zifu_instances,
                          GxJingzhengPipelineResult& result,
                          const std::string& heat_number,
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
    std::unique_ptr<InstanceSegmenter> seg_;
    std::unique_ptr<Classifier> direction_cls_;
    std::unique_ptr<OCRRecognizer> ocr_;

    // gangbiao 分支复用
    std::unique_ptr<TiebiaoPipeline> tiebiao_pipeline_;
};

} // namespace Pipeline
} // namespace JHDeepCore
