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

struct LabelDisplayInfo {
    cv::Mat rotated_image;
    std::vector<CharCropInfo> char_crops;
    std::vector<std::string> ocr_texts;
    std::string label_type;
    int direction_flag = 0;
    std::string matched_luhao;
    // 方向矫正前的字符片段 (从 rotated_image 裁出的原始 crop)
    std::vector<cv::Mat> char_images_before_flip;
    // 方向矫正后的字符片段 (flip 180 后的 crop，即实际送 OCR 的图像)
    std::vector<cv::Mat> char_images_after_flip;
};

class TiebiaoPipeline {
public:
    explicit TiebiaoPipeline(const TiebiaoConfig& config);

    // 独立运行：用 label_seg_ 自己切 label
    TiebiaoResult process(const cv::Mat& image,
                          int station_id,
                          const std::string& heat_number,
                          bool verbose = false);

    // gx gangbiao 分支复用：跳过 label_seg_，直接用 gx 已切好的 gangbiao bbox
    // 作为 label ROI（每个 bbox 当作一个 circle 标签牌）
    TiebiaoResult process(const cv::Mat& image,
                          int station_id,
                          const std::string& heat_number,
                          const std::vector<cv::Rect>& gangbiao_bboxes,
                          bool verbose = false);

private:
    std::vector<std::pair<cv::Mat, int>> detectLabels(const cv::Mat& image);

    // 共享核心：给定 label ROI 列表 (crop, class_id)，跑字符分割/方向/OCR/炉号匹配
    TiebiaoResult runLabels(const cv::Mat& image,
                            const std::vector<std::pair<cv::Mat, int>>& labels,
                            const std::string& heat_number,
                            bool verbose,
                            TiebiaoResult& result);

    void detectAndRotateChars(const cv::Mat& label_roi,
                               cv::Mat& rotated_image,
                               std::vector<CharCropInfo>& char_crops);

    int classifyDirection(const std::vector<cv::Mat>& char_images);

    std::vector<std::string> recognizeChars(const std::vector<CharCropInfo>& crops);

    cv::Mat createAnnotatedImage(const cv::Mat& src_img,
                                  const std::vector<LabelDisplayInfo>& labels);

    void warmup();

    std::unique_ptr<InstanceSegmenter> label_seg_;
    std::unique_ptr<InstanceSegmenter> char_seg_;
    std::unique_ptr<OCRRecognizer> ocr_;
    std::unique_ptr<Classifier> direction_cls_;
    TiebiaoConfig config_;
};

} // namespace Pipeline
} // namespace JHDeepCore
