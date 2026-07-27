#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "JHDeepCore.h"
#include "jhdeepcore_utils/config_loader.h"

namespace JHDeepCore {
namespace inference {

class BaseInference {
  public:
    BaseInference(const std::string &model_path, const std::string &device = "cpu",
                  const std::vector<std::string> &class_names = {});

    virtual ~BaseInference() = default;

    virtual bool LoadModel() = 0;

    cv::Mat PreprocessImageCommon(const cv::Mat &image);
    cv::Mat PreprocessImageDetection(const cv::Mat &image);

    ClassificationResult ProcessClassificationOutput(const std::vector<float> &output);
    SegmentationResult ProcessSegmentationOutput(const std::vector<float> &output,
                                                 const std::vector<int64_t> &output_shape);
    DetectionResult ProcessDetectionOutput(const std::vector<float> &output, const std::vector<int64_t> &output_shape);
    InstanceSegmentationResult ProcessInstanceSegmentationOutput(
        const std::vector<float> &detection_output, const std::vector<int64_t> &detection_output_shape,
        const std::vector<float> &protos_output, const std::vector<int64_t> &protos_output_shape);
    std::vector<cv::Mat> ProcessInstanceMasks(const std::vector<float> &protos,
                                                const std::vector<std::vector<float>> &masks_in,
                                                const std::vector<cv::Rect> &bboxes, int mask_dim, int mask_h,
                                                int mask_w);

    std::string GetDevice() const { return device_; }
    std::string GetModelPath() const { return model_path_; }
    utils::ModelConfig GetConfig() const { return config_; }
    void SetThresholds(float conf, float iou) { conf_threshold_ = conf; iou_threshold_ = iou; }

    // 返回最近一次 InferBatch* 的分段耗时统计（preprocess/tensor/run + device）
    // 默认返回零；子类（OnnxInference）覆盖以提供真实数据
    virtual InferenceTiming lastBatchTiming() const { return InferenceTiming{}; }

  protected:
    std::string model_path_;
    std::string device_;
    std::vector<std::string> class_names_;
    utils::ModelConfig config_;

    cv::Size original_image_size_;

    float conf_threshold_;
    float iou_threshold_;
    cv::Point2i letterbox_pad_;
    float letterbox_gain_;

    cv::Mat Letterbox(const cv::Mat &img, const cv::Size &new_shape, cv::Point2i &pad, float &gain);
    cv::Mat ResizeMaskToOriginal(const cv::Mat &mask, const cv::Size &original_size);
    std::string GetClassName(int class_id) const;
    std::vector<float> Softmax(const std::vector<float> &logits);
};

} // namespace inference
} // namespace JHDeepCore
