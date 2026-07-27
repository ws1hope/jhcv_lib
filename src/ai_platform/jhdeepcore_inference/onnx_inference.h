#pragma once

#include "jhdeepcore_inference/base_inference.h"
#include <memory>
#include <vector>

#ifdef ONNXRUNTIME_FOUND
#include <onnxruntime_cxx_api.h>
#endif

namespace JHDeepCore {
namespace inference {

class OnnxInference : public BaseInference {
  public:
    OnnxInference(const std::string &model_path, const std::string &device = "cpu",
                  const std::vector<std::string> &class_names = {}, bool warmup = true);

    ~OnnxInference() override;

    bool LoadModel() override;

    ClassificationResult InferSingle(const cv::Mat &image);
    std::vector<ClassificationResult> InferBatch(const std::vector<cv::Mat> &images);

    SegmentationResult InferSingleSegmentation(const cv::Mat &image);
    std::vector<SegmentationResult> InferBatchSegmentation(const std::vector<cv::Mat> &images);

    DetectionResult InferSingleDetection(const cv::Mat &image);
    std::vector<DetectionResult> InferBatchDetection(const std::vector<cv::Mat> &images);

    InstanceSegmentationResult InferSingleInstanceSegmentation(const cv::Mat &image);
    std::vector<InstanceSegmentationResult> InferBatchInstanceSegmentation(const std::vector<cv::Mat> &images);

    void WarmupModel(int iterations = 5);

    InferenceTiming lastBatchTiming() const override { return batch_timing_; }

  private:
#ifdef ONNXRUNTIME_FOUND
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;

    std::string input_name_;
    std::vector<int64_t> input_shape_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> output_shapes_;
#endif

    bool model_loaded_;
    bool warmup_enabled_;

    // 最近一次 InferBatch* 的分段耗时（每次 InferBatch* 起点复位，InferSingle* 累加）
    InferenceTiming batch_timing_;

    // 预分配缓冲区，避免每次推理重复申请释放
    std::vector<float> input_buffer_;
    std::vector<const char *> output_names_cstr_;
    std::vector<const char *> input_names_cstr_;

    void PreprocessForOnnx(const cv::Mat &image);
    std::vector<float> RunInference(const std::vector<float> &input_data);
};

} // namespace inference
} // namespace JHDeepCore
