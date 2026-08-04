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
    Ort::MemoryInfo memory_info_;        // CPU allocator 描述（非 split 路径建 CPU tensor）
    Ort::MemoryInfo cuda_mem_info_;      // Cuda allocator 描述（split 路径建 GPU tensor）
    // 按 input_on_gpu_（由 prepareInput 设置）返 cuda_mem_info_ 或 memory_info_，保证 ptr 与 MemoryInfo 一致。
    const Ort::MemoryInfo &inputMemInfo() const;

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

    // device=="cuda" && JHDEEP_H2D_SPLIT==1：输入走真实 GPU tensor 路径（H2D/run/D2H 三段
    // 分开 cudaEvent 计时）。否则走原 CPU tensor 路径（Run 内部自行 H2D/D2H，不拆分）。
    bool useGpuTensor() const;
    // split 路径：真实 cudaMemcpyAsync(H2D)+cudaEvent 拷进 cuda_input_，累计 h2d_ms，返回 GPU 指针；
    // 非 split：原样返回（const_cast）CPU 指针，供建 CPU tensor。
    float *prepareInput(const float *data, size_t count);
    // split 路径：真实 cudaMemcpyAsync(D2H)+cudaEvent 把 GPU 输出拷进 dst，累计 d2h_ms；
    // 非 split：src 为 CPU 指针，直接 std::copy 进 dst。
    void readOutput(const float *src, size_t count, std::vector<float> &dst);
    // 按需分配/扩容 cuda_input_（仅 USE_CUDA 下实际分配）
    void ensureCudaInput(size_t float_count);

    float *cuda_input_ = nullptr;   // GPU 输入缓冲（split 路径作为 Run 的真实输入）
    size_t cuda_input_count_ = 0;
    // 由 prepareInput 设置、inputMemInfo/readOutput 读取：本次输入是否实际落在 GPU。
    // cudaMalloc 失败时 prepareInput 回退 CPU，此标志为 false，保证 ptr 与 MemoryInfo 一致。
    bool input_on_gpu_ = false;
};

} // namespace inference
} // namespace JHDeepCore
